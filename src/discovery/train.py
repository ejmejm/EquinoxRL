from functools import partial
import math
import tempfile
import time
from typing import Callable, NamedTuple, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import omegaconf
from omegaconf import DictConfig
import optax
from tqdm import tqdm

from envs.envs import create_env
from envs.equinox_env import EquinoxEnv
from models import ActorCriticModel, get_feature_extractor_cls, get_activation_fn
from training import TrainState, TrajectoryData, apply_grads
from utils import tree_replace


# jax.config.update("jax_debug_nans", True)


def create_model(
        key: PRNGKeyArray,
        obs_dim: Tuple[int],
        n_actions: int,
        model_config: DictConfig,
        half_precision: bool = False,
    ) -> eqx.Module:
    """Create a model from a config."""

    keys = jax.random.split(key, 2)

    feature_extractor_cls = get_feature_extractor_cls(model_config.feature_extractor.type)
    feature_extractor = feature_extractor_cls(
        key = keys[0],
        input_dim = obs_dim,
        output_dim = model_config.feature_extractor.embedding_dim,
        activation_fn = get_activation_fn(model_config.feature_extractor.activation),
    )

    ac_config = model_config.actor_critic
    model = ActorCriticModel(
        key = keys[1],
        feature_extractor = feature_extractor,
        action_dim = n_actions,
        actor_layer_sizes = ac_config.actor_layer_sizes,
        critic_layer_sizes = ac_config.critic_layer_sizes,
        activation_fn = get_activation_fn(ac_config.activation),
    )

    if half_precision:
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model)

    return model


def create_optimizer(
    model: eqx.Module, optimizer_config: DictConfig,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optimizer = optax.adam(optimizer_config.learning_rate)
    opt_state = optimizer.init(model)
    return optimizer, opt_state


# Gather data
# Calculate losses
# Calculate gradients
# Update model


def gather_data(
        key: PRNGKeyArray,
        env_state_and_obs: Tuple[PyTree, PyTree],
        env_step_fn: Callable,
        model: eqx.Module,
        forward_fn: Callable[[eqx.Module, ArrayLike], Tuple[ArrayLike, ArrayLike]],
        rollout_length: int,
        half_precision: bool = False,
    ) -> Tuple[Tuple[PyTree, PyTree], TrajectoryData]:
    def predict_and_step_fn(state, _):
        env_state, obs = state
            
        act_logits, value = forward_fn(model, obs)
        action = jax.random.categorical(key, act_logits, axis=-1)
        
        env_state, new_obs, reward, done, info = env_step_fn(env_state, action)
        print(new_obs.shape, reward, done, info)
        if half_precision:
            new_obs = new_obs.astype(jnp.bfloat16)
            reward = reward.astype(jnp.bfloat16)
        
        return (env_state, new_obs), TrajectoryData(obs, action, new_obs, reward, done, value, info)
    
    # TODO: Add support for non-jittable env step fn with a for loop
    env_state_and_obs, train_sequences = jax.lax.scan(
        predict_and_step_fn, env_state_and_obs, length=rollout_length)

    return env_state_and_obs, train_sequences


# def gen_summary_metrics(
#         metrics: Dict[str, float],
#         prefix: str = '',
#     ) -> Dict[str, float]:
#     return {f'{prefix}_{k}': v for k, v in metrics.items()}


@partial(jax.jit, static_argnums=(3,))
def batch_train_iter(
        train_state: TrainState,
        model: eqx.Module,
        trajectory_data: TrajectoryData,
        train_config: DictConfig,
    ):
    # Trajectory data element shapes: (rollout_length, n_envs, ...) -> (n_envs, rollout_length, ...)
    trajectory_data = jax.tree.map(jnp.transpose, trajectory_data)

    # policy_grads, policy_metrics = train_reinforce_step(
    #     train_state: TrainState,
    #     model: ActorCriticModel,
    #     obs: Array,
    #     action: Array,
    #     reward: Array,
    # )

    # value_grads, value_metrics = train_value_fn_step(
    #     train_state: TrainState,
    #     model: eqx.Module,
    #     obs: Array,
    #     reward: Array,
    #     done: Array,
    # )

    # # batch_loss_and_grads = jax.vmap(supervised_loss_and_grads, (None, 0, 0))
    # # losses, grads, accuracies, rnn_states = batch_loss_and_grads(model, rnn_states, train_sequences)
    # # loss = jnp.mean(losses)
    # # accuracy = jnp.mean(accuracies)
    # metrics = {
    #     **{f'policy/{k}': v for k, v in policy_metrics.items()},
    #     **{f'value/{k}': v for k, v in value_metrics.items()},
    # }
    
    # grads_sum = jnp.zeros_like(policy_grads)
    # for grads in [policy_grads, value_grads]:
    #     grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads) # Average over gradients
    #     grads_sum += grads
    
    # train_state, model = apply_grads(
    #     train_state,
    #     model,
    #     grads_sum,
    # )

    # # Update train state
    # train_state = tree_replace(train_state, train_step=train_state.train_step + 1)

    metrics = {'loss': jnp.array([0.0, 1.0])}

    return train_state, model, metrics


def train_loop(
        train_state: TrainState,
        env_state_and_obs: Tuple[PyTree, PyTree], # (env_state, obs)
        model: eqx.Module,
        env_step_fn: Callable,
        train_config: DictConfig,
        train_steps: int,
        half_precision: bool = False,
    ):
    # This may be reduntant if this entire function is jitted, but it is necessary for
    # when the environment is not jittable, and hence, this entire function is not jitted
    rollout_forward_fn = jax.jit(jax.vmap(type(model).__call__, in_axes=(None, 0)))
    
    def rollout_and_train_step(carry, _):
        train_state, env_state_and_obs, model = carry

        rollout_key, rng = jax.random.split(train_state.rng)
        train_state = tree_replace(train_state, rng=rng)

        # Gather data
        env_state_and_obs, trajectory_data = gather_data(
            rollout_key, env_state_and_obs, env_step_fn, model, rollout_forward_fn,
            train_config.per_env_batch_size, half_precision,
        )

        # Train on data
        train_state, model, metrics = batch_train_iter(
            train_state, model, trajectory_data, train_config)

        return (train_state, env_state_and_obs, model), metrics

    # TODO: Add support for non-jittable env step fn with a for loop
    (train_state, env_state_and_obs, model), metrics = jax.lax.scan(
        rollout_and_train_step, (train_state, env_state_and_obs, model), length=train_steps)
    return train_state, env_state_and_obs, model, metrics


def train(
        key: PRNGKeyArray,
        train_state: TrainState,
        env: EquinoxEnv,
        model: eqx.Module,
        config: DictConfig,
    ):
    train_config = config.train
    effective_batch_size = train_config.per_env_batch_size * config.env.n_envs
    steps_per_log = train_config.log_interval // effective_batch_size
    total_steps = train_config.steps // effective_batch_size
    
    env_steps_passed = 0
    train_steps_passed = 0
    
    # If the env is jittable, we can jit the entire train loop
    _train_loop = train_loop
    if env.jittable:
        _train_loop = jax.jit(train_loop, static_argnums=(3, 4, 5, 6))
    
    # If there are multiple environments, I need to make sure that the model forward fn for rollouts can handle multiple obs
    # If the env is not jittable, and hence the data collection loop is not jittable, I need to make sure that the same function is also jitted
    # Maybe the answer here is to just vmap it and jit it for all cases, then I don't need to worry about different cases


    half_precision = config.get('half_precision', False)
    
    # Initialize the env state
    if env.jittable:
        keys = jax.random.split(key, config.env.n_envs + 1)
        env_keys, key = keys[:-1], keys[-1]
        env_state, obs = jax.vmap(env.reset)(env_keys)
        env_step_fn = jax.vmap(env.step)
    else:
        raise ValueError('Only jittable environments are supported for now.')
    
    if half_precision:
        obs = obs.astype(jnp.bfloat16)
    env_state_and_obs = (env_state, obs)
    
    with tqdm(total=train_config.steps) as pbar:
        for _ in range(total_steps // steps_per_log):
            train_state, env_state_and_obs, model, metrics = _train_loop(
                train_state, env_state_and_obs, model, env_step_fn,
                train_config, steps_per_log, half_precision,
            )

            avg_loss = jnp.nanmean(metrics['loss'])

            if jnp.isnan(avg_loss):
                print('Loss is nan, breakpoint!')

            train_steps_passed += steps_per_log
            env_steps_passed += steps_per_log * effective_batch_size

            pbar.update(steps_per_log * effective_batch_size)
            pbar.set_postfix({
                'avg_loss': avg_loss,
                'train_steps': train_steps_passed,
                'env_steps': env_steps_passed
            })

            if config.wandb.get('enabled', False):
                import wandb
                wandb.log({
                    'loss': avg_loss,
                    'train_step': train_steps_passed,
                    'env_step': env_steps_passed
                })


def to_half_precision(x: jax.Array):
    """Convert array to bf16 if it's a full precision float."""
    if x.dtype == jnp.float32:
        return x.astype(jnp.bfloat16)
    return x


def validate_config(config: DictConfig):
    pass


@hydra.main(config_path='conf', config_name='train_base')
def main(config: DictConfig) -> None:
    print('Config:\n', config)
    validate_config(config)

    if config.wandb.get('enabled', False):
        import wandb
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        wandb.init(entity=config.wandb.get('entity'), project=config.wandb.get('project'))

    # Cache jitted functions
    if config.get('cache_jit', False):
        compilation_cache.set_cache_dir(tempfile.tempdir)

    rng = jax.random.PRNGKey(config.get('seed', time.time_ns()))
    model_key, env_key, rng = jax.random.split(rng, 3)

    # Prepare environment(s)
    env = create_env(config.env)

    # Prepare model
    model = create_model(
        key = model_key,
        obs_dim = env.observation_space,
        n_actions = env.num_actions,
        model_config = config.model,
        half_precision = config.half_precision,
    )
    print('# Model params:', sum(jax.tree.leaves(jax.tree.map(lambda x: math.prod(x.shape), model))))

    print(model)

    # Prepare optimizer
    optimizer, opt_state = create_optimizer(model, config.optimizer)
    
    print(optimizer)


    # Train
    train_state = TrainState(rng, opt_state, optimizer.update)
    train(env_key, train_state, env, model, config)


if __name__ == '__main__':
    main()