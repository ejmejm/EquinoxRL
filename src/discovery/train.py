from functools import partial
import math
import tempfile
import time
from typing import Callable, NamedTuple, Tuple

import equinox as eqx
import hydra
import jax
import jax.nn as jnn
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
from training import apply_grads, compute_reinforce_grads, TrainState, TrajectoryData
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
        (env_state, obs), rng = state
        act_key, random_key, act_mask_key, rng = jax.random.split(rng, 4)
            
        act_logits, value = forward_fn(model, obs)
        model_action = jax.random.categorical(act_key, act_logits, axis=-1)
        
        # 10% chance of taking random action
        random_action = jax.random.randint(random_key, shape=model_action.shape, minval=0, maxval=act_logits.shape[-1])
        random_mask = jax.random.uniform(act_mask_key, shape=model_action.shape) < 0.1
        action = jnp.where(random_mask, random_action, model_action)
        action = model_action
        
        env_state, new_obs, reward, done, info = env_step_fn(env_state, action)
        if half_precision:
            new_obs = new_obs.astype(jnp.bfloat16)
            reward = reward.astype(jnp.bfloat16)
            done = done.astype(jnp.bfloat16)
        
        return ((env_state, new_obs), rng), TrajectoryData(obs, action, new_obs, reward, done, value, info)
    
    # TODO: Add support for non-jittable env step fn with a for loop
    scan_state = (env_state_and_obs, key)
    scan_state, train_sequences = jax.lax.scan(
        predict_and_step_fn, scan_state, length=rollout_length)
    env_state_and_obs = scan_state[0]

    return env_state_and_obs, train_sequences


@jax.jit
def batch_train_iter(
        train_state: TrainState,
        model: eqx.Module,
        trajectory_data: TrajectoryData,
    ):
    # Trajectory data element shapes: (rollout_length, n_envs, ...) -> (n_envs, rollout_length, ...)
    trajectory_data = jax.tree.map(partial(jnp.swapaxes, axis1=0, axis2=1), trajectory_data)

    # Compute gradients and update model
    grads, metrics = compute_reinforce_grads(train_state, model, trajectory_data)
    train_state, model = apply_grads(train_state, model, grads)
    train_state = tree_replace(train_state, train_step=train_state.train_step + 1)

    return train_state, model, metrics


def train_loop(
        train_state: TrainState,
        env_state_and_obs: Tuple[PyTree, PyTree], # (env_state, obs)
        model: eqx.Module,
        env_step_fn: Callable,
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
            train_state.config.per_env_batch_size, half_precision,
        )

        # Train on data
        train_state, model, metrics = batch_train_iter(
            train_state, model, trajectory_data)

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
        _train_loop = jax.jit(train_loop, static_argnums=(3, 4, 5))

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
                train_state, env_state_and_obs, model, env_step_fn, steps_per_log, half_precision)

            avg_loss = jnp.nanmean(metrics['loss'])
            avg_reward = jnp.nanmean(metrics['avg_reward'])

            if jnp.isnan(avg_loss):
                print('Loss is nan, breakpoint!')

            train_steps_passed += steps_per_log
            env_steps_passed += steps_per_log * effective_batch_size

            pbar.update(steps_per_log * effective_batch_size)
            pbar.set_postfix({
                'avg_reward': avg_reward,
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


    # Train
    train_state = TrainState(rng, opt_state, optimizer.update, config.train)
    train(env_key, train_state, env, model, config)


if __name__ == '__main__':
    main()
