import math
import tempfile
import time
from typing import Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache
from jaxtyping import ArrayLike, PRNGKeyArray
import numpy as np
import omegaconf
from omegaconf import DictConfig
import optax
from tqdm import tqdm

from discovery.envs.envs import create_env
from discovery.envs.base_env import BaseEnv, VectorizedGymLikeEnv
from discovery.evaluation import evaluate_agent
from discovery.models import ActorCriticModel, get_feature_extractor_cls, get_activation_fn
from discovery.training import train_loop, TrainState


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


def to_half_precision(x: jax.Array) -> jax.Array:
    """Convert array to bf16 if it's a full precision float."""
    if x.dtype == jnp.float32:
        return x.astype(jnp.bfloat16)
    return x


def validate_config(config: DictConfig) -> None:
    pass


def log_trajectories(trajectories: ArrayLike) -> None:
    """Log trajectories to wandb.
    
    Args:
        trajectories: A list of trajectories, each of shape (S, C, H, W).
    """
    import wandb
    videos = []
    for trajectory in trajectories:
        video = (trajectory * 255).astype(np.uint8)
        videos.append(wandb.Video(video, fps=4))
    wandb.log({'eval/trajectory': videos})


def log_training_metrics(config, log_metrics, train_config, trajectories):
    if config.wandb.get('enabled', False):
        import wandb
        for k, v in log_metrics.items():
            if isinstance(v, list):
                log_metrics[k] = wandb.Histogram(v, num_bins=32)
        wandb.log(log_metrics)
        if train_config.get('log_trajectories', False) and trajectories is not None:
            log_trajectories(trajectories)


def train(
        key: PRNGKeyArray,
        train_state: TrainState,
        env: BaseEnv,
        model: eqx.Module,
        config: DictConfig,
    ) -> None:
    
    ### Initialize training loop variables ###
    
    train_config = config.train
    effective_batch_size = train_config.per_env_batch_size * config.env.n_envs
    steps_per_log = train_config.log_interval // effective_batch_size
    total_steps = train_config.steps // effective_batch_size
    last_eval = 0
    
    env_steps_passed = 0
    train_steps_passed = 0
    
    ### Jitting ###
    
    _train_loop = train_loop
    _eval_loop = evaluate_agent
    if env.jittable:
        _train_loop = jax.jit(train_loop, static_argnums=(3, 4, 5, 7))
        _eval_loop = jax.jit(evaluate_agent, static_argnums=(3, 4))
    forward_fn = jax.jit(jax.vmap(type(model).__call__, in_axes=(None, 0)))

    half_precision = config.get('half_precision', False)
    
    ### Initialize the env states ###
    
    if env.jittable:
        keys = jax.random.split(key, config.env.n_envs + 1)
        env_keys, key = keys[:-1], keys[-1]
        env_state, obs = jax.vmap(env.reset)(env_keys)
        env_step_fn = jax.vmap(env.step)
    else:
        assert isinstance(env, VectorizedGymLikeEnv), "Non-jittable environments must be pre-vectorized!"
        env_state, obs = env.reset()
        env_step_fn = env.step
    
    if half_precision:
        obs = obs.astype(jnp.bfloat16)
    env_state_and_obs = (env_state, obs)
    
    with tqdm(total=train_config.steps) as pbar:
        for _ in range(total_steps // steps_per_log):
            
            ### Train ###
            
            train_state, env_state_and_obs, model, train_metrics = _train_loop(
                train_state, env_state_and_obs, model, env_step_fn,
                steps_per_log, half_precision, env.jittable, forward_fn,
            )
            
            ### Compute metrics ###

            avg_loss = jnp.nanmean(train_metrics['loss'])
            avg_reward = jnp.nanmean(train_metrics['avg_reward'])

            if jnp.isnan(avg_loss):
                print('Loss is nan, breakpoint!')

            train_steps_passed += steps_per_log
            env_steps_passed += steps_per_log * effective_batch_size
            
            log_metrics = dict(
                avg_reward = avg_reward,
                loss = avg_loss,
                train_step = train_steps_passed,
                env_step = env_steps_passed,
            )
            log_metrics = {f'train/{k}': v for k, v in log_metrics.items()}
  
            ### Evaluation ###
  
            eval_interval = train_config.get('eval_interval', None)
            if eval_interval > 0 and env_steps_passed - last_eval > eval_interval:
                eval_key, key = jax.random.split(key)
                eval_metrics, trajectories = _eval_loop(eval_key, env, model, forward_fn, config)
                log_metrics.update({f'eval/{k}': v for k, v in eval_metrics.items()})
                last_eval = env_steps_passed
            else:
                trajectories = None
            
            ### Logging ###
            
            pbar.update(steps_per_log * effective_batch_size)
            pbar.set_postfix({
                k[6:]: f'{v:.3f}' 
                for k, v in log_metrics.items()
                if np.isscalar(v) and k.startswith('train/')
            })

            log_training_metrics(config, log_metrics, train_config, trajectories)


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
    print(model)
    print('# Model params:', sum(jax.tree.leaves(jax.tree.map(lambda x: math.prod(x.shape), model))))

    # Prepare optimizer
    optimizer, opt_state = create_optimizer(model, config.optimizer)

    ### Training & optional profiling ###
    
    train_state = TrainState(rng, opt_state, optimizer.update, config.train)
    
    profiling_enabled = config.get('profile', False)
    
    if profiling_enabled:
        import cProfile
        import pstats
        import io
    
        # Profile the train function
        profiler = cProfile.Profile()
        profiler.enable()
    
    train(env_key, train_state, env, model, config)
    
    if profiling_enabled:
        profiler.disable()
    
        # Save profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        with open('train_profile.txt', 'w') as f:
            f.write(s.getvalue())
        
        # Save profiling results in a format suitable for snakeviz
        profiler.dump_stats('train_profile.prof')


if __name__ == '__main__':
    main()
