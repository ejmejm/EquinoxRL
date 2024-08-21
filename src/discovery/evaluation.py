from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import numpy as np
from omegaconf import DictConfig

from .envs.base_env import BaseEnv, VectorizedGymLikeEnv
from .rollouts import gather_data, TrajectoryData


import numpy as np

def compute_metrics(train_sequences: TrajectoryData):
    """Compute metrics from a trajectory.
    
    Metrics include:
        - Average reward
        - Average episode return
        - Average episode length
    """
    avg_reward = np.mean(train_sequences.reward)
    return_sums = [0 for _ in range(len(train_sequences.reward[0]))]
    last_done_step = [-1 for _ in range(len(train_sequences.reward[0]))]
    episode_returns = []
    episode_lengths = []
    
    for step in range(len(train_sequences.reward)):
        for env_idx in range(len(train_sequences.reward[step])):
            return_sums[env_idx] += train_sequences.reward[step][env_idx]
            if train_sequences.done[step][env_idx]:
                episode_returns.append(return_sums[env_idx])
                episode_lengths.append(step - last_done_step[env_idx] - 1)
                return_sums[env_idx] = 0
                last_done_step[env_idx] = step
    
    # Handle cases where there are no completed episodes
    if episode_returns:
        metrics = {
            "avg_reward": avg_reward,
            "episode_return": episode_returns,
            "episode_return_mean": np.mean(episode_returns),
            "episode_return_std": np.std(episode_returns),
            "episode_length": episode_lengths,
            "episode_length_mean": np.mean(episode_lengths),
            "episode_length_std": np.std(episode_lengths),
        }
    else:
        metrics = {
            "avg_reward": avg_reward,
            "episode_return": [],
            "episode_return_mean": np.nan,
            "episode_return_std": np.nan,
            "episode_length": [],
            "episode_length_mean": np.nan,
            "episode_length_std": np.nan,
        }
    
    return metrics



def evaluate_agent(
        key: PRNGKeyArray,
        env: BaseEnv,
        model: eqx.Module,
        forward_fn: Callable,
        config: DictConfig,
    ) -> Tuple[Dict[str, float], np.ndarray]:
    train_config = config.train

    half_precision = config.get('half_precision', False)
    
    # Initialize the env state
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
    
    rollout_key, rng = jax.random.split(key, 2)
    _, train_sequences = gather_data(
            rollout_key, env_state_and_obs, env_step_fn, model, forward_fn,
            train_config.eval_steps, half_precision, env.jittable,
        )
    
    # Compute metrics
    metrics = compute_metrics(train_sequences)
    
    # Collect trajectories
    sample_trajectories = train_sequences.obs.swapaxes(0, 1)
    sample_trajectories = sample_trajectories[:train_config.n_trajectories, :train_config.log_trajectory_length]
    
    return metrics, sample_trajectories
