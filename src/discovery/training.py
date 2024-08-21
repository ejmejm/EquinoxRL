from functools import partial
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import optax

from discovery.models import ActorCriticModel
from discovery.rollouts import gather_data, TrajectoryData
from discovery.utils import scan_or_loop, tree_replace


EPSILON = 1e-6


class TrainState(eqx.Module):
    rng: PRNGKeyArray
    opt_state: optax.OptState
    train_step: Array
    tx_update_fn: Callable = eqx.field(static=True)
    config: Dict = eqx.field(static=True) # Train config

    def __init__(
            self,
            rng: PRNGKeyArray,
            opt_state: optax.OptState,
            tx_update_fn: Callable,
            config: Dict,
        ):
        self.rng = rng
        self.opt_state = opt_state
        self.tx_update_fn = tx_update_fn
        self.train_step = jnp.array(0)
        self.config = config


def calculate_gaes(trajectory_data: TrajectoryData, gamma: float, lambda_decay: float) -> Array:
    """Calculate the Generalized Advantage Estimation (GAE) for a trajectory.
    
    Args:
        trajectory_data: The trajectory with elements of shape (n_envs, rollout_len, ...).
        gamma: The discount factor.
        lambda_decay: The decay factor.

    Returns:
        An array of GAE values of shape (n_envs, rollout_len).
    """
    rewards = trajectory_data.reward
    dones = trajectory_data.done
    values = trajectory_data.value
    
    deltas = rewards + (1 - dones) * gamma * values[:, 1:] - values[:, :-1]
    
    def calc_gae(gae, delta):
        gae = delta + lambda_decay * gamma * gae
        return gae, gae
    
    _, gaes = jax.lax.scan(jax.vmap(calc_gae), jnp.zeros_like(deltas[:, 0]), deltas.T, reverse=True)

    return gaes.T


def compute_reinforce_grads(
        train_state: TrainState,
        model: ActorCriticModel,
        trajectory_data: TrajectoryData,
        target_model: ActorCriticModel = None,
    ) -> Tuple[ActorCriticModel, Dict]:
    """Perform a single training step.
    
    Args:
        train_state: The current training state.
        model: The current model.
        trajectory_data: Trajectory data with elements of shape (n_envs, rollout_len, ...).
        target_model: The target model.

    Returns:
        The weight updates and a dictionary of metrics.
    """
    if target_model is not None:
        raise NotImplementedError('Target model not implemented')
    
    gamma = train_state.config['gamma']
    lambda_decay = train_state.config['lambda_decay']
    
    final_obs = trajectory_data.new_obs[:, -1]
    final_values = jax.vmap(model.value)(final_obs)
    all_values = jnp.concatenate([trajectory_data.value, final_values[:, None]], axis=1)
    trajectory_data = tree_replace(trajectory_data, value=all_values)
    
    gaes = calculate_gaes(trajectory_data, gamma, lambda_decay)

    def reinforce_loss_fn(model, gaes, obs, acts):
        act_logits, values = jax.vmap(model)(obs)
        
        # Gather the log probs of the actions
        log_probs = jnp.log(jax.nn.softmax(act_logits) + EPSILON)
        act_log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze(1)
        
        # Calculate losses
        td_error = gaes + jax.lax.stop_gradient(values) - values
        policy_loss = -jnp.mean(act_log_probs * gaes)
        value_loss = jnp.mean(td_error ** 2)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'loss': total_loss,
        }

    grad_fn = jax.grad(reinforce_loss_fn, has_aux=True)

    flat_gaes = gaes.flatten()
    flat_obs = trajectory_data.obs.reshape((-1,) + trajectory_data.obs.shape[2:])
    flat_acts = trajectory_data.action.flatten()

    grads, loss_metrics = grad_fn(model, flat_gaes, flat_obs, flat_acts)
    metrics = {
        **loss_metrics,
        'avg_reward': jnp.mean(trajectory_data.reward),
    }

    return grads, metrics


def apply_grads(
        train_state: TrainState,
        model: eqx.Module,
        grads: eqx.Module,
    ):
    # Replace nan grads with 0
    grads = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grads)
    updates, new_opt_state = train_state.tx_update_fn(grads, train_state.opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    train_state = tree_replace(train_state, opt_state=new_opt_state)

    return train_state, new_model


if __name__ == '__main__':
    # Test the GAE calculation
    rewards = jnp.array([
        [1.0, 0.0, 2.0, -1.0],
        [0.5, 1.5, 0.0, 1.0]
    ])
    values = jnp.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1.5, 2.5, 1.0, 0.0]
    ])
    dones = jnp.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ])

    trajectory_data = TrajectoryData(
        obs=None, action=None, new_obs=None, reward=rewards,
        done=dones, value=values, info=None,
    )

    gamma = 0.99
    lambda_decay = 0.95

    gaes = calculate_gaes(trajectory_data, gamma, lambda_decay)
    print(gaes)
    

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
        env_jittable: bool = True,
        forward_fn: Callable = None,
    ):
    # This may be reduntant if this entire function is jitted, but it is necessary for
    # when the environment is not jittable, and hence, this entire function is not jitted
    rollout_forward_fn = forward_fn or jax.jit(jax.vmap(type(model).__call__, in_axes=(None, 0)))
    
    def rollout_and_train_step(carry, _):
        train_state, env_state_and_obs, model = carry

        rollout_key, rng = jax.random.split(train_state.rng)
        train_state = tree_replace(train_state, rng=rng)

        # Gather data
        env_state_and_obs, trajectory_data = gather_data(
            rollout_key, env_state_and_obs, env_step_fn, model, rollout_forward_fn,
            train_state.config.per_env_batch_size, half_precision, env_jittable,
        )
        
        if not env_jittable:
            trajectory_data = jax.tree.map(jnp.array, trajectory_data)

        # Train on data
        train_state, model, metrics = batch_train_iter(
            train_state, model, trajectory_data)

        return (train_state, env_state_and_obs, model), metrics

    (train_state, env_state_and_obs, model), metrics = scan_or_loop(
        env_jittable, rollout_and_train_step, (train_state, env_state_and_obs, model), length=train_steps)
    return train_state, env_state_and_obs, model, metrics