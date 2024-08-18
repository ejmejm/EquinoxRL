from typing import Callable, Dict, NamedTuple, Tuple

import equinox as eqx
import gymnasium as gym
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import optax

from discovery.models import ActorCriticModel
from discovery.utils import tree_replace


EPSILON = 1e-6


TrajectoryData = NamedTuple(
    'TrajectoryData',
    obs = ArrayLike,
    action = ArrayLike,
    new_obs = ArrayLike,
    reward = ArrayLike,
    done = ArrayLike,
    value = ArrayLike,
    info = PyTree,
)


class TrainState(eqx.Module):
    rng: PRNGKeyArray
    opt_state: optax.OptState
    train_step: Array
    tx_update_fn: Callable = eqx.field(static=True)

    def __init__(
            self,
            rng: PRNGKeyArray,
            opt_state: optax.OptState,
            tx_update_fn: Callable,
        ):
        self.rng = rng
        self.opt_state = opt_state
        self.tx_update_fn = tx_update_fn
        self.train_step = jnp.array(0)


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
    
    gamma = 0.98
    
    final_obs = trajectory_data.new_obs[:, -1]
    final_values = jax.vmap(model.value)(final_obs)
    rewards = trajectory_data.reward
    dones = trajectory_data.done.astype(jnp.bool_)
    
    # returns = jax.stack([returns, final_values[:, None]], axis=-1)
    # TODO: Check if I can do this with an associative scan
    # It wouldn't be a huge performance boost, but I do want to learn how to use associative scans
    def discounted_return_fn(next_vals, x):
        rewards, dones = x
        return_vals = rewards + (1 - dones) * gamma * next_vals
        return return_vals, return_vals
    
    # (2, n_envs, rollout_len) -> (rollout_len[reversed], 2, n_envs)
    reward_and_dones = jnp.stack([rewards, dones]).transpose(2, 0, 1)[::-1]
    _, returns = jax.lax.scan(discounted_return_fn, final_values, reward_and_dones)
    returns = returns[::-1].transpose() # -> (n_envs, rollout_len)

    def reinforce_loss_fn(model, returns, obs, acts):
        act_logits, values = jax.vmap(model)(obs)
        
        # Gather the log probs of the actions
        log_probs = jnp.log(jax.nn.softmax(act_logits) + EPSILON)
        act_log_probs = jnp.take_along_axis(log_probs, acts[:, None], axis=1).squeeze(1)
        
        # Calculate losses
        td_error = returns - values
        policy_loss = -jnp.mean(act_log_probs * jax.lax.stop_gradient(td_error))
        value_loss = jnp.mean(td_error ** 2)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        return total_loss, {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'loss': total_loss,
        }

    grad_fn = jax.grad(reinforce_loss_fn, has_aux=True)

    flat_returns = returns.flatten()
    flat_obs = trajectory_data.obs.reshape((-1,) + trajectory_data.obs.shape[2:])
    flat_acts = trajectory_data.action.flatten()

    grads, loss_metrics = grad_fn(model, flat_returns, flat_obs, flat_acts)
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
