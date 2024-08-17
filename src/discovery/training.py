from typing import Callable, NamedTuple, Tuple

import chex
import equinox as eqx
import gymnasium as gym
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree
import optax

from discovery.models import ActorCriticModel
from discovery.utils import tree_replace

# from cl_agent.models import ActorCriticModel, LSTMState
# from cl_agent.utils import tree_replace
# from cl_agent.training.recurrent_backprop import reinforce_train_on_sequence
# from cl_agent.training.swift_td import SwiftTDState, swift_td_step



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


# def train_reinforce_step(
#         train_state: TrainState,
#         model: ActorCriticModel,
#         obs: Array,
#         action: Array,
#         reward: Array,
#         target_model: ActorCriticModel = None,
#     ) -> Tuple[TrainState, PyTree, eqx.Module]:
#     """Perform a single training step.
    
#     Args:
#         train_state: The current training state.
#         model: The current model.
#         obs: Observation at step t+1 from the environment.
#         action: Action at step t+1 from the environment.
#         reward: Reward at step t+1 from the environment.
#         target_model: The target model.

#     Returns:
#         The new training state and model.
#     """
#     updates = {}

#     # Update observation and cumulant history
#     obs_history = jnp.roll(train_state.obs_history, shift=-1, axis=0)
#     obs_history = obs_history.at[-1].set(obs)
#     updates['obs_history'] = obs_history

#     def reinforce_noop():
#         weight_updates = jax.tree.map(lambda x: jnp.zeros_like(x), model)
#         dummy_rnn_state = jax.tree.map(lambda x: jnp.zeros_like(x), rnn_state)
#         return weight_updates, train_state.opt_state, dummy_rnn_state, 0.0

#     # Calculate backprop (+ optimizer) updates for the entire model when it's time to update the features
#     weight_updates, new_opt_state, _, reinforce_loss = jax.lax.cond(
#         # Should train condition:
#         (train_state.feature_update_freq > 0) and (train_state.train_step % train_state.feature_update_freq == 0),
#         # If train function:
#         lambda: reinforce_train_on_sequence(
#             model, target_model, train_state.opt_state, train_state.tx_update_fn, train_state.gamma,
#             train_state.history_rnn_state, obs_history, action, reward),
#         # Noop function:
#         reinforce_noop,
#     )
#     updates['opt_state'] = new_opt_state

#     # Update the old RNN state used for TBPTT
#     updates['history_rnn_state'] = model.feature_extractor(obs_history[0], train_state.history_rnn_state)[0]

#     # Calculate updates to the SwiftTD final value layer weights
#     if train_state.use_swift_td:
#         weights = model.critic[-1].weight
#         new_rnn_state, features = model.value_features(obs_history[-1], rnn_state)
#         new_swift_td_state, value_linear_weights, td_error = swift_td_step(
#             train_state.swift_td_state, weights, features, reward)
#         swift_td_loss = jnp.square(td_error)
#         updates['swift_td_state'] = new_swift_td_state

#         swift_td_updates = value_linear_weights - weights
#         weight_updates = eqx.tree_at(lambda x: x.critic[-1].weight, weight_updates, swift_td_updates)
#     else:
#         new_rnn_state = model.value_features(obs_history[-1], rnn_state)[0]
#         swift_td_loss = 0

#     # Increment train step
#     updates['train_step'] = train_state.train_step + 1

#     # Update the train state
#     train_state = tree_replace(train_state, **updates)

#     # Return updated state, rnn_state, and model
#     return train_state, new_rnn_state, weight_updates, reinforce_loss, swift_td_loss


# def train_loop(
#         train_state: TrainState,
#         rnn_state: LSTMState,
#         model: ActorCriticModel,
#         env: gym.Env,
#         curr_obs: Array,
#         n_steps: int,
#     ) -> Tuple[TrainState, LSTMState, ActorCriticModel]:
#     """Perform a training loop.
    
#     Args:
#         train_state: The current training state.
#         rnn_state: The current RNN state.
#         model: The current model.
#         obs: Observation at step t+1 from the environment.
#         reward: Reward at step t+1 from the environment.
#         n_steps: Number of steps to train for.

#     Returns:
#         The new training state, RNN state, and model.
#     """
#     def env_train_step(state, _):
#         train_state, rnn_state, model, target_model, env, prev_obs, update_buffer = state
#         act_key, rng = jax.random.split(train_state.rng)

#         new_rnn_state, act_logits = model.act_logits(prev_obs, rnn_state)

#         action = jax.random.categorical(act_key, act_logits)
#         env, obs, reward = env.step(action)[:3]
#         train_state = tree_replace(train_state, rng=rng)

#         train_state, _, weight_updates, reinforce_loss, swift_td_loss = train_reinforce_step(
#             train_state, new_rnn_state, model, target_model, obs, action, reward)
#         update_buffer = jax.tree.map(
#             lambda x, y: x + y,
#             update_buffer, weight_updates,
#         )

#         def apply_updates(model, update_buffer):
#             update_buffer = jax.tree.map(
#                 lambda x: x, #/ train_state.feature_update_freq,
#                 update_buffer,
#             )
#             return eqx.apply_updates(model, update_buffer)

#         model, update_buffer = jax.lax.cond(
#             train_state.train_step % train_state.feature_update_freq == 0,
#             lambda: (apply_updates(model, update_buffer), jax.tree.map(lambda x: jnp.zeros_like(x), model)),
#             lambda: (model, update_buffer),
#         )
        
#         return (train_state, new_rnn_state, model, target_model, env, obs, update_buffer), {
#             'reinforce_loss': reinforce_loss,
#             'swift_td_loss': swift_td_loss,
#             'reward': reward,
#         }

#     update_buffer = jax.tree.map(lambda x: jnp.zeros_like(x), model)
#     (train_state, rnn_state, model, target_model, env, obs, update_buffer), metrics = jax.lax.scan(
#         env_train_step, (train_state, rnn_state, model, model, env, curr_obs, update_buffer), length=n_steps)

#     return train_state, rnn_state, model, env, obs, metrics


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
    from cl_agent.models import FeatureExtractor
    from optax import adam

    rng = jax.random.PRNGKey(0)
    fe_key, ac_key, rng = jax.random.split(rng, 3)

    # Dummy test data
    obs = jnp.ones((10,))
    action = jnp.array(1)
    reward = jnp.array(1.0)

    # Initialize the model and train state
    feature_extractor = FeatureExtractor(fe_key, obs.shape[0], [32, 32], 64, [1])
    action_dim = 3
    gamma = 0.9
    feature_dim = 32
    actor_layer_sizes = [64, 32]
    critic_layer_sizes = [64, feature_dim]

    model = ActorCriticModel(ac_key, feature_extractor, action_dim, actor_layer_sizes, critic_layer_sizes)

    swift_td_state = SwiftTDState(feature_dim, gamma=gamma)

    optimizer = adam(3e-4)
    opt_state = optimizer.init(model)

    train_state = TrainState(
        model = model,
        opt_state = opt_state,
        tx_update_fn = optimizer.update,
        obs_shape = obs.shape,
        gamma = gamma,
        swift_td_state = swift_td_state,
        feature_update_freq = 3,
        use_swift_td = True,
    )

    rnn_state = model.init_rnn_state()
    jit_reinforce_step = jax.jit(train_reinforce_step)

    # Call the function
    for _ in range(20):
        train_state, rnn_state, model, reinforce_loss, swift_td_loss = jit_reinforce_step(
            train_state, rnn_state, model, obs, action, reward)
        print(f"Train step: {train_state.train_step}, Reinforce Loss: {reinforce_loss}, SwiftTD loss: {swift_td_loss}")