from functools import partial
from typing import Callable, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PRNGKeyArray, PyTree

from discovery.utils import scan_or_loop


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


@partial(jax.jit, static_argnums=(1,))
def determine_action(model, forward_fn, obs, rng):
    act_key, random_key, act_mask_key, rng = jax.random.split(rng, 4)
        
    act_logits, value = forward_fn(model, obs)
    model_action = jax.random.categorical(act_key, act_logits, axis=-1)
    
    # 10% chance of taking random action
    random_action = jax.random.randint(random_key, shape=model_action.shape, minval=0, maxval=act_logits.shape[-1])
    random_mask = jax.random.uniform(act_mask_key, shape=model_action.shape) < 0.1
    action = jnp.where(random_mask, random_action, model_action)
    action = model_action
    
    return action, value, rng


def gather_data(
        key: PRNGKeyArray,
        env_state_and_obs: Tuple[PyTree, PyTree],
        env_step_fn: Callable,
        model: eqx.Module,
        forward_fn: Callable[[eqx.Module, ArrayLike], Tuple[ArrayLike, ArrayLike]],
        rollout_length: int,
        half_precision: bool = False,
        env_jittable: bool = True,
    ) -> Tuple[Tuple[PyTree, PyTree], TrajectoryData]:

    def predict_and_step_fn(state, _):
        (env_state, obs), rng = state
        
        action, value, rng = determine_action(model, forward_fn, obs, rng)
        
        if not env_jittable:
            action = np.array(action)
        
        env_state, new_obs, reward, done, info = env_step_fn(env_state, action)
        if half_precision:
            new_obs = new_obs.astype(jnp.bfloat16)
            reward = reward.astype(jnp.bfloat16)
            done = done.astype(jnp.bfloat16)
        
        return ((env_state, new_obs), rng), TrajectoryData(obs, action, new_obs, reward, done, value, info)

    scan_state = (env_state_and_obs, key)
    scan_state, train_sequences = scan_or_loop(
        env_jittable, predict_and_step_fn, scan_state, length=rollout_length)
    env_state_and_obs = scan_state[0]

    return env_state_and_obs, train_sequences
