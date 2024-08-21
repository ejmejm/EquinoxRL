import jax.numpy as jnp
from xminigrid.environment import Environment as XMEnvironment
from xminigrid.wrappers import Wrapper as XMWrapper


class XMObsDownscaleWrapper(XMWrapper):
    def __init__(self, env: XMEnvironment, downscale_factor: int = 4):
        super().__init__(env)
        self.downscale_factor = downscale_factor
    
    def observation_shape(self, params):
        orig_shape = self._env.observation_shape(params)
        new_shape = (orig_shape[0] // self.downscale_factor, orig_shape[1] // self.downscale_factor, orig_shape[2])
        return new_shape

    def __convert_obs(self, env_state):
        new_obs = env_state.observation[::self.downscale_factor, ::self.downscale_factor, :]
        env_state = env_state.replace(observation=new_obs)
        return env_state

    def reset(self, params, key):
        env_state = self._env.reset(params, key)
        env_state = self.__convert_obs(env_state)
        return env_state

    def step(self, params, env_state, action):
        env_state = self._env.step(params, env_state, action)
        env_state = self.__convert_obs(env_state)
        return env_state


class XMRGBPreprocessWrapper(XMWrapper):
    """
    Performs the following preprocessing on the observation:
        1. Transposes the observation from (H, W, C) to (C, H, W)
        2. Converts the observation to float32
        3. Divides the observation by 255
    """
    def observation_shape(self, params):
        orig_shape = self._env.observation_shape(params)
        new_shape = (orig_shape[2], orig_shape[0], orig_shape[1])
        return new_shape

    def __convert_obs(self, env_state):
        transposed_obs = jnp.transpose(env_state.observation, (2, 0, 1))
        float_obs = transposed_obs.astype(jnp.float32)
        normalized_obs = float_obs / 255.0
        env_state = env_state.replace(observation=normalized_obs)
        return env_state

    def reset(self, params, key):
        env_state = self._env.reset(params, key)
        env_state = self.__convert_obs(env_state)
        return env_state

    def step(self, params, env_state, action):
        env_state = self._env.step(params, env_state, action)
        env_state = self.__convert_obs(env_state)
        return env_state
