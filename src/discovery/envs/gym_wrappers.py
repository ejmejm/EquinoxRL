import gymnasium as gym
import numpy as np


class RescaleObservationRange(gym.Wrapper):
    """Rescales the observation range (by default to [0, 1] for images)."""
    
    def __init__(self, env, scale_factor=1.0/255.0):
        super().__init__(env)
        self.scale_factor = scale_factor
        self.observation_space = gym.spaces.Box(
            low = self.env.observation_space.low * self.scale_factor,
            high = self.env.observation_space.high * self.scale_factor,
            shape = self.env.observation_space.shape,
            dtype = np.float32,
        )

    def observation(self, obs):
        return obs * self.scale_factor

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        return self.observation(obs), reward, done, truncation, info


class SwapChannelToFirstAxis(gym.Wrapper):
    """Swaps the channel axis (last axis) to the first axis of the observation space."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = (old_shape[-1],) + old_shape[:-1]
        new_low = self.env.observation_space.low.reshape(new_shape)
        new_high = self.env.observation_space.high.reshape(new_shape)
        self.observation_space = gym.spaces.Box(
            low = new_low,
            high = new_high,
            shape = new_shape,
            dtype = self.env.observation_space.dtype,
        )
        self.new_axes = (len(old_shape) - 1,) + tuple(range(len(old_shape) - 1))

    def observation(self, obs):
        return np.transpose(obs, self.new_axes)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        return self.observation(obs), reward, done, truncation, info


class ExtraTerminalTransition(gym.Wrapper):
    """Adds an extra terminal transition to the environment DMLab-style.
    
    When using vectorized environments, the final observation in gym-like envs is
    provided via the `info` dict, which for ease-of-use is ignored.
    This adds an extra terminal transition to the environment, which transitions
    to the same state as the final observation with a reward of 0.
    This way no special handling is required for the final observation in
    gym-like environments.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.last_done = False
        self.last_obs = None
        self.last_truncated = False
        
    def reset(self, **kwargs):
        self.last_done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.last_done:
            return self.last_obs, 0.0, True, self.last_truncated, {}
        
        obs, reward, done, truncation, info = self.env.step(action)
        
        self.last_done = done
        self.last_obs = obs
        self.last_truncated = truncation
        
        return obs, reward, False, False, info

