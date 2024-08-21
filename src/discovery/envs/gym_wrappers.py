import gymnasium as gym
import numpy as np


class RescaleObservationRange(gym.Wrapper):
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
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        return self.observation(obs), reward, done, truncation, info


class SwapChannelToFirstAxis(gym.Wrapper):
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

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        return self.observation(obs), reward, done, truncation, info
