# This file contains wrappers that convert every type of environment to a
# unified interface based on Equinox modules.

from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
from jaxtyping import PyTree, PRNGKeyArray


class BaseEnv():
    """Shared interface for all environments."""

    def reset(self, rng: Any) -> Tuple[PyTree, PyTree]:
        """Reset the environment and return the initial observation.

        Returns:
            A tuple containing two elements:
            (new_env, observation)
        """
        raise NotImplementedError

    def step(self, env_state: Any, action: int) -> Tuple[Any, Any, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Returns:
            A tuple containing five elements:
            (new_env, observation, reward, done, info)
        """
        raise NotImplementedError

    @property
    def observation_space(self) -> Tuple[int, ...]:
        """Return the observation space of the environment.

        Returns:
            A gym.Space object representing the observation space.
        """
        raise NotImplementedError

    @property
    def action_space(self) -> int:
        """Return the action space of the environment.

        Returns:
            A gym.Space object representing the action space.
        """
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        """Return the number of actions in the environment.

        Returns:
            An integer representing the number of actions.
        """
        raise NotImplementedError


class EquinoxEnvInterface(BaseEnv, eqx.Module):
    """Interface for Equinox environments with a shared gym-like interface."""
    jittable: bool = eqx.field(static=True, default=False)


class GymLikeEnvInterface(BaseEnv):
    """Interface for gym-like environments"""
    
    def __init__(self):
        self.jittable = False


##### Gymnax #####


try:
    import gymnax
    from gymnax.environments.environment import Environment as GymnaxEnv


    class GymnaxEqxEnv(EquinoxEnvInterface):
        """Wraps a gymnax environment to make it usable with Equinox."""
        jittable: bool = eqx.field(static=True, default=True)
        env: GymnaxEnv = eqx.field(static=True)
        env_params: Optional[gymnax.EnvParams] = eqx.field(static=True)

        def __init__(self, env: GymnaxEnv, env_params: Optional[gymnax.EnvParams] = None):
            self.env = env
            self.env_params = env_params

        def reset(self, rng: PRNGKeyArray):
            obs, state = self.env.reset(rng, self.env_params)
            return state, obs

        def step(self, env_state: gymnax.EnvState, action: int):
            obs, state, reward, done, info = self.env.step(
                env_state, action, self.env_params)
            return state, obs, reward, done, info

        @property
        def observation_space(self):
            return self.env.observation_space(params=self.env_params)

        @property
        def action_space(self):
            return self.env.action_space(params=self.env_params)

        def __getattr__(self, name):
            return getattr(self.env, name)

except ImportError:
    pass


##### XLand MiniGrid #####


try:
    from xminigrid.environment import Environment as XMEnvironment, EnvParams as XMEnvParams
    from xminigrid.types import TimeStep as XMTimeStep


    class XMinigridEqxEnv(EquinoxEnvInterface):
        """Wraps an XLand-MiniGrid environment to make it usable with Equinox."""
        jittable: bool = eqx.field(static=True, default=True)
        env: XMEnvironment = eqx.field(static=True)
        env_params: XMEnvParams = eqx.field(static=True)

        def __init__(self, env: XMEnvironment, env_params: XMEnvParams):
            self.env = env
            self.env_params = env_params

        def reset(self, rng: PRNGKeyArray):
            env_state = self.env.reset(self.env_params, rng)
            return env_state, env_state.observation

        def step(self, env_state: XMTimeStep, action: int):
            env_state = self.env.step(self.env_params, env_state, action=action)
            return env_state, env_state.observation, env_state.reward, env_state.last(), {}

        @property
        def observation_space(self):
            return self.env.observation_shape(self.env_params)

        @property
        def action_space(self):
            return self.env.num_actions(self.env_params)
        
        @property
        def num_actions(self):
            return self.env.num_actions(self.env_params)

        def render(self):
            return self.env.render(self.env_params, self.timestep)

        def __getattr__(self, name):
            return getattr(self.env, name)

except ImportError:
    pass


##### Gymnasium #####


try:
    import gymnasium as gym
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    
    
    class GymLikeEnv(GymLikeEnvInterface):
        """Wraps and vecotizes gymnasium environments."""

        def __init__(self, env: gym.Env):
            super().__init__()
            self.env = env

        def reset(self, rng):
            obs, info = self.env.reset()
            return {}, obs

        def step(self, env_state, action: int):
            obs, reward, done, truncated, info = self.env.step(action)
            return env_state, obs, reward, done, info

        @property
        def observation_space(self):
            return self.env.observation_space.shape

        @property
        def action_space(self):
            return self.env.action_space.n
        
        @property
        def num_actions(self):
            return self.env.action_space.n

        def render(self):
            return self.env.render()

        def __getattr__(self, name):
            return getattr(self.env, name)


    class VectorizedGymLikeEnv(GymLikeEnvInterface):
        """Wraps and vecotizes gym-like environments."""

        def __init__(self, make_env: Callable[..., gym.Env], n_envs: int, vec_env_type: str = "dummy"):
            super().__init__()
            if vec_env_type.lower() == "dummy":
                self.env = DummyVecEnv([make_env for _ in range(n_envs)])
                self._observation_space = self.env.envs[0].observation_space
                self._action_space = self.env.envs[0].action_space
            elif vec_env_type.lower() == "subproc":
                self.env = SubprocVecEnv([make_env for _ in range(n_envs)])
                self.env.remotes[0].send(("get_spaces", None))
                self._observation_space, self._action_space = self.env.remotes[0].recv()
            else:
                raise ValueError(f"Invalid vectorized environment type: {vec_env_type}")

        def reset(self, rng=None):
            obs = self.env.reset()
            return {}, obs

        def step(self, env_state, action: int):
            obs, reward, done, info = self.env.step(action)
            return env_state, obs, reward, done, {}

        @property
        def observation_space(self):
            return self._observation_space.shape

        @property
        def action_space(self):
            return self._action_space.n
        
        @property
        def num_actions(self):
            return self._action_space.n

        def render(self):
            raise NotImplementedError("Vectorized environments do not support rendering!")

        def __getattr__(self, name):
            return getattr(self.env, name)

except ImportError:
    pass