from typing import Any, Dict, Optional, Tuple

import equinox as eqx
from jaxtyping import PyTree, PRNGKeyArray


class EquinoxEnv(eqx.Module):
    """Interface for Equinox environments with a shared gym-like interface."""
    jittable: bool = eqx.field(static=True, default=False)

    def reset(self, rng: PRNGKeyArray) -> Tuple[PyTree, PyTree]:
        """Reset the environment and return the initial observation.

        Returns:
            A tuple containing two elements:
            (new_env, observation)
        """
        raise NotImplementedError

    def step(self, env_state: PyTree, action: int) -> Tuple[PyTree, PyTree, float, bool, Dict[str, Any]]:
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


##### Gymnax #####


try:
    import gymnax
    from gymnax.environments.environment import Environment as GymnaxEnv

    class GymnaxEqxEnv(EquinoxEnv):
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

    class XMinigridEqxEnv(EquinoxEnv):
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