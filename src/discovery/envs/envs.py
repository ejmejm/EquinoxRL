from omegaconf import DictConfig

from .base_env import BaseEnv, VectorizedGymLikeEnv, XMinigridEqxEnv


VALID_ENV_TYPES = ['xminigrid']


def resolve_xminigrid_wrapper(name: str):
    """Converts a wrapper class name into an xminigrid wrapper class."""
    if name.lower() == 'rgbimgobservationwrapper':
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper
        return RGBImgObservationWrapper
    else:
        import xminigrid.wrappers
        from . import eqx_wrappers as custom_eqx_wrappers
        if hasattr(xminigrid.wrappers, name):
            return getattr(xminigrid.wrappers, name)
        elif hasattr(custom_eqx_wrappers, name):
            return getattr(custom_eqx_wrappers, name)

    raise ValueError(f"Invalid wrapper name for xminigrid: {name}")


def resolve_gym_wrapper(name: str):
    """Converts a wrapper class name into a gymnasium wrapper class."""
    import gymnasium.wrappers
    from . import gym_wrappers as custom_gym_wrappers
    if hasattr(gymnasium.wrappers, name):
        return getattr(gymnasium.wrappers, name)
    elif hasattr(custom_gym_wrappers, name):
        return getattr(custom_gym_wrappers, name)

    raise ValueError(f"Invalid wrapper name for a gymnasium environment: {name}")


def create_env(env_config: DictConfig) -> BaseEnv:
    """Higher level function to create an environment from a config."""

    if env_config.type.lower() == 'xminigrid':
        import xminigrid
        env, env_params = xminigrid.make(env_config.name)
        for wrapper_name in env_config.get('wrappers', []):
            wrapper_class = resolve_xminigrid_wrapper(wrapper_name)
            env = wrapper_class(env)
        env = XMinigridEqxEnv(env, env_params)
        
    elif env_config.type.lower() == 'collect_objects':
        from .new_envs.collect_objects import CollectObjectsEnv
        
        def make_env_fn():
            env = CollectObjectsEnv()
            for wrapper_name in env_config.get('wrappers', []):
                wrapper_class = resolve_gym_wrapper(wrapper_name)
                env = wrapper_class(env)
            return env
        
        env = VectorizedGymLikeEnv(make_env_fn, env_config.n_envs)
        
    elif env_config.type.lower() == 'gymnasium':
        pass
        # env = VectorEnv()
        # env = gym.make(env_config.name)

    else:
        raise ValueError(
            f"Invalid environment type: {env_config.type}"
            f"Supported types: {VALID_ENV_TYPES}"
        )

    return env