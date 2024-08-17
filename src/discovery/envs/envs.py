from omegaconf import DictConfig

from .equinox_env import EquinoxEnv, XMinigridEqxEnv
from discovery.utils import tree_replace


VALID_ENV_TYPES = ['xminigrid']


def resolve_xminigrid_wrapper(name: str):
    """Converts a wrapper class name into an xminigrid wrapper class."""
    if name.lower() == 'rgbimgobservationwrapper':
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper
        return RGBImgObservationWrapper
    else:
        import xminigrid.wrappers
        from . import wrappers as custom_wrappers
        if hasattr(xminigrid.wrappers, name):
            return getattr(xminigrid.wrappers, name)
        elif hasattr(custom_wrappers, name):
            return getattr(custom_wrappers, name)

    raise ValueError(f"Invalid wrapper name for xminigrid: {name}")


def create_env(env_config: DictConfig) -> EquinoxEnv:
    """Higher level function to create an environment from a config."""

    if env_config.type.lower() == 'xminigrid':
        import xminigrid
        env, env_params = xminigrid.make(env_config.name)
        for wrapper_name in env_config.get('wrappers', []):
            wrapper_class = resolve_xminigrid_wrapper(wrapper_name)
            env = wrapper_class(env)
        env = XMinigridEqxEnv(env, env_params)

    else:
        raise ValueError(
            f"Invalid environment type: {env_config.type}"
            f"Supported types: {VALID_ENV_TYPES}"
        )
        
    if env_config.get('disable_jit', False):
        env = tree_replace(env, jittable=False)
    
    return env