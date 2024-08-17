from typing import Callable, List, Optional, Sequence, Tuple

import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


FeatureExtractor = eqx.Module


class CNNFeatureExtractor(FeatureExtractor):
    input_dim: Tuple[int] = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    activation_fn: eqx.Module = eqx.field(static=True)
    
    conv_layers: List[nn.Conv2d]
    linear_layer: nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        input_dim: Sequence[int],
        output_dim: int,
        activation_fn: Optional[Callable] = None,
    ):
        """A convolutional feature extractor that outputs a fixed sized feature vector.
        
        Args:
            key (PRNGKeyArray): The PRNG key used to initialize weights.
            input_dim (Sequence[int]): The shape of the input image.
            output_dim (int): The size of the output feature vector.
            activation_fn (eqx.Module, optional): The activation function. Defaults to jax.nn.gelu.
        """
        assert len(input_dim) == 3, "Input dimension must be 3D (C, H, W)"

        activation_fn = activation_fn or jax.nn.gelu

        filters = (8, 6, 4)
        strides = (2, 2, 2)
        channels = (input_dim[0], 64, 64, 64)
        
        keys = jax.random.split(key, len(filters) + 1)

        self.input_dim = tuple(input_dim)
        self.output_dim = output_dim
        self.activation_fn = activation_fn
  
        self.conv_layers = []
        for i in range(len(filters)):
            self.conv_layers.append(nn.Conv2d(channels[i], channels[i+1], filters[i], strides[i], key=keys[i]))

        sample_input = jnp.ones(input_dim, dtype=jnp.float32)
        z = sample_input
        for layer in self.conv_layers:
            z = layer(z)
            
        conv_out_dim = z.size
        self.linear_layer = nn.Linear(conv_out_dim, output_dim, key=keys[-1])

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        z = x
        for layer in self.conv_layers:
            z = layer(z)
            z = self.activation_fn(z)
        out = self.linear_layer(z.flatten())
        return out


def make_mlp(
        key: PRNGKeyArray,
        layer_sizes: Sequence[int],
        activation_fn: Optional[Callable] = None,
    ) -> List[eqx.Module]:
    """Create a multi-layer perceptron with the given layer sizes and activation function.
    
    Args:
        key (PRNGKeyArray): The PRNG key used to initialize weights.
        layer_sizes (Sequence[int]): The sizes of the hidden layers.
        activation_fn (Callable, optional): The activation function. Defaults to jax.nn.gelu.
    """
    activation_fn = activation_fn or jax.nn.gelu
    class ActivationFn(eqx.Module):
        def __call__(self, x: Array, *, key = None) -> Array:
            return activation_fn(x)

    layers = []
    for i in range(1, len(layer_sizes)):
        last_layer = i == len(layer_sizes) - 1
        layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i], key=key, use_bias=not last_layer))
        if not last_layer:
            layers.append(ActivationFn())
    return layers


class ActorCriticModel(eqx.Module):
    action_dim: int = eqx.field(static=True)
    activation_fn: eqx.Module = eqx.field(static=True)

    feature_extractor: FeatureExtractor
    actor: eqx.Module
    critic: eqx.Module

    def __init__(
            self,
            key: Array,
            feature_extractor: FeatureExtractor,
            action_dim: int,
            actor_layer_sizes: Sequence[int],
            critic_layer_sizes: Sequence[int],
            activation_fn: Optional[Callable] = None,
        ):
        """A model that combines a feature extractor, actor, and critic.
        
        Args:
            key (Array): The PRNG key used to initialize weights.
            feature_extractor (FeatureExtractor): The feature extractor.
            action_dim (int): The (1D) size of the action space.
            actor_layer_sizes (Sequence[int]): The sizes of the hidden layers in the actor network.
            critic_layer_sizes (Sequence[int]): The sizes of the hidden layers in the critic network.
            activation_fn (Callable, optional): The activation function. Defaults to jax.nn.gelu.
        """
        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        self.activation_fn = activation_fn or jax.nn.gelu

        gen_keys = jax.random.split(key, 3)
        input_dim = feature_extractor.output_dim
        self.actor = nn.Sequential(make_mlp(
            key = gen_keys[0],
            layer_sizes = [input_dim] + list(actor_layer_sizes) + [action_dim],
            activation_fn = activation_fn,
        ))
        self.critic = nn.Sequential(make_mlp(
            key = gen_keys[1],
            layer_sizes = [input_dim] + list(critic_layer_sizes) + [1],
            activation_fn = activation_fn,
        ))

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Tuple[Array, Array]:
        z = self.feature_extractor(x)
        act_logits = self.actor(z)
        value = self.critic(z)[0]
        return act_logits, value
    
    def act_logits(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        z = self.feature_extractor(x)
        act_logits = self.actor(z)
        return act_logits
    
    def value(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        z = self.feature_extractor(x)
        value = self.critic(z)[0]
        return value


ACTIVATION_FN_MAP = {
    'gelu': jax.nn.gelu,
    'relu': jax.nn.relu,
    'tanh': jnp.tanh,
}


FEATURE_EXTRACTOR_MAP = {
    'cnn': CNNFeatureExtractor,
}


def get_activation_fn(name: str) -> Callable:
    name = name.lower()
    if name not in ACTIVATION_FN_MAP:
        raise ValueError(
            f"Unknown activation function: '{name}'. "
            f"Available types are: {', '.join(ACTIVATION_FN_MAP.keys())}"
        )
    return ACTIVATION_FN_MAP[name]


def get_feature_extractor_cls(name: str) -> FeatureExtractor:
    name = name.lower()
    if name not in FEATURE_EXTRACTOR_MAP:
        raise ValueError(
            f"Unknown feature extractor type: '{name}'. "
            f"Available types are: {', '.join(FEATURE_EXTRACTOR_MAP.keys())}"
        )
    return FEATURE_EXTRACTOR_MAP[name]
