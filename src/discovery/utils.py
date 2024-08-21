from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree
import numpy as np


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces the values of a tree with the provided keyword arguments."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)


def map_stack(xs: ArrayLike) -> ArrayLike:
    def stack_nested(items):
        if isinstance(items[0], dict):
            return {k: stack_nested([item[k] for item in items]) for k in items[0]}
        elif isinstance(items[0], (list, tuple)):
            stacked_elements = [stack_nested(subitems) for subitems in zip(*items)]
            return type(items[0])(*stacked_elements)
        else:
            return np.stack(items, axis=0)
    
    return stack_nested(xs)


@jax.jit
def jax_map_stack(xs: ArrayLike) -> ArrayLike:
    return jax.tree.map(lambda *args: jnp.stack(args), *xs)


def scan_or_loop(
        use_scan: bool,
        f: Callable,
        init: Any,
        xs: ArrayLike | None = None,
        length: int | None = None,
        reverse: bool = False,
    ) -> tuple[Any, ArrayLike]:
    """If scan is True, use jax.lax.scan, otherwise use a for loop."""

    if use_scan:
        return jax.lax.scan(f, init, xs, length, reverse)

    if xs is None:
        xs = [None] * length

    if reverse:
        xs = xs[::-1]

    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    if reverse:
        ys = ys[::-1]

    return carry, map_stack(ys)