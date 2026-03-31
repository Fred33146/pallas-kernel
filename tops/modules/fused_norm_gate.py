import jax
import jax.numpy as jnp
from flax import nnx


class FusedRMSNormGated(nnx.Module):
    """RMSNorm + SiLU gate fusion (JAX/Flax NNX version).

    output = RMSNorm(x) * silu(g)
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nnx.Param(jnp.ones(hidden_size))
        else:
            self.weight = None

    def __call__(self, x: jax.Array, g: jax.Array) -> jax.Array:
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        g = g.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x = x * rms
        if self.weight is not None:
            x = x * self.weight.value.astype(jnp.float32)
        x = x * jax.nn.silu(g)
        return x.astype(input_dtype)
