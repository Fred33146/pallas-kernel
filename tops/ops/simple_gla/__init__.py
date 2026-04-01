import enum
import jax

from .chunk import chunk_simple_gla_bwd as chunk_simple_gla_bwd
from .chunk import chunk_simple_gla_fwd as chunk_simple_gla_fwd
from .fused_chunk import fused_chunk_simple_gla_bwd as fused_chunk_simple_gla_bwd
from .fused_chunk import fused_chunk_simple_gla_fwd as fused_chunk_simple_gla_fwd
from .fused_recurrent import fused_recurrent_simple_gla as fused_recurrent_simple_gla
from .naive import simple_gla_naive as simple_gla_naive


class SimpleGLAKernelMode(enum.Enum):
    """Simple GLA kernel implementation mode."""

    CHUNK = "chunk"
    FUSED_CHUNK = "fused_chunk"


def simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK
):
    fn = None
    if mode == SimpleGLAKernelMode.CHUNK:
        fn = chunk_simple_gla_fwd
    elif mode == SimpleGLAKernelMode.FUSED_CHUNK:
        fn = fused_chunk_simple_gla_fwd        
    else:
        raise Exception(f"mode {mode} not support")

    return fn(
            q, k, v, 
            g=g, 
            g_gamma=g_gamma,
            scale=scale,
            h0=h0,
            use_ht=use_ht,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_seqlens_dev=cu_seqlens_dev,
            chunk_size=chunk_size
        )


def simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    dht: jax.Array | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK
):
    fn = None
    if mode == SimpleGLAKernelMode.CHUNK:
        fn = chunk_simple_gla_bwd
    elif mode == SimpleGLAKernelMode.FUSED_CHUNK:
        fn = fused_chunk_simple_gla_bwd
    else:
        raise Exception(f"mode {mode} not support")

    return fn(
            q, k, v, 
            do=do,
            g=g, 
            g_gamma=g_gamma,
            scale=scale,
            h0=h0,
            dht=dht,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cu_seqlens_dev=cu_seqlens_dev,
            chunk_size=chunk_size
        )