
from tops.ops.common.fused_chunk import fused_chunk_fwd
import jax

def fused_chunk_simple_gla_fwd(
    q: jax.Array,                       # [B, T, H, K]
    k: jax.Array,                       # [B, T, H, K]
    v: jax.Array,                          # [B, T, H, V]
    *,
    g: jax.Array | None = None,          # [B, T, H]  chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,    # [H]         per-head fixed decay rate
    h0: jax.Array | None = None,         # [B, H, K, V] initial hidden state
    scale: float | None = None,
    use_ht: bool = False,
    chunk_size: int = 64,
    interpret: bool | None = None,
):
    return fused_chunk_fwd(
        q,
        k,
        v,
        g=g,
        g_gamma=g_gamma,
        h0=h0,
        scale=scale,
        use_ht=use_ht,
        chunk_size=chunk_size,
        interpret=interpret
    )

def fused_chunk_simple_gla_bwd():
    pass