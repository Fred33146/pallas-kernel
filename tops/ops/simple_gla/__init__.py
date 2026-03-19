import jax
import jax.numpy as jnp


def simple_gla_naive(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens_cpu: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Naive JAX implementation of GLA for testing."""
    _B, _T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    scale = float(scale)
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32) if g is not None else None
    g_gamma = g_gamma.astype(jnp.float32) if g_gamma is not None else None
    initial_state = (
        initial_state.astype(jnp.float32) if initial_state is not None else None
    )
    N = len(cu_seqlens_cpu) - 1 if cu_seqlens_cpu is not None else _B

    assert (g is None) or (g.ndim == 4)
    assert (g_gamma is None) or (g_gamma.ndim == 1)

    if cu_seqlens_cpu is not None:
        q = q.reshape(1, -1, H, K)
        k = k.reshape(1, -1, H, K)
        v = v.reshape(1, -1, H, V)
        g = g.reshape(1, -1, H, K) if g is not None else None

    B = q.shape[0]
    T = q.shape[1]

    if g is None:
        if g_gamma is not None:
            g = jnp.broadcast_to(g_gamma[None, None, :, None], (B, T, H, K))
        else:
            g = jnp.zeros((B, T, H, K), dtype=jnp.float32)
    else:
        if g_gamma is not None:
            g = g + g_gamma[None, None, :, None]

    q = q.reshape(-1, H, K)
    k = k.reshape(-1, H, K)
    v = v.reshape(-1, H, V)
    if g is not None:
        g = g.reshape(-1, H, K)
    if initial_state is not None:
        initial_state = initial_state.reshape(N, H, K, V)

    q = q * scale
    o = jnp.zeros((B * T, H, V), dtype=jnp.float32)
    S = jnp.zeros((N, H, K, V), dtype=jnp.float32)
    if initial_state is not None:
        S += initial_state

    for i_b in range(N):
        if cu_seqlens_cpu is not None:
            bos, eos = cu_seqlens_cpu[i_b], cu_seqlens_cpu[i_b + 1]
        else:
            bos = i_b * T
            eos = (i_b + 1) * T

        t = eos - bos
        s = S[i_b]  # [H, K, V]
        for i_t in range(t):
            q_b = q[bos:eos][i_t]  # [H, K]
            k_b = k[bos:eos][i_t]
            v_b = v[bos:eos][i_t]  # [H, V]
            g_b = g[bos:eos][i_t]  # [H, K]

            g_b = jnp.exp(g_b)
            kv = k_b[..., None] * v_b[:, None, :]
            s = s * g_b[..., None] + kv
            o_b = (q_b[..., None] * s).sum(axis=1)  # [H, V]

            o = o.at[bos + i_t].set(o_b)
        S = S.at[i_b].set(s)
    if not output_final_state:
        S = None

    return o.reshape(B, T, H, V), S
