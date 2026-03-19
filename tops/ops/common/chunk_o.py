import jax
import jax.numpy as jnp


def chunk_fwd_o(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    v: jax.Array,       # [B, T, H, V]
    h: jax.Array,       # [NT_total, H, K, V]
    *,
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunk forward output computation (pure JAX reference).

    O_c = scale * ( Q_c @ H_c * exp(g_c) + causal(Q_c K_c^T * exp(g_row - g_col)) V_c )

    Note: when cu_seqlens is provided, each sequence length must be
    a multiple of chunk_size so that chunk boundaries align with
    sequence boundaries. Under this constraint the varlen case
    reduces to the standard path — just reshape h from
    [NT_total, H, K, V] to [B, NT, H, K, V].
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
      scale = K ** -0.5

    assert scale is not None
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size={C}"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % chunk_size == 0).all(), "All sequence lengths must be divisible by chunk_size"

    h = h.reshape(B, NT, H, K, V)

    # Reshape into chunks and transpose for batched matmul
    # [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    v_c = v.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)

    # Inter-chunk: Q_c @ H_c -> [B, NT, H, C, V]
    o_inter = jnp.zeros((B, NT, H, C, V), dtype=jnp.float32)
    A = jnp.zeros((B, NT, H, C, C), dtype=jnp.float32)
    o_inter += jnp.matmul(q_c, h, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Intra-chunk: Q_c @ K_c^T -> [B, NT, H, C, C]
    A += jnp.matmul(q_c, jnp.swapaxes(k_c, -2, -1), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Apply scalar gate g
    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        o_inter = o_inter * jnp.exp(g_c)[..., None]
        A = A * jnp.exp(g_c[..., :, None] - g_c[..., None, :])

    # Apply per-head fixed decay g_gamma
    if g_gamma is not None:
        ramp = g_gamma[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C]
        o_inter = o_inter * jnp.exp(ramp)[None, None, :, :, None]
        A = A * jnp.exp(ramp[..., :, None] - ramp[..., None, :])[None, None]

    # Causal mask (lower triangular: i >= j)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(causal_mask, A, 0.0)

    # Intra: A @ V_c -> [B, NT, H, C, V]
    o_intra = jnp.matmul(A.astype(v_c.dtype), v_c, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Combine
    o = (o_inter + o_intra) * scale

    # [B, NT, H, C, V] -> [B, NT, C, H, V] -> [B, T, H, V]
    o = o.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return o
