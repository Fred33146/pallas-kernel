"""Pallas fused backward kernel for simple GLA (g_gamma only).

Simple GLA uses per-head scalar gates g_gamma: [H] instead of per-element
gates gk: [B,T,H,K]. The gate for position t within a chunk is:
    b_g[t] = gamma * (t + 1)

This kernel computes dq, dk, dv in a single fused pass (no dg needed since
g_gamma is a hyperparameter with no gradient).
"""

import functools

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tops.ops.utils import exp

def chunk_simple_gla_bwd_kernel(
    q_ref, k_ref, v_ref, g_gamma_ref, h_ref, a_ref, do_ref, dh_ref,
    dq_ref, dk_ref, dv_ref,
    *,
    BT: int,
    scale: float,
):
    """Fused backward kernel for simple GLA with g_gamma.

    Grid: (H, total_NT)
    Refs (after block spec indexing):
      q_ref/k_ref: (1, 1, BT, K)
      v_ref/do_ref: (1, 1, BT, V)
      g_gamma_ref: [H] in SMEM
      h_ref/dh_ref: (1, 1, K, V)
      a_ref: (1, 1, BT, BT)
      dq_ref/dk_ref: (1, 1, BT, K)
      dv_ref: (1, 1, BT, V)
    """
    b_q = q_ref[0, 0]    # (BT, K)
    b_k = k_ref[0, 0]    # (BT, K)
    b_v = v_ref[0, 0]    # (BT, V)
    b_h = h_ref[0, 0].astype(jnp.float32)    # (K, V)
    b_a = a_ref[0, 0].astype(jnp.float32)    # (BT, BT)
    b_do = do_ref[0, 0]  # (BT, V)
    b_dh = dh_ref[0, 0].astype(jnp.float32)  # (K, V)

    # Build per-position decay from g_gamma
    head_idx = pl.program_id(0)
    b_gamma = g_gamma_ref[head_idx]
    b_g = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)  # [BT]
    b_gn = b_g[BT - 1]  # scalar — last position decay

    # 1. dA = do @ v^T * scale, masked lower-triangular
    b_A_do = b_do.astype(b_v.dtype)
    b_dA = jnp.dot(b_A_do, b_v.T, precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32) * scale
    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_dA = jnp.where(mask, b_dA, 0.0)

    # 2. dv = A^T @ do + k_decay @ dh
    b_a_masked = jnp.where(mask, b_a, 0.0)
    b_dv_intra = jnp.dot(b_a_masked.T.astype(b_do.dtype), b_do,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    k_decay = (b_k * jnp.exp(b_gn - b_g)[:, None]).astype(b_k.dtype)
    b_dv_inter = jnp.dot(k_decay, b_dh.astype(k_decay.dtype),
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    b_dv = b_dv_intra + b_dv_inter
    dv_ref[0, 0] = b_dv.astype(dv_ref.dtype)

    # 3. dq = dA @ k_neg * exp(b_g) + do @ h^T * scale * exp(b_g)
    k_neg = (b_k * jnp.exp(-b_g)[:, None]).astype(b_k.dtype)
    b_dq_intra = jnp.dot(b_dA.astype(k_neg.dtype), k_neg,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(b_g)[:, None]
    b_dq_inter = jnp.dot(b_do, b_h.astype(b_do.dtype).T,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * (scale * jnp.exp(b_g)[:, None])
    b_dq = b_dq_intra + b_dq_inter
    dq_ref[0, 0] = b_dq.astype(dq_ref.dtype)

    # 4. dk = dA^T @ q_pos * exp(-b_g) + v @ dh^T * exp(b_gn - b_g)
    q_pos = (b_q * jnp.exp(b_g)[:, None]).astype(b_q.dtype)
    b_dk_intra = jnp.dot(b_dA.T.astype(q_pos.dtype), q_pos,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(-b_g)[:, None]
    b_dk_inter = jnp.dot(b_v, b_dh.astype(b_v.dtype).T,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32) * jnp.exp(b_gn - b_g)[:, None]
    b_dk = b_dk_intra + b_dk_inter
    dk_ref[0, 0] = b_dk.astype(dk_ref.dtype)


def chunk_simple_gla_bwd_o_pl(
    q: jax.Array,   # [B, T, H, K]
    k: jax.Array,   # [B, T, H, K]
    v: jax.Array,   # [B, T, H, V]
    g_gamma: jax.Array,  # [H]
    h: jax.Array,   # [B, NT, H, K, V]
    A: jax.Array,   # [B, T, H, BT]
    do: jax.Array,  # [B, T, H, V]
    dh: jax.Array,  # [B, NT, H, K, V]
    scale: float,
    chunk_size: int,
):
    """Launcher for the fused simple GLA backward kernel.

    Returns: (dq, dk, dv)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    # Reshape: [B, T, H, D] -> [H, total_NT, BT, D]
    def _reshape_bt(x, D):
        return x.reshape(B, NT, BT, H, D).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, D)

    _q = _reshape_bt(q, K)
    _k = _reshape_bt(k, K)
    _v = _reshape_bt(v, V)
    _do = _reshape_bt(do, V)
    _A = A.reshape(B, NT, BT, H, BT).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, BT)
    # h/dh: [B, NT, H, K, V] -> [H, total_NT, K, V]
    _h = h.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _dh = dh.transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)

    # BlockSpecs
    grid = (H, total_NT)
    spec_K = pl.BlockSpec([1, 1, BT, K], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_V = pl.BlockSpec([1, 1, BT, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_h = pl.BlockSpec([1, 1, K, V], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_A = pl.BlockSpec([1, 1, BT, BT], index_map=lambda h, nt: (h, nt, 0, 0))
    spec_gamma = pl.BlockSpec(memory_space=pltpu.SMEM)

    dq_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], q.dtype)
    dk_shape = jax.ShapeDtypeStruct([H, total_NT, BT, K], k.dtype)
    dv_shape = jax.ShapeDtypeStruct([H, total_NT, BT, V], v.dtype)

    dq, dk, dv = pl.pallas_call(
        functools.partial(chunk_simple_gla_bwd_kernel, BT=BT, scale=scale),
        grid=grid,
        out_shape=[dq_shape, dk_shape, dv_shape],
        in_specs=[spec_K, spec_K, spec_V, spec_gamma, spec_h, spec_A, spec_V, spec_h],
        out_specs=[spec_K, spec_K, spec_V],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
    )(_q, _k, _v, g_gamma, _h, _A, _do, _dh)

    # Post-process: (H, total_NT, BT, D) -> (B, T, H, D)
    def _unreshape(x, D):
        x = x.reshape(H, B, NT, BT, D)
        x = x.transpose(1, 0, 2, 3, 4)  # (B, H, NT, BT, D)
        x = x.reshape(B, H, T, D)
        return x.transpose(0, 2, 1, 3)   # (B, T, H, D)

    dq = _unreshape(dq, K)
    dk = _unreshape(dk, K)
    dv = _unreshape(dv, V)

    return dq, dk, dv

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
        o_inter = o_inter * exp(g_c)[..., None]
        A = A * exp(g_c[..., :, None] - g_c[..., None, :])

    # Apply per-head fixed decay g_gamma
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        ramp = g_gamma_f32[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C] float32
        o_inter = o_inter * exp(ramp)[None, None, :, :, None]
        A = A * exp(ramp[..., :, None] - ramp[..., None, :])[None, None]

    # Causal mask (lower triangular: i >= j)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(causal_mask, A, 0.0)

    # Intra: A @ V_c -> [B, NT, H, C, V]
    o_intra = jnp.matmul(A.astype(v_c.dtype), v_c, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Combine
    o = (o_inter + o_intra) * scale

    # [B, NT, H, C, V] -> [B, NT, C, H, V] -> [B, T, H, V]
    o = o.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return o.astype(v.dtype)  # Cast back to input dtype


def chunk_bwd_dv(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    do: jax.Array,      # [B, T, H, V]
    dh: jax.Array,      # [NT_total, H, K, V]
    *,
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Backward: gradient of v (pure JAX reference).

    dv = dv_inter + dv_intra
      dv_inter[j] = sum_k  k[j,k] * dh[k,v] * exp(-g[j] + g_last)   (inter-chunk)
      dv_intra[j] = sum_{i>=j} A[j,i] * do[i]                        (intra-chunk)

    where A[j,i] = sum_k k[j,k]*q[i,k] * exp(g[i]-g[j]) * scale,  masked j<=i

    Note: when cu_seqlens_cpu is provided, each sequence length must be
    a multiple of chunk_size.
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
        scale = K ** -0.5

    assert T % C == 0
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all()

    dh = dh.reshape(B, NT, H, K, V)

    # Reshape into chunks: [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)   # [B, NT, H, C, K]
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)   # [B, NT, H, C, K]
    do_c = do.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)  # [B, NT, H, C, V]

    # Inter-chunk: K_c @ dH -> [B, NT, H, C, V]
    dv_inter = jnp.matmul(
        k_c, dh,
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Intra-chunk attention: K_c @ Q_c^T -> [B, NT, H, C, C]
    # A[j, i] = k[j] @ q[i]^T  (note: row=key, col=query)
    A = jnp.matmul(
        k_c, jnp.swapaxes(q_c, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Gate: g is chunk-local cumsum [B, T, H]
    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        g_last = g_c[..., -1:]  # [B, NT, H, 1]
        # Inter-chunk gate: exp(-g_j + g_last)
        dv_inter = dv_inter * exp(-g_c + g_last)[..., None]
        # Intra-chunk gate: A[j,i] *= exp(g_i - g_j)
        # row=j (key), col=i (query), so: g_c[..., None, :] - g_c[..., :, None]
        A = A * exp(g_c[..., None, :] - g_c[..., :, None])

    # Gate: g_gamma — per-head fixed decay [H]
    if g_gamma is not None:
        ramp = g_gamma[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C]
        ramp_last = ramp[..., -1:]  # [H, 1]
        dv_inter = dv_inter * exp(-ramp + ramp_last)[None, None, :, :, None]
        A = A * exp(ramp[..., None, :] - ramp[..., :, None])[None, None]

    # Upper-triangular mask: keep j <= i  (row=j, col=i)
    upper_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(upper_mask, A, 0.0) * scale

    # Intra-chunk: A @ do -> [B, NT, H, C, V]
    dv_intra = jnp.matmul(
        A.astype(do_c.dtype), do_c,
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    dv = dv_inter + dv_intra

    # [B, NT, H, C, V] -> [B, NT, C, H, V] -> [B, T, H, V]
    dv = dv.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return dv


def chunk_bwd_dqkwg(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    v: jax.Array,       # [B, T, H, V]
    h: jax.Array,       # [NT_total, H, K, V]
    do: jax.Array,      # [B, T, H, V]
    dh: jax.Array,      # [NT_total, H, K, V]
    *,
    w: jax.Array | None = None,        # [B, T, H, K] (optional, for delta rule)
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    dv: jax.Array | None = None,       # [B, T, H, V] (optional, for efficiency/delta rule)
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    """Backward: gradients of q, k, w and g (pure JAX reference).

    Matches Triton ``chunk_bwd_kernel_dqkwg`` semantics exactly.

    For Gated Linear Attention / RetNet:
      dq = (do @ h^T * exp(g) + causal(ds) @ k) * scale
      dk = v @ dh^T * exp(-g + g_last) + causal(ds)^T @ q

    For Delta Rule (when w and dv are provided):
      dw = -(dv @ h^T)

    where ds = causal(do @ v^T * exp(g_row - g_col)) * scale.

    Returns (dq, dk, dw, dg).

    Note: when g is provided, the returned dg is the *raw* per-position
    gradient (with dg_last folded into the last position of each chunk,
    but NOT reverse-cumsummed).  The caller must apply reverse cumsum
    to obtain the final gate gradient.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
        scale = K ** -0.5

    assert scale is not None
    assert T % C == 0
    assert (cu_seqlens is None) or (cu_seqlens % C == 0).all()

    h = h.reshape(B, NT, H, K, V)
    dh = dh.reshape(B, NT, H, K, V)

    # Reshape into chunks: [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    v_c = v.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)
    do_c = do.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)

    # Inter-chunk: dq_inter = do @ h^T -> [B, NT, H, C, K]
    dq_inter = jnp.matmul(
        do_c, jnp.swapaxes(h, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Inter-chunk: dk_inter = v @ dh^T -> [B, NT, H, C, K]
    dk_inter = jnp.matmul(
        v_c, jnp.swapaxes(dh, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Delta Rule specific backward to w
    dw = None
    if w is not None and dv is not None:
        dv_c = dv.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)
        dw_c = jnp.matmul(
            dv_c, jnp.swapaxes(h, -2, -1),
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dw = -dw_c.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K).astype(w.dtype)

    # Intra-chunk: ds = do @ v^T -> [B, NT, H, C, C]
    ds = jnp.matmul(
        do_c, jnp.swapaxes(v_c, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Causal mask (lower triangular: i >= j)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))

    dg = None

    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        g_last_per_chunk = g_c[..., -1:]  # [B, NT, H, 1]

        # Triton V-loop: b_dg_last += sum(b_h * b_dh) across all V,K blocks
        # Then after the loop: b_dg_last *= exp(b_g_last)
        dg_last_term = jnp.sum(h * dh, axis=(-2, -1))  # [B, NT, H]
        dg_last_term = dg_last_term * exp(g_last_per_chunk[..., 0])  # [B, NT, H]

        # 1. dq with gate
        # b_dq = b_dq * exp(b_g)[:, None] * scale
        dq_gate = dq_inter * exp(g_c)[..., None] * scale
        # b_dg += tl.sum(b_dq * b_q, axis=1)
        dg_from_dq = jnp.sum(dq_gate * q_c, axis=-1)  # [B, NT, H, C]

        # 2. dk with gate
        # b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        dk_gate = dk_inter * exp(-g_c + g_last_per_chunk)[..., None]
        # b_dg -= tl.sum(b_k * b_dk, axis=1)
        dg_from_dk = -jnp.sum(k_c * dk_gate, axis=-1)   # [B, NT, H, C]
        # b_dg_last += tl.sum(b_dk * b_k)
        dg_last_term = dg_last_term + jnp.sum(dk_gate * k_c, axis=(-2, -1))  # [B, NT, H]

        # 3. ds with gate and causal mask
        # b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        ds_gate = jnp.where(causal_mask, ds * exp(g_c[..., :, None] - g_c[..., None, :]), 0.0) * scale

        # b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        qk = jnp.matmul(
            q_c, jnp.swapaxes(k_c, -2, -1),
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        ds2 = ds_gate * qk

        # b_dg += tl.sum(b_ds2, axis=1)  — row-sum (reduce over key dim j)
        # b_dg -= tl.sum(b_ds2, axis=0)  — col-sum (reduce over query dim i)
        dg_from_ds = jnp.sum(ds2, axis=-1) - jnp.sum(ds2, axis=-2)  # [B, NT, H, C]

        # Intra-chunk: dq += ds @ k, dk += ds^T @ q
        ds_cast = ds_gate.astype(k_c.dtype)
        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )

        dq = dq_gate + dq_intra
        dk = dk_gate + dk_intra

        # Assemble per-position dg, fold dg_last into the last position of each chunk.
        # Triton: b_dg = tl.where(o_t < last, b_dg, b_dg + b_dg_last)
        # NOTE: reverse cumsum is NOT applied here — matches Triton kernel output.
        #       The caller must apply reverse cumsum to obtain the final gate gradient.
        dg_c = dg_from_dq + dg_from_dk + dg_from_ds   # [B, NT, H, C]
        dg_c = dg_c.at[..., -1].add(dg_last_term)

        # [B, NT, H, C] -> [B, NT, C, H] -> [B, T, H]
        dg = dg_c.transpose(0, 1, 3, 2).reshape(B, T, H)

    elif g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        ramp = g_gamma_f32[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C]
        ramp_last = ramp[:, -1:]  # [H, 1]

        # dq_inter with gamma
        dq_inter = dq_inter * exp(ramp)[None, None, :, :, None] * scale
        # dk_inter with gamma
        dk_inter = dk_inter * exp(-ramp + ramp_last)[None, None, :, :, None]

        # ds with gamma
        ds = jnp.where(
            causal_mask,
            ds * exp(ramp[..., :, None] - ramp[..., None, :])[None, None],
            0.0,
        ) * scale

        # Intra-chunk
        ds_cast = ds.astype(k_c.dtype)
        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )

        dq = dq_inter + dq_intra
        dk = dk_inter + dk_intra

    else:
        # No gate
        ds = jnp.where(causal_mask, ds, 0.0)
        ds_cast = ds.astype(k_c.dtype)

        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        # Note: Triton applies scale to dk_intra * scale
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        ) * scale

        dq = (dq_inter + dq_intra) * scale
        dk = dk_inter + dk_intra

    # [B, NT, H, C, K] -> [B, NT, C, H, K] -> [B, T, H, K]
    dq = dq.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)

    return dq.astype(q.dtype), dk.astype(k.dtype), dw, dg
