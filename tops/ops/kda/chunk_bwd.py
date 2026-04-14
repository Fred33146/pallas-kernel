import jax.numpy as jnp
import jax

from tops.ops.utils import exp2
from tops.utils import assert_shape
from jax.experimental import pallas as pl
from functools import partial

from tops.ops.utils import is_tpu_runtime
from tops.utils import assert_shape_or_none

def _chunk_kda_bwd_wy_dqkg_fused_kernel(
    # --- 11 inputs ---
    q_ref, k_ref, v_ref, vn_ref, g_ref,
    beta_ref, A_ref, h_ref, do_ref, dh_ref, dv_ref,
    # --- 6 outputs ---
    dq_ref, dk_ref, dv2_ref, dg_ref, db_ref, dA_ref,
    *, scale, BT, BK, BV, NK, NV, K, V,
):
    """Pallas kernel body — processes one (chunk, batch*head) tile.

    K and V are tiled into BK and BV blocks via jax.lax.fori_loop.
    """
    # Each Ref has a leading singleton dim from the BlockSpec: [1, ...]
    bq   = q_ref[0]       # [BT, K]
    bk   = k_ref[0]       # [BT, K]
    bv   = v_ref[0]       # [BT, V]
    bvn  = vn_ref[0]      # [BT, V]
    bg   = g_ref[0]       # [BT, K]
    bb   = beta_ref[0]    # [BT]
    bA   = A_ref[0].T     # [BT, BT] — transposed (matching Triton load)
    bh   = h_ref[0]       # [K, V]
    bdh  = dh_ref[0]      # [K, V]
    bdo  = do_ref[0]      # [BT, V]
    bdv  = dv_ref[0]      # [BT, V]

    o_t = jnp.arange(BT)
    m_last  = (o_t == BT - 1).astype(jnp.float32)
    m_lower = o_t[:, None] > o_t[None, :]

    # --- Helper: slice along last dim ---
    def _sl2d_t(x, start, size):
        """x[..., start:start+size] for 2D [BT, D]."""
        return jax.lax.dynamic_slice(x, (0, start), (BT, size))

    def _sl2d_kv(x, rs, cs, rsize, csize):
        """x[rs:rs+rsize, cs:cs+csize] for 2D [K, V]."""
        return jax.lax.dynamic_slice(x, (rs, cs), (rsize, csize))

    # --- Inner V loop body (for fixed i_k) ---
    def v_loop_body(i_v, carry):
        b_dq_k, b_dk_k, b_dw_k, b_dgk, b_dA_acc, b_dv2, b_db_acc, i_k = carry
        vs = i_v * BV

        bvn_blk = _sl2d_t(bvn, vs, BV)     # [BT, BV]
        bdo_blk = _sl2d_t(bdo, vs, BV)     # [BT, BV]
        bdv_blk = _sl2d_t(bdv, vs, BV)     # [BT, BV]
        bh_blk  = _sl2d_kv(bh, i_k * BK, vs, BK, BV)   # [BK, BV]
        bdh_blk = _sl2d_kv(bdh, i_k * BK, vs, BK, BV)  # [BK, BV]

        b_dgk  = b_dgk + (bh_blk * bdh_blk).sum(axis=1)
        b_dq_k = b_dq_k + bdo_blk @ bh_blk.T
        b_dk_k = b_dk_k + bvn_blk @ bdh_blk.T
        b_dw_k = b_dw_k + bdv_blk @ bh_blk.T

        # i_k == 0 branch
        bv_blk = _sl2d_t(bv, vs, BV)                    # [BT, BV]
        new_dA = b_dA_acc + bdv_blk @ bv_blk.T
        b_dvb  = bA @ bdv_blk                            # [BT, BV]
        new_dv2 = jax.lax.dynamic_update_slice(
            b_dv2, b_dvb * bb[:, None], (0, vs))
        new_db = b_db_acc + (b_dvb * bv_blk).sum(axis=1)
        # Only apply the i_k==0 updates when i_k is actually 0
        b_dA_acc = jnp.where(i_k == 0, new_dA, b_dA_acc)
        b_dv2    = jnp.where(i_k == 0, new_dv2, b_dv2)
        b_db_acc = jnp.where(i_k == 0, new_db, b_db_acc)

        return (b_dq_k, b_dk_k, b_dw_k, b_dgk, b_dA_acc, b_dv2, b_db_acc, i_k)

    # --- Outer K loop body ---
    def k_loop_body(i_k, carry):
        b_dA_acc, b_db_acc, b_dv2, b_dq_full, b_dk_full, b_dg_full = carry
        ks = i_k * BK

        bk_blk = _sl2d_t(bk, ks, BK)       # [BT, BK]
        bg_blk = _sl2d_t(bg, ks, BK)        # [BT, BK]
        bgn    = bg_blk[-1, :]               # [BK]

        b_dq_k = jnp.zeros((BT, BK), jnp.float32)
        b_dk_k = jnp.zeros((BT, BK), jnp.float32)
        b_dw_k = jnp.zeros((BT, BK), jnp.float32)
        b_dgk  = jnp.zeros((BK,), jnp.float32)

        # V loop
        init_v = (b_dq_k, b_dk_k, b_dw_k, b_dgk, b_dA_acc, b_dv2, b_db_acc, i_k)
        b_dq_k, b_dk_k, b_dw_k, b_dgk, b_dA_acc, b_dv2, b_db_acc, _ = \
            jax.lax.fori_loop(0, NV, v_loop_body, init_v)

        # Gate application for this K block
        gk_exp = exp2(bg_blk)
        gb     = gk_exp * bb[:, None]
        b_dgk  = b_dgk * exp2(bgn)
        b_dq_k = b_dq_k * gk_exp * scale
        b_dk_k = b_dk_k * exp2(bgn[None, :] - bg_blk)

        kg = bk_blk * gk_exp
        b_dw_k = -b_dw_k
        b_dA_acc = b_dA_acc + b_dw_k @ kg.T

        dkgb = bA @ b_dw_k
        b_db_acc = b_db_acc + (dkgb * kg).sum(axis=1)

        bq_blk = _sl2d_t(bq, ks, BK)
        kdk    = bk_blk * b_dk_k
        b_dgk  = b_dgk + kdk.sum(axis=0)

        b_dg_k = (bq_blk * b_dq_k - kdk
                  + m_last[:, None] * b_dgk[None, :]
                  + kg * dkgb * bb[:, None])
        b_dk_k = b_dk_k + dkgb * gb

        b_dq_full = jax.lax.dynamic_update_slice(b_dq_full, b_dq_k, (0, ks))
        b_dk_full = jax.lax.dynamic_update_slice(b_dk_full, b_dk_k, (0, ks))
        b_dg_full = jax.lax.dynamic_update_slice(b_dg_full, b_dg_k, (0, ks))

        return (b_dA_acc, b_db_acc, b_dv2, b_dq_full, b_dk_full, b_dg_full)

    # Initialize accumulators
    b_dA_acc  = jnp.zeros((BT, BT), jnp.float32)
    b_db_acc  = jnp.zeros((BT,), jnp.float32)
    b_dv2     = jnp.zeros((BT, V), jnp.float32)
    b_dq_full = jnp.zeros((BT, K), jnp.float32)
    b_dk_full = jnp.zeros((BT, K), jnp.float32)
    b_dg_full = jnp.zeros((BT, K), jnp.float32)

    init_k = (b_dA_acc, b_db_acc, b_dv2, b_dq_full, b_dk_full, b_dg_full)
    b_dA_acc, b_db_acc, b_dv2, b_dq_full, b_dk_full, b_dg_full = \
        jax.lax.fori_loop(0, NK, k_loop_body, init_k)

    # Post-process dA
    b_dA_acc = jnp.where(m_lower, b_dA_acc * bb[None, :], 0.0)
    b_dA_acc = b_dA_acc @ bA
    b_dA_acc = bA @ b_dA_acc
    b_dA_acc = jnp.where(m_lower, -b_dA_acc, 0.0)

    # Store outputs
    dq_ref[0]  = b_dq_full
    dk_ref[0]  = b_dk_full
    dv2_ref[0] = b_dv2
    dg_ref[0]  = b_dg_full
    db_ref[0]  = b_db_acc
    dA_ref[0]  = b_dA_acc


def chunk_kda_bwd_wy_dqkg_fused_kernel(
    q, k, v, v_new, g, beta, A, h, do, dh, dv,
    scale, chunk_size=64, block_K=None, block_V=None,
):
    """
    JAX Pallas implementation of chunk_kda_bwd_wy_dqkg_fused.

    Args:
        q:      [B, T, H, K]  query tensor.
        k:      [B, T, H, K]  key tensor.
        v:      [B, T, H, V]  original value tensor.
        v_new:  [B, T, H, V]  WY-transformed value tensor.
        g:      [B, T, H, K]  log-space cumsum gate (base-2 scaled).
        beta:   [B, T, H]     WY beta coefficients.
        A:      [B, T, H, BT] Akk inverse matrix (chunk_size = BT).
        h:      [B*NT, H, K, V]  per-chunk hidden states.
        do:     [B, T, H, V]  output gradient.
        dh:     [B*NT, H, K, V]  hidden state gradients.
        dv:     [B, T, H, V]  value gradient.
        scale:  float          softmax scaling factor.
        chunk_size: int        chunk size (BT). T must be divisible by chunk_size.
        block_K: K-dimension tile size (BK). Defaults to K. Must divide K.
        block_V: V-dimension tile size (BV). Defaults to V. Must divide V.

    Returns:
        dq:  [B, T, H, K]   query gradient (float32).
        dk:  [B, T, H, K]   key gradient (float32).
        dv2: [B, T, H, V]   value gradient (same dtype as v).
        db:  [B, T, H]      beta gradient (float32).
        dg:  [B, T, H, K]   gate gradient (float32).
        dA:  [B, T, H, BT]  Akk inverse gradient (float32).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT

    # =================== input shape assertions ===================
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(v_new, (B, T, H, V), "v_new")
    assert_shape(g, (B, T, H, K), "g")
    assert_shape(beta, (B, T, H), "beta")
    assert_shape(A, (B, T, H, BT), "A")
    assert_shape(h, (B , NT, H, K, V), "h")
    assert_shape(do, (B, T, H, V), "do")
    assert_shape(dh, (B , NT, H, K, V), "dh")
    assert_shape(dv, (B, T, H, V), "dv")

    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

    BK = block_K if block_K is not None else K
    BV = block_V if block_V is not None else V
    assert K % BK == 0, f"K={K} must be divisible by block_K={BK}"
    assert V % BV == 0, f"V={V} must be divisible by block_V={BV}"
    # ==============================================================
    NK = K // BK
    NV = V // BV
    BH = B * H

    # --- Reshape: [B, T, H, X] -> [BH*NT, BT, X] -----------------------
    def _r_bthx(x, d):
        return (x.reshape(B, NT, BT, H, d)
                 .transpose(0, 3, 1, 2, 4)
                 .reshape(BH * NT, BT, d))

    def _r_bth(x):
        return (x.reshape(B, NT, BT, H)
                 .transpose(0, 3, 1, 2)
                 .reshape(BH * NT, BT))

    def _r_h(x):
        return (x.transpose(0, 2, 1, 3, 4)
                 .reshape(BH * NT, K, V))

    q_r    = _r_bthx(q, K)
    k_r    = _r_bthx(k, K)
    v_r    = _r_bthx(v, V)
    vn_r   = _r_bthx(v_new, V)
    g_r    = _r_bthx(g, K)
    beta_r = _r_bth(beta)
    A_r    = _r_bthx(A, BT)
    h_r    = _r_h(h)
    do_r   = _r_bthx(do, V)
    dh_r   = _r_h(dh)
    dv_r   = _r_bthx(dv, V)

    total = BH * NT

    def _spec3(d1, d2):
        return pl.BlockSpec(block_shape=(1, d1, d2),
                            index_map=lambda idx: (idx, 0, 0))

    def _spec2(d1):
        return pl.BlockSpec(block_shape=(1, d1),
                            index_map=lambda idx: (idx, 0))

    in_specs = [
        _spec3(BT, K),   # q
        _spec3(BT, K),   # k
        _spec3(BT, V),   # v
        _spec3(BT, V),   # v_new
        _spec3(BT, K),   # g
        _spec2(BT),      # beta
        _spec3(BT, BT),  # A
        _spec3(K, V),    # h
        _spec3(BT, V),   # do
        _spec3(K, V),    # dh
        _spec3(BT, V),   # dv
    ]

    out_specs = [
        _spec3(BT, K),   # dq
        _spec3(BT, K),   # dk
        _spec3(BT, V),   # dv2
        _spec3(BT, K),   # dg
        _spec2(BT),      # db
        _spec3(BT, BT),  # dA
    ]

    out_shape = [
        jax.ShapeDtypeStruct((total, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((total, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((total, BT, V), jnp.float32),
        jax.ShapeDtypeStruct((total, BT, K), jnp.float32),
        jax.ShapeDtypeStruct((total, BT),    jnp.float32),
        jax.ShapeDtypeStruct((total, BT, BT), jnp.float32),
    ]

    kernel = partial(_chunk_kda_bwd_wy_dqkg_fused_kernel,
                      scale=scale, BT=BT, BK=BK, BV=BV, NK=NK, NV=NV, K=K, V=V)

    interpret = not is_tpu_runtime()

    dq_r, dk_r, dv2_r, dg_r, db_r, dA_r = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(total,),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
    )(q_r, k_r, v_r, vn_r, g_r, beta_r, A_r, h_r, do_r, dh_r, dv_r)

    # --- Reshape back: [BH*NT, BT, X] -> [B, T, H, X] --------------------
    def _ir_bthx(x, d):
        return (x.reshape(B, H, NT, BT, d)
                 .transpose(0, 2, 3, 1, 4)
                 .reshape(B, T, H, d))

    def _ir_bth(x):
        return (x.reshape(B, H, NT, BT)
                 .transpose(0, 2, 3, 1)
                 .reshape(B, T, H))

    return (_ir_bthx(dq_r, K),
            _ir_bthx(dk_r, K),
            _ir_bthx(dv2_r, V),
            _ir_bth(db_r),
            _ir_bthx(dg_r, K),
            _ir_bthx(dA_r, BT))


# =====================================================================
# chunk_kda_bwd_dAv  —  Pallas kernel
# =====================================================================

def _chunk_kda_bwd_dAv_kernel(
    v_ref, A_ref, do_ref,
    dA_ref, dv_ref,
    *, scale, BT, BV, NV, V,
):
    """Pallas kernel body for dAv — processes one (chunk, batch*head) tile.

    Computes dA and dv from the output gradient do, the attention matrix A,
    and the value tensor v.  V is tiled into BV blocks via fori_loop.
    """
    bv  = v_ref[0]    # [BT, V]
    bA  = A_ref[0]    # [BT, BT]
    bdo = do_ref[0]   # [BT, V]

    o_t = jnp.arange(BT)
    m_causal = (o_t[:, None] >= o_t[None, :])
    bA_masked = jnp.where(m_causal, bA, 0.0)

    def _sl2d(x, start, size):
        return jax.lax.dynamic_slice(x, (0, start), (BT, size))

    def v_loop_body(i_v, carry):
        b_dA_acc, b_dv = carry
        vs = i_v * BV

        b_v_blk  = _sl2d(bv, vs, BV)    # [BT, BV]
        b_do_blk = _sl2d(bdo, vs, BV)   # [BT, BV]

        # dA += do @ v^T
        b_dA_acc = b_dA_acc + b_do_blk.astype(jnp.float32) @ b_v_blk.astype(jnp.float32).T

        # dv = A^T @ do
        b_dv_blk = bA_masked.T.astype(bdo.dtype) @ b_do_blk  # [BT, BV]
        b_dv = jax.lax.dynamic_update_slice(b_dv, b_dv_blk, (0, vs))

        return b_dA_acc, b_dv

    b_dA = jnp.zeros((BT, BT), jnp.float32)
    b_dv = jnp.zeros((BT, V), bdo.dtype)

    b_dA, b_dv = jax.lax.fori_loop(0, NV, v_loop_body, (b_dA, b_dv))

    # Apply causal mask and scale
    b_dA = jnp.where(m_causal, b_dA * scale, 0.0)

    dA_ref[0] = b_dA
    dv_ref[0] = b_dv


def chunk_kda_bwd_dAv_kernel(
    q, k, v, do, A,
    scale, chunk_size=64, block_V=None,
):
    """JAX Pallas implementation of chunk_kda_bwd_dAv.

    Computes the attention gradient dA and value gradient dv for the KDA
    backward pass.

    Args:
        q:     [B, T, H, K]   query tensor.
        k:     [B, T, H, K]   key tensor.
        v:     [B, T, H, V]   value tensor (v_new in the full backward).
        do:    [B, T, H, V]   output gradient.
        A:     [B, T, H, BT]  attention matrix (Aqk).
        scale: float           softmax scaling factor.
        chunk_size: int        chunk size (BT). T must be divisible by chunk_size.
        block_V: V-dimension tile size (BV). Defaults to V. Must divide V.

    Returns:
        dA: [B, T, H, BT]  attention gradient (float32).
        dv: [B, T, H, V]   value gradient (same dtype as do).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT

    # =================== input shape assertions ===================
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(v, (B, T, H, V), "v")
    assert_shape(do, (B, T, H, V), "do")
    assert_shape(A, (B, T, H, BT), "A")
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"

    BV = block_V if block_V is not None else V
    assert V % BV == 0, f"V={V} must be divisible by block_V={BV}"
    # ==============================================================
    NV = V // BV
    BH = B * H

    # --- Reshape: [B, T, H, X] -> [BH*NT, BT, X] ---
    def _r_bthx(x, d):
        return (x.reshape(B, NT, BT, H, d)
                 .transpose(0, 3, 1, 2, 4)
                 .reshape(BH * NT, BT, d))

    v_r  = _r_bthx(v, V)
    A_r  = _r_bthx(A, BT)
    do_r = _r_bthx(do, V)

    total = BH * NT

    def _spec3(d1, d2):
        return pl.BlockSpec(block_shape=(1, d1, d2),
                            index_map=lambda idx: (idx, 0, 0))

    in_specs = [
        _spec3(BT, V),   # v
        _spec3(BT, BT),  # A
        _spec3(BT, V),   # do
    ]

    out_specs = [
        _spec3(BT, BT),  # dA
        _spec3(BT, V),   # dv
    ]

    out_shape = [
        jax.ShapeDtypeStruct((total, BT, BT), jnp.float32),
        jax.ShapeDtypeStruct((total, BT, V), do.dtype),
    ]

    kernel = partial(_chunk_kda_bwd_dAv_kernel,
                      scale=scale, BT=BT, BV=BV, NV=NV, V=V)

    interpret = not is_tpu_runtime()

    dA_r, dv_r = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(total,),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
    )(v_r, A_r, do_r)

    # --- Reshape back: [BH*NT, BT, X] -> [B, T, H, X] ---
    def _ir_bthx(x, d):
        return (x.reshape(B, H, NT, BT, d)
                 .transpose(0, 2, 3, 1, 4)
                 .reshape(B, T, H, d))

    return _ir_bthx(dA_r, BT), _ir_bthx(dv_r, V)


# =====================================================================
# chunk_gated_delta_rule_bwd_dhu  —  Pallas kernel
# =====================================================================

def _chunk_gated_delta_rule_bwd_dhu_kernel(
    q_ref, k_ref, w_ref, gk_ref, do_ref, dv_ref, dht_ref,
    dh_ref, dh0_ref, dv2_ref,
    *, scale, BT, K, V, NT,
):
    """Pallas kernel body for dhu — processes one (batch*head) tile.

    Performs the reverse scan through chunks, computing per-chunk hidden state
    gradient dh, initial state gradient dh0, and the updated value gradient dv2.
    """
    # Refs are [1, NT, BT, D] or [1, K, V] shaped due to BlockSpec
    bq  = q_ref[0]    # [NT, BT, K]
    bk  = k_ref[0]    # [NT, BT, K]
    bw  = w_ref[0]    # [NT, BT, K]
    bgk = gk_ref[0]   # [NT, BT, K]
    bdo = do_ref[0]   # [NT, BT, V]
    bdv = dv_ref[0]   # [NT, BT, V]
    bdht = dht_ref[0]  # [K, V]

    # Reverse scan through chunks
    def scan_fn(carry, t_rev):
        b_dh = carry  # [K, V]
        i_t = NT - 1 - t_rev

        b_q_t  = bq[i_t]    # [BT, K]
        b_k_t  = bk[i_t]    # [BT, K]
        b_w_t  = bw[i_t]    # [BT, K]
        b_gk_t = bgk[i_t]   # [BT, K]
        b_do_t = bdo[i_t]   # [BT, V]
        b_dv_t = bdv[i_t]   # [BT, V]

        # Store current dh before updating
        dh_out = b_dh  # [K, V]

        # Last position gate value
        b_gk_last = b_gk_t[BT - 1]  # [K]

        # dv2 = k @ dh + dv
        b_dv2 = (b_k_t.astype(jnp.float32) @ b_dh.astype(jnp.float32)
                 + b_dv_t.astype(jnp.float32))  # [BT, V]

        # Update dh: apply gk gating
        b_dh = b_dh * exp2(b_gk_last[:, None])  # [K, V]

        # Accumulate: dh += q^T @ do * scale - w^T @ dv2
        b_dh = b_dh + (
            b_q_t.astype(jnp.float32).T @ b_do_t.astype(jnp.float32) * scale
            - b_w_t.astype(jnp.float32).T @ b_dv2
        )

        return b_dh, (dh_out, b_dv2)

    init_dh = bdht.astype(jnp.float32)

    final_dh, (dh_all, dv2_all) = jax.lax.scan(scan_fn, init_dh, jnp.arange(NT))
    # dh_all is [NT, K, V] in reverse order; reverse to natural order
    dh_all = dh_all[::-1]
    dv2_all = dv2_all[::-1]

    dh_ref[0] = dh_all         # [NT, K, V]
    dh0_ref[0] = final_dh      # [K, V] — gradient of initial state
    dv2_ref[0] = dv2_all       # [NT, BT, V]


def chunk_gated_delta_rule_bwd_dhu_kernel(
    q, k, w, gk, h0, dht, do, dv,
    scale, chunk_size=64,
):
    """JAX Pallas implementation of chunk_gated_delta_rule_bwd_dhu.

    Backward recurrence through chunks computing per-chunk hidden state
    gradient dh and the updated value gradient dv2.

    Args:
        q:    [B, T, H, K]   gated query tensor.
        k:    [B, T, H, K]   gated key tensor.
        w:    [B, T, H, K]   delta-rule erase weight.
        gk:   [B, T, H, K]   per-key gate (cumsum'd, log2-space).
        h0:   [B, H, K, V]   initial hidden state or None.
        dht:  [B, H, K, V]   gradient of final state or None.
        do:   [B, T, H, V]   output gradient.
        dv:   [B, T, H, V]   value gradient from dAv stage.
        scale: float          softmax scaling factor.
        chunk_size: int       chunk size (BT). T must be divisible by chunk_size.

    Returns:
        dh:  [B, NT, H, K, V]  per-chunk hidden state gradient.
        dh0: [B, H, K, V]      gradient of initial state or None.
        dv2: [B, T, H, V]      updated value gradient.
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    NT = T // BT

    # =================== input shape assertions ===================
    assert_shape(q, (B, T, H, K), "q")
    assert_shape(k, (B, T, H, K), "k")
    assert_shape(w, (B, T, H, K), "w")
    assert_shape(gk, (B, T, H, K), "gk")
    assert_shape(do, (B, T, H, V), "do")
    assert_shape(dv, (B, T, H, V), "dv")
    assert_shape_or_none(h0, (B, H, K, V), "h0")
    assert_shape_or_none(dht, (B, H, K, V), "dht")
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    # ==============================================================
    BH = B * H

    # --- Reshape: [B, T, H, X] -> [BH, NT, BT, X] ---
    def _r_bthx(x, d):
        return (x.reshape(B, NT, BT, H, d)
                 .transpose(0, 3, 1, 2, 4)
                 .reshape(BH, NT, BT, d))

    q_r  = _r_bthx(q, K)
    k_r  = _r_bthx(k, K)
    w_r  = _r_bthx(w, K)
    gk_r = _r_bthx(gk, K)
    do_r = _r_bthx(do, V)
    dv_r = _r_bthx(dv, V)

    # dht: [B, H, K, V] -> [BH, K, V]
    if dht is not None:
        dht_r = dht.reshape(BH, K, V)
    else:
        dht_r = jnp.zeros((BH, K, V), dtype=jnp.float32)

    def _spec_ntbd(d):
        return pl.BlockSpec(block_shape=(1, NT, BT, d),
                            index_map=lambda idx: (idx, 0, 0, 0))

    def _spec_kv():
        return pl.BlockSpec(block_shape=(1, K, V),
                            index_map=lambda idx: (idx, 0, 0))

    in_specs = [
        _spec_ntbd(K),   # q
        _spec_ntbd(K),   # k
        _spec_ntbd(K),   # w
        _spec_ntbd(K),   # gk
        _spec_ntbd(V),   # do
        _spec_ntbd(V),   # dv
        _spec_kv(),      # dht
    ]

    out_specs = [
        pl.BlockSpec(block_shape=(1, NT, K, V),
                     index_map=lambda idx: (idx, 0, 0, 0)),   # dh
        _spec_kv(),                                            # dh0
        pl.BlockSpec(block_shape=(1, NT, BT, V),
                     index_map=lambda idx: (idx, 0, 0, 0)),   # dv2
    ]

    out_shape = [
        jax.ShapeDtypeStruct((BH, NT, K, V), jnp.float32),    # dh
        jax.ShapeDtypeStruct((BH, K, V), jnp.float32),        # dh0
        jax.ShapeDtypeStruct((BH, NT, BT, V), jnp.float32),   # dv2
    ]

    kernel = partial(_chunk_gated_delta_rule_bwd_dhu_kernel,
                      scale=scale, BT=BT, K=K, V=V, NT=NT)

    interpret = not is_tpu_runtime()

    dh_r, dh0_r, dv2_r = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(BH,),
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=interpret,
    )(q_r, k_r, w_r, gk_r, do_r, dv_r, dht_r)

    # --- Reshape back ---
    # dh: [BH, NT, K, V] -> [B, H, NT, K, V] -> [B, NT, H, K, V]
    dh = dh_r.reshape(B, H, NT, K, V).transpose(0, 2, 1, 3, 4)

    # dh0: [BH, K, V] -> [B, H, K, V]
    dh0 = dh0_r.reshape(B, H, K, V) if h0 is not None else None

    # dv2: [BH, NT, BT, V] -> [B, H, NT, BT, V] -> [B, T, H, V]
    dv2 = (dv2_r.reshape(B, H, NT, BT, V)
                 .transpose(0, 2, 3, 1, 4)
                 .reshape(B, T, H, V))

    return dh, dh0, dv2