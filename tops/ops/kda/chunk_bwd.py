import jax.numpy as jnp
import jax

from tops.ops.utils import exp2
from tops.utils import assert_shape
from jax.experimental import pallas as pl
from functools import partial

from tops.ops.utils import is_tpu_runtime

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