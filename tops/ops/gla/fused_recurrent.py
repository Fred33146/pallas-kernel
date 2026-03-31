import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tops.ops.utils import get_interpret
from tops.utils import cdiv, align_up, pad_to_multiple


# ============================================================================
# Pure JAX reference implementation
# ============================================================================


def fused_recurrent_gla_fwd_ref(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | None = None,
):
    """Pure JAX implementation of fused_recurrent_gla_fwd.

    Recurrence per time step t (or reversed when reverse=True):
        h_t = decay(h_{t-1}) + k_t @ v_t
        o_t = q_t . h_t

    Decay gates (all in log domain, kernel applies exp internally):
        g        : [B, T, H]    scalar log-gate per (batch, time, head)
        g_gamma  : [H]          per-head constant log-gate (same every step)
        gk       : [B, T, H, K] key-wise log-gate
        gv       : [B, T, H, V] value-wise log-gate

    Args:
        q:             [B, T, H, K]
        k:             [B, T, H, K]
        v:             [B, T, H, V]
        g:             [B, T, H]    scalar log-gate (optional)
        g_gamma:       [H]          per-head constant log-gate (optional)
        gk:            [B, T, H, K] key-wise log-gate (optional)
        gv:            [B, T, H, V] value-wise log-gate (optional)
        scale:         scalar, default K^-0.5
        initial_state: [N, H, K, V]
        output_final_state: whether to return final hidden state
        reverse:       if True, iterate time steps from T-1 to 0
        cu_seqlens:    [N+1] cumulative sequence lengths (varlen mode, requires B=1)

    Returns:
        o:  [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5

    USE_G = g is not None
    USE_G_GAMMA = g_gamma is not None
    USE_GK = gk is not None
    USE_GV = gv is not None

    q_f = q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)
    g_f = g.astype(jnp.float32) if USE_G else None
    g_gamma_f = g_gamma.astype(jnp.float32) if USE_G_GAMMA else None
    gk_f = gk.astype(jnp.float32) if USE_GK else None
    gv_f = gv.astype(jnp.float32) if USE_GV else None

    o = jnp.zeros((B, T, H, V), dtype=jnp.float32)

    ht_list = []

    def _run_seq(batch_idx, bos, seq_len, o_arr):
        """Run recurrence for one sequence, return (final_h, updated_o)."""
        if initial_state is not None:
            h = initial_state[batch_idx].astype(jnp.float32)  # [H, K, V]
        else:
            h = jnp.zeros((H, K, V), dtype=jnp.float32)

        time_range = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        b = 0 if cu_seqlens is not None else batch_idx

        for i_t in time_range:
            t_idx = bos + i_t

            q_t = q_f[b, t_idx] * scale  # [H, K]
            k_t = k_f[b, t_idx]  # [H, K]
            v_t = v_f[b, t_idx]  # [H, V]

            if USE_G:
                h = h * jnp.exp(g_f[b, t_idx])[:, None, None]
            if USE_G_GAMMA:
                h = h * jnp.exp(g_gamma_f)[:, None, None]
            if USE_GK:
                h = h * jnp.exp(gk_f[b, t_idx])[:, :, None]
            if USE_GV:
                h = h * jnp.exp(gv_f[b, t_idx])[:, None, :]

            h = h + k_t[:, :, None] * v_t[:, None, :]

            o_arr = o_arr.at[b, t_idx].set((h * q_t[:, :, None]).sum(1))

        return h, o_arr

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        for i_n in range(N):
            bos = int(cu_seqlens[i_n])
            eos = int(cu_seqlens[i_n + 1])
            h_final, o = _run_seq(i_n, bos, eos - bos, o)
            if output_final_state:
                ht_list.append(h_final)
    else:
        for i_n in range(B):
            h_final, o = _run_seq(i_n, 0, T, o)
            if output_final_state:
                ht_list.append(h_final)

    ht = jnp.stack(ht_list, axis=0) if output_final_state else None
    return o, ht


# ============================================================================
# Pallas kernel
# ============================================================================
# TODO(0xaskr) support large tokens by tiling in time dimension
def _fused_recurrent_gla_fwd_kernel(
    # in
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    gv: jax.Array,
    h0: jax.Array,
    seqlens: jax.Array,
    # out
    o: jax.Array,
    ht: jax.Array,
    # static args
    SCALE: float,
    T: int,
    B: int,
    H: int,
    K: int,
    V: int,
    BK: int,
    BV: int,
    USE_G: bool,
    USE_GK: bool,
    USE_GV: bool,
    USE_INIT_STATE: bool,
    OUTPUT_FINAL_STATE: bool,
    REVERSE: bool,
    USE_SEQLENS: bool,
):
    assert K % BK == 0, f"K must be a multiple of BK={BK}, got {K}"
    assert V % BV == 0, f"V must be a multiple of BV={BV}, got {V}"
    q = q.reshape(T, BK)
    k = k.reshape(T, BK)
    v = v.reshape(T, BV)
    if USE_GK:
        gk = gk.reshape(T, BK)
    if USE_GV:
        gv = gv.reshape(T, BV)
    if USE_INIT_STATE:
        h0 = h0.reshape(BK, BV)

    if USE_SEQLENS:
        idx_nh = pl.program_id(2)
        seq_idx = idx_nh // H
        bos = seqlens[seq_idx]
        eos = seqlens[seq_idx + 1]

        # Zero-init the output block: on TPU, output blocks are not
        # guaranteed to be zero-initialized.  Each varlen grid cell only
        # writes to [bos, eos), so the remaining positions must be
        # explicitly cleared before the sum-over-N reduction in the launcher.
        def _zero_output(t, _):
            o[0, 0, 0, t, 0:BV] = jnp.zeros(BV, dtype=o.dtype)
            return _

        jax.lax.fori_loop(0, T, _zero_output, None)
    else:
        bos = 0
        eos = T

    b_h = jnp.zeros((BK, BV), dtype=jnp.float32)
    if USE_INIT_STATE:
        b_h += h0[...].astype(jnp.float32)

    def body(idx_t, b_h):
        if REVERSE:
            t = (eos - 1) - (idx_t - bos)
        else:
            t = idx_t
        if USE_GK:
            b_gk = gk[t, 0:BK].astype(jnp.float32)
            b_h = b_h * jnp.exp(b_gk[:, None])
        if USE_GV:
            b_gv = gv[t, 0:BV].astype(jnp.float32)
            b_h = b_h * jnp.exp(b_gv[None, :])
        b_q = q[t, 0:BK].astype(jnp.float32) * SCALE
        b_k = k[t, 0:BK].astype(jnp.float32)
        b_v = v[t, 0:BV].astype(jnp.float32)

        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = jnp.sum(b_o, axis=0)
        o[0, 0, 0, t, 0:BV] = b_o.astype(o.dtype)
        return b_h

    b_h = jax.lax.fori_loop(bos, eos, body, b_h)

    if OUTPUT_FINAL_STATE:
        ht[0, 0, :, :] = b_h.astype(ht.dtype)


# ============================================================================
# Pallas launcher
# ============================================================================


def _fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
    use_gk: bool = False,
    use_gv: bool = False,
    use_init_state: bool = False,
    use_final_state: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1
    BK, BV = 128, 128
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    origin_K = K
    origin_V = V
    K = align_up(K, BK)
    V = align_up(V, BV)

    h0 = initial_state
    ht = jnp.zeros((N, H, K, V), dtype=jnp.float32)
    o = jnp.zeros([NK, H, N, T, V], dtype=jnp.float32)
    o_spec = jax.ShapeDtypeStruct(o.shape, o.dtype)
    ht_spec = jax.ShapeDtypeStruct(ht.shape, ht.dtype)

    # [B, T, H, K] -> [H, B, T, K]
    q_trans = pad_to_multiple(q, BK, 3, 0).transpose(2, 0, 1, 3)
    k_trans = pad_to_multiple(k, BK, 3, 0).transpose(2, 0, 1, 3)
    v_trans = pad_to_multiple(v, BV, 3, 0).transpose(2, 0, 1, 3)
    gk_trans = pad_to_multiple(gk, BK, 3, 0).transpose(2, 0, 1, 3) if use_gk else None
    gv_trans = pad_to_multiple(gv, BV, 3, 0).transpose(2, 0, 1, 3) if use_gv else None
    # h0: [N, H, K, V] -> transpose(1,0,2,3) -> [H, N, K, V]
    h0_trans = (
        pad_to_multiple(h0, [BK, BV], [2, 3], 0).transpose(1, 0, 2, 3)
        if use_init_state
        else None
    )

    USE_SEQLENS = cu_seqlens is not None

    if USE_SEQLENS:
        # varlen: B=1, batch dim is always 0
        def qk_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, 0, 0, idx_k)

        def v_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, 0, 0, idx_v)
    else:

        def qk_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, idx_nh // H, 0, idx_k)

        def v_index_map(idx_v, idx_k, idx_nh):
            return (idx_nh % H, idx_nh // H, 0, idx_v)

    # o: [NK, H, N, T, V] — N=B for non-varlen, N=num_seqs for varlen
    def o_index_map(idx_v, idx_k, idx_nh):
        return (idx_k, idx_nh % H, idx_nh // H, 0, idx_v)

    # h0_trans: [H, N, K, V]
    def h0_index_map(idx_v, idx_k, idx_nh):
        return (idx_nh % H, idx_nh // H, idx_k, idx_v)

    # ht: [N, H, K, V] (N and H swapped vs h0_trans)
    def ht_index_map(idx_v, idx_k, idx_nh):
        return (idx_nh // H, idx_nh % H, idx_k, idx_v)

    q_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map)
    k_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map)
    v_blockspec = pl.BlockSpec([1, 1, T, BV], v_index_map)
    gk_blockspec = pl.BlockSpec([1, 1, T, BK], qk_index_map) if use_gk else None
    gv_blockspec = pl.BlockSpec([1, 1, T, BV], v_index_map) if use_gv else None
    h0_blockspec = (
        pl.BlockSpec([1, 1, BK, BV], h0_index_map) if use_init_state else None
    )
    seqlens_blockspec = (
        pl.BlockSpec([N + 1], lambda *idx: (0,), memory_space=pltpu.TPUMemorySpace.SMEM)
        if USE_SEQLENS
        else None
    )

    o_blockspec = pl.BlockSpec([1, 1, 1, T, BV], o_index_map)
    ht_blockspec = pl.BlockSpec([1, 1, BK, BV], ht_index_map)
    call_func = functools.partial(
        _fused_recurrent_gla_fwd_kernel,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=False,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        USE_INIT_STATE=initial_state is not None,
        OUTPUT_FINAL_STATE=output_final_state,
        REVERSE=reverse,
        SCALE=scale,
        USE_SEQLENS=USE_SEQLENS,
    )

    grid = (NV, NK, N * H)
    results = pl.pallas_call(
        call_func,
        out_shape=[o_spec, ht_spec],
        grid=grid,
        in_specs=[
            q_blockspec,
            k_blockspec,
            v_blockspec,
            gk_blockspec,
            gv_blockspec,
            h0_blockspec,
            seqlens_blockspec,
        ],
        out_specs=[o_blockspec, ht_blockspec],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
        interpret=get_interpret(),
    )(q_trans, k_trans, v_trans, gk_trans, gv_trans, h0_trans, cu_seqlens)
    o, ht = results
    o = o.transpose(0, 2, 3, 1, 4)  # [NK, H, N, T, V] -> [NK, N, T, H, V]
    o = o.sum(0)  # [N, T, H, V]
    if USE_SEQLENS:
        # In varlen mode, each seq wrote to non-overlapping T ranges; sum over N to merge
        o = o.sum(0, keepdims=True)  # [1, T, H, V]
    o = o[:, :, :, :origin_V]  # [B, T, H, V]
    ht = ht[:, :, :origin_K, :origin_V] if use_final_state else None
    return o, ht


# ============================================================================
# Entry point: validation + dispatch
# ============================================================================


def fused_recurrent_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
    scale = scale if scale is not None else K**-0.5

    assert k.shape == (B, T, H, K), f"Expected k shape {(B, T, H, K)}, got {k.shape}"
    assert v.shape == (B, T, H, V), f"Expected v shape {(B, T, H, V)}, got {v.shape}"
    assert (gk is None) or (gk.shape == (B, T, H, K)), (
        f"Expected gk shape {(B, T, H, K)}, got {gk.shape}"
    )
    assert (gv is None) or (gv.shape == (B, T, H, V)), (
        f"Expected gv shape {(B, T, H, V)}, got {gv.shape}"
    )
    assert (initial_state is None) or (initial_state.shape == (N, H, K, V)), (
        f"Expected initial_state shape {(N, H, K, V)}, got {initial_state.shape}"
    )
    assert (cu_seqlens is None) or (B == 1), (
        f"Batch size must be 1 when using cu_seqlens, got {B}"
    )
    assert scale is not None, (
        "ignore pylance warning about unused variable `scale`, which is actually used in the kernel call"
    )

    o, ht = _fused_recurrent_gla_fwd(
        q=q,
        k=k,
        v=v,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        use_gk=gk is not None,
        use_gv=gv is not None,
        use_init_state=initial_state is not None,
        use_final_state=output_final_state,
    )
    return o, ht


# ============================================================================
# Public API matching FLA signature
# ============================================================================


def fused_recurrent_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused recurrent GLA — public API matching FLA signature.

    Wraps fused_recurrent_gla_fwd, mapping gk/gv to the internal API.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        gk: [B, T, H, K] key-wise log-gate (optional)
        gv: [B, T, H, V] value-wise log-gate (optional)
        scale: scalar, default K^-0.5
        initial_state: [N, H, K, V]
        output_final_state: whether to return final hidden state
        reverse: if True, iterate time steps from T-1 to 0
        cu_seqlens: [N+1] cumulative sequence lengths

    Returns:
        o: [B, T, H, V]
        ht: [N, H, K, V] if output_final_state else None
    """
    return fused_recurrent_gla_fwd(
        q=q,
        k=k,
        v=v,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


# ============================================================================
# Backward Pallas kernels
# ============================================================================
# Split into two kernels because TPU Pallas does not support reading from
# output refs or scatter updates on VMEM arrays inside fori_loop.
#
# Kernel 1 (_bwd_dq): forward replay → writes dq
# Kernel 2 (_bwd_main): forward replay for h_final + backward pass → reads
#   dq as INPUT, writes dk, dv, dgk, dgv, dh0


def _fused_recurrent_gla_bwd_kernel(
    # in
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    gv: jax.Array,
    h0: jax.Array,
    do: jax.Array,
    dht: jax.Array,
    seqlens: jax.Array,
    # out
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    dh0: jax.Array,
    ht: jax.Array,
    # static args
    SCALE: float,
    T: int,
    H: int,
    K: int,
    V: int,
    USE_GK: bool,
    USE_GV: bool,
    USE_INIT_STATE: bool,
    USE_DHT: bool,
    REVERSE: bool,
    USE_SEQLENS: bool,
):
    q = q.reshape(T, K)
    k = k.reshape(T, K)
    v = v.reshape(T, V)
    do = do.reshape(T, V)
    if USE_GK:
        gk = gk.reshape(T, K)
    if USE_GV:
        gv = gv.reshape(T, V)
    if USE_INIT_STATE:
        h0 = h0.reshape(K, V)
    if USE_DHT:
        dht = dht.reshape(K, V)

    if USE_SEQLENS:
        idx_nh = pl.program_id(0)
        seq_idx = idx_nh // H
        bos = seqlens[seq_idx]
        eos = seqlens[seq_idx + 1]

        def _zero_out(t, _):
            dq[0, 0, t, 0:K] = jnp.zeros(K, dtype=dq.dtype)
            dk[0, 0, t, 0:K] = jnp.zeros(K, dtype=dk.dtype)
            dv[0, 0, t, 0:V] = jnp.zeros(V, dtype=dv.dtype)
            return _

        jax.lax.fori_loop(0, T, _zero_out, None)
    else:
        bos = 0
        eos = T

    # ---- Pass 1: forward replay → dq ----
    b_h = jnp.zeros((K, V), dtype=jnp.float32)
    if USE_INIT_STATE:
        b_h += h0[...].astype(jnp.float32)

    def pass1_body(idx_t, b_h):
        if REVERSE:
            t = (eos - 1) - (idx_t - bos)
        else:
            t = idx_t
        if USE_GK:
            b_h = b_h * jnp.exp(gk[t, 0:K].astype(jnp.float32))[:, None]
        if USE_GV:
            b_h = b_h * jnp.exp(gv[t, 0:V].astype(jnp.float32))[None, :]
        b_k = k[t, 0:K].astype(jnp.float32)
        b_v = v[t, 0:V].astype(jnp.float32)
        b_h = b_h + b_k[:, None] * b_v[None, :]
        b_do = do[t, 0:V].astype(jnp.float32)
        b_dq = SCALE * jnp.sum(b_h * b_do[None, :], axis=1)
        dq[0, 0, t, 0:K] = b_dq.astype(dq.dtype)
        return b_h

    b_h_final = jax.lax.fori_loop(bos, eos, pass1_body, b_h)

    # Write h_final (final hidden state from forward replay)
    ht[0, 0, :, :] = b_h_final.astype(ht.dtype)

    # ---- Pass 2: backward pass → dk, dv, dh0 ----
    b_dh = jnp.zeros((K, V), dtype=jnp.float32)
    if USE_DHT:
        b_dh = dht[...].astype(jnp.float32)

    def pass2_body(idx_t, b_dh):
        if REVERSE:
            t = idx_t
        else:
            t = (eos - 1) - (idx_t - bos)

        b_q = q[t, 0:K].astype(jnp.float32)
        b_k = k[t, 0:K].astype(jnp.float32)
        b_v = v[t, 0:V].astype(jnp.float32)
        b_do = do[t, 0:V].astype(jnp.float32)

        # Output gradient contribution
        b_dh = b_dh + SCALE * b_q[:, None] * b_do[None, :]

        # dk, dv
        b_dk = jnp.sum(b_dh * b_v[None, :], axis=1)  # [K]
        b_dv = jnp.sum(b_dh * b_k[:, None], axis=0)  # [V]
        dk[0, 0, t, 0:K] = b_dk.astype(dk.dtype)
        dv[0, 0, t, 0:V] = b_dv.astype(dv.dtype)

        # Propagate dh through gate
        if USE_GK:
            b_dh = b_dh * jnp.exp(gk[t, 0:K].astype(jnp.float32))[:, None]
        if USE_GV:
            b_dh = b_dh * jnp.exp(gv[t, 0:V].astype(jnp.float32))[None, :]

        return b_dh

    b_dh = jax.lax.fori_loop(bos, eos, pass2_body, b_dh)

    # Write dh0 (gradient w.r.t. initial state)
    dh0[0, 0, :, :] = b_dh.astype(dh0.dtype)


# ============================================================================
# Backward Pallas launcher
# ============================================================================


def _fused_recurrent_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    scale: float,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    dht: jax.Array | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
    use_gk: bool = False,
    use_gv: bool = False,
    use_init_state: bool = False,
    use_dht: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1
    BK, BV = 128, 128
    origin_K = K
    origin_V = V
    K = align_up(K, BK)
    V = align_up(V, BV)

    USE_SEQLENS = cu_seqlens is not None

    # Transpose inputs: [B, T, H, D] -> [H, B, T, D]
    q_trans = pad_to_multiple(q, BK, 3, 0).transpose(2, 0, 1, 3)
    k_trans = pad_to_multiple(k, BK, 3, 0).transpose(2, 0, 1, 3)
    v_trans = pad_to_multiple(v, BV, 3, 0).transpose(2, 0, 1, 3)
    do_trans = pad_to_multiple(do, BV, 3, 0).transpose(2, 0, 1, 3)
    gk_trans = (
        pad_to_multiple(gk, BK, 3, 0).transpose(2, 0, 1, 3) if use_gk else None
    )
    gv_trans = (
        pad_to_multiple(gv, BV, 3, 0).transpose(2, 0, 1, 3) if use_gv else None
    )
    # State tensors: [N, H, K, V] -> [H, N, K, V]
    h0_trans = (
        pad_to_multiple(initial_state, [BK, BV], [2, 3], 0).transpose(1, 0, 2, 3)
        if use_init_state
        else None
    )
    dht_trans = (
        pad_to_multiple(dht, [BK, BV], [2, 3], 0).transpose(1, 0, 2, 3)
        if use_dht
        else None
    )

    # Index maps — inputs
    if USE_SEQLENS:
        def in_qk_map(idx_nh):
            return (idx_nh % H, 0, 0, 0)

        def in_v_map(idx_nh):
            return (idx_nh % H, 0, 0, 0)
    else:
        def in_qk_map(idx_nh):
            return (idx_nh % H, idx_nh // H, 0, 0)

        def in_v_map(idx_nh):
            return (idx_nh % H, idx_nh // H, 0, 0)

    def state_map(idx_nh):
        return (idx_nh % H, idx_nh // H, 0, 0)

    # Index maps — outputs: always use (h, n, ...)
    def out_qk_map(idx_nh):
        return (idx_nh % H, idx_nh // H, 0, 0)

    def out_v_map(idx_nh):
        return (idx_nh % H, idx_nh // H, 0, 0)

    # BlockSpecs — inputs
    q_bs = pl.BlockSpec([1, 1, T, K], in_qk_map)
    k_bs = pl.BlockSpec([1, 1, T, K], in_qk_map)
    v_bs = pl.BlockSpec([1, 1, T, V], in_v_map)
    do_bs = pl.BlockSpec([1, 1, T, V], in_v_map)
    gk_bs = pl.BlockSpec([1, 1, T, K], in_qk_map) if use_gk else None
    gv_bs = pl.BlockSpec([1, 1, T, V], in_v_map) if use_gv else None
    h0_bs = pl.BlockSpec([1, 1, K, V], state_map) if use_init_state else None
    dht_bs = pl.BlockSpec([1, 1, K, V], state_map) if use_dht else None
    seqlens_bs = (
        pl.BlockSpec(
            [N + 1], lambda idx_nh: (0,),
            memory_space=pltpu.TPUMemorySpace.SMEM,
        )
        if USE_SEQLENS
        else None
    )

    # BlockSpecs — outputs
    dq_bs = pl.BlockSpec([1, 1, T, K], out_qk_map)
    dk_bs = pl.BlockSpec([1, 1, T, K], out_qk_map)
    dv_bs = pl.BlockSpec([1, 1, T, V], out_v_map)
    dh0_bs = pl.BlockSpec([1, 1, K, V], state_map)
    ht_bs = pl.BlockSpec([1, 1, K, V], state_map)

    # Output shapes
    dq_shape = jax.ShapeDtypeStruct([H, N, T, K], jnp.float32)
    dk_shape = jax.ShapeDtypeStruct([H, N, T, K], jnp.float32)
    dv_shape = jax.ShapeDtypeStruct([H, N, T, V], jnp.float32)
    dh0_shape = jax.ShapeDtypeStruct([H, N, K, V], jnp.float32)
    ht_shape = jax.ShapeDtypeStruct([H, N, K, V], jnp.float32)

    grid = (N * H,)

    # ---- Single kernel: forward replay → dq, backward → dk, dv, dh0 ----
    bwd_call = functools.partial(
        _fused_recurrent_gla_bwd_kernel,
        SCALE=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        USE_GK=use_gk,
        USE_GV=use_gv,
        USE_INIT_STATE=use_init_state,
        USE_DHT=use_dht,
        REVERSE=reverse,
        USE_SEQLENS=USE_SEQLENS,
    )
    dq_out, dk_out, dv_out, dh0_out, ht_out = pl.pallas_call(
        bwd_call,
        out_shape=[dq_shape, dk_shape, dv_shape, dh0_shape, ht_shape],
        grid=grid,
        in_specs=[
            q_bs, k_bs, v_bs, gk_bs, gv_bs, h0_bs,
            do_bs, dht_bs, seqlens_bs,
        ],
        out_specs=[dq_bs, dk_bs, dv_bs, dh0_bs, ht_bs],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
        interpret=get_interpret(),
    )(
        q_trans, k_trans, v_trans, gk_trans, gv_trans, h0_trans,
        do_trans, dht_trans, cu_seqlens,
    )

    # Transpose: [H, N, T, D] -> [N, T, H, D]
    dq_out = dq_out.transpose(1, 2, 0, 3)
    dk_out = dk_out.transpose(1, 2, 0, 3)
    dv_out = dv_out.transpose(1, 2, 0, 3)

    if USE_SEQLENS:
        # Sum over N to merge non-overlapping writes from different sequences
        dq_out = dq_out.sum(0, keepdims=True)
        dk_out = dk_out.sum(0, keepdims=True)
        dv_out = dv_out.sum(0, keepdims=True)

    # Slice back to original dimensions
    dq_out = dq_out[:, :, :, :origin_K]
    dk_out = dk_out[:, :, :, :origin_K]
    dv_out = dv_out[:, :, :, :origin_V]

    # dh0: [H, N, K, V] -> [N, H, K, V]
    dh0_out = dh0_out.transpose(1, 0, 2, 3)
    dh0_out = dh0_out[:, :, :origin_K, :origin_V]

    # ht (h_final): [H, N, K, V] -> [N, H, K, V]
    ht_out = ht_out.transpose(1, 0, 2, 3)
    ht_out = ht_out[:, :, :origin_K, :origin_V]

    return dq_out, dk_out, dv_out, dh0_out, ht_out


# ============================================================================
# Backward entry point
# ============================================================================


def _compute_gate_grads(
    q, k, v, dq, dk, dv, do, fwd_o, dht, h_final,
    use_gk, use_gv, use_dht,
    reverse, cu_seqlens,
):
    """Compute dgk/dgv using the algebraic identity + suffix/prefix cumsum.

    dgk_t = correction_k + cumsum_{s from t to end_of_segment}(q_s*dq_s - k_s*dk_s)
    dgv_t = correction_v + cumsum_{s from t to end_of_segment}(o_s*do_s - v_s*dv_s)

    For reverse=False: suffix cumsum (right to left within each segment)
    For reverse=True: prefix cumsum (left to right within each segment)
    """
    B, T, H, _ = q.shape

    def _segmented_cumsum(diff, reverse_scan):
        """Compute prefix or suffix cumsum, respecting segment boundaries."""
        if reverse_scan:
            # Suffix cumsum: sum from t to end of segment
            full_suffix = jnp.flip(
                jnp.cumsum(jnp.flip(diff, axis=1), axis=1), axis=1
            )  # [B, T, H, D]
            if cu_seqlens is not None:
                seg_idx = jnp.searchsorted(cu_seqlens[1:], jnp.arange(T), side='right')
                eos_vals = cu_seqlens[seg_idx + 1]  # [T]
                safe_eos = jnp.minimum(eos_vals, T - 1)
                suffix_at_eos = full_suffix[0, safe_eos, :, :]  # [T, H, D]
                suffix_at_eos = jnp.where(
                    (eos_vals < T)[:, None, None], suffix_at_eos, 0.0
                )
                return full_suffix - suffix_at_eos[None, :, :, :]
            return full_suffix
        else:
            # Prefix cumsum: sum from start of segment to t
            full_prefix = jnp.cumsum(diff, axis=1)  # [B, T, H, D]
            if cu_seqlens is not None:
                seg_idx = jnp.searchsorted(cu_seqlens[1:], jnp.arange(T), side='right')
                bos_vals = cu_seqlens[seg_idx]  # [T]
                safe_bos = jnp.maximum(bos_vals - 1, 0)
                prefix_before_seg = full_prefix[0, safe_bos, :, :]  # [T, H, D]
                prefix_before_seg = jnp.where(
                    (bos_vals > 0)[:, None, None], prefix_before_seg, 0.0
                )
                return full_prefix - prefix_before_seg[None, :, :, :]
            return full_prefix

    # For non-reverse mode, backward pass goes right-to-left → suffix cumsum
    # For reverse mode, backward pass goes left-to-right → prefix cumsum
    use_suffix = not reverse

    dgk_out = None
    dgv_out = None

    if use_gk:
        diff_k = q * dq - k * dk  # [B, T, H, K]
        dgk_out = _segmented_cumsum(diff_k, reverse_scan=use_suffix)
        if use_dht and h_final is not None:
            correction_k = jnp.sum(dht * h_final, axis=-1)  # [N, H, K]
            if cu_seqlens is not None:
                # Varlen: apply per-segment correction
                seg_idx = jnp.searchsorted(cu_seqlens[1:], jnp.arange(T), side='right')
                corr_per_t = correction_k[seg_idx, :, :]  # [T, H, K]
                dgk_out = dgk_out + corr_per_t[None, :, :, :]
            else:
                dgk_out = dgk_out + correction_k[:, None, :, :]

    if use_gv:
        diff_v = fwd_o * do - v * dv  # [B, T, H, V]
        dgv_out = _segmented_cumsum(diff_v, reverse_scan=use_suffix)
        if use_dht and h_final is not None:
            correction_v = jnp.sum(dht * h_final, axis=-2)  # [N, H, V]
            if cu_seqlens is not None:
                seg_idx = jnp.searchsorted(cu_seqlens[1:], jnp.arange(T), side='right')
                corr_per_t = correction_v[seg_idx, :, :]  # [T, H, V]
                dgv_out = dgv_out + corr_per_t[None, :, :, :]
            else:
                dgv_out = dgv_out + correction_v[:, None, :, :]

    return dgk_out, dgv_out


def fused_recurrent_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    o: jax.Array | None = None,
    do: jax.Array | None = None,
    dht: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array,
           jax.Array | None, jax.Array | None, jax.Array | None]:
    """Backward for fused recurrent GLA — Pallas TPU kernel.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        gk: [B, T, H, K] key-wise log-gate (optional)
        gv: [B, T, H, V] value-wise log-gate (optional)
        o: [B, T, H, V] forward output (required when gv is used)
        do: [B, T, H, V] gradient of output
        dht: [N, H, K, V] gradient of final state (optional)
        scale: scalar, default K^-0.5
        initial_state: [N, H, K, V]
        reverse: if True, iterate time steps from T-1 to 0
        cu_seqlens: [N+1] cumulative sequence lengths

    Returns:
        dq, dk, dv, dgk, dgv, dh0
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else B
    scale = scale if scale is not None else K**-0.5

    assert k.shape == (B, T, H, K)
    assert v.shape == (B, T, H, V)
    assert do is not None and do.shape == (B, T, H, V)
    assert (gk is None) or (gk.shape == (B, T, H, K))
    assert (gv is None) or (gv.shape == (B, T, H, V))
    assert (initial_state is None) or (initial_state.shape == (N, H, K, V))
    assert (dht is None) or (dht.shape == (N, H, K, V))
    assert (cu_seqlens is None) or (B == 1)

    use_gk = gk is not None
    use_gv = gv is not None
    use_init_state = initial_state is not None
    use_dht = dht is not None

    # o is required for dgv
    assert not use_gv or o is not None, "o (forward output) is required when gv is used"

    # Pallas kernel: forward replay → dq + h_final, backward → dk, dv, dh0
    dq, dk, dv, dh0_out, h_final = _fused_recurrent_gla_bwd(
        q=q, k=k, v=v, do=do,
        scale=scale,
        gk=gk, gv=gv,
        initial_state=initial_state, dht=dht,
        reverse=reverse, cu_seqlens=cu_seqlens,
        use_gk=use_gk, use_gv=use_gv,
        use_init_state=use_init_state, use_dht=use_dht,
    )

    # Gate gradients via algebraic identity + cumsum (done in JAX, not Pallas)
    dgk_out, dgv_out = _compute_gate_grads(
        q=q, k=k, v=v, dq=dq, dk=dk, dv=dv, do=do,
        fwd_o=o, dht=dht, h_final=h_final,
        use_gk=use_gk, use_gv=use_gv, use_dht=use_dht,
        reverse=reverse, cu_seqlens=cu_seqlens,
    )

    dh0_out = dh0_out if use_init_state else None

    return dq, dk, dv, dgk_out, dgv_out, dh0_out
