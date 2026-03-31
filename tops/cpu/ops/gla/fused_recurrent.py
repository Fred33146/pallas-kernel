"""JAX CPU reference for GLA fused recurrent with FLA-exact dtype behavior.

Precisely matches the FLA Triton fused_recurrent kernels
(fla/ops/common/fused_recurrent.py) dtype behavior.

Dtype contract (matching FLA Triton for bf16/fp16/fp32; all fp64 for fp64):

  Forward (fused_recurrent_fwd_kernel):
    All inputs loaded as fp32:  tl.load(...).to(tl.float32)
    Hidden state b_h:           fp32 accumulator
    Output o:                   fp32 (fused_recurrent_fwd returns fp32)
    Final state ht:             fp32
    Public API (fused_recurrent_gla) casts o to q.dtype,
    matching FusedRecurrentFunction.forward.

  Backward (fused_recurrent_bwd_kernel):
    Pass 1 (forward replay → dq):
      h:  fp32 accumulator (same as forward)
      dq: fp32, then dq.to(q.dtype) in autograd wrapper
    Pass 2 (backward → dk, dv, dgk, dgv, dh0):
      dh:  fp32 accumulator
      dk:  fp32, then dk.to(k.dtype)
      dv:  fp32, then dv.to(v.dtype)
      dgk: fp32 running suffix sum, then dgk.to(gk.dtype)
      dgv: fp32 running suffix sum, then dgv.to(gv.dtype)
      dh0: h0.dtype (typically fp32)

  fp64 mode: all computation in fp64, no precision casts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tops.cpu.ops import cpu_reference


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


# =============================================================================
# Forward
# =============================================================================


@cpu_reference
def fused_recurrent_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Forward pass for fused recurrent GLA — FLA-exact dtype behavior.

    Recurrence per timestep (gate applied before outer product):
        if gk: h = h * exp(gk_t[:, None])
        if gv: h = h * exp(gv_t[None, :])
        h = h + k_t[:, None] * v_t[None, :]
        o_t = scale * sum_k(h * q_t[:, None])

    Dtype behavior (matching FLA Triton fused_recurrent_fwd_kernel):
      - All inputs loaded as fp32 inside kernel
      - Hidden state h: fp32 accumulator
      - Output o: fp32 (matching FLA's Python launcher; caller casts to q.dtype)
      - Final state ht: fp32
      - fp64 mode: all fp64, no casts

    Note: Returns fp32 output (before cast), matching FLA's fused_recurrent_fwd.
    The public API fused_recurrent_gla casts o to q.dtype, mirroring
    FusedRecurrentFunction.forward. Users who need backward with gv should
    save the fp32 o from this function for fused_recurrent_bwd.

    Args:
        q:  [B, T, H, K] — Queries
        k:  [B, T, H, K] — Keys
        v:  [B, T, H, V] — Values
        gk: [B, T, H, K] — Per-key gate in log-space (optional)
        gv: [B, T, H, V] — Per-value gate in log-space (optional)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V] — Initial hidden state
        output_final_state: Whether to return final hidden state
        reverse: If True, iterate from T-1 to 0
        cu_seqlens: [N+1] — Cumulative sequence lengths (requires B=1)

    Returns:
        o:  [B, T, H, V] in fp32/fp64 (accumulator dtype)
        ht: [N, H, K, V] in fp32/fp64, or None
    """
    acc_dt = _acc_dtype(q.dtype)

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"

    if scale is None:
        scale = K**-0.5

    USE_GK = gk is not None
    USE_GV = gv is not None

    # Triton kernel: all loads cast to tl.float32
    q_f = q.astype(acc_dt)
    k_f = k.astype(acc_dt)
    v_f = v.astype(acc_dt)
    gk_f = gk.astype(acc_dt) if USE_GK else None
    gv_f = gv.astype(acc_dt) if USE_GV else None

    # Triton: o = q.new_empty(NK, *v.shape, dtype=torch.float32)
    o = jnp.zeros((B, T, H, V), dtype=acc_dt)
    ht_list = [] if output_final_state else None

    for i_n in range(N):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
            b = 0
        else:
            bos, eos = 0, T
            b = i_n

        # Triton: b_h = tl.zeros([BK, BV], dtype=tl.float32)
        h = jnp.zeros((H, K, V), dtype=acc_dt)
        if initial_state is not None:
            # Triton: b_h += tl.load(p_h0).to(tl.float32)
            h = h + initial_state[i_n].astype(acc_dt)

        time_range = range(eos - 1, bos - 1, -1) if reverse else range(bos, eos)

        for t in time_range:
            # Triton: b_gk = tl.load(p_gk).to(tl.float32); b_h *= exp(b_gk[:, None])
            if USE_GK:
                h = h * jnp.exp(gk_f[b, t, :, :, None])
            if USE_GV:
                h = h * jnp.exp(gv_f[b, t, :, None, :])

            # Triton: b_h += b_k[:, None] * b_v[None, :]
            h = h + k_f[b, t, :, :, None] * v_f[b, t, :, None, :]

            # Triton: b_o = sum(b_h * b_q[:, None], axis=0); store(p_o, b_o)
            o_t = jnp.sum(h * (q_f[b, t, :, :, None] * scale), axis=1)  # [H, V]
            o = o.at[b, t].set(o_t)

        if output_final_state:
            ht_list.append(h)

    ht = jnp.stack(ht_list, axis=0) if output_final_state else None

    # FLA fused_recurrent_fwd (Python launcher) returns fp32 o and ht.
    # The cast to q.dtype happens in FusedRecurrentFunction.forward / fused_recurrent_gla.
    return o, ht


# =============================================================================
# Backward
# =============================================================================


@cpu_reference
def fused_recurrent_bwd(
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
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array,
           jax.Array | None, jax.Array | None, jax.Array | None]:
    """Backward pass for fused recurrent GLA — FLA-exact dtype behavior.

    Two-pass algorithm matching fused_recurrent_bwd_kernel:
      Pass 1 (forward replay → dq):
        Replay forward recurrence, compute dq at each timestep.
      Pass 2 (backward → dk, dv, dgk, dgv, dh0):
        Backward through recurrence with running gate gradient accumulators.

    Gate gradient formula (running suffix sum):
        dgk_t = sum_{s>=t}(q_s * dq_s - k_s * dk_s) + sum_v(h_final * dht)
        dgv_t = sum_{s>=t}(o_s * do_s - v_s * dv_s) + sum_k(h_final * dht)

    Args:
        q:  [B, T, H, K]
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        gk: [B, T, H, K] — Per-key gate (optional)
        gv: [B, T, H, V] — Per-value gate (optional)
        o:  [B, T, H, V] — Forward output in fp32 (required when gv is used).
            Use the fp32 output from fused_recurrent_fwd directly, not the
            q.dtype output from fused_recurrent_gla, to match FLA precision.
        do: [B, T, H, V] — Gradient of output
        dht: [N, H, K, V] — Gradient of final state (optional)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V]
        reverse: If True, iterate from T-1 to 0
        cu_seqlens: [N+1] — Cumulative sequence lengths

    Returns:
        dq:  [B, T, H, K] in q.dtype
        dk:  [B, T, H, K] in k.dtype
        dv:  [B, T, H, V] in v.dtype
        dgk: [B, T, H, K] in gk.dtype, or None
        dgv: [B, T, H, V] in gv.dtype, or None
        dh0: [N, H, K, V] in fp32/fp64, or None
    """
    orig_dtype = q.dtype
    acc_dt = _acc_dtype(orig_dtype)

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"

    if scale is None:
        scale = K**-0.5

    USE_GK = gk is not None
    USE_GV = gv is not None

    assert do is not None
    if USE_GV:
        assert o is not None, "o (forward output) required when gv is used"

    # Cast all inputs to accumulator dtype
    q_f = q.astype(acc_dt)
    k_f = k.astype(acc_dt)
    v_f = v.astype(acc_dt)
    do_f = do.astype(acc_dt)
    gk_f = gk.astype(acc_dt) if USE_GK else None
    gv_f = gv.astype(acc_dt) if USE_GV else None
    o_f = o.astype(acc_dt) if USE_GV else None

    # Allocate outputs in accumulator dtype (Triton: dtype=torch.float32)
    dq = jnp.zeros((B, T, H, K), dtype=acc_dt)
    dk = jnp.zeros((B, T, H, K), dtype=acc_dt)
    dv = jnp.zeros((B, T, H, V), dtype=acc_dt)
    dgk_out = jnp.zeros((B, T, H, K), dtype=acc_dt) if USE_GK else None
    dgv_out = jnp.zeros((B, T, H, V), dtype=acc_dt) if USE_GV else None
    dh0_list = [] if initial_state is not None else None

    for i_n in range(N):
        if cu_seqlens is not None:
            bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
            b = 0
        else:
            bos, eos = 0, T
            b = i_n

        # ================================================================
        # Pass 1: forward replay → dq (and h_final for gate gradients)
        # Matches Triton pass 1 in fused_recurrent_bwd_kernel
        # ================================================================
        h = jnp.zeros((H, K, V), dtype=acc_dt)
        if initial_state is not None:
            h = h + initial_state[i_n].astype(acc_dt)

        fwd_range = range(eos - 1, bos - 1, -1) if reverse else range(bos, eos)
        for t in fwd_range:
            if USE_GK:
                h = h * jnp.exp(gk_f[b, t, :, :, None])
            if USE_GV:
                h = h * jnp.exp(gv_f[b, t, :, None, :])
            h = h + k_f[b, t, :, :, None] * v_f[b, t, :, None, :]
            # Triton: b_dq = SCALE * sum(b_h * b_do[None, :], axis=1)
            dq_t = scale * jnp.sum(h * do_f[b, t, :, None, :], axis=2)  # [H, K]
            dq = dq.at[b, t].set(dq_t)

        h_final = h  # [H, K, V] — needed for gate gradient initialization

        # ================================================================
        # Pass 2: backward → dk, dv, dgk, dgv, dh0
        # Matches Triton pass 2 in fused_recurrent_bwd_kernel
        # ================================================================
        dh = jnp.zeros((H, K, V), dtype=acc_dt)
        if dht is not None:
            dh = dh + dht[i_n].astype(acc_dt)

        # Triton: b_dgk = tl.sum(b_h * b_dh, 1) — initial from h_final * dht
        if USE_GK:
            dgk_running = jnp.sum(h_final * dh, axis=2)  # [H, K]
        if USE_GV:
            dgv_running = jnp.sum(h_final * dh, axis=1)  # [H, V]

        # Backward: opposite direction of forward
        bwd_range = range(bos, eos) if reverse else range(eos - 1, bos - 1, -1)
        for t in bwd_range:
            # Triton: b_dh += (b_q * scale)[:, None] * b_do[None, :]
            dh = dh + (q_f[b, t, :, :, None] * scale) * do_f[b, t, :, None, :]

            # Triton: b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
            dk_t = jnp.sum(dh * v_f[b, t, :, None, :], axis=2)  # [H, K]
            # Triton: b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
            dv_t = jnp.sum(dh * k_f[b, t, :, :, None], axis=1)  # [H, V]
            dk = dk.at[b, t].set(dk_t)
            dv = dv.at[b, t].set(dv_t)

            # Gate gradients (running suffix sum) + gate application to dh
            if USE_GK:
                # Triton: b_dgk += b_q * b_dq - b_k * b_dk
                dgk_running = dgk_running + q_f[b, t] * dq[b, t] - k_f[b, t] * dk_t
                dgk_out = dgk_out.at[b, t].set(dgk_running)
                # Triton: b_dh *= exp(b_gk)[:, None]
                dh = dh * jnp.exp(gk_f[b, t, :, :, None])

            if USE_GV:
                # Triton: if i_k == 0: b_dgv += b_o * b_do
                #         b_dgv -= b_v * b_dv
                # (no tiling in CPU ref, so o*do is added directly)
                dgv_running = dgv_running + o_f[b, t] * do_f[b, t] - v_f[b, t] * dv_t
                dgv_out = dgv_out.at[b, t].set(dgv_running)
                # Triton: b_dh *= exp(b_gv)[None, :]
                dh = dh * jnp.exp(gv_f[b, t, :, None, :])

        if initial_state is not None:
            dh0_list.append(dh)

    # Triton: dh0 = torch.empty_like(h0), store b_dh.to(p_dh0.dtype.element_ty)
    dh0 = jnp.stack(dh0_list, axis=0) if initial_state is not None else None
    if dh0 is not None:
        dh0 = dh0.astype(initial_state.dtype)

    # FLA FusedRecurrentFunction.backward casts:
    #   dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
    #   dgk already .to(gk) in fused_recurrent_bwd, dgv .to(gv)
    #   dh0 stays h0.dtype (empty_like)
    # fp64 mode: no casts (all fp64 → fp64 is no-op)
    dq = dq.astype(q.dtype)
    dk = dk.astype(k.dtype)
    dv = dv.astype(v.dtype)
    if USE_GK:
        dgk_out = dgk_out.astype(gk.dtype)
    if USE_GV:
        dgv_out = dgv_out.astype(gv.dtype)

    return dq, dk, dv, dgk_out, dgv_out, dh0


# =============================================================================
# Public API
# =============================================================================


@cpu_reference
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
    cu_seqlens: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused recurrent GLA — public API matching FLA signature.

    Wraps fused_recurrent_fwd, casting output to q.dtype (mirrors
    FusedRecurrentFunction.forward in FLA). For backward with gv,
    call fused_recurrent_fwd directly to obtain fp32 o, then pass
    it to fused_recurrent_bwd.

    Args:
        q:  [B, T, H, K] — Queries
        k:  [B, T, H, K] — Keys
        v:  [B, T, H, V] — Values
        gk: [B, T, H, K] — Per-key gate in log-space (optional)
        gv: [B, T, H, V] — Per-value gate in log-space (optional)
        scale: Scaling factor, default K^{-0.5}
        initial_state: [N, H, K, V] — Initial hidden state
        output_final_state: Whether to return final hidden state
        reverse: If True, iterate from T-1 to 0
        cu_seqlens: [N+1] — Cumulative sequence lengths

    Returns:
        o:  [B, T, H, V] in q.dtype
        ht: [N, H, K, V] in fp32/fp64, or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert k.shape == (B, T, H, K), f"k shape {k.shape} != expected {(B, T, H, K)}"
    assert v.shape[:3] == (B, T, H), f"v shape {v.shape} incompatible with q"
    assert gk is None or gk.shape == (B, T, H, K), (
        f"gk shape {gk.shape} != expected {(B, T, H, K)}"
    )
    assert gv is None or gv.shape == (B, T, H, V), (
        f"gv shape {gv.shape} != expected {(B, T, H, V)}"
    )
    assert initial_state is None or initial_state.shape == (N, H, K, V), (
        f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"
    )
    assert cu_seqlens is None or B == 1, (
        f"cu_seqlens requires B=1, got B={B}"
    )

    o, ht = fused_recurrent_fwd(
        q=q, k=k, v=v, gk=gk, gv=gv, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        reverse=reverse, cu_seqlens=cu_seqlens,
    )
    # FLA FusedRecurrentFunction.forward: return o.to(q.dtype), ht
    return o.astype(q.dtype), ht
