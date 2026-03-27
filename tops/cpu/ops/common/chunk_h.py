"""Shared chunk_fwd_h and chunk_bwd_dh for GLA / Simple GLA CPU reference.

Matches FLA Triton chunk_fwd_kernel_h / chunk_bwd_kernel_dh behavior exactly,
including the variable-reuse semantics when both USE_G and USE_G_GAMMA are active.

WARNING: passing both g and g_gamma simultaneously is NOT recommended.
  FLA's Triton kernel has a variable-reuse bug in chunk_fwd_h / chunk_bwd_dh:
  USE_G overwrites b_g with loaded g values inside the loop, then
  USE_G_GAMMA's exp(b_g_last - b_g) uses those g values instead of
  gamma*pos. Since g_last_gamma (~-19) minus g (~-32) can be positive
  (~+13), exp(+13) >> 1 and the state explodes. This CPU ref replicates
  that behavior for FLA-exactness, but callers should use only one of
  g or g_gamma. The backward orchestrator (chunk_simple_gla_bwd) explicitly
  rejects the "both" case because chunk_bwd_dqkwg/chunk_bwd_dv treat
  g vs g_gamma as mutually exclusive (if/elif).

Gate flags (matching FLA Triton constexpr parameters):
  USE_G:       g is not None      — per-head scalar gate [B, T, H] (gates v)
  USE_G_GAMMA: g_gamma is not None — fixed per-head decay [H] (gates v)
  USE_GK:      gk is not None     — per-element K-dim gate [B, T, H, K] (gates k)
"""

from __future__ import annotations

import jax.numpy as jnp

from tops.cpu.ops.common.utils import acc_dtype, dot


def chunk_fwd_h(
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    g: jnp.ndarray | None = None,
    g_gamma: jnp.ndarray | None = None,
    gk: jnp.ndarray | None = None,
    h0: jnp.ndarray | None = None,
    output_final_state: bool = False,
    states_in_fp32: bool = False,
    chunk_size: int = 64,
    original_T: int | None = None,
    cu_seqlens: jnp.ndarray | None = None,
    orig_seqlens: list[int] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Inter-chunk hidden state propagation (shared GLA / Simple GLA).

    Processes chunks sequentially, propagating hidden state h through
    gate-decayed outer products of k and v.

    Gate behavior:
      USE_G (g):       scalar per-head gate on v — b_v *= exp(g_last - g)
      USE_G_GAMMA:     fixed per-head decay on v — b_v *= exp(gamma_last - b_g)
                       where b_g comes from USE_G when both active (FLA reuse)
      USE_GK (gk):     per-element gate on k — b_k *= exp(gk_last - gk)

    Args:
        k:  [B, T, H, K] — keys (input dtype)
        v:  [B, T, H, V] — values (input dtype)
        g:  [B, T, H]    — chunk-local cumsummed scalar gates (fp32). Optional.
        g_gamma: [H]     — fixed per-head log-decay. Optional.
        gk: [B, T, H, K] — chunk-local cumsummed K-dim gates (fp32). Optional.
        h0: [B, H, K, V] or [N, H, K, V] — initial hidden state (fp32). Optional.
            When cu_seqlens is provided, shape is [N, H, K, V] (one per sequence).
        output_final_state: whether to return final state
        states_in_fp32: if True, store h in fp32; else in k.dtype
        chunk_size: block size
        original_T: original unpadded T for g_gamma chunk_len. Defaults to T.
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length
            packing. When provided, B must be 1 and hidden state resets at
            each sequence boundary. h0/ht become [N, H, K, V].
        orig_seqlens: per-sequence original (unpadded) lengths for g_gamma
            chunk_len computation in varlen mode. Optional.

    Returns:
        h:  [B, NT, H, K, V] — hidden states
        ht: [B, H, K, V] or [N, H, K, V] (fp32) or None
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    acc = acc_dtype(k.dtype)
    if original_T is None:
        original_T = T

    # --- Varlen setup ---
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
        N = len(cu_seqlens) - 1
        chunk_starts = [i * C for i in range(NT)]
        # Map each chunk index to its sequence index
        cu_list = [int(cu_seqlens[j]) for j in range(N + 1)]
        chunk_to_seq = []
        for cs in chunk_starts:
            seq_id = 0
            for j in range(1, N + 1):
                if cs < cu_list[j]:
                    seq_id = j - 1
                    break
            chunk_to_seq.append(seq_id)

    store_dtype = acc if k.dtype == jnp.float64 else (jnp.float32 if states_in_fp32 else k.dtype)

    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    if g is not None:
        g_c = g.reshape(B, NT, C, H)
    if gk is not None:
        gk_c = gk.reshape(B, NT, C, H, K)

    h_list = []
    if is_varlen:
        h = jnp.zeros((1, H, K, V), dtype=acc)
        if h0 is not None:
            h = h + h0[0:1].astype(acc)
        ht_list = [None] * N
    else:
        h = jnp.zeros((B, H, K, V), dtype=acc)
        if h0 is not None:
            h = h + h0.astype(acc)

    # FLA: b_g initialized before loop when USE_G_GAMMA
    if g_gamma is not None:
        gamma = g_gamma.astype(acc)
        b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]

    for i in range(NT):
        # --- Varlen: reset h at sequence boundaries ---
        if is_varlen:
            seq_id = chunk_to_seq[i]
            chunk_start = i * C
            if chunk_start == cu_list[seq_id]:
                # First chunk of this sequence: reset h
                h = jnp.zeros((1, H, K, V), dtype=acc)
                if h0 is not None:
                    h = h + h0[seq_id : seq_id + 1].astype(acc)

        h_list.append(h.astype(store_dtype))

        b_k = k_c[:, i]  # [B, C, H, K]
        b_v = v_c[:, i]  # [B, C, H, V]

        # --- USE_G: scalar gate on v ---
        if g is not None:
            b_g = g_c[:, i]           # [B, C, H] — this is the "b_g" FLA uses
            b_g_last = b_g[:, -1]     # [B, H]
            h = h * jnp.exp(b_g_last[:, :, None, None])
            b_v = (b_v * jnp.exp(b_g_last[:, None, :, None] - b_g[..., None])).astype(v.dtype)

        # --- USE_G_GAMMA: fixed per-head decay on v ---
        if g_gamma is not None:
            if is_varlen and orig_seqlens is not None:
                seg_start = cu_list[seq_id]
                seg_original_T = orig_seqlens[seq_id]
                local_i = (i * C - seg_start) // C
                chunk_len = min(C, seg_original_T - local_i * C)
            else:
                chunk_len = min(C, original_T - i * C)
            g_last_gamma = gamma * chunk_len  # [H]
            h = h * jnp.exp(g_last_gamma[None, :, None, None])
            # FLA variable reuse: when USE_G is also active, b_g_gamma was
            # initialized before the loop but USE_G overwrote b_g with loaded
            # g values. So exp(g_last_gamma - b_g) uses g values, not gamma*pos.
            # When USE_G is not active, use b_g_gamma (gamma*pos).
            if g is not None:
                # Match FLA: USE_G_GAMMA uses g values as b_g
                b_g_reuse = b_g  # [B, C, H] from USE_G block
                b_v = (b_v * jnp.exp(g_last_gamma[None, None, :, None] - b_g_reuse[..., None])).astype(v.dtype)
            else:
                b_g_clamped = jnp.where(
                    jnp.arange(C)[:, None] < chunk_len,
                    b_g_gamma,
                    g_last_gamma[None, :],
                )
                b_v = (b_v * jnp.exp(g_last_gamma[None, None, :, None] - b_g_clamped[None, :, :, None])).astype(v.dtype)

        # --- USE_GK: per-element K-dim gate on k ---
        if gk is not None:
            gc_gk = gk_c[:, i]       # [B, C, H, K]
            gk_last = gc_gk[:, -1]   # [B, H, K]
            h = h * jnp.exp(gk_last[:, :, :, None])
            b_k = (b_k * jnp.exp(gk_last[:, None, :, :] - gc_gk)).astype(k.dtype)

        h = h + dot("bchk,bchv->bhkv", b_k, b_v, acc)

        # NOTE: FLA Triton keeps b_h in fp32 registers throughout the loop.
        # tl.store(p_h, b_h.to(p_h.dtype)) is write-only to HBM — the
        # accumulator is never read back, so no round-trip quantization occurs.
        # The h_list entries (for chunk_fwd_o) are already cast to store_dtype
        # above, matching FLA's HBM stores.

        # --- Varlen: store ht at last chunk of each sequence ---
        if is_varlen and output_final_state:
            # Check if this is the last chunk of the current sequence
            next_chunk_start = (i + 1) * C
            if i == NT - 1 or next_chunk_start >= cu_list[seq_id + 1]:
                ht_list[seq_id] = h[0]

    h_all = jnp.stack(h_list, axis=1)  # [B, NT, H, K, V]
    if is_varlen:
        ht = jnp.stack(ht_list) if output_final_state else None  # [N, H, K, V]
    else:
        ht = h if output_final_state else None
    return h_all, ht


def chunk_bwd_dh(
    q: jnp.ndarray,
    do: jnp.ndarray,
    *,
    g: jnp.ndarray | None = None,
    g_gamma: jnp.ndarray | None = None,
    gk: jnp.ndarray | None = None,
    h0: jnp.ndarray | None = None,
    dht: jnp.ndarray | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    original_T: int | None = None,
    cu_seqlens: jnp.ndarray | None = None,
    orig_seqlens: list[int] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Backward hidden state gradient propagation (shared GLA / Simple GLA).

    Reverse iteration through chunks. Matches FLA chunk_bwd_kernel_dh.

    Variable-reuse when USE_G + USE_G_GAMMA:
      USE_G sets b_g to loaded g values; USE_G_GAMMA's exp(b_g) then uses
      those g values, not gamma*pos. This matches FLA's Triton kernel exactly.

    Args:
        q:   [B, T, H, K] — queries (input dtype)
        do:  [B, T, H, V] — output gradient (input dtype)
        g:   [B, T, H]    — cumsummed scalar gates (fp32). Optional.
        g_gamma: [H]      — fixed per-head log-decay. Optional.
        gk:  [B, T, H, K] — cumsummed K-dim gates (fp32). Optional.
        h0:  [B, H, K, V] or [N, H, K, V] — initial state (determines if dh0 returned).
            When cu_seqlens is provided, shape is [N, H, K, V].
        dht: [B, H, K, V] or [N, H, K, V] — terminal state gradient. Optional.
            When cu_seqlens is provided, shape is [N, H, K, V].
        scale: scaling factor
        chunk_size: block size
        original_T: original unpadded T. Defaults to T.
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length
            packing. When provided, B must be 1 and gradient resets at
            each sequence boundary.
        orig_seqlens: per-sequence original (unpadded) lengths for g_gamma
            chunk_len computation in varlen mode. Optional.

    Returns:
        dh:  [B, NT, H, K, V] — acc dtype
        dh0: [B, H, K, V] or [N, H, K, V] or None
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C
    acc = acc_dtype(q.dtype)
    if original_T is None:
        original_T = T

    # --- Varlen setup ---
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
        N = len(cu_seqlens) - 1
        chunk_starts = [i * C for i in range(NT)]
        cu_list = [int(cu_seqlens[j]) for j in range(N + 1)]
        chunk_to_seq = []
        for cs in chunk_starts:
            seq_id = 0
            for j in range(1, N + 1):
                if cs < cu_list[j]:
                    seq_id = j - 1
                    break
            chunk_to_seq.append(seq_id)

    q_c = q.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    if g is not None:
        g_c = g.reshape(B, NT, C, H)
    if gk is not None:
        gk_c = gk.reshape(B, NT, C, H, K)

    dh_list = [None] * NT
    if is_varlen:
        # Start from the last sequence's dht
        last_seq = chunk_to_seq[NT - 1]
        dh = jnp.zeros((1, H, K, V), dtype=acc)
        if dht is not None:
            dh = dh + dht[last_seq : last_seq + 1].astype(acc)
        dh0_list = [None] * N
    else:
        dh = jnp.zeros((B, H, K, V), dtype=acc)
        if dht is not None:
            dh = dh + dht.astype(acc)

    # FLA: b_g initialized before loop when USE_G_GAMMA
    if g_gamma is not None:
        gamma = g_gamma.astype(acc)
        b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]

    for i in range(NT - 1, -1, -1):
        # --- Varlen: reset dh at end-of-sequence boundary (reverse order) ---
        if is_varlen:
            seq_id = chunk_to_seq[i]
            # Check if this is the last chunk of this sequence
            next_chunk_start = (i + 1) * C
            is_last_chunk = (i == NT - 1) or (next_chunk_start >= cu_list[seq_id + 1])
            if is_last_chunk:
                dh = jnp.zeros((1, H, K, V), dtype=acc)
                if dht is not None:
                    dh = dh + dht[seq_id : seq_id + 1].astype(acc)

        dh_list[i] = dh

        b_q = q_c[:, i]      # [B, C, H, K]
        b_do = do_c[:, i]    # [B, C, H, V]

        # --- USE_GK: per-element gate ---
        if gk is not None:
            gc_gk = gk_c[:, i]       # [B, C, H, K]
            gk_last = gc_gk[:, -1]   # [B, H, K]
            b_q = (b_q * scale * jnp.exp(gc_gk)).astype(q.dtype)
            dh = dh * jnp.exp(gk_last[:, :, :, None])
        else:
            b_q = (b_q * scale).astype(q.dtype)

        # --- USE_G: scalar gate ---
        if g is not None:
            b_g = g_c[:, i]           # [B, C, H]
            g_last = b_g[:, -1]       # [B, H]
            b_q = (b_q * jnp.exp(b_g[..., None])).astype(q.dtype)
            dh = dh * jnp.exp(g_last[:, :, None, None])

        # --- USE_G_GAMMA: fixed per-head decay ---
        if g_gamma is not None:
            if is_varlen and orig_seqlens is not None:
                seg_start = cu_list[seq_id]
                seg_original_T = orig_seqlens[seq_id]
                local_i = (i * C - seg_start) // C
                chunk_len = min(C, seg_original_T - local_i * C)
            else:
                chunk_len = min(C, original_T - i * C)
            g_last_gamma = gamma * chunk_len  # [H]
            # FLA variable reuse: when USE_G is also active, b_g_gamma was
            # set before the loop but USE_G overwrote b_g. So exp(b_g) uses
            # g values. When USE_G is not active, use b_g_gamma.
            if g is not None:
                b_q = (b_q * jnp.exp(b_g[..., None])).astype(q.dtype)
            else:
                b_q = (b_q * jnp.exp(b_g_gamma[None, :, :, None])).astype(q.dtype)
            dh = dh * jnp.exp(g_last_gamma[None, :, None, None])

        dh = dh + dot("bchk,bchv->bhkv", b_q, b_do.astype(q.dtype), acc)

        # --- Varlen: store dh0 at first chunk of each sequence ---
        if is_varlen:
            chunk_start = i * C
            if chunk_start == cu_list[seq_id]:
                dh0_list[seq_id] = dh[0]

    dh_all = jnp.stack(dh_list, axis=1)  # [B, NT, H, K, V]
    if is_varlen:
        dh0 = jnp.stack(dh0_list) if h0 is not None else None  # [N, H, K, V]
    else:
        dh0 = dh if h0 is not None else None
    return dh_all, dh0
