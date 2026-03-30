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

from tops.cpu.ops.common.utils import acc_dtype, cdiv, dot, read_chunk


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
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Inter-chunk hidden state propagation (shared GLA / Simple GLA).

    Processes chunks sequentially, propagating hidden state h through
    gate-decayed outer products of k and v.

    Gate behavior:
      USE_G (g):       scalar per-head gate on v — b_v *= exp(g_last - g)
      USE_G_GAMMA:     fixed per-head decay on v — b_v *= exp(gamma_last - b_g)
                       where b_g comes from USE_G when both active (FLA reuse)
      USE_GK (gk):     per-element gate on k — b_k *= exp(gk_last - gk)

    When cu_seqlens is provided, iterates per-sequence (matching Triton's
    chunk_fwd_kernel_h grid over N*H). Each sequence is processed
    independently with its own hidden state. Partial last chunks are
    handled via read_chunk zero-padding.

    Args:
        k:  [B, T, H, K] — keys (input dtype). Packed layout when varlen.
        v:  [B, T, H, V] — values (input dtype). Packed layout when varlen.
        g:  [B, T, H]    — chunk-local cumsummed scalar gates (fp32). Optional.
        g_gamma: [H]     — fixed per-head log-decay. Optional.
        gk: [B, T, H, K] — chunk-local cumsummed K-dim gates (fp32). Optional.
        h0: [B, H, K, V] or [N, H, K, V] — initial hidden state (fp32). Optional.
            When cu_seqlens is provided, shape is [N, H, K, V] (one per sequence).
        output_final_state: whether to return final state
        states_in_fp32: if True, store h in fp32; else in k.dtype
        chunk_size: block size
        original_T: original unpadded T for g_gamma chunk_len. Defaults to T.
            Not used when cu_seqlens is provided (derived from valid_len).
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length
            packing. When provided, B must be 1 and hidden state resets at
            each sequence boundary. h0/ht become [N, H, K, V].

    Returns:
        h:  [B, total_NT, H, K, V] — hidden states (total_NT chunks across
            all sequences when varlen, else T//chunk_size)
        ht: [B, H, K, V] or [N, H, K, V] (fp32) or None
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    acc = acc_dtype(k.dtype)
    if original_T is None:
        original_T = T

    store_dtype = acc if k.dtype == jnp.float64 else (jnp.float32 if states_in_fp32 else k.dtype)

    # FLA: b_g initialized before loop when USE_G_GAMMA
    if g_gamma is not None:
        gamma = g_gamma.astype(acc)
        b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]

    # =====================================================================
    # Varlen path: per-sequence iteration (matches Triton chunk_fwd_kernel_h)
    # =====================================================================
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
        N = len(cu_seqlens) - 1
        cu_list = [int(cu_seqlens[j]) for j in range(N + 1)]

        h_list = []
        ht_list = [None] * N

        for i_n in range(N):
            bos, eos = cu_list[i_n], cu_list[i_n + 1]
            T_seq = eos - bos
            NT_seq = cdiv(T_seq, C)

            # Initialize h for this sequence
            h = jnp.zeros((1, H, K, V), dtype=acc)
            if h0 is not None:
                h = h + h0[i_n : i_n + 1].astype(acc)

            for i_t in range(NT_seq):
                h_list.append(h.astype(store_dtype))

                start = bos + i_t * C
                valid_len = min(C, T_seq - i_t * C)

                b_k = read_chunk(k, start, valid_len, C)  # [1, C, H, K]
                b_v = read_chunk(v, start, valid_len, C)  # [1, C, H, V]

                # --- USE_G: scalar gate on v ---
                b_g = None
                if g is not None:
                    b_g = read_chunk(g, start, valid_len, C)  # [1, C, H]
                    b_g_last = g[:, start + valid_len - 1 : start + valid_len]  # [1, 1, H]
                    b_g_last = b_g_last[:, 0]  # [1, H]
                    h = h * jnp.exp(b_g_last[:, :, None, None])
                    b_v = (b_v * jnp.exp(b_g_last[:, None, :, None] - b_g[..., None])).astype(v.dtype)

                # --- USE_G_GAMMA: fixed per-head decay on v ---
                if g_gamma is not None:
                    chunk_len = valid_len
                    g_last_gamma = gamma * chunk_len  # [H]
                    h = h * jnp.exp(g_last_gamma[None, :, None, None])
                    if g is not None:
                        b_g_reuse = b_g  # [1, C, H] from USE_G block
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
                    b_gk = read_chunk(gk, start, valid_len, C)  # [1, C, H, K]
                    gk_last = gk[:, start + valid_len - 1 : start + valid_len]  # [1, 1, H, K]
                    gk_last = gk_last[:, 0]  # [1, H, K]
                    h = h * jnp.exp(gk_last[:, :, :, None])
                    b_k = (b_k * jnp.exp(gk_last[:, None, :, :] - b_gk)).astype(k.dtype)

                h = h + dot("bchk,bchv->bhkv", b_k, b_v, acc)

            # Store final state for this sequence
            if output_final_state:
                ht_list[i_n] = h[0]

        h_all = jnp.stack(h_list, axis=1)  # [1, total_NT, H, K, V]
        ht = jnp.stack(ht_list) if output_final_state else None  # [N, H, K, V]
        return h_all, ht

    # =====================================================================
    # Non-varlen path: standard uniform-chunk iteration
    # =====================================================================
    NT = T // C
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    if g is not None:
        g_c = g.reshape(B, NT, C, H)
    if gk is not None:
        gk_c = gk.reshape(B, NT, C, H, K)

    h_list = []
    h = jnp.zeros((B, H, K, V), dtype=acc)
    if h0 is not None:
        h = h + h0.astype(acc)

    for i in range(NT):
        h_list.append(h.astype(store_dtype))

        b_k = k_c[:, i]  # [B, C, H, K]
        b_v = v_c[:, i]  # [B, C, H, V]

        # --- USE_G: scalar gate on v ---
        if g is not None:
            b_g = g_c[:, i]           # [B, C, H]
            b_g_last = b_g[:, -1]     # [B, H]
            h = h * jnp.exp(b_g_last[:, :, None, None])
            b_v = (b_v * jnp.exp(b_g_last[:, None, :, None] - b_g[..., None])).astype(v.dtype)

        # --- USE_G_GAMMA: fixed per-head decay on v ---
        if g_gamma is not None:
            chunk_len = min(C, original_T - i * C)
            g_last_gamma = gamma * chunk_len  # [H]
            h = h * jnp.exp(g_last_gamma[None, :, None, None])
            if g is not None:
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

    h_all = jnp.stack(h_list, axis=1)  # [B, NT, H, K, V]
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
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Backward hidden state gradient propagation (shared GLA / Simple GLA).

    Reverse iteration through chunks. Matches FLA chunk_bwd_kernel_dh.

    Variable-reuse when USE_G + USE_G_GAMMA:
      USE_G sets b_g to loaded g values; USE_G_GAMMA's exp(b_g) then uses
      those g values, not gamma*pos. This matches FLA's Triton kernel exactly.

    When cu_seqlens is provided, iterates per-sequence in reverse order
    (matching Triton's chunk_bwd_kernel_dh). Each sequence is processed
    independently. Partial last chunks handled via read_chunk zero-padding.

    Args:
        q:   [B, T, H, K] — queries (input dtype). Packed layout when varlen.
        do:  [B, T, H, V] — output gradient (input dtype). Packed layout when varlen.
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
            Not used when cu_seqlens is provided (derived from valid_len).
        cu_seqlens: [N+1] cumulative sequence lengths for variable-length
            packing. When provided, B must be 1 and gradient resets at
            each sequence boundary.

    Returns:
        dh:  [B, total_NT, H, K, V] — acc dtype
        dh0: [B, H, K, V] or [N, H, K, V] or None
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    acc = acc_dtype(q.dtype)
    if original_T is None:
        original_T = T

    # FLA: b_g initialized before loop when USE_G_GAMMA
    if g_gamma is not None:
        gamma = g_gamma.astype(acc)
        b_g_gamma = gamma[None, :] * jnp.arange(1, C + 1, dtype=acc)[:, None]  # [C, H]

    # =====================================================================
    # Varlen path: per-sequence reverse iteration
    # =====================================================================
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
        N = len(cu_seqlens) - 1
        cu_list = [int(cu_seqlens[j]) for j in range(N + 1)]

        # Build flat dh_list indexed by global chunk index
        # First compute total_NT and per-sequence chunk counts
        seq_nt = []
        total_NT = 0
        for i_n in range(N):
            T_seq = cu_list[i_n + 1] - cu_list[i_n]
            nt = cdiv(T_seq, C)
            seq_nt.append(nt)
            total_NT += nt

        dh_list = [None] * total_NT
        dh0_list = [None] * N

        # Compute starting global chunk index for each sequence
        seq_offsets = [0] * N
        offset = 0
        for i_n in range(N):
            seq_offsets[i_n] = offset
            offset += seq_nt[i_n]

        for i_n in range(N - 1, -1, -1):
            bos, eos = cu_list[i_n], cu_list[i_n + 1]
            T_seq = eos - bos
            NT_seq = seq_nt[i_n]
            base_idx = seq_offsets[i_n]

            # Initialize dh for this sequence from dht
            dh = jnp.zeros((1, H, K, V), dtype=acc)
            if dht is not None:
                dh = dh + dht[i_n : i_n + 1].astype(acc)

            for i_t in range(NT_seq - 1, -1, -1):
                global_idx = base_idx + i_t
                dh_list[global_idx] = dh

                start = bos + i_t * C
                valid_len = min(C, T_seq - i_t * C)

                b_q = read_chunk(q, start, valid_len, C)    # [1, C, H, K]
                b_do = read_chunk(do, start, valid_len, C)  # [1, C, H, V]

                # --- USE_GK: per-element gate ---
                if gk is not None:
                    b_gk = read_chunk(gk, start, valid_len, C)  # [1, C, H, K]
                    gk_last = gk[:, start + valid_len - 1 : start + valid_len][:, 0]  # [1, H, K]
                    b_q = (b_q * scale * jnp.exp(b_gk)).astype(q.dtype)
                    dh = dh * jnp.exp(gk_last[:, :, :, None])
                else:
                    b_q = (b_q * scale).astype(q.dtype)

                # --- USE_G: scalar gate ---
                if g is not None:
                    b_g = read_chunk(g, start, valid_len, C)  # [1, C, H]
                    g_last = g[:, start + valid_len - 1 : start + valid_len][:, 0]  # [1, H]
                    b_q = (b_q * jnp.exp(b_g[..., None])).astype(q.dtype)
                    dh = dh * jnp.exp(g_last[:, :, None, None])

                # --- USE_G_GAMMA: fixed per-head decay ---
                if g_gamma is not None:
                    chunk_len = valid_len
                    g_last_gamma = gamma * chunk_len  # [H]
                    if g is not None:
                        b_q = (b_q * jnp.exp(b_g[..., None])).astype(q.dtype)
                    else:
                        b_q = (b_q * jnp.exp(b_g_gamma[None, :, :, None])).astype(q.dtype)
                    dh = dh * jnp.exp(g_last_gamma[None, :, None, None])

                dh = dh + dot("bchk,bchv->bhkv", b_q, b_do.astype(q.dtype), acc)

            # Store dh0 for this sequence (after all chunks processed)
            if h0 is not None:
                dh0_list[i_n] = dh[0]

        dh_all = jnp.stack(dh_list, axis=1)  # [1, total_NT, H, K, V]
        dh0 = jnp.stack(dh0_list) if h0 is not None else None  # [N, H, K, V]
        return dh_all, dh0

    # =====================================================================
    # Non-varlen path: standard uniform-chunk iteration
    # =====================================================================
    NT = T // C

    q_c = q.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    if g is not None:
        g_c = g.reshape(B, NT, C, H)
    if gk is not None:
        gk_c = gk.reshape(B, NT, C, H, K)

    dh_list = [None] * NT
    dh = jnp.zeros((B, H, K, V), dtype=acc)
    if dht is not None:
        dh = dh + dht.astype(acc)

    for i in range(NT - 1, -1, -1):
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
            chunk_len = min(C, original_T - i * C)
            g_last_gamma = gamma * chunk_len  # [H]
            if g is not None:
                b_q = (b_q * jnp.exp(b_g[..., None])).astype(q.dtype)
            else:
                b_q = (b_q * jnp.exp(b_g_gamma[None, :, :, None])).astype(q.dtype)
            dh = dh * jnp.exp(g_last_gamma[None, :, None, None])

        dh = dh + dot("bchk,bchv->bhkv", b_q, b_do.astype(q.dtype), acc)

    dh_all = jnp.stack(dh_list, axis=1)  # [B, NT, H, K, V]
    dh0 = dh if h0 is not None else None
    return dh_all, dh0
