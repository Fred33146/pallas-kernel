"""Torch CPU reference for simple_gla (lightning_attn) chunk attention.

Simple GLA uses per-head scalar gates instead of per-element gates:
  - g: [B, T, H] — data-dependent per-head gate (optional)
  - g_gamma: [H] — data-independent fixed per-head log-decay (optional)
  - At least one of g or g_gamma must be provided.

All formulas are identical to GLA, except the gate is scalar per head
and broadcasts over the K dimension.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Helpers (reused from gla/chunk.py)
# =============================================================================


def pad_varlen_seqs(
    tensors: list[torch.Tensor],
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> tuple[list[torch.Tensor], torch.Tensor, list[int] | None, list[int] | None]:
    N = len(cu_seqlens) - 1
    orig_seqlens = torch.diff(cu_seqlens).tolist()
    padded_seqlens = [((L + chunk_size - 1) // chunk_size) * chunk_size for L in orig_seqlens]

    if orig_seqlens == padded_seqlens:
        return tensors, cu_seqlens, None, None

    padded = [[] for _ in tensors]
    for i in range(N):
        bos = cu_seqlens[i].item()
        L = orig_seqlens[i]
        pad = padded_seqlens[i] - L
        for j, t in enumerate(tensors):
            seg = t[:, bos:bos + L]
            # Determine pad width based on tensor ndim
            # [1, L, H, K] -> pad last 2 dims with 0, then time dim
            pad_args = [0] * ((t.ndim - 2) * 2) + [0, pad]
            padded[j].append(F.pad(seg, pad_args) if pad > 0 else seg)

    padded_tensors = [torch.cat(p, dim=1) for p in padded]
    offsets = [0]
    for pl in padded_seqlens:
        offsets.append(offsets[-1] + pl)
    new_cu_seqlens = torch.tensor(offsets, dtype=torch.long)

    return padded_tensors, new_cu_seqlens, orig_seqlens, padded_seqlens


def unpad_varlen_seqs(
    tensor: torch.Tensor,
    orig_seqlens: list[int],
    padded_seqlens: list[int],
) -> torch.Tensor:
    parts = []
    offset = 0
    for L, PL in zip(orig_seqlens, padded_seqlens):
        parts.append(tensor[:, offset:offset + L])
        offset += PL
    return torch.cat(parts, dim=1)


# =============================================================================
# Build effective gc from g and/or g_gamma
# =============================================================================


def build_gc(
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    chunk_size: int,
    T: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Build chunk-local cumsum gate gc [B, T, H] from g and/or g_gamma.

    Args:
        g: [B, T, H] — raw log-space per-head gates, or None
        g_gamma: [H] — fixed per-head log-decay, or None
        chunk_size: block size
        T: padded sequence length (must be multiple of chunk_size)
        cu_seqlens: optional varlen boundaries

    Returns:
        gc: [B, T, H] — chunk-local cumsum of effective gate
    """
    C = chunk_size
    assert T % C == 0

    gc = None

    if g is not None:
        B, T_g, H = g.shape
        assert T_g == T
        NT = T // C

        if cu_seqlens is not None:
            assert B == 1
            N = len(cu_seqlens) - 1
            gc = torch.zeros_like(g)
            for i in range(N):
                bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                seg = g[:, bos:eos]
                NT_seg = (eos - bos) // C
                gc[:, bos:eos] = seg.view(1, NT_seg, C, H).cumsum(dim=2).view(1, eos - bos, H)
        else:
            gc = g.view(B, NT, C, H).cumsum(dim=2).view(B, T, H)

    if g_gamma is not None:
        H = g_gamma.shape[0]
        # gamma_pos[t_in_chunk] = gamma * (t_in_chunk + 1), for t_in_chunk in [0, C-1]
        # Shape: [C, H]
        pos = torch.arange(1, C + 1, dtype=g_gamma.dtype, device=g_gamma.device).unsqueeze(-1)  # [C, 1]
        gamma_gc = pos * g_gamma.unsqueeze(0)  # [C, H]

        if gc is not None:
            B, _, H = gc.shape
            NT = T // C
            # Add gamma positional decay to each chunk
            gc = gc.view(B, NT, C, H) + gamma_gc[None, None, :, :]  # broadcast
            gc = gc.view(B, T, H)
        else:
            # g_gamma only — tile the positional decay for all chunks
            NT = T // C
            gc = gamma_gc[None, None, :, :].expand(1, NT, C, H).reshape(1, T, H)

    assert gc is not None, "At least one of g or g_gamma must be provided"
    return gc


# =============================================================================
# Forward: chunk_fwd_h (simple_gla version)
# =============================================================================


def chunk_simple_gla_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inter-chunk hidden state propagation for simple_gla.

    Args:
        k:  [B, T, H, K]
        v:  [B, T, H, V]
        gc: [B, T, H] — chunk-local cumsum of effective gate
        h0: [B, H, K, V] or [N, H, K, V]
        output_final_state: whether to return final state
        cu_seqlens: [N+1]
        chunk_size: block size

    Returns:
        h:  [B, NT, H, K, V]
        ht: [B, H, K, V] or None
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        h_list, ht_list = [], []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = h0[i:i+1] if h0 is not None else None
            h_seg, ht_seg = chunk_simple_gla_fwd_h(
                k[:, bos:eos], v[:, bos:eos], gc[:, bos:eos],
                h0=h0_i, output_final_state=output_final_state, chunk_size=chunk_size)
            h_list.append(h_seg)
            if ht_seg is not None:
                ht_list.append(ht_seg.squeeze(0))
        h_all = torch.cat(h_list, dim=1)
        ht = torch.stack(ht_list) if ht_list else None
        return h_all, ht

    NT = T // C

    k_c = k.view(B, NT, C, H, K)
    v_c = v.view(B, NT, C, H, V)
    gc_c = gc.view(B, NT, C, H)

    h = k.new_zeros(B, H, K, V, dtype=torch.float32)
    if h0 is not None:
        h = h + h0.float()

    h_list = []
    for i in range(NT):
        h_list.append(h.clone())

        gc_i = gc_c[:, i]          # [B, C, H]
        ki = k_c[:, i]             # [B, C, H, K]
        vi = v_c[:, i]             # [B, C, H, V]

        g_total = gc_i[:, -1]      # [B, H]

        # h = h * exp(g_total) + Σ_j k_j * exp(g_total - gc_j) ⊗ v_j
        # g_total: [B, H] -> broadcast over K, V
        h = h * g_total[:, :, None, None].exp()
        # k_state: [B, C, H, K], gate broadcast over K
        k_state = ki * (g_total[:, None, :] - gc_i).unsqueeze(-1).exp()
        h = h + torch.einsum("bchk,bchv->bhkv", k_state, vi)

    h_all = torch.stack(h_list, dim=1)  # [B, NT, H, K, V]
    ht = h if output_final_state else None
    return h_all, ht


# =============================================================================
# Forward: intra-chunk attention
# =============================================================================


def chunk_simple_gla_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Intra-chunk attention matrix with causal mask for simple_gla.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gc: [B, T, H] — chunk-local cumsum of gate
        scale: scaling factor
        chunk_size: block size

    Returns:
        A: [B, T, H, BT] (float32)
    """
    B, T, H, K = q.shape
    BT = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        A_list = []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            A_seg = chunk_simple_gla_fwd_intra(
                q[:, bos:eos], k[:, bos:eos], gc[:, bos:eos],
                scale, chunk_size=chunk_size)
            A_list.append(A_seg)
        return torch.cat(A_list, dim=1)

    NT = T // BT

    q_c = q.view(B, NT, BT, H, K)
    k_c = k.view(B, NT, BT, H, K)
    gc_c = gc.view(B, NT, BT, H)   # [B, NT, BT, H]

    # Numerical stability: reference point g_n (first row of each chunk)
    g_n = gc_c[:, :, 0:1, :]  # [B, NT, 1, H]
    q_gated = q_c * (gc_c - g_n).unsqueeze(-1).exp()   # broadcast over K
    k_gated = k_c * (g_n - gc_c).unsqueeze(-1).exp()   # broadcast over K

    A = torch.einsum("bnihk,bnjhk->bnihj", q_gated, k_gated) * scale

    return A.reshape(B, T, H, BT)


# =============================================================================
# Forward: output (inter + intra)
# =============================================================================


def chunk_simple_gla_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    gc: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Output = inter-chunk + intra-chunk for simple_gla.

    Args:
        q: [B, T, H, K]
        v: [B, T, H, V]
        gc: [B, T, H] — chunk-local cumsum of gate
        A: [B, T, H, BT]
        h: [B, NT, H, K, V]
        scale: scaling factor

    Returns:
        o: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    q = q.reshape(-1, C, H, K)
    v = v.reshape(-1, C, H, V)
    gc = gc.reshape(-1, C, H)
    h = h.reshape(-1, H, K, V)
    A = A.reshape(-1, C, H, C)

    # q_gated: gate broadcast over K
    qg = q * gc.unsqueeze(-1).exp()

    # Inter-chunk
    o_inter = scale * torch.einsum("nchk,nhkv->nchv", qg, h)

    # Intra-chunk
    o_intra = torch.einsum("nihj,njhv->nihv", A, v)

    o = (o_inter + o_intra).reshape(B, T, H, V)
    return o


# =============================================================================
# Forward orchestrator
# =============================================================================


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Simple GLA forward orchestrator.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H] — raw per-head log-space gate, or None
        g_gamma: [H] — fixed per-head log-decay, or None
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        output_final_state: bool
        cu_seqlens: [N+1] or None
        chunk_size: block size

    Returns:
        (gc, A, h, ht, o)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    pad_tensors = [q, k, v]
    if g is not None:
        pad_tensors.append(g)

    if cu_seqlens is not None:
        assert B == 1
        pad_tensors, cu_seqlens, orig_seqlens, padded_seqlens = pad_varlen_seqs(
            pad_tensors, cu_seqlens, C
        )
        if g is not None:
            q, k, v, g = pad_tensors
        else:
            q, k, v = pad_tensors
    else:
        T_padded = ((T + C - 1) // C) * C
        if T_padded > T:
            pad = T_padded - T
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            if g is not None:
                g = F.pad(g, (0, 0, 0, pad))

    T_padded = q.shape[1]
    NT = T_padded // C

    # --- build gc ---
    gc = build_gc(g, g_gamma, C, T_padded, cu_seqlens)
    # Ensure gc has correct batch size (g_gamma-only case returns B=1)
    if gc.shape[0] == 1 and B > 1:
        gc = gc.expand(B, -1, -1)

    # --- forward ---
    h, ht = chunk_simple_gla_fwd_h(
        k, v, gc=gc, h0=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    A = chunk_simple_gla_fwd_intra(q, k, gc=gc, scale=scale, cu_seqlens=cu_seqlens, chunk_size=C)
    # Causal mask
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=A.device))
    A = A.view(B, -1, C, H, C).masked_fill(~causal_mask[None, None, :, None, :], 0.0).reshape(B, T_padded, H, C)

    o = chunk_simple_gla_fwd_o(q, v, gc=gc, A=A, h=h, scale=scale, cu_seqlens=cu_seqlens, chunk_size=C)

    # --- unpadding ---
    if orig_seqlens is not None:
        gc = unpad_varlen_seqs(gc, orig_seqlens, padded_seqlens)
        o = unpad_varlen_seqs(o, orig_seqlens, padded_seqlens)
        A = unpad_varlen_seqs(A, orig_seqlens, padded_seqlens)
    else:
        o = o[:, :T]
        gc = gc[:, :T]
        A = A[:, :T]

    return gc, A, h, ht, o


# =============================================================================
# Backward: dh
# =============================================================================


def chunk_simple_gla_bwd_dh(
    q: torch.Tensor,
    gc: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backward hidden state gradient propagation for simple_gla.

    Args:
        q:   [B, T, H, K]
        gc:  [B, T, H] — chunk-local cumsum of gate
        do:  [B, T, H, V]
        h0:  [B, H, K, V] or [N, H, K, V]
        dht: [B, H, K, V] or [N, H, K, V]
        scale: scaling factor
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V]
        dh0: [B, H, K, V] or None
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dh_list, dh0_list = [], []
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            h0_i = h0[i:i+1] if h0 is not None else None
            dht_i = dht[i:i+1] if dht is not None else None
            dh_seg, dh0_seg = chunk_simple_gla_bwd_dh(
                q[:, bos:eos], gc[:, bos:eos], do[:, bos:eos],
                h0=h0_i, dht=dht_i, scale=scale, chunk_size=chunk_size)
            dh_list.append(dh_seg)
            if dh0_seg is not None:
                dh0_list.append(dh0_seg.squeeze(0))
        dh = torch.cat(dh_list, dim=1)
        dh0 = torch.stack(dh0_list) if dh0_list else None
        return dh, dh0

    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    do_c = do.view(B, NT, C, H, V)
    gc_c = gc.view(B, NT, C, H)

    dh = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if dht is not None:
        dh = dh + dht.float()

    dh_list = [None] * NT
    for i in range(NT - 1, -1, -1):
        dh_list[i] = dh.clone()

        b_q = q_c[:, i]         # [B, C, H, K]
        b_do = do_c[:, i]       # [B, C, H, V]
        gc_i = gc_c[:, i]       # [B, C, H]
        g_total = gc_i[:, -1]   # [B, H]

        # q_hat = q * exp(gc) * scale, gate broadcast over K
        b_q_hat = b_q * gc_i.unsqueeze(-1).exp() * scale
        # dh decay: broadcast g_total over K, V
        dh = dh * g_total[:, :, None, None].exp()
        dh = dh + torch.einsum("bchk,bchv->bhkv", b_q_hat, b_do)

    dh_all = torch.stack(dh_list, dim=1)
    dh0 = dh if (h0 is not None or dht is not None) else None
    return dh_all, dh0


# =============================================================================
# Backward: dA
# =============================================================================


def chunk_simple_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Gradient of intra-chunk attention matrix (identical to GLA — no gate involved)."""
    B, T, H, V = v.shape
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dA = torch.zeros(1, T, H, C, dtype=v.dtype, device=v.device)
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            dA[:, bos:eos] = chunk_simple_gla_bwd_dA(v[:, bos:eos], do[:, bos:eos], scale, chunk_size=chunk_size)
        return dA

    NT = T // C
    v_c = v.view(B, NT, C, H, V)
    do_c = do.view(B, NT, C, H, V)

    dA = torch.einsum("bnihv,bnjhv->bnihj", do_c, v_c) * scale
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    dA = torch.where(causal_mask[None, None, :, None, :], dA, 0.0)
    return dA.reshape(B, T, H, C)


# =============================================================================
# Backward: dv
# =============================================================================


def chunk_simple_gla_bwd_dv(
    k: torch.Tensor,
    gc: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Gradient of v for simple_gla.

    Args:
        k:  [B, T, H, K]
        gc: [B, T, H] — chunk-local cumsum of gate
        A:  [B, T, H, C]
        do: [B, T, H, V]
        dh: [B, NT, H, K, V]
        chunk_size: block size

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = k.shape
    V = do.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dv = torch.zeros(1, T, H, V, dtype=do.dtype, device=do.device)
        chunk_offset = 0
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            NT_seg = (eos - bos) // C
            dv[:, bos:eos] = chunk_simple_gla_bwd_dv(
                k[:, bos:eos], gc[:, bos:eos], A[:, bos:eos],
                do[:, bos:eos], dh[:, chunk_offset:chunk_offset+NT_seg],
                chunk_size=chunk_size)
            chunk_offset += NT_seg
        return dv

    NT = T // C

    k_c = k.view(B, NT, C, H, K)
    gc_c = gc.view(B, NT, C, H)
    do_c = do.view(B, NT, C, H, V)
    A_c = A.view(B, NT, C, H, C)

    # Intra: dv[j] = sum_{i>=j} A[i,j] * do[i]
    dv_intra = torch.einsum("bnihj,bnihv->bnjhv", A_c, do_c)

    # Inter: k_decay @ dh
    gn = gc_c[:, :, -1, :]  # [B, NT, H]
    # k_decay: [B, NT, C, H, K], gate broadcast over K
    k_decay = k_c * (gn[:, :, None, :] - gc_c).unsqueeze(-1).exp()
    dv_inter = torch.einsum("bnchk,bnhkv->bnchv", k_decay, dh)

    return (dv_intra + dv_inter).reshape(B, T, H, V)


# =============================================================================
# Backward: dq, dk (intra-chunk from dA)
# =============================================================================


def chunk_simple_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gc: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intra-chunk dq, dk from dA for simple_gla.

    Args:
        q, k: [B, T, H, K]
        gc:   [B, T, H] — chunk-local cumsum of gate
        dA:   [B, T, H, C]

    Returns:
        dq, dk: [B, T, H, K]
    """
    B, T, H, K = q.shape
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dq_out = torch.zeros_like(q)
        dk_out = torch.zeros_like(k)
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            dq_seg, dk_seg = chunk_simple_gla_bwd_dqk_intra(
                q[:, bos:eos], k[:, bos:eos], gc[:, bos:eos],
                dA[:, bos:eos], chunk_size=chunk_size)
            dq_out[:, bos:eos] = dq_seg
            dk_out[:, bos:eos] = dk_seg
        return dq_out, dk_out

    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    gc_c = gc.view(B, NT, C, H)       # [B, NT, C, H]
    dA_c = dA.view(B, NT, C, H, C)

    # k_neg = k * exp(-gc), gc broadcast over K
    k_neg = k_c * (-gc_c).unsqueeze(-1).exp()
    dq = gc_c.unsqueeze(-1).exp() * torch.einsum("bnihj,bnjhk->bnihk", dA_c, k_neg)

    q_pos = q_c * gc_c.unsqueeze(-1).exp()
    dk = (-gc_c).unsqueeze(-1).exp() * torch.einsum("bnihj,bnihk->bnjhk", dA_c, q_pos)

    return dq.reshape(B, T, H, K), dk.reshape(B, T, H, K)


# =============================================================================
# Backward: dq, dk (inter) + dg
# =============================================================================


def chunk_simple_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    gc: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float,
    has_g: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Inter-chunk dq, dk + gate gradient dg for simple_gla.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        h:    [B, NT, H, K, V]
        gc:   [B, T, H] — chunk-local cumsum of gate
        do:   [B, T, H, V]
        dh:   [B, NT, H, K, V]
        dq, dk: [B, T, H, K] — intra-chunk gradients
        scale: scaling factor
        has_g: whether g was provided (if False, no dg computed)

    Returns:
        dq, dk: [B, T, H, K]
        dg:     [B, T, H] or None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if cu_seqlens is not None:
        assert B == 1
        N = len(cu_seqlens) - 1
        dq_out = torch.zeros_like(q)
        dk_out = torch.zeros_like(k)
        dg_out = torch.zeros(B, T, H, dtype=q.dtype, device=q.device) if has_g else None
        chunk_offset = 0
        for i in range(N):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            NT_seg = (eos - bos) // C
            dq_seg, dk_seg, dg_seg = chunk_simple_gla_bwd_dqkg(
                q[:, bos:eos], k[:, bos:eos], v[:, bos:eos],
                h[:, chunk_offset:chunk_offset+NT_seg],
                gc[:, bos:eos], do[:, bos:eos],
                dh[:, chunk_offset:chunk_offset+NT_seg],
                dq[:, bos:eos], dk[:, bos:eos],
                scale, has_g, chunk_size=chunk_size)
            dq_out[:, bos:eos] = dq_seg
            dk_out[:, bos:eos] = dk_seg
            if dg_seg is not None:
                dg_out[:, bos:eos] = dg_seg
            chunk_offset += NT_seg
        return dq_out, dk_out, dg_out

    NT = T // C

    q_c = q.view(B, NT, C, H, K)
    k_c = k.view(B, NT, C, H, K)
    v_c = v.view(B, NT, C, H, V)
    gc_c = gc.view(B, NT, C, H)       # [B, NT, C, H]
    do_c = do.view(B, NT, C, H, V)
    dq_c = dq.view(B, NT, C, H, K)
    dk_c = dk.view(B, NT, C, H, K)

    gn = gc_c[:, :, -1, :]  # [B, NT, H]

    # Inter-chunk dq: scale * exp(gc) * (do @ h^T)
    # gc broadcast over K
    dq_inter = scale * gc_c.unsqueeze(-1).exp() * torch.einsum("bnchv,bnhkv->bnchk", do_c, h)

    # Inter-chunk dk: exp(gn - gc) * (v @ dh^T)
    dk_inter = (gn[:, :, None, :] - gc_c).unsqueeze(-1).exp() * torch.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh
    )

    dq_total = dq_c + dq_inter
    dk_total = dk_c + dk_inter

    # Gate gradient (only if g was provided)
    dg = None
    if has_g:
        # dgk_inter: chunk-level constant, shape [B, NT, H]
        # Path 4: h_n decay
        # Path 5: k_decay
        dgk_inter = (
            gn[:, :, :, None].exp() * torch.einsum("bnhkv,bnhkv->bnhk", h, dh)
        ).sum(-1) + (dk_inter * k_c).sum(dim=2).sum(dim=-1)
        # [B, NT, H]

        # dg_raw = sum_K(q * dq_total - k * dk_total), shape [B, NT, C, H]
        dg_raw = (q_c * dq_total - k_c * dk_total).sum(dim=-1)

        # reverse cumsum + inter constant
        dg = dg_raw.flip(2).cumsum(2).flip(2) + dgk_inter[:, :, None, :]

        dg = dg.reshape(B, T, H)

    return (
        dq_total.reshape(B, T, H, K),
        dk_total.reshape(B, T, H, K),
        dg,
    )


# =============================================================================
# Backward orchestrator
# =============================================================================


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Simple GLA backward orchestrator.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H] — raw per-head log-space gate, or None
        g_gamma: [H] — fixed per-head log-decay, or None
        scale: scaling factor
        initial_state: [B, H, K, V] or None
        do:   [B, T, H, V]
        dht:  [B, H, K, V] or None
        cu_seqlens: [N+1] or None
        chunk_size: block size

    Returns:
        (dq, dk, dv, dg, dh0)
        dg is [B, T, H] if g was provided, else None
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    has_g = g is not None

    # --- padding ---
    orig_seqlens = None
    padded_seqlens = None

    pad_tensors = [q, k, v, do]
    if g is not None:
        pad_tensors.append(g)

    if cu_seqlens is not None:
        assert B == 1
        pad_tensors, cu_seqlens, orig_seqlens, padded_seqlens = pad_varlen_seqs(
            pad_tensors, cu_seqlens, C
        )
        if g is not None:
            q, k, v, do, g = pad_tensors
        else:
            q, k, v, do = pad_tensors
    else:
        T_padded = ((T + C - 1) // C) * C
        if T_padded > T:
            pad = T_padded - T
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            do = F.pad(do, (0, 0, 0, 0, 0, pad))
            if g is not None:
                g = F.pad(g, (0, 0, 0, pad))

    T_padded = q.shape[1]

    # --- build gc & recompute h ---
    gc = build_gc(g, g_gamma, C, T_padded, cu_seqlens)
    if gc.shape[0] == 1 and B > 1:
        gc = gc.expand(B, -1, -1)

    h, _ = chunk_simple_gla_fwd_h(
        k, v, gc, h0=initial_state, output_final_state=False,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    # --- backward ---
    dh, dh0 = chunk_simple_gla_bwd_dh(
        q, gc, do, h0=initial_state, dht=dht, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    # Recompute A for dv
    A = chunk_simple_gla_fwd_intra(q, k, gc, scale, cu_seqlens=cu_seqlens, chunk_size=C)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=A.device))
    A = A.view(B, -1, C, H, C).masked_fill(~causal_mask[None, None, :, None, :], 0.0).reshape(B, T_padded, H, C)

    dv = chunk_simple_gla_bwd_dv(k, gc, A, do, dh, cu_seqlens=cu_seqlens, chunk_size=C)

    dA = chunk_simple_gla_bwd_dA(v, do, scale, cu_seqlens=cu_seqlens, chunk_size=C)
    dq, dk = chunk_simple_gla_bwd_dqk_intra(q, k, gc, dA, cu_seqlens=cu_seqlens, chunk_size=C)

    dq, dk, dg = chunk_simple_gla_bwd_dqkg(
        q, k, v, h, gc, do, dh, dq, dk, scale, has_g,
        cu_seqlens=cu_seqlens, chunk_size=C,
    )

    # --- unpadding ---
    if orig_seqlens is not None:
        dq = unpad_varlen_seqs(dq, orig_seqlens, padded_seqlens)
        dk = unpad_varlen_seqs(dk, orig_seqlens, padded_seqlens)
        dv = unpad_varlen_seqs(dv, orig_seqlens, padded_seqlens)
        if dg is not None:
            dg = unpad_varlen_seqs(dg, orig_seqlens, padded_seqlens)
    else:
        dq = dq[:, :T]
        dk = dk[:, :T]
        dv = dv[:, :T]
        if dg is not None:
            dg = dg[:, :T]

    return dq, dk, dv, dg, dh0