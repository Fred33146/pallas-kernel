"""JAX CPU reference for GLA chunk operations with FLA-triton-exact dtype behavior.

This implementation precisely matches the mixed-precision behavior of the
FLA Triton kernels, including:
- Keeping inputs in their original dtype (bf16/fp16/fp32), NO blanket upcast
- Selective upcasting to fp32 for accumulation and exp operations
- Explicit casting back to input dtype where Triton does so

fp64 mode: When inputs are fp64, all precision casts are skipped (they become
no-ops) and accumulation uses fp64 throughout. This provides a high-precision
reference that exceeds Triton's fp32 accumulation.

Dtype contract (matching FLA Triton for bf16/fp16/fp32; all fp64 for fp64):
  Forward:
    g_cumsum: fp32 (chunk_local_cumsum output)     [fp64 mode: fp64]
    h:        k.dtype if states_in_fp32=False, else fp32  [fp64 mode: fp64]
    A:        fp32 (intra-chunk attention matrix)   [fp64 mode: fp64]
    o:        v.dtype                               [fp64 mode: fp64]
    ht:       fp32 (final hidden state)             [fp64 mode: fp64]
  Backward:
    h (recomputed): fp32 (states_in_fp32=True)      [fp64 mode: fp64]
    dh:       fp32                                  [fp64 mode: fp64]
    dA:       fp32                                  [fp64 mode: fp64]
    dq, dk:   fp32 (from chunk_gla_bwd)             [fp64 mode: fp64]
    dv:       do.dtype                              [fp64 mode: fp64]
    dg:       fp32 (NOT cast to g.dtype — FLA keeps fp32)  [fp64 mode: fp64]
    dh0:      fp32                                  [fp64 mode: fp64]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from tops.cpu.ops import cpu_reference


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _pad_to_multiple(x: jnp.ndarray, multiple: int, axis: int) -> jnp.ndarray:
    length = x.shape[axis]
    remainder = length % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, pad_len)
    return jnp.pad(x, pad_widths)


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


# =============================================================================
# Sub-function 1: chunk_local_cumsum
# =============================================================================


def chunk_local_cumsum(
    g: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Chunk-local cumulative sum of gates.

    FLA Triton: internal cumsum in fp32, output_dtype defaults to torch.float.

    Args:
        g: [B, T, H, K] — log-space gates (T must be a multiple of chunk_size)
        chunk_size: block size

    Returns:
        g_cumsum: [B, T, H, K] — fp32
    """
    B, T, H, K = g.shape
    C = chunk_size
    assert T % C == 0, (
        f"T ({T}) must be a multiple of chunk_size ({C})"
    )
    NT = T // C
    acc = _acc_dtype(g.dtype)
    g_cast = g.astype(acc)
    g_cumsum = g_cast.reshape(B, NT, C, H, K).cumsum(axis=2).reshape(B, T, H, K)
    return g_cumsum


# =============================================================================
# Sub-function 2: chunk_fwd_h
# =============================================================================


def chunk_fwd_h(
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    h0: jnp.ndarray | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    states_in_fp32: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Inter-chunk hidden state propagation.

    FLA Triton dtype behavior:
    - Internal accumulator: always fp32
    - k_decay = (k * exp(gk_last - gk)).astype(k.dtype)
    - Matmul k_decay^T @ v: input dtype operands, fp32 accumulation
    - h_all: k.dtype if states_in_fp32=False, fp32 if True
    - ht: always fp32

    Args:
        k:  [B, T, H, K] — keys (input dtype)
        v:  [B, T, H, V] — values (input dtype)
        gk: [B, T, H, K] — chunk-local cumsum (fp32)
        h0: [B, H, K, V] — initial hidden state (fp32, optional)
        output_final_state: whether to return final state
        chunk_size: block size
        states_in_fp32: if True, store h in fp32; else in k.dtype

    Returns:
        h:  [B, NT, H, K, V]
        ht: [B, H, K, V] (fp32) or None
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(k.dtype)

    # fp64: store_dtype always fp64; else: fp32 if states_in_fp32, k.dtype otherwise
    store_dtype = acc if k.dtype == jnp.float64 else (jnp.float32 if states_in_fp32 else k.dtype)

    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gk_c = gk.reshape(B, NT, C, H, K)

    h_list = []
    h = jnp.zeros((B, H, K, V), dtype=acc)
    if h0 is not None:
        h = h + h0.astype(acc)

    for i in range(NT):
        h_list.append(h.astype(store_dtype))

        gc = gk_c[:, i]      # [B, C, H, K] acc dtype
        ki = k_c[:, i]       # [B, C, H, K] input dtype
        vi = v_c[:, i]       # [B, C, H, V] input dtype
        gk_last = gc[:, -1]  # [B, H, K] acc dtype

        # Decay existing state
        h = h * jnp.exp(gk_last[:, :, :, None])

        # k_decay: triton does (k * exp(...)).to(k.dtype)
        k_decay = (ki * jnp.exp(gk_last[:, None, :, :] - gc)).astype(k.dtype)

        # k_decay^T @ v: input dtype operands, acc dtype accumulation
        h = h + jnp.einsum(
            "bchk,bchv->bhkv", k_decay, vi,
            precision=lax.Precision.HIGHEST,
            preferred_element_type=acc,
        )

    h_all = jnp.stack(h_list, axis=1)  # [B, NT, H, K, V] in store_dtype
    ht = h if output_final_state else None  # acc dtype
    return h_all, ht


# =============================================================================
# Sub-function 3: chunk_gla_fwd_intra_gk
# =============================================================================


def chunk_gla_fwd_intra_gk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    g: jnp.ndarray,
    scale: float,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Intra-chunk attention matrix.

    FLA Triton: g_cumsum is fp32, so q*exp(g) and k*exp(-g) are fp32
    (natural type promotion). Matmul in fp32. A output: always fp32.

    Args:
        q: [B, T, H, K] — queries (input dtype)
        k: [B, T, H, K] — keys (input dtype)
        g: [B, T, H, K] — chunk-local cumsum (fp32)
        scale: scaling factor
        chunk_size: block size

    Returns:
        A: [B, T, H, C] — fp32
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(q.dtype)

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    g_c = g.reshape(B, NT, C, H, K)

    # Reference point for numerical stability
    g_n = g_c[:, :, 0:1, :, :]

    # input_dtype * fp32_exp → fp32 (natural promotion); fp64 stays fp64
    q_gated = q_c * jnp.exp(g_c - g_n)
    k_gated = k_c * jnp.exp(g_n - g_c)

    A = jnp.einsum(
        "bnihk,bnjhk->bnihj", q_gated, k_gated,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    ) * scale

    return A.reshape(B, T, H, C)


# =============================================================================
# Sub-function 4: chunk_gla_fwd_o_gk
# =============================================================================


def chunk_gla_fwd_o_gk(
    q: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    A: jnp.ndarray,
    h: jnp.ndarray,
    scale: float,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Combine inter-chunk and intra-chunk to produce output.

    FLA Triton dtype behavior:
    - qg = (q * exp(g_fp32)).to(q.dtype) — explicit cast back!
    - o_inter: einsum(qg_input_dtype, h_input_dtype) → fp32 accum
    - A cast to v.dtype for intra matmul
    - o stored in v.dtype

    Args:
        q: [B, T, H, K] — queries (input dtype)
        v: [B, T, H, V] — values (input dtype)
        g: [B, T, H, K] — chunk-local cumsum (fp32)
        A: [B, T, H, C] — attention matrix (fp32)
        h: [B, NT, H, K, V] — hidden states (k.dtype or fp32)
        scale: scaling factor
        chunk_size: block size

    Returns:
        o: [B, T, H, V] — v.dtype
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    acc = _acc_dtype(q.dtype)

    q_c = q.reshape(-1, C, H, K)
    v_c = v.reshape(-1, C, H, V)
    g_c = g.reshape(-1, C, H, K)
    h_c = h.reshape(-1, H, K, V)
    A_c = A.reshape(-1, C, H, C)

    # qg = (q * exp(g_fp32)).to(q.dtype) — triton explicit cast back
    qg = (q_c * jnp.exp(g_c)).astype(q.dtype)

    # Inter: qg(input_dtype) @ h.to(qg.dtype), acc dtype accumulation
    o_inter = scale * jnp.einsum(
        "nchk,nhkv->nchv", qg, h_c.astype(qg.dtype),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    # Intra: A.to(v.dtype) @ v, acc dtype accumulation
    A_v = A_c.astype(v.dtype)
    o_intra = jnp.einsum(
        "nihj,njhv->nihv", A_v, v_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    o = (o_inter + o_intra).astype(v.dtype)
    return o.reshape(B, T, H, V)


# =============================================================================
# Orchestrator: chunk_gla_fwd
# =============================================================================


def chunk_gla_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    g_cumsum: jnp.ndarray | None,
    scale: float,
    initial_state: jnp.ndarray | None,
    output_final_state: bool,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    """Chunk GLA forward orchestrator. No blanket dtype upcast.

    Returns:
        (g_cumsum, A, h, ht, o) — o unpadded to original T
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Padding
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
        q = _pad_to_multiple(q, C, axis=1)
        k = _pad_to_multiple(k, C, axis=1)
        v = _pad_to_multiple(v, C, axis=1)
        g = _pad_to_multiple(g, C, axis=1)

    NT = T_padded // C

    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, C)

    # Forward: states_in_fp32=False → h in k.dtype
    h, ht = chunk_fwd_h(
        k, v, gk=g_cumsum, h0=initial_state,
        output_final_state=output_final_state, chunk_size=C,
        states_in_fp32=False,
    )

    # A: fp32
    A = chunk_gla_fwd_intra_gk(q, k, g=g_cumsum, scale=scale, chunk_size=C)

    # Causal mask
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A_5d = A.reshape(B, -1, C, H, C)
    A = jnp.where(
        causal_mask[None, None, :, None, :], A_5d, 0.0
    ).reshape(B, T_padded, H, C)

    # Output: v.dtype
    o = chunk_gla_fwd_o_gk(q, v, g=g_cumsum, A=A, h=h, scale=scale, chunk_size=C)

    # Unpad
    o = o[:, :T]
    g_cumsum_out = g_cumsum[:, :T]
    A_out = A[:, :T]

    return g_cumsum_out, A_out, h, ht, o


# =============================================================================
# Backward sub-function 1: chunk_bwd_dh
# =============================================================================


def chunk_bwd_dh(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    gk: jnp.ndarray,
    do: jnp.ndarray,
    h0: jnp.ndarray | None = None,
    dht: jnp.ndarray | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Backward hidden state gradient propagation.

    FLA Triton: states_in_fp32=True → dh in fp32, dh0 in fp32.
    Triton casts q_hat back to input dtype before matmul:
      b_q = (b_q * scale).to(b_q.dtype)
      b_q = (b_q * exp(b_gk)).to(b_q.dtype)
      tl.dot(b_q, b_do.to(b_q.dtype))

    Args:
        q:   [B, T, H, K] — input dtype
        gk:  [B, T, H, K] — acc dtype (cumsum)
        do:  [B, T, H, V] — input dtype
        h0:  [B, H, K, V] — fp32 or None
        dht: [B, H, K, V] — fp32 or None
        scale: scaling factor
        chunk_size: block size

    Returns:
        dh:  [B, NT, H, K, V] — acc dtype (fp32 or fp64)
        dh0: [B, H, K, V] — acc dtype or None
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(q.dtype)

    q_c = q.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    gk_c = gk.reshape(B, NT, C, H, K)

    dh_list = [None] * NT
    dh = jnp.zeros((B, H, K, V), dtype=acc)
    if dht is not None:
        dh = dh + dht.astype(acc)

    for i in range(NT - 1, -1, -1):
        dh_list[i] = dh

        b_q = q_c[:, i]      # input dtype
        b_do = do_c[:, i]    # input dtype
        gc = gk_c[:, i]      # acc dtype
        gk_last = gc[:, -1]  # acc dtype

        # Triton: (q * scale).to(q.dtype); (q * exp(gc)).to(q.dtype)
        # Cast back to input dtype before matmul (for fp64, these are no-ops)
        b_q_hat = (b_q * jnp.exp(gc) * scale).astype(q.dtype)

        # Decay
        dh = dh * jnp.exp(gk_last[:, :, :, None])

        # Triton: tl.dot(b_q, b_do.to(b_q.dtype)) — both in input dtype
        dh = dh + jnp.einsum(
            "bchk,bchv->bhkv", b_q_hat, b_do.astype(q.dtype),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=acc,
        )

    dh_all = jnp.stack(dh_list, axis=1)  # [B, NT, H, K, V] acc dtype
    dh0 = dh if h0 is not None else None
    return dh_all, dh0


# =============================================================================
# Backward sub-function 2: chunk_gla_bwd_dA
# =============================================================================


def chunk_gla_bwd_dA(
    v: jnp.ndarray,
    do: jnp.ndarray,
    scale: float,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Gradient of intra-chunk attention matrix.

    FLA Triton: dA output in fp32. Matmul accumulates in fp32.

    Returns:
        dA: [B, T, H, C] — fp32, lower-triangular masked
    """
    B, T, H, V = v.shape
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(v.dtype)

    v_c = v.reshape(B, NT, C, H, V)
    do_c = do.reshape(B, NT, C, H, V)

    dA = jnp.einsum(
        "bnihv,bnjhv->bnihj", do_c, v_c,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    ) * scale

    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    dA = jnp.where(causal_mask[None, None, :, None, :], dA, 0.0)
    return dA.reshape(B, T, H, C)


# =============================================================================
# Backward sub-function 3: chunk_gla_bwd_dv
# =============================================================================


def chunk_gla_bwd_dv(
    k: jnp.ndarray,
    g_cumsum: jnp.ndarray,
    A: jnp.ndarray,
    do: jnp.ndarray,
    dh: jnp.ndarray,
    chunk_size: int = 64,
) -> jnp.ndarray:
    """Gradient of v.

    FLA Triton dtype:
    - Intra: dot(A_fp32, do.to(fp32)) — fp32, allow_tf32=False
    - Inter: k_decay.to(k.dtype) @ dh.to(k.dtype) — input dtype, fp32 accum
    - dv output: do.dtype

    Returns:
        dv: [B, T, H, V] — do.dtype
    """
    B, T, H, K = k.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(k.dtype)

    k_c = k.reshape(B, NT, C, H, K)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    A_c = A.reshape(B, NT, C, H, C)

    # Intra: A(acc) @ do.to(acc), allow_tf32=False → HIGHEST precision
    dv_intra = jnp.einsum(
        "bnihj,bnihv->bnjhv", A_c, do_c.astype(acc),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    # Inter: k_decay @ dh
    gn = gc_c[:, :, -1, :, :]
    k_decay = (k_c * jnp.exp(gn[:, :, None, :, :] - gc_c)).astype(k.dtype)

    dv_inter = jnp.einsum(
        "bnchk,bnhkv->bnchv", k_decay, dh.astype(k.dtype),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    dv = (dv_intra + dv_inter).astype(do.dtype)
    return dv.reshape(B, T, H, V)


# =============================================================================
# Backward sub-function 4: chunk_gla_bwd_dqk_intra
# =============================================================================


def chunk_gla_bwd_dqk_intra(
    q: jnp.ndarray,
    k: jnp.ndarray,
    g_cumsum: jnp.ndarray,
    dA: jnp.ndarray,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Intra-chunk dq, dk from dA.

    FLA Triton: k, gk explicitly cast to fp32. dq, dk output in fp32.

    Returns:
        dq, dk: [B, T, H, K] — fp32
    """
    B, T, H, K = q.shape
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(q.dtype)

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    dA_c = dA.reshape(B, NT, C, H, C)

    # k and gk explicitly cast to acc dtype in triton intra kernel
    k_neg = k_c.astype(acc) * jnp.exp(-gc_c)
    dq = jnp.exp(gc_c) * jnp.einsum(
        "bnihj,bnjhk->bnihk", dA_c, k_neg,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    q_pos = q_c.astype(acc) * jnp.exp(gc_c)
    dk = jnp.exp(-gc_c) * jnp.einsum(
        "bnihj,bnihk->bnjhk", dA_c, q_pos,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    return dq.reshape(B, T, H, K), dk.reshape(B, T, H, K)


# =============================================================================
# Backward sub-function 5: chunk_gla_bwd_dqkg
# =============================================================================


def chunk_gla_bwd_dqkg(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    h: jnp.ndarray,
    g_cumsum: jnp.ndarray,
    do: jnp.ndarray,
    dh: jnp.ndarray,
    dq: jnp.ndarray,
    dk: jnp.ndarray,
    scale: float,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inter-chunk dq, dk + gate gradient dg.

    FLA Triton dtype:
    - h: fp32 (states_in_fp32=True in bwd). Cast to do.dtype for matmul.
    - dh: fp32. Cast to v.dtype for matmul.
    - All intermediates in fp32.

    Returns:
        dq, dk, dg: [B, T, H, K] — fp32
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C
    acc = _acc_dtype(q.dtype)

    q_c = q.reshape(B, NT, C, H, K)
    k_c = k.reshape(B, NT, C, H, K)
    v_c = v.reshape(B, NT, C, H, V)
    gc_c = g_cumsum.reshape(B, NT, C, H, K)
    do_c = do.reshape(B, NT, C, H, V)
    dq_c = dq.reshape(B, NT, C, H, K)
    dk_c = dk.reshape(B, NT, C, H, K)

    gn = gc_c[:, :, -1, :, :]  # [B, NT, H, K] acc dtype

    # dq_inter: do(input) @ h.to(do.dtype), * scale * exp(gc)
    dq_inter = scale * jnp.exp(gc_c) * jnp.einsum(
        "bnchv,bnhkv->bnchk", do_c, h.astype(do.dtype),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    # dk_inter: v(input) @ dh.to(v.dtype), * exp(gn - gc)
    dk_inter = jnp.exp(gn[:, :, None, :, :] - gc_c) * jnp.einsum(
        "bnchv,bnhkv->bnchk", v_c, dh.astype(v.dtype),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=acc,
    )

    dq_total = dq_c + dq_inter
    dk_total = dk_c + dk_inter

    # Gate gradient
    dgk_inter = (
        jnp.exp(gn) * jnp.einsum(
            "bnhkv,bnhkv->bnhk", h.astype(acc), dh,
        )
        + jnp.sum(dk_inter * k_c.astype(acc), axis=2)
    )

    dg_raw = q_c.astype(acc) * dq_total - k_c.astype(acc) * dk_total
    # Reverse cumsum: flip → cumsum → flip
    dg = (
        jnp.cumsum(dg_raw[:, :, ::-1], axis=2)[:, :, ::-1]
        + dgk_inter[:, :, None, :, :]
    )

    return (
        dq_total.reshape(B, T, H, K),
        dk_total.reshape(B, T, H, K),
        dg.reshape(B, T, H, K),
    )


# =============================================================================
# Backward orchestrator: chunk_gla_bwd
# =============================================================================


def chunk_gla_bwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    g_cumsum: jnp.ndarray | None,
    scale: float,
    initial_state: jnp.ndarray | None,
    h: jnp.ndarray | None,
    A: jnp.ndarray | None,
    do: jnp.ndarray,
    dht: jnp.ndarray | None,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Chunk GLA backward orchestrator.

    FLA Triton:
    - Recomputes h with states_in_fp32=True (fp32)
    - dh: fp32
    - Returns: dq(fp32), dk(fp32), dv(do.dtype), dg(fp32), dh0(fp32)

    Returns:
        (dq, dk, dv, dg, dh0)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    # Padding
    T_padded = _cdiv(T, C) * C
    if T_padded > T:
        q = _pad_to_multiple(q, C, axis=1)
        k = _pad_to_multiple(k, C, axis=1)
        v = _pad_to_multiple(v, C, axis=1)
        g = _pad_to_multiple(g, C, axis=1)
        do = _pad_to_multiple(do, C, axis=1)

    # Recompute g_cumsum
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, C)

    # Recompute h with states_in_fp32=True (FLA backward behavior)
    if h is None:
        h, _ = chunk_fwd_h(
            k, v, g_cumsum, h0=initial_state,
            output_final_state=False, chunk_size=C,
            states_in_fp32=True,
        )

    # dh: fp32
    dh, dh0 = chunk_bwd_dh(
        q, k, v, gk=g_cumsum, do=do, h0=initial_state,
        dht=dht, scale=scale, chunk_size=C,
    )

    # Recompute A if not provided
    if A is None:
        A = chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
        causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
        A = jnp.where(
            causal_mask[None, None, :, None, :],
            A.reshape(B, -1, C, H, C), 0.0
        ).reshape(B, T_padded, H, C)
    else:
        A = A.reshape(B, T_padded, H, C)

    # dv: do.dtype
    dv = chunk_gla_bwd_dv(k, g_cumsum, A, do, dh, chunk_size=C)

    # dA: fp32
    dA = chunk_gla_bwd_dA(v, do, scale, chunk_size=C)

    # dq, dk intra: fp32
    dq, dk = chunk_gla_bwd_dqk_intra(q, k, g_cumsum, dA, chunk_size=C)

    # dq, dk inter + dg: fp32
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q, k, v, h, g_cumsum, do, dh, dq, dk, scale, chunk_size=C,
    )

    # Unpad
    dq = dq[:, :T]
    dk = dk[:, :T]
    dv = dv[:, :T]
    dg = dg[:, :T]

    # Match FLA output dtypes: dq(fp32), dk(fp32), dv(do.dtype), dg(fp32), dh0(fp32)
    # FLA keeps dg in fp32 (g_cumsum dtype), does NOT cast to g.dtype

    return dq, dk, dv, dg, dh0


# =============================================================================
# Public API: chunk_gla
# =============================================================================


@cpu_reference
def chunk_gla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray | None = None,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Chunk GLA with FLA-triton-exact dtype behavior.

    No blanket upcast — inputs stay in their original dtype.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H, K] — log-space gates (any dtype); if None, defaults to zeros (no decay)
        scale: scaling factor (default: K^-0.5)
        initial_state: [B, H, K, V] (fp32)
        output_final_state: whether to return final state
        chunk_size: block size

    Returns:
        (o, final_state) — o in v.dtype, final_state in fp32 or None
    """
    B, T, H, K = q.shape
    if g is None:
        g = jnp.zeros_like(q)
    if scale is None:
        scale = K ** -0.5
    if initial_state is not None:
        initial_state = initial_state.astype(_acc_dtype(q.dtype))

    _, _, _, ht, o = chunk_gla_fwd(
        q, k, v, g, None, scale, initial_state,
        output_final_state, chunk_size,
    )
    return o, ht if output_final_state else None
