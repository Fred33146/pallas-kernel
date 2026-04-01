"""JAX CPU reference for Multi-head Latent Attention (MLA) from DeepSeek-V2.

Precisely matches the FLA layer implementation (fla/layers/mla.py) dtype behavior:

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  RMSNorm:
    Internal computation: fp32                        [fp64 mode: fp64]
    Output: cast back to input dtype                  [fp64 mode: fp64]
  RoPE:
    cos/sin: same dtype as input                      [fp64 mode: fp64]
    Output: same dtype as input                       [fp64 mode: fp64]
  Linear projections:
    Output: same dtype as input                       [fp64 mode: fp64]
  Attention:
    Scores (q @ k^T): fp32 for numerical stability   [fp64 mode: fp64]
    softmax: fp32                                     [fp64 mode: fp64]
    Output (attn @ v): cast back to v.dtype           [fp64 mode: fp64]
  Final output:
    o: same dtype as hidden input                     [fp64 mode: fp64]

Reference:
  - FLA: fla/layers/mla.py (MultiheadLatentAttention)
  - Paper: DeepSeek-V2 (arXiv 2405.04434)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tops.cpu.ops import cpu_reference


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


# =============================================================================
# Utility: yarn_get_mscale
# =============================================================================


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Compute YaRN attention scaling factor.

    Matches FLA's ``yarn_get_mscale`` used when ``rope_scaling`` is configured.

    Formula:
        mscale_out = 0.1 * mscale * log(scale) + 1.0   if scale > 1
        mscale_out = 1.0                                 otherwise

    Args:
        scale:  float — RoPE scaling factor (from rope_scaling["factor"])
        mscale: float — mscale coefficient (from rope_scaling["mscale_all_dim"])

    Returns:
        float — Multiplicative adjustment for attention scaling
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# =============================================================================
# Sub-function 1: rms_norm
# =============================================================================


def rms_norm(
    x: jax.Array,
    weight: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Functional RMSNorm — fp32 internal computation, cast back to input dtype.

    Matches FLA's RMSNorm(dtype=torch.float32) used in MLA projection paths.

    Formula: x * rsqrt(mean(x^2) + eps) * weight

    Args:
        x:      [..., D] — Input tensor, any dtype
        weight: [D]      — Learnable scale parameter
        eps:    float     — Numerical stability constant

    Returns:
        out:    [..., D] — Same dtype as input x

    Dtype behavior:
        - x cast to fp32 (or fp64) for computation
        - weight applied in fp32
        - Result cast back to original input dtype
    """
    assert x.shape[-1] == weight.shape[0], (
        f"x last dim {x.shape[-1]} != weight dim {weight.shape[0]}"
    )
    orig_dtype = x.dtype
    acc_dt = _acc_dtype(orig_dtype)
    x_f = x.astype(acc_dt)
    rms = jnp.sqrt(jnp.mean(x_f ** 2, axis=-1, keepdims=True) + eps)
    x_normed = x_f / rms
    out = x_normed * weight.astype(acc_dt)
    return out.astype(orig_dtype)


# =============================================================================
# Sub-function 2: precompute_freqs_cis
# =============================================================================


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE cos/sin frequency tables.

    Matches FLA's RotaryEmbedding._update_cos_sin_cache() — position indices
    and inverse frequencies computed in fp32 for precision.

    Args:
        dim:         int   — Rotary embedding dimension (qk_rope_head_dim)
        max_seq_len: int   — Maximum sequence length
        theta:       float — RoPE base frequency (default 10000.0)

    Returns:
        cos: [max_seq_len, dim // 2] — Cosine frequencies in fp32
        sin: [max_seq_len, dim // 2] — Sine frequencies in fp32
    """
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    # FLA: inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    # FLA: freqs = outer(t, inv_freq)
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [max_seq_len, dim // 2]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin


# =============================================================================
# Sub-function 3: apply_rotary_emb
# =============================================================================


def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
) -> jax.Array:
    """Apply rotary position embedding (non-interleaved / GPT-NeoX style).

    Matches FLA's rotary_embedding_ref() with interleaved=False:
        x0, x1 = x[..., :D//2], x[..., D//2:]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

    Args:
        x:   [..., T, D] — Input tensor. Last dim is the rotary dimension.
        cos: [..., T, D // 2] — Cosine frequencies (from precompute_freqs_cis)
        sin: [..., T, D // 2] — Sine frequencies (from precompute_freqs_cis)

    Returns:
        out: [..., T, D] — Same shape and dtype as input x

    Dtype behavior:
        - cos/sin should be cast to x.dtype before application
        - Output preserves x.dtype
    """
    D = x.shape[-1]
    assert D % 2 == 0, f"RoPE dimension must be even, got {D}"
    R = D // 2
    assert cos.shape[-1] == R, f"cos last dim {cos.shape[-1]} != D//2={R}"
    assert sin.shape[-1] == R, f"sin last dim {sin.shape[-1]} != D//2={R}"

    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)

    # Broadcast cos/sin to match x shape: [..., T, R]
    # cos/sin are [T, R], need to broadcast for batch/head dims
    x0 = x[..., :R]
    x1 = x[..., R:]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return jnp.concatenate([out0, out1], axis=-1)


# =============================================================================
# Helper: _get_rope_cos_sin
# =============================================================================


def _get_rope_cos_sin(
    cos: jax.Array,
    sin: jax.Array,
    T: int,
    seqlen_offset: int | jax.Array = 0,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Extract position-aware RoPE cos/sin for the current sequence.

    Handles three scenarios matching FLA's RotaryEmbedding:
      1. Default (no offset, no varlen): cos[:T] — simple slice.
      2. Scalar offset (KV cache): cos[offset:offset+T] via lax.dynamic_slice.
      3. cu_seqlens (varlen): local position per sequence, cos[local_pos].
      4. Per-batch tensor offset (mask + cache): position_ids = offset[:, None] + arange(T).

    All paths are jax.jit friendly (static shapes, dynamic indices).

    Args:
        cos:           [max_seq_len, R] — Precomputed cosine table (fp32)
        sin:           [max_seq_len, R] — Precomputed sine table (fp32)
        T:             int — Current sequence length
        seqlen_offset: int or [B] — Position offset for KV cache continuation
        cu_seqlens:    [N+1] or None — Cumulative sequence lengths for varlen

    Returns:
        cos_out: [T, 1, R] or [B, T, 1, R] — Broadcast-ready cos
        sin_out: [T, 1, R] or [B, T, 1, R] — Broadcast-ready sin
    """
    R = cos.shape[-1]

    if cu_seqlens is not None:
        # Varlen: compute local positions within each packed sequence
        # cu_seqlens = [0, 3, 7, 10] → seq_ids = [0,0,0, 1,1,1,1, 2,2,2]
        seq_ids = jnp.sum(jnp.arange(T)[None, :] >= cu_seqlens[1:, None], axis=0)  # [T]
        seq_starts = cu_seqlens[seq_ids]  # [T] — start of each token's sequence
        local_pos = jnp.arange(T) - seq_starts  # [T] — position within sequence
        cos_out = cos[local_pos]  # [T, R]
        sin_out = sin[local_pos]  # [T, R]
        return cos_out.reshape(T, 1, R), sin_out.reshape(T, 1, R)

    if isinstance(seqlen_offset, (int, float)) and seqlen_offset == 0:
        # Default: simple slice
        cos_out = cos[:T]
        sin_out = sin[:T]
        return cos_out.reshape(T, 1, R), sin_out.reshape(T, 1, R)

    if isinstance(seqlen_offset, (int, float)):
        # Scalar offset (KV cache): jit-safe dynamic slice
        cos_out = jax.lax.dynamic_slice(cos, (seqlen_offset, 0), (T, R))
        sin_out = jax.lax.dynamic_slice(sin, (seqlen_offset, 0), (T, R))
        return cos_out.reshape(T, 1, R), sin_out.reshape(T, 1, R)

    # Per-batch tensor offset: seqlen_offset is [B]
    # position_ids = offset[:, None] + arange(T) → [B, T]
    position_ids = seqlen_offset[:, None] + jnp.arange(T)[None, :]  # [B, T]
    cos_out = cos[position_ids]  # [B, T, R]
    sin_out = sin[position_ids]  # [B, T, R]
    return cos_out.reshape(-1, T, 1, R), sin_out.reshape(-1, T, 1, R)


# =============================================================================
# Sub-function 4: mla_project_q
# =============================================================================


def mla_project_q(
    hidden: jax.Array,
    w_dq: jax.Array | None,
    w_uq: jax.Array | None,
    q_norm_weight: jax.Array | None,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    cos: jax.Array,
    sin: jax.Array,
    eps: float = 1e-5,
    *,
    w_q: jax.Array | None = None,
    seqlen_offset: int | jax.Array = 0,
    cu_seqlens: jax.Array | None = None,
) -> jax.Array:
    """MLA Query projection with optional LoRA compression.

    Two paths matching FLA's q_proj:
      - LoRA (w_dq is not None):
            Linear(D, qlr) -> RMSNorm(fp32) -> Linear(qlr, H*qk_head_dim)
      - Direct (w_dq is None, w_q provided):
            Linear(D, H*qk_head_dim)

    Then: reshape -> split nope/rope -> apply RoPE -> concat.

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dq:             [q_lora_rank, D_model] or None — Down-projection (LoRA)
        w_uq:             [H * qk_head_dim, q_lora_rank] or None — Up-projection (LoRA)
        q_norm_weight:    [q_lora_rank] or None — RMSNorm weight (LoRA)
        num_heads:        int — Number of attention heads
        qk_nope_head_dim: int — Non-rotary QK head dimension
        qk_rope_head_dim: int — Rotary QK head dimension
        cos:              [max_seq_len, qk_rope_head_dim // 2] — RoPE cosine
        sin:              [max_seq_len, qk_rope_head_dim // 2] — RoPE sine
        eps:              float — RMSNorm epsilon
        w_q:              [H * qk_head_dim, D_model] or None — Direct projection
        seqlen_offset:    int or [B] — RoPE position offset (KV cache)
        cu_seqlens:       [N+1] or None — Varlen cumulative sequence lengths

    Returns:
        q: [B, T, num_heads, qk_head_dim] — Query tensor with RoPE applied
    """
    B, T, D = hidden.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if w_dq is not None:
        # LoRA path: down → RMSNorm → up
        q_lora_rank = w_dq.shape[0]
        assert w_dq.shape == (q_lora_rank, D), (
            f"w_dq shape {w_dq.shape} != ({q_lora_rank}, {D})"
        )
        assert w_uq is not None and w_uq.shape == (num_heads * qk_head_dim, q_lora_rank), (
            f"w_uq shape {getattr(w_uq, 'shape', None)} != ({num_heads * qk_head_dim}, {q_lora_rank})"
        )
        assert q_norm_weight is not None and q_norm_weight.shape == (q_lora_rank,), (
            f"q_norm_weight shape {getattr(q_norm_weight, 'shape', None)} != ({q_lora_rank},)"
        )
        # FLA: c_q = Linear(hidden_size, q_lora_rank)(hidden)
        c_q = hidden @ w_dq.T  # [B, T, q_lora_rank]
        # FLA: RMSNorm(q_lora_rank, dtype=torch.float32)
        c_q = rms_norm(c_q, q_norm_weight, eps)
        # FLA: q = Linear(q_lora_rank, num_heads * qk_head_dim)(c_q)
        q = c_q @ w_uq.T  # [B, T, num_heads * qk_head_dim]
    else:
        # Direct projection path (q_lora_rank is None in FLA)
        assert w_q is not None, "Either w_dq (LoRA) or w_q (direct) must be provided"
        assert w_q.shape == (num_heads * qk_head_dim, D), (
            f"w_q shape {w_q.shape} != ({num_heads * qk_head_dim}, {D})"
        )
        # FLA: q_proj = Linear(hidden_size, num_heads * qk_head_dim)
        q = hidden @ w_q.T  # [B, T, num_heads * qk_head_dim]

    # FLA: q = rearrange(q, '... (h d) -> ... h d', d=qk_head_dim)
    q = q.reshape(B, T, num_heads, qk_head_dim)
    # FLA: q_pass, q_rot = split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope = q[..., :qk_nope_head_dim]
    q_rope = q[..., qk_nope_head_dim:]

    # FLA: q_rot, _ = self.rotary(q_rot, ..., seqlen_offset=..., cu_seqlens=...)
    cos_b, sin_b = _get_rope_cos_sin(cos, sin, T, seqlen_offset, cu_seqlens)
    q_rope = apply_rotary_emb(q_rope, cos_b, sin_b)

    # FLA: q = torch.cat((q_pass, q_rot), dim=-1)
    q = jnp.concatenate([q_nope, q_rope], axis=-1)
    return q


# =============================================================================
# Sub-function 5: mla_project_kv
# =============================================================================


def mla_project_kv(
    hidden: jax.Array,
    w_dkv: jax.Array,
    w_ukv: jax.Array,
    kv_norm_weight: jax.Array,
    w_kr: jax.Array,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    qk_rope_head_dim: int,
    cos: jax.Array,
    sin: jax.Array,
    eps: float = 1e-5,
    *,
    seqlen_offset: int | jax.Array = 0,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """MLA KV projection: LoRA compress -> RMSNorm(fp32) -> expand/split + separate RoPE key.

    Matches FLA's kv_proj = nn.Sequential(
        Linear(hidden_size, kv_lora_rank),
        RMSNorm(kv_lora_rank, dtype=float32),
        Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)),
    ) and k_rope = Linear(hidden_size, qk_rope_head_dim).

    Computation flow:
        c_kv = hidden @ w_dkv.T                             # [B, T, kv_lora_rank]
        c_kv = rms_norm(c_kv, kv_norm_weight, eps)           # fp32 normalize
        kv = c_kv @ w_ukv.T                                 # [B, T, H*(nope+v)]
        kv = reshape(kv, [B, T, H, nope+v])
        k_nope, v = split(kv, [nope, v])

        k_rope = hidden @ w_kr.T                             # [B, T, rope_dim]
        k_rope = reshape(k_rope, [B, T, 1, rope_dim])
        k_rope = apply_rotary_emb(k_rope, cos, sin)
        k_rope = broadcast(k_rope, num_heads)                # shared across heads

        k = concat(k_nope, k_rope)                           # [B, T, H, qk_head_dim]

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dkv:            [kv_lora_rank, D_model] — KV down-projection weight
        w_ukv:            [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank] — KV up-projection
        kv_norm_weight:   [kv_lora_rank] — KV RMSNorm weight
        w_kr:             [qk_rope_head_dim, D_model] — Key-rope projection weight
        num_heads:        int
        qk_nope_head_dim: int
        v_head_dim:       int
        qk_rope_head_dim: int
        cos:              [max_seq_len, qk_rope_head_dim // 2] — RoPE cosine
        sin:              [max_seq_len, qk_rope_head_dim // 2] — RoPE sine
        eps:              float
        seqlen_offset:    int or [B] — RoPE position offset (KV cache)
        cu_seqlens:       [N+1] or None — Varlen cumulative sequence lengths

    Returns:
        k: [B, T, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        v: [B, T, num_heads, v_head_dim]
    """
    B, T, D = hidden.shape
    kv_lora_rank = w_dkv.shape[0]
    kv_dim_per_head = qk_nope_head_dim + v_head_dim

    assert w_dkv.shape == (kv_lora_rank, D), (
        f"w_dkv shape {w_dkv.shape} != ({kv_lora_rank}, {D})"
    )
    assert w_ukv.shape == (num_heads * kv_dim_per_head, kv_lora_rank), (
        f"w_ukv shape {w_ukv.shape} != ({num_heads * kv_dim_per_head}, {kv_lora_rank})"
    )
    assert kv_norm_weight.shape == (kv_lora_rank,), (
        f"kv_norm_weight shape {kv_norm_weight.shape} != ({kv_lora_rank},)"
    )
    assert w_kr.shape == (qk_rope_head_dim, D), (
        f"w_kr shape {w_kr.shape} != ({qk_rope_head_dim}, {D})"
    )

    # FLA: c_kv = Linear(hidden_size, kv_lora_rank)(hidden)
    c_kv = hidden @ w_dkv.T  # [B, T, kv_lora_rank]
    # FLA: RMSNorm(kv_lora_rank, dtype=torch.float32)
    c_kv = rms_norm(c_kv, kv_norm_weight, eps)
    # FLA: kv = Linear(kv_lora_rank, num_heads * kv_dim_per_head)(c_kv)
    kv = c_kv @ w_ukv.T  # [B, T, num_heads * kv_dim_per_head]

    # FLA: k_pass = rearrange(kv, '... (h d) -> ... h d', d=kv_dim_per_head)
    kv = kv.reshape(B, T, num_heads, kv_dim_per_head)
    # FLA: k_pass, v = split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
    k_nope = kv[..., :qk_nope_head_dim]
    v = kv[..., qk_nope_head_dim:]

    # k_rope: separate single-head projection from hidden (not from compressed c_kv)
    # because RoPE is position-dependent and cannot survive LoRA compression
    k_rope = hidden @ w_kr.T  # [B, T, qk_rope_head_dim]
    k_rope = k_rope.reshape(B, T, 1, qk_rope_head_dim)
    cos_b, sin_b = _get_rope_cos_sin(cos, sin, T, seqlen_offset, cu_seqlens)
    k_rope = apply_rotary_emb(k_rope, cos_b, sin_b)
    k_rope = jnp.broadcast_to(k_rope, (B, T, num_heads, qk_rope_head_dim))
    k = jnp.concatenate([k_nope, k_rope], axis=-1)
    return k, v


# =============================================================================
# Sub-function 6: causal_softmax_attention
# =============================================================================


def causal_softmax_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float | None = None,
    *,
    window_size: int | None = None,
    attention_mask: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
) -> jax.Array:
    """Causal softmax attention — reference implementation with full feature set.

    Supports:
      - Asymmetric T_q != T_k (KV cache decode)
      - Sliding window attention (window_size)
      - Padding mask (attention_mask)
      - Variable-length packed sequences (cu_seqlens)

    Matches FLA's three attention paths:
      1. attention_mask → mask-based causal attention
      2. cu_seqlens → block-diagonal causal attention (B=1)
      3. Neither → standard full causal attention

    attention_mask and cu_seqlens are mutually exclusive.

    Args:
        q:              [B, T_q, H, D_qk] — Query tensor
        k:              [B, T_k, H, D_qk] — Key tensor
        v:              [B, T_k, H, D_v]  — Value tensor
        scale:          float — Scaling factor, default = D_qk ** -0.5
        window_size:    int or None — Sliding window size (FLA: window_size-1 left context)
        attention_mask: [B, T_k] or None — 0/1 padding mask (1=valid, 0=pad)
        cu_seqlens:     [N+1] or None — Cumulative sequence lengths (B must be 1)

    Returns:
        o: [B, T_q, H, D_v] — Output tensor, same dtype as v

    Dtype behavior:
        - Attention scores computed in fp32 (softmax numerical stability)
        - Output cast back to v.dtype
        - fp64 mode: all computation in fp64
    """
    assert q.ndim == 4, f"q must be 4D [B,T_q,H,D], got {q.ndim}D"
    assert k.ndim == 4, f"k must be 4D [B,T_k,H,D], got {k.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T_k,H,D], got {v.ndim}D"
    assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size mismatch"
    assert q.shape[2] == k.shape[2] == v.shape[2], "Head count mismatch"
    assert q.shape[-1] == k.shape[-1], (
        f"q head dim {q.shape[-1]} != k head dim {k.shape[-1]}"
    )
    assert k.shape[1] == v.shape[1], (
        f"k seq len {k.shape[1]} != v seq len {v.shape[1]}"
    )
    assert not (attention_mask is not None and cu_seqlens is not None), (
        "attention_mask and cu_seqlens are mutually exclusive"
    )
    if cu_seqlens is not None:
        assert q.shape[0] == 1, (
            f"cu_seqlens requires B=1, got B={q.shape[0]}"
        )

    B, T_q, H, D_qk = q.shape
    T_k = k.shape[1]
    D_v = v.shape[-1]
    orig_v_dtype = v.dtype
    acc_dt = _acc_dtype(q.dtype)

    if scale is None:
        scale = D_qk ** -0.5

    q_f = q.astype(acc_dt)
    k_f = k.astype(acc_dt)
    v_f = v.astype(acc_dt)

    # Transpose to [B, H, T, D]
    q_f = jnp.transpose(q_f, (0, 2, 1, 3))  # [B, H, T_q, D_qk]
    k_f = jnp.transpose(k_f, (0, 2, 1, 3))  # [B, H, T_k, D_qk]
    v_f = jnp.transpose(v_f, (0, 2, 1, 3))  # [B, H, T_k, D_v]

    # Attention scores: [B, H, T_q, T_k]
    attn = jnp.matmul(q_f, jnp.transpose(k_f, (0, 1, 3, 2))) * scale

    # --- Build mask ---
    if cu_seqlens is not None:
        # Block-diagonal causal mask for varlen packed sequences
        # Compute sequence IDs for each position
        pos_k = jnp.arange(T_k)
        seq_ids = jnp.sum(pos_k[None, :] >= cu_seqlens[1:, None], axis=0)  # [T_k]
        seq_starts = cu_seqlens[seq_ids]  # [T_k]
        local_pos = pos_k - seq_starts  # [T_k]

        # Same sequence AND causal within sequence
        same_seq = seq_ids[None, :] == seq_ids[:, None]  # [T_k, T_k]
        local_causal = local_pos[None, :] <= local_pos[:, None]  # [T_k, T_k]
        mask = same_seq & local_causal  # [T_k, T_k]

        # Window size within sequence
        if window_size is not None:
            in_window = (local_pos[:, None] - local_pos[None, :]) < window_size
            mask = mask & in_window

        # Apply mask: [T_q, T_k] → broadcast to [B, H, T_q, T_k]
        attn = jnp.where(mask[None, None, :, :], attn, float('-inf'))
    else:
        # Standard causal mask (supports T_q != T_k for KV cache)
        q_pos = jnp.arange(T_q)
        k_pos = jnp.arange(T_k)
        offset = T_k - T_q  # offset for cache (past tokens)

        causal = (q_pos[:, None] + offset) >= k_pos[None, :]  # [T_q, T_k]

        if window_size is not None:
            in_window = (q_pos[:, None] + offset - k_pos[None, :]) < window_size
            causal = causal & in_window

        # Start with causal mask
        mask = causal  # [T_q, T_k]

        if attention_mask is not None:
            # attention_mask: [B, T_k], 1=valid, 0=pad
            pad_mask = attention_mask[:, None, None, :] > 0  # [B, 1, 1, T_k]
            attn = jnp.where(mask[None, None, :, :] & pad_mask, attn, float('-inf'))
        else:
            attn = jnp.where(mask[None, None, :, :], attn, float('-inf'))

    # Softmax in accumulator dtype (numerically stable)
    attn = attn - jnp.max(attn, axis=-1, keepdims=True)
    attn = jnp.exp(attn)
    attn = attn / jnp.sum(attn, axis=-1, keepdims=True)

    # Apply attention: [B, H, T_q, D_v]
    o = jnp.matmul(attn, v_f)

    # Transpose back: [B, H, T_q, D_v] -> [B, T_q, H, D_v]
    o = jnp.transpose(o, (0, 2, 1, 3))

    return o.astype(orig_v_dtype)


# =============================================================================
# Sub-function 7: mla_forward
# =============================================================================


@cpu_reference
def mla_forward(
    hidden: jax.Array,
    w_dq: jax.Array | None,
    w_uq: jax.Array | None,
    q_norm_weight: jax.Array | None,
    w_dkv: jax.Array,
    w_ukv: jax.Array,
    kv_norm_weight: jax.Array,
    w_kr: jax.Array,
    w_o: jax.Array,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    cos: jax.Array,
    sin: jax.Array,
    eps: float = 1e-5,
    *,
    w_q: jax.Array | None = None,
    rope_scaling: dict | None = None,
    past_k: jax.Array | None = None,
    past_v: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    window_size: int | None = None,
    seqlen_offset: int | jax.Array = 0,
    apply_mscale: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """Full MLA forward pass: project Q/K/V -> causal attention -> output projection.

    Orchestrates all MLA sub-functions matching FLA's MultiheadLatentAttention.forward().

    Computation flow (matching FLA):
        1. q = mla_project_q(hidden, ...)             [B, T, H, qk_head_dim]
        2. k, v = mla_project_kv(hidden, ...)          [B, T, H, qk_head_dim], [B, T, H, v_head_dim]
        3. (optional) k = concat(past_k, k); v = concat(past_v, v)
        4. o = causal_softmax_attention(q, k, v, ...)  [B, T, H, v_head_dim]
        5. output = reshape(o) @ w_o.T                 [B, T, D_model]

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dq:             [q_lora_rank, D_model] or None — Query down-projection (LoRA)
        w_uq:             [H * qk_head_dim, q_lora_rank] or None — Query up-projection (LoRA)
        q_norm_weight:    [q_lora_rank] or None — Query RMSNorm weight (LoRA)
        w_dkv:            [kv_lora_rank, D_model] — KV down-projection
        w_ukv:            [H * (qk_nope_head_dim + v_head_dim), kv_lora_rank] — KV up-projection
        kv_norm_weight:   [kv_lora_rank] — KV RMSNorm weight
        w_kr:             [qk_rope_head_dim, D_model] — Key-rope projection
        w_o:              [D_model, H * v_head_dim] — Output projection
        num_heads:        int
        qk_nope_head_dim: int — Non-rotary QK head dimension
        qk_rope_head_dim: int — Rotary QK head dimension
        v_head_dim:       int — Value head dimension
        cos:              [max_seq_len, qk_rope_head_dim // 2]
        sin:              [max_seq_len, qk_rope_head_dim // 2]
        eps:              float — RMSNorm epsilon (FLA default: 1e-5)
        w_q:              [H * qk_head_dim, D_model] or None — Direct Q projection (no LoRA)
        rope_scaling:     dict or None — RoPE scaling config with "factor", "mscale_all_dim", "rope_type"
        past_k:           [B, T_past, H, D_qk] or None — Cached keys for incremental decoding
        past_v:           [B, T_past, H, D_v] or None — Cached values for incremental decoding
        attention_mask:   [B, T_k] or None — Padding mask (1=valid, 0=pad)
        cu_seqlens:       [N+1] or None — Varlen cumulative sequence lengths
        window_size:      int or None — Sliding window attention size
        seqlen_offset:    int or [B] — RoPE position offset for KV cache
        apply_mscale:     bool — If True, apply YaRN mscale to attention scaling
                          (DeepSeek paper behavior). Default False matches FLA runtime
                          where mscale is computed but not passed to flash_attn.

    Returns:
        If past_k/past_v are provided:
            (output, k, v) — output: [B, T, D_model], k/v: updated cache tensors
        Otherwise:
            output: [B, T, D_model] — Same dtype as input hidden
    """
    B, T, D = hidden.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    use_cache = past_k is not None

    assert w_o.shape == (D, num_heads * v_head_dim), (
        f"w_o shape {w_o.shape} != ({D}, {num_heads * v_head_dim})"
    )

    # Auto-derive seqlen_offset from cache length when not explicitly set
    # FLA: seqlen_offset = past_key_values.get_seq_length(layer_idx)
    if use_cache and isinstance(seqlen_offset, (int, float)) and seqlen_offset == 0:
        seqlen_offset = past_k.shape[1]

    # FLA: when attention_mask is present with cache, adjust offset per batch
    # seqlen_offset += prepare_lens_from_mask(mask) - mask.shape[-1]
    if attention_mask is not None and use_cache:
        actual_lens = jnp.sum(attention_mask, axis=-1, dtype=jnp.int32)  # [B]
        padded_len = attention_mask.shape[-1]
        seqlen_offset = seqlen_offset + actual_lens - padded_len  # [B]

    # Compute attention scaling
    # FLA: self.scaling = self.qk_head_dim ** (-0.5)
    # NOTE: FLA computes mscale but does NOT pass it to flash_attn (dead code).
    # Default (apply_mscale=False) matches FLA runtime; True matches DeepSeek paper.
    scale = qk_head_dim ** -0.5
    if apply_mscale and rope_scaling is not None and rope_scaling.get("rope_type", "default") != "default":
        mscale_val = yarn_get_mscale(
            rope_scaling["factor"],
            rope_scaling.get("mscale_all_dim", 0),
        )
        scale = scale * mscale_val * mscale_val

    # 1. Project queries
    q = mla_project_q(
        hidden, w_dq, w_uq, q_norm_weight,
        num_heads, qk_nope_head_dim, qk_rope_head_dim,
        cos, sin, eps,
        w_q=w_q,
        seqlen_offset=seqlen_offset,
        cu_seqlens=cu_seqlens,
    )  # [B, T, H, qk_head_dim]

    # 2. Project keys and values
    k, v = mla_project_kv(
        hidden, w_dkv, w_ukv, kv_norm_weight, w_kr,
        num_heads, qk_nope_head_dim, v_head_dim, qk_rope_head_dim,
        cos, sin, eps,
        seqlen_offset=seqlen_offset,
        cu_seqlens=cu_seqlens,
    )  # k: [B, T, H, qk_head_dim], v: [B, T, H, v_head_dim]

    # 3. KV cache: concatenate past with current
    # FLA: k_cached, v_cached = past_key_values.update(attn_state=(k, v), ...)
    if use_cache:
        assert past_v is not None, "past_k and past_v must be provided together"
        k = jnp.concatenate([past_k, k], axis=1)  # [B, T_past+T, H, D_qk]
        v = jnp.concatenate([past_v, v], axis=1)  # [B, T_past+T, H, D_v]

    # 4. Causal softmax attention
    o = causal_softmax_attention(
        q, k, v, scale,
        window_size=window_size,
        attention_mask=attention_mask,
        cu_seqlens=cu_seqlens,
    )  # [B, T, H, v_head_dim]

    # 5. Output projection
    # FLA: o = o.reshape(B, T, -1); o = self.o_proj(o)
    o = o.reshape(B, T, num_heads * v_head_dim)
    output = o @ w_o.T  # [B, T, D_model]

    if use_cache:
        return output, k, v
    return output
