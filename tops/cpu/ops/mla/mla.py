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

import jax.numpy as jnp

from tops.cpu.ops import cpu_reference


def _acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


# =============================================================================
# Sub-function 1: rms_norm
# =============================================================================


def rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """Apply rotary position embedding (non-interleaved / GPT-NeoX style).

    Matches FLA's rotary_embedding_ref() with interleaved=False:
        x0, x1 = x[..., :D//2], x[..., D//2:]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

    Args:
        x:   [..., T, D] — Input tensor. Last dim is the rotary dimension.
        cos: [T, D // 2] — Cosine frequencies (from precompute_freqs_cis)
        sin: [T, D // 2] — Sine frequencies (from precompute_freqs_cis)

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
# Sub-function 4: mla_project_q
# =============================================================================


def mla_project_q(
    hidden: jnp.ndarray,
    w_dq: jnp.ndarray,
    w_uq: jnp.ndarray,
    q_norm_weight: jnp.ndarray,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """MLA Query projection: LoRA compress -> RMSNorm(fp32) -> expand -> split -> RoPE.

    Matches FLA's q_proj = nn.Sequential(
        Linear(hidden_size, q_lora_rank),
        RMSNorm(q_lora_rank, dtype=float32),
        Linear(q_lora_rank, num_heads * qk_head_dim),
    )

    Computation flow:
        c_q = hidden @ w_dq.T                              # [B, T, q_lora_rank]
        c_q = rms_norm(c_q, q_norm_weight, eps)             # fp32 normalize
        q = c_q @ w_uq.T                                   # [B, T, num_heads * qk_head_dim]
        q = reshape(q, [B, T, H, qk_head_dim])
        q_nope, q_rope = split(q)
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        q = concat(q_nope, q_rope)

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dq:             [q_lora_rank, D_model] — Down-projection weight (PyTorch convention)
        w_uq:             [num_heads * qk_head_dim, q_lora_rank] — Up-projection weight
        q_norm_weight:    [q_lora_rank] — RMSNorm weight
        num_heads:        int — Number of attention heads
        qk_nope_head_dim: int — Non-rotary QK head dimension
        qk_rope_head_dim: int — Rotary QK head dimension
        cos:              [T, qk_rope_head_dim // 2] — RoPE cosine frequencies
        sin:              [T, qk_rope_head_dim // 2] — RoPE sine frequencies
        eps:              float — RMSNorm epsilon

    Returns:
        q: [B, T, num_heads, qk_head_dim] — Query tensor with RoPE applied
    """
    B, T, D = hidden.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_lora_rank = w_dq.shape[0]

    assert w_dq.shape == (q_lora_rank, D), (
        f"w_dq shape {w_dq.shape} != ({q_lora_rank}, {D})"
    )
    assert w_uq.shape == (num_heads * qk_head_dim, q_lora_rank), (
        f"w_uq shape {w_uq.shape} != ({num_heads * qk_head_dim}, {q_lora_rank})"
    )
    assert q_norm_weight.shape == (q_lora_rank,), (
        f"q_norm_weight shape {q_norm_weight.shape} != ({q_lora_rank},)"
    )

    # FLA: c_q = Linear(hidden_size, q_lora_rank)(hidden)
    c_q = hidden @ w_dq.T  # [B, T, q_lora_rank]
    # FLA: RMSNorm(q_lora_rank, dtype=torch.float32)
    c_q = rms_norm(c_q, q_norm_weight, eps)
    # FLA: q = Linear(q_lora_rank, num_heads * qk_head_dim)(c_q)
    q = c_q @ w_uq.T  # [B, T, num_heads * qk_head_dim]

    # FLA: q = rearrange(q, '... (h d) -> ... h d', d=qk_head_dim)
    q = q.reshape(B, T, num_heads, qk_head_dim)
    # FLA: q_pass, q_rot = split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_nope = q[..., :qk_nope_head_dim]
    q_rope = q[..., qk_nope_head_dim:]

    # FLA: q_rot, _ = self.rotary(q_rot, ...)
    # cos/sin are [T, R//2], q_rope is [B, T, H, rope_dim]
    # Need to broadcast: add head dim to cos/sin -> [T, 1, R//2]
    cos_b = cos[:T, :].reshape(T, 1, -1)  # [T, 1, R//2]
    sin_b = sin[:T, :].reshape(T, 1, -1)  # [T, 1, R//2]
    q_rope = apply_rotary_emb(q_rope, cos_b, sin_b)

    # FLA: q = torch.cat((q_pass, q_rot), dim=-1)
    q = jnp.concatenate([q_nope, q_rope], axis=-1)
    return q


# =============================================================================
# Sub-function 5: mla_project_kv
# =============================================================================


def mla_project_kv(
    hidden: jnp.ndarray,
    w_dkv: jnp.ndarray,
    w_ukv: jnp.ndarray,
    kv_norm_weight: jnp.ndarray,
    w_kr: jnp.ndarray,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    qk_rope_head_dim: int,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    eps: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        cos:              [T, qk_rope_head_dim // 2] — RoPE cosine frequencies
        sin:              [T, qk_rope_head_dim // 2] — RoPE sine frequencies
        eps:              float

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

    # FLA: k_rot = self.k_rope(hidden)  — separate linear projection
    # FLA: k_rot = rearrange(k_rot, 'b t d -> b t 1 d')
    k_rope = hidden @ w_kr.T  # [B, T, qk_rope_head_dim]
    k_rope = k_rope.reshape(B, T, 1, qk_rope_head_dim)

    # FLA: q_rot, k_rot = self.rotary(q_rot, k_rot, ...)
    cos_b = cos[:T, :].reshape(T, 1, -1)  # [T, 1, R//2]
    sin_b = sin[:T, :].reshape(T, 1, -1)  # [T, 1, R//2]
    k_rope = apply_rotary_emb(k_rope, cos_b, sin_b)

    # FLA: k_rot = repeat(k_rot, 'b t 1 d -> b t h d', h=num_heads)
    k_rope = jnp.broadcast_to(k_rope, (B, T, num_heads, qk_rope_head_dim))

    # FLA: k = torch.cat((k_pass, k_rot), dim=-1)
    k = jnp.concatenate([k_nope, k_rope], axis=-1)
    return k, v


# =============================================================================
# Sub-function 6: causal_softmax_attention
# =============================================================================


def causal_softmax_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    scale: float | None = None,
) -> jnp.ndarray:
    """Standard causal softmax attention — reference implementation.

    Computes: o = softmax(q @ k^T * scale + causal_mask) @ v

    This matches FlashAttention's output but uses explicit materialization
    of the attention matrix (O(T^2) memory). Suitable for CPU reference testing.

    Args:
        q: [B, T, H, D_qk] — Query tensor
        k: [B, T, H, D_qk] — Key tensor
        v: [B, T, H, D_v]  — Value tensor
        scale: float         — Scaling factor, default = D_qk ** -0.5

    Returns:
        o: [B, T, H, D_v] — Output tensor, same dtype as v

    Dtype behavior:
        - Attention scores computed in fp32 (softmax numerical stability)
        - Output cast back to v.dtype
        - fp64 mode: all computation in fp64
    """
    assert q.ndim == 4, f"q must be 4D [B,T,H,D], got {q.ndim}D"
    assert k.ndim == 4, f"k must be 4D [B,T,H,D], got {k.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T,H,D], got {v.ndim}D"
    assert q.shape[:3] == k.shape[:3], (
        f"q shape {q.shape[:3]} != k shape {k.shape[:3]}"
    )
    assert q.shape[-1] == k.shape[-1], (
        f"q head dim {q.shape[-1]} != k head dim {k.shape[-1]}"
    )
    assert k.shape[:3] == v.shape[:3], (
        f"k shape {k.shape[:3]} != v shape {v.shape[:3]}"
    )

    B, T, H, D_qk = q.shape
    D_v = v.shape[-1]
    orig_v_dtype = v.dtype
    acc_dt = _acc_dtype(q.dtype)

    if scale is None:
        scale = D_qk ** -0.5

    q_f = q.astype(acc_dt)
    k_f = k.astype(acc_dt)
    v_f = v.astype(acc_dt)

    # Compute attention scores: [B, H, T, T]
    # q: [B, T, H, D] -> [B, H, T, D]
    q_f = jnp.transpose(q_f, (0, 2, 1, 3))
    k_f = jnp.transpose(k_f, (0, 2, 1, 3))
    v_f = jnp.transpose(v_f, (0, 2, 1, 3))

    attn = jnp.matmul(q_f, jnp.transpose(k_f, (0, 1, 3, 2))) * scale  # [B, H, T, T]

    # Causal mask: upper triangle = -inf
    mask = jnp.triu(jnp.full((T, T), float('-inf'), dtype=acc_dt), k=1)
    attn = attn + mask

    # Softmax in accumulator dtype
    attn = jnp.exp(attn - jnp.max(attn, axis=-1, keepdims=True))
    attn = attn / jnp.sum(attn, axis=-1, keepdims=True)

    # Apply attention: [B, H, T, D_v]
    o = jnp.matmul(attn, v_f)

    # Transpose back: [B, H, T, D_v] -> [B, T, H, D_v]
    o = jnp.transpose(o, (0, 2, 1, 3))

    return o.astype(orig_v_dtype)


# =============================================================================
# Sub-function 7: mla_forward
# =============================================================================


@cpu_reference
def mla_forward(
    hidden: jnp.ndarray,
    w_dq: jnp.ndarray,
    w_uq: jnp.ndarray,
    q_norm_weight: jnp.ndarray,
    w_dkv: jnp.ndarray,
    w_ukv: jnp.ndarray,
    kv_norm_weight: jnp.ndarray,
    w_kr: jnp.ndarray,
    w_o: jnp.ndarray,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Full MLA forward pass: project Q/K/V -> causal attention -> output projection.

    Orchestrates all MLA sub-functions to compute the complete forward pass.

    Computation flow:
        1. q = mla_project_q(hidden, ...)             [B, T, H, qk_head_dim]
        2. k, v = mla_project_kv(hidden, ...)          [B, T, H, qk_head_dim], [B, T, H, v_head_dim]
        3. o = causal_softmax_attention(q, k, v)       [B, T, H, v_head_dim]
        4. output = reshape(o) @ w_o.T                 [B, T, D_model]

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dq:             [q_lora_rank, D_model] — Query down-projection
        w_uq:             [num_heads * qk_head_dim, q_lora_rank] — Query up-projection
        q_norm_weight:    [q_lora_rank] — Query RMSNorm weight
        w_dkv:            [kv_lora_rank, D_model] — KV down-projection
        w_ukv:            [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank] — KV up-projection
        kv_norm_weight:   [kv_lora_rank] — KV RMSNorm weight
        w_kr:             [qk_rope_head_dim, D_model] — Key-rope projection
        w_o:              [D_model, num_heads * v_head_dim] — Output projection
        num_heads:        int
        qk_nope_head_dim: int — Non-rotary QK head dimension
        qk_rope_head_dim: int — Rotary QK head dimension
        v_head_dim:       int — Value head dimension
        cos:              [max_seq_len, qk_rope_head_dim // 2]
        sin:              [max_seq_len, qk_rope_head_dim // 2]
        eps:              float

    Returns:
        output: [B, T, D_model] — Same dtype as input hidden
    """
    B, T, D = hidden.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    assert w_o.shape == (D, num_heads * v_head_dim), (
        f"w_o shape {w_o.shape} != ({D}, {num_heads * v_head_dim})"
    )

    # 1. Project queries
    q = mla_project_q(
        hidden, w_dq, w_uq, q_norm_weight,
        num_heads, qk_nope_head_dim, qk_rope_head_dim,
        cos, sin, eps,
    )  # [B, T, H, qk_head_dim]

    # 2. Project keys and values
    k, v = mla_project_kv(
        hidden, w_dkv, w_ukv, kv_norm_weight, w_kr,
        num_heads, qk_nope_head_dim, v_head_dim, qk_rope_head_dim,
        cos, sin, eps,
    )  # k: [B, T, H, qk_head_dim], v: [B, T, H, v_head_dim]

    # 3. Causal softmax attention
    # FLA uses scaling = qk_head_dim ** -0.5
    scale = qk_head_dim ** -0.5
    o = causal_softmax_attention(q, k, v, scale)  # [B, T, H, v_head_dim]

    # 4. Output projection
    # FLA: o = o.reshape(B, T, -1); o = self.o_proj(o)
    o = o.reshape(B, T, num_heads * v_head_dim)
    output = o @ w_o.T  # [B, T, D_model]

    return output
