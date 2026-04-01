"""PyTorch gold reference for MLA (Multi-head Latent Attention).

Pure PyTorch CPU implementation mirroring the JAX CPU ref
(tops/cpu/ops/mla/mla.py) for cross-validation testing.

Dtype contract (matching FLA for bf16/fp16/fp32; all fp64 for fp64):
  RMSNorm: fp32 internal, output = input dtype
  RoPE: cos/sin cast to input dtype, output = input dtype
  Attention: fp32 scores/softmax, output = v.dtype
  Final output: same dtype as hidden input
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _acc_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return torch.float64 if input_dtype == torch.float64 else torch.float32


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Functional RMSNorm — fp32 internal computation, cast back to input dtype.

    Formula: x * rsqrt(mean(x^2) + eps) * weight

    Args:
        x:      [..., D] — Input tensor
        weight: [D]      — Scale parameter
        eps:    float     — Numerical stability constant

    Returns:
        out: [..., D] — Same dtype as input x
    """
    orig_dtype = x.dtype
    acc_dt = _acc_dtype(orig_dtype)
    x_f = x.to(acc_dt)
    rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_normed = x_f / rms
    out = x_normed * weight.to(acc_dt)
    return out.to(orig_dtype)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin frequency tables in fp32.

    Args:
        dim:         int — Rotary embedding dimension
        max_seq_len: int — Maximum sequence length
        theta:       float — Base frequency

    Returns:
        cos: [max_seq_len, dim // 2]
        sin: [max_seq_len, dim // 2]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding (non-interleaved / GPT-NeoX style).

    Args:
        x:   [..., T, D] — Input tensor
        cos: [..., T, D // 2] — Cosine frequencies
        sin: [..., T, D // 2] — Sine frequencies

    Returns:
        out: [..., T, D] — Same shape and dtype as input x
    """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    R = x.shape[-1] // 2
    x0 = x[..., :R]
    x1 = x[..., R:]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.cat([out0, out1], dim=-1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Compute YaRN attention scaling factor."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def gold_mla_forward(
    hidden: torch.Tensor,
    w_dq: torch.Tensor | None,
    w_uq: torch.Tensor | None,
    q_norm_weight: torch.Tensor | None,
    w_dkv: torch.Tensor,
    w_ukv: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    w_kr: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-5,
    *,
    w_q: torch.Tensor | None = None,
    rope_scaling: dict | None = None,
    window_size: int | None = None,
    seqlen_offset: int = 0,
    apply_mscale: bool = False,
) -> torch.Tensor:
    """Full MLA forward in pure PyTorch — mirrors JAX CPU ref.

    Computation flow (matching FLA's MultiheadLatentAttention.forward):
        1. Q projection (LoRA or direct) → split → RoPE
        2. KV projection (LoRA) + K-rope → split → RoPE → broadcast
        3. Causal softmax attention (with optional window/mask)
        4. Output projection

    Args:
        hidden:           [B, T, D_model] — Input hidden states
        w_dq:             [q_lora_rank, D_model] or None — Query down-projection
        w_uq:             [H * qk_head_dim, q_lora_rank] or None — Query up-projection
        q_norm_weight:    [q_lora_rank] or None — Query RMSNorm weight
        w_dkv:            [kv_lora_rank, D_model] — KV down-projection
        w_ukv:            [H * (nope + v), kv_lora_rank] — KV up-projection
        kv_norm_weight:   [kv_lora_rank] — KV RMSNorm weight
        w_kr:             [rope_dim, D_model] — Key-rope projection
        w_o:              [D_model, H * v_head_dim] — Output projection
        num_heads:        int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim:       int
        cos:              [max_seq_len, rope_dim // 2]
        sin:              [max_seq_len, rope_dim // 2]
        eps:              float
        w_q:              [H * qk_head_dim, D_model] or None — Direct Q projection
        rope_scaling:     dict or None — RoPE scaling config
        window_size:      int or None — Sliding window size
        seqlen_offset:    int — RoPE position offset

    Returns:
        output: [B, T, D_model] — Same dtype as input hidden
    """
    B, T, D = hidden.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    acc_dt = _acc_dtype(hidden.dtype)

    # --- Q projection ---
    if w_dq is not None:
        c_q = F.linear(hidden, w_dq)  # [B, T, q_lora_rank]
        c_q = rms_norm(c_q, q_norm_weight, eps)
        q = F.linear(c_q, w_uq)  # [B, T, H * qk_head_dim]
    else:
        q = F.linear(hidden, w_q)  # [B, T, H * qk_head_dim]

    q = q.reshape(B, T, num_heads, qk_head_dim)
    q_nope = q[..., :qk_nope_head_dim]
    q_rope = q[..., qk_nope_head_dim:]

    # --- KV projection ---
    c_kv = F.linear(hidden, w_dkv)  # [B, T, kv_lora_rank]
    c_kv = rms_norm(c_kv, kv_norm_weight, eps)
    kv = F.linear(c_kv, w_ukv)  # [B, T, H * (nope + v)]

    kv = kv.reshape(B, T, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv[..., :qk_nope_head_dim]
    v = kv[..., qk_nope_head_dim:]

    # K-rope: separate projection + broadcast
    k_rope = F.linear(hidden, w_kr)  # [B, T, rope_dim]
    k_rope = k_rope.reshape(B, T, 1, qk_rope_head_dim)

    # --- RoPE ---
    cos_b = cos[seqlen_offset:seqlen_offset + T, :].reshape(T, 1, -1)
    sin_b = sin[seqlen_offset:seqlen_offset + T, :].reshape(T, 1, -1)
    q_rope = apply_rotary_emb(q_rope, cos_b, sin_b)
    k_rope = apply_rotary_emb(k_rope, cos_b, sin_b)

    # Assemble
    k_rope = k_rope.expand(B, T, num_heads, qk_rope_head_dim)
    q = torch.cat([q_nope, q_rope], dim=-1)  # [B, T, H, qk_head_dim]
    k = torch.cat([k_nope, k_rope], dim=-1)  # [B, T, H, qk_head_dim]

    # --- Attention ---
    scale = qk_head_dim ** -0.5
    if apply_mscale and rope_scaling is not None and rope_scaling.get("rope_type", "default") != "default":
        mscale_val = yarn_get_mscale(
            rope_scaling["factor"], rope_scaling.get("mscale_all_dim", 0),
        )
        scale = scale * mscale_val * mscale_val

    # Transpose to [B, H, T, D]
    q_f = q.to(acc_dt).transpose(1, 2)
    k_f = k.to(acc_dt).transpose(1, 2)
    v_f = v.to(acc_dt).transpose(1, 2)

    attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # [B, H, T, T]

    # Causal mask
    q_pos = torch.arange(T)
    k_pos = torch.arange(T)
    causal = q_pos[:, None] >= k_pos[None, :]  # [T, T]

    if window_size is not None:
        in_window = (q_pos[:, None] - k_pos[None, :]) < window_size
        causal = causal & in_window

    attn = attn.masked_fill(~causal[None, None, :, :], float('-inf'))

    # Softmax
    attn = attn - attn.max(dim=-1, keepdim=True).values
    attn = torch.exp(attn)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    o = torch.matmul(attn, v_f)  # [B, H, T, D_v]
    o = o.transpose(1, 2)  # [B, T, H, D_v]
    o = o.to(v.dtype)

    # --- Output projection ---
    o = o.reshape(B, T, num_heads * v_head_dim)
    output = F.linear(o, w_o)  # [B, T, D_model]

    return output
