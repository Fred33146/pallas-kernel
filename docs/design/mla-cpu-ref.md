# MLA CPU Reference — Design Spec

**Date:** 2026-03-31
**Status:** Draft (pre-implementation)

## Goal

Implement a JAX CPU reference for Multi-head Latent Attention (MLA) that precisely matches FLA's `fla/layers/mla.py` dtype behavior and computation flow.

## Step 1: FLA Source Code Research

### Key Finding: MLA Has No Custom Triton Kernel

After searching `fla/ops/` (38 subdirectories), **there is no `fla/ops/mla/` directory**. MLA is fundamentally different from GLA/Simple GLA:

| Aspect | GLA / Simple GLA | MLA |
|--------|-----------------|-----|
| Attention type | Linear attention (gated recurrence) | **Standard softmax attention** |
| Core innovation | Chunk decomposition of linear recurrence | **Low-rank KV compression** |
| FLA Triton kernel | `fla/ops/gla/chunk.py` (custom Triton) | **None** — calls `flash_attn` |
| Recurrence formula | `h_t = h_{t-1} * exp(g_t) + k_t^T @ v_t` | N/A (no recurrence) |

FLA's MLA layer (`fla/layers/mla.py`) directly calls `flash_attn_func()` / `flash_attn_varlen_func()` for the attention computation. No Triton kernel to port.

**Implication:** The `/write-cpu-kernel` workflow (port Triton → JAX CPU) does not apply. Instead, we reference `fla/layers/mla.py` (PyTorch layer code) directly.

### FLA Source Files

| File | Role |
|------|------|
| `fla/layers/mla.py` | `MultiheadLatentAttention` — main reference |
| `fla/modules/rotary.py` | `RotaryEmbedding`, `rotary_embedding_ref` |
| `fla/models/mla/configuration_mla.py` | Hyperparameter definitions |
| `flash_attn.flash_attn_func` | External attention backend (fp16/bf16 only) |

### FLA MLA Forward Flow Analysis

From reading `fla/layers/mla.py` line by line:

```python
class MultiheadLatentAttention(nn.Module):
    def forward(self, hidden, attention_mask=None, past_key_values=None, ...):
        # 1. Query projection
        if q_lora_rank is not None:
            q = self.q_proj(hidden)              # Linear(D, q_lora_rank)
            #   → RMSNorm(q_lora_rank, dtype=float32)
            #   → Linear(q_lora_rank, H * qk_head_dim)
        else:
            q = self.q_proj(hidden)              # Linear(D, H * qk_head_dim)

        q = rearrange(q, '... (h d) -> ... h d', d=qk_head_dim)
        q_pass, q_rot = split(q, [qk_nope_head_dim, qk_rope_head_dim])

        # 2. KV projection
        kv = self.kv_proj(hidden)                # Sequential:
        #   Linear(D, kv_lora_rank)
        #   → RMSNorm(kv_lora_rank, dtype=float32)
        #   → Linear(kv_lora_rank, H * (qk_nope_head_dim + v_head_dim))
        kv = rearrange(kv, '... (h d) -> ... h d', d=kv_dim_per_head)
        k_pass, v = split(kv, [qk_nope_head_dim, v_head_dim])

        # 3. Separate K-rope (NOT from compressed c_kv)
        k_rot = self.k_rope(hidden)              # Linear(D, qk_rope_head_dim)
        k_rot = rearrange(k_rot, 'b t d -> b t 1 d')

        # 4. Apply RoPE
        q_rot, k_rot = self.rotary(q_rot, k_rot, seqlen_offset=..., cu_seqlens=...)
        k_rot = repeat(k_rot, 'b t 1 d -> b t h d', h=num_heads)

        # 5. Concatenate
        q = cat((q_pass, q_rot), dim=-1)
        k = cat((k_pass, k_rot), dim=-1)

        # 6. Attention (delegates to flash_attn)
        o = flash_attn_func(q, k, v, causal=True, window_size=...)

        # 7. Output projection
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)                       # Linear(H * v_head_dim, D)
```

### Dtype Behavior Extraction

Traced from FLA source code — every dtype transition point:

```
Function: q_proj / kv_proj (nn.Sequential with RMSNorm)
├── Input: hidden [input_dtype]
├── Linear:        output = input_dtype (PyTorch default)
├── RMSNorm:
│   ├── Internal:  cast to fp32 (dtype=torch.float32 in constructor)
│   ├── Normalize: fp32
│   └── Output:    cast back to input_dtype
├── Linear:        output = input_dtype
└── Output: input_dtype

Function: k_rope (nn.Linear)
├── Input: hidden [input_dtype]
└── Output: input_dtype

Function: rotary_embedding (fla/modules/rotary.py)
├── Input: x [input_dtype], cos/sin [fp32 precomputed]
├── Internal:
│   ├── cos/sin cast to x.dtype              ← KEY: not fp32 computation
│   ├── x0 * cos - x1 * sin                  ← in input_dtype
│   └── x0 * sin + x1 * cos                  ← in input_dtype
└── Output: input_dtype

Function: flash_attn_func
├── Input: q, k, v [input_dtype]              ← fp16 or bf16 ONLY
├── Internal: CUDA kernel (proprietary)
├── Scores: fp32 accumulation (hardware)
├── Softmax: fp32
└── Output: input_dtype

Function: o_proj (nn.Linear)
├── Input: o [input_dtype]
└── Output: input_dtype
```

**fp64 mode:** Not supported by FLA/flash_attn. Our CPU reference will support fp64 with all-fp64 computation (no precision promotion needed).

## Step 2: Design Decisions

### Conventions

- **Matmul precision:** JAX default precision is sufficient for CPU reference (no explicit `precision=HIGHEST` needed, unlike GLA's Triton-aligned einsum). All attention scores computed in accumulator dtype (fp32 or fp64).
- **Weight dtype:** Projection weights (w_dq, w_uq, w_dkv, w_ukv, w_kr, w_o) follow input dtype via `hidden @ w.T`. Norm weights (q_norm_weight, kv_norm_weight) are always stored in fp32, cast to accumulator dtype during computation.
- **`@cpu_reference` decorator:** Only the top-level public function (`mla_forward`) gets this decorator. Internal sub-functions (rms_norm, apply_rotary_emb, etc.) do not — they inherit the CPU context from the decorated caller.

### Why k_rope Must Bypass LoRA Compression

RoPE encodes absolute position via `cos(pos * freq)` / `sin(pos * freq)`. If k_rope were derived from compressed `c_kv`, position information would be entangled with LoRA compression, making it impossible to cache only `c_kv` at inference time.

The separation enables MLA's KV cache optimization:
- **Standard MHA cache**: `B × T × H × (qk_dim + v_dim)` — scales with num_heads
- **MLA cache**: `B × T × (kv_lora_rank + qk_rope_dim)` — independent of num_heads

k_rope is projected as **single-head** then broadcast — position information is identical across heads, saving `(H-1) × rope_dim` in compute and storage.

### Sub-function Split Strategy

Since MLA has no chunk decomposition, sub-functions are split by **projection pipeline** (not by chunk operations like GLA):

| # | Sub-function | FLA Counterpart | Purpose | Public? |
|---|-------------|----------------|---------|---------|
| 1 | `rms_norm` | `nn.RMSNorm(dtype=float32)` | fp32 internal normalization | Yes |
| 2 | `precompute_freqs_cis` | `RotaryEmbedding._update_cos_sin_cache` | RoPE frequency tables | Yes |
| 3 | `apply_rotary_emb` | `rotary_embedding_ref` (non-interleaved) | Position encoding | Yes |
| 4 | `yarn_get_mscale` | `yarn_get_mscale` in FLA layers | YaRN attention scaling factor | Yes |
| 5 | `_get_rope_cos_sin` | `RotaryEmbedding.forward` position logic | Position-aware cos/sin dispatch | No (internal) |
| 6 | `mla_project_q` | `q_proj` Sequential + RoPE | Query LoRA **or direct** + RoPE | Yes |
| 7 | `mla_project_kv` | `kv_proj` Sequential + `k_rope` + RoPE | KV LoRA + separate rope key | Yes |
| 8 | `causal_softmax_attention` | `flash_attn_func` / `flash_attn_varlen_func` | Reference softmax attention | Yes |
| 9 | `mla_forward` | `MultiheadLatentAttention.forward` | Full forward orchestrator | Yes (@cpu_reference) |

**Exported via `__init__.py`:** All "Yes" functions above.

### Query Projection: Two Paths

FLA's `MultiheadLatentAttention` supports two Q projection modes:

1. **LoRA path** (`q_lora_rank is not None`):
   `hidden → Linear(D, q_lora_rank) → RMSNorm(fp32) → Linear(q_lora_rank, H*qk_head_dim)`
2. **Direct path** (`q_lora_rank is None`):
   `hidden → Linear(D, H*qk_head_dim)`

Our `mla_project_q` must support both: accept either `w_dq` + `w_uq` + `q_norm_weight` (LoRA) or `w_q` (direct).

### RoPE Position Handling

FLA's `RotaryEmbedding` handles 4 position scenarios. Our `_get_rope_cos_sin` must replicate all of them:

| Scenario | FLA Code | JAX CPU Approach |
|----------|---------|-----------------|
| Default (no offset) | `cos[:T]` | `cos[:T]` — simple slice |
| Scalar offset (KV cache) | `cos[offset:offset+T]` | `jax.lax.dynamic_slice` — jit-safe |
| Per-batch tensor offset | `position_ids = offset[:,None] + arange(T)` | Same — fancy indexing |
| Varlen (cu_seqlens) | Triton kernel with `chunk_indices` | Vectorized searchsorted: `sum(arange(T) >= cu_seqlens[1:])` |

### Attention: What to Replace flash_attn With

`flash_attn_func` is a CUDA-only optimized kernel. Our CPU reference replaces it with explicit softmax attention supporting all modes FLA uses:

| FLA API | CPU Reference Equivalent |
|---------|------------------------|
| `flash_attn_func(q, k, v, causal=True)` | Standard causal mask: `q_pos >= k_pos` |
| `flash_attn_func(..., window_size=(W-1, 0))` | Add sliding window: `(q_pos - k_pos) < W` |
| `flash_attn_varlen_func(..., cu_seqlens_q, ...)` | Block-diagonal mask: `same_seq AND local_causal` |
| Asymmetric `T_q != T_k` (decode) | Offset-based causal: `(q_pos + T_k - T_q) >= k_pos` |

flash_attn only supports fp16/bf16. Our CPU reference additionally supports fp32 and fp64.

### KV Cache and seqlen_offset

FLA's MLA supports incremental decoding via KV cache. Our reference must replicate these behaviors:

1. **Auto-derive seqlen_offset from cache length:**
   When `past_k` is provided and `seqlen_offset == 0`, automatically set `seqlen_offset = past_k.shape[1]` (number of cached tokens). This matches FLA's `past_key_values.get_seq_length()`.

2. **Adjust offset for padded sequences:**
   When `attention_mask` is present with cache, per-batch offset adjustment is needed:
   ```
   actual_lens = sum(attention_mask, axis=-1)    # [B] — non-padding token count
   seqlen_offset += actual_lens - padded_len     # [B] — converts scalar to per-batch
   ```
   This accounts for padding tokens that shouldn't contribute to RoPE positions.

3. **Return format with cache:**
   When `past_k/past_v` are provided, return `(output, k_full, v_full)` where `k_full` and `v_full` are **full concatenated cache tensors** `[B, T_past+T_new, H, D]`, not just the new tokens.

### Attention Scaling with YaRN mscale

FLA computes mscale but does **NOT** pass it to `flash_attn` (dead code in FLA runtime). Our reference provides an `apply_mscale` flag:
- `apply_mscale=False` (default): matches FLA runtime behavior, `scale = qk_head_dim ** -0.5`
- `apply_mscale=True`: matches DeepSeek paper, `scale = qk_head_dim ** -0.5 * mscale²`

The mscale is **squared** because it applies to both Q and K sides of the attention score `(Q·mscale) @ (K·mscale)^T = Q @ K^T · mscale²`.

### Varlen Constraints

When `cu_seqlens` is provided:
- **B must equal 1** — sequences are packed into a single batch dimension `[1, T_total, ...]`
- `cu_seqlens` is `[N+1]` array defining sequence boundaries: `[0, T1, T1+T2, ..., T_total]`
- `attention_mask` and `cu_seqlens` are **mutually exclusive**

This B=1 requirement is an **implicit assumption in FLA** — not enforced by an explicit assert, but implied by the `.squeeze(0)` / `.unsqueeze(0)` pattern in [fla/layers/mla.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mla.py):

```python
# FLA source: squeeze removes batch dim, flash_attn_varlen_func expects [T_total, H, D]
o = flash_attn_varlen_func(
    q.squeeze(0), k.squeeze(0), v.squeeze(0),
    cu_seqlens_q=..., cu_seqlens_k=...,
    max_seqlen_q=..., max_seqlen_k=...,
).unsqueeze(0)
```

`flash_attn_varlen_func` requires inputs without a batch dimension — multiple sequences are described solely via `cu_seqlens`. FLA's `.squeeze(0)` would silently produce wrong results if B>1. Our CPU reference adds an explicit assert to fail fast.

### Validation Strategy

Since there's no Triton kernel to directly compare against, validation is split into two independent paths:

```
Path A: Correctness (any machine, no GPU required)
  JAX CPU reference ←→ PyTorch gold reference (gold_mla.py)
  - Both are independent implementations of the same algorithm
  - fp32 tolerance: 1e-5 (same-hardware, same-precision)
  - fp64 tolerance: 1e-7

Path B: FLA alignment (requires GPU + flash_attn)
  JAX CPU reference ←→ FLA MultiheadLatentAttention (flash_attn on GPU)
  - Cross-hardware: CPU vs GPU → hardware-level rounding differences
  - fp16 tolerance: 5e-3 (matches project standard for GLA/KDA)
  - bf16 tolerance: 5e-2 (matches project standard for GLA/KDA)
  - fp32/fp64: N/A (flash_attn only supports fp16/bf16)
```

### @cpu_reference Decorator

Lesson learned from GLA: on GPU machines, JAX defaults to GPU computation with TF32 precision, causing false mismatches against PyTorch CPU gold reference. The `@cpu_reference` decorator from `tops/cpu/ops/__init__.py` solves this by:
1. Moving all JAX array inputs to CPU device
2. Setting `jax.default_device(cpu)` context for computation
3. Ensuring deterministic CPU execution

Only `mla_forward` (public entry point) gets this decorator. Sub-functions do not — they inherit the CPU context from the decorated caller.

## Step 3: Proposed File Structure

```
tops/cpu/ops/mla/
├── __init__.py              # exports all sub-functions + mla_forward
└── mla.py                   # all sub-functions + @cpu_reference mla_forward

tests/src/ops/
└── gold_mla.py              # PyTorch gold reference (CPU, no flash_attn dep)

tests/ref/mla/
├── __init__.py
└── test_mla.py              # multi-level test suite
```

## Step 4: Proposed Test Structure

```
Level 1: Dtype verification (TestDtypes)
  └── Each sub-function preserves correct dtype (bf16/fp16/fp32/fp64)

Level 2: Cross-validation (TestCrossValidation)
  └── JAX CPU vs PyTorch gold — fp32: 1e-5, fp64: 1e-7

Level 3: Feature tests
  ├── TestQLoraNone      — direct Q projection path
  ├── TestWindowSize     — sliding window restricts attention
  ├── TestKVCache        — prefill-then-decode matches full forward
  ├── TestAttentionMask  — padding tokens excluded correctly
  ├── TestCuSeqlens      — varlen sequence independence
  ├── TestSeqlenOffset   — RoPE position shifting
  └── TestMscale         — YaRN mscale computation and effect

Level 4: GPU validation (TestVsFLA, requires CUDA + flash_attn)
  └── JAX CPU vs FLA flash_attn — fp16: 5e-3, bf16: 5e-2
```

## Step 5: MLA vs GLA/KDA Complexity Comparison

| Dimension | GLA | KDA (Delta Rule) | MLA |
|-----------|-----|-----------------|-----|
| Sub-function count | ~8 (fwd 4 + bwd 4) | ~10 (fwd 5 + bwd 5) | **~5** (projections + attention) |
| Chunk decomposition | Required | Required (matrix solve) | **Not needed** |
| Backward pass | Hand-written per sub-function | Hand-written per sub-function | **`jax.grad` auto-diff** |
| Dtype alignment | Trace every Triton cast | Trace every Triton cast | **Simple** (RMSNorm fp32 + softmax fp32) |
| Core algorithm | Linear recurrence | Delta rule + triangular solve | **Standard softmax** (textbook) |
| FLA Triton kernel | Yes, must trace line by line | Yes, must trace line by line | **None** — reference `fla/layers/` |

MLA has significantly less implementation complexity than GLA or KDA because the core attention is standard softmax (no custom kernel porting required).

## FLA Source Reference

- [`fla/layers/mla.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mla.py) — `MultiheadLatentAttention` (main reference)
- [`fla/modules/rotary.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/rotary.py) — `RotaryEmbedding`, `rotary_embedding_ref`
- [`fla/models/mla/configuration_mla.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mla/configuration_mla.py) — hyperparameter definitions
- `flash_attn.flash_attn_func` — attention kernel called by FLA
- Paper: DeepSeek-V2 (arXiv 2405.04434)
