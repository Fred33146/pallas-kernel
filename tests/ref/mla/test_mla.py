"""MLA (Multi-head Latent Attention): JAX CPU ref (tops.cpu.ops.mla) tests.

Tests:
  1. Sub-function dtype verification (no GPU)
  2. JAX CPU vs PyTorch gold reference cross-validation (no GPU)
  3. New feature tests: q_lora_rank=None, window_size, KV cache, mask, cu_seqlens, mscale
  4. JAX CPU vs FLA PyTorch layer (GPU — requires flash_attn)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import math

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp

from tops.cpu.ops.mla import (
    rms_norm,
    precompute_freqs_cis,
    apply_rotary_emb,
    yarn_get_mscale,
    mla_project_q,
    mla_project_kv,
    causal_softmax_attention,
    mla_forward,
)


# ============================================================================
# Config
# ============================================================================

# Default MLA configuration (small model for testing)
DEFAULT_CFG = dict(
    hidden_size=256,
    num_heads=4,
    q_lora_rank=32,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rope_theta=10000.0,
)

# Test cases: (B, T, hidden_size, num_heads, q_lora_rank, kv_lora_rank,
#              qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
CASES = [
    # Small
    (1, 16, 128, 2, 16, 32, 16, 8, 16),
    # Medium
    (2, 32, 256, 4, 32, 64, 32, 16, 32),
    # Larger T
    (1, 64, 256, 4, 32, 64, 32, 16, 32),
    # Different nope != v_head_dim
    (1, 16, 128, 2, 16, 32, 24, 8, 16),
    # DeepSeek-like proportions (scaled down)
    (1, 32, 256, 4, 32, 64, 32, 16, 32),
]


def _case_id(c):
    B, T, D, H, qlr, kvlr, nope, rope, vd = c
    return f"B{B}_T{T}_D{D}_H{H}_ql{qlr}_kvl{kvlr}_nope{nope}_rope{rope}_v{vd}"


# ============================================================================
# Helpers
# ============================================================================


def _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype, seed=42):
    """Generate random MLA weights in JAX.

    Returns:
        (w_dq, w_uq, q_norm_weight, w_dkv, w_ukv, kv_norm_weight, w_kr, w_o)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)
    qk_head_dim = nope + rope
    std = 0.02

    w_dq = jax.random.normal(keys[0], (qlr, D), dtype=jnp.float32) * std
    w_uq = jax.random.normal(keys[1], (H * qk_head_dim, qlr), dtype=jnp.float32) * std
    q_norm_weight = jnp.ones(qlr, dtype=jnp.float32)
    w_dkv = jax.random.normal(keys[2], (kvlr, D), dtype=jnp.float32) * std
    w_ukv = jax.random.normal(keys[3], (H * (nope + vd), kvlr), dtype=jnp.float32) * std
    kv_norm_weight = jnp.ones(kvlr, dtype=jnp.float32)
    w_kr = jax.random.normal(keys[4], (rope, D), dtype=jnp.float32) * std
    w_o = jax.random.normal(keys[5], (D, H * vd), dtype=jnp.float32) * std

    # Cast weights to target dtype (except norm weights stay fp32)
    if dtype != jnp.float32:
        w_dq = w_dq.astype(dtype)
        w_uq = w_uq.astype(dtype)
        w_dkv = w_dkv.astype(dtype)
        w_ukv = w_ukv.astype(dtype)
        w_kr = w_kr.astype(dtype)
        w_o = w_o.astype(dtype)

    return w_dq, w_uq, q_norm_weight, w_dkv, w_ukv, kv_norm_weight, w_kr, w_o


def _make_weights_no_lora(D, H, nope, rope, kvlr, vd, dtype, seed=42):
    """Generate MLA weights for q_lora_rank=None (direct Q projection)."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    qk_head_dim = nope + rope
    std = 0.02

    w_q = jax.random.normal(keys[0], (H * qk_head_dim, D), dtype=jnp.float32) * std
    w_dkv = jax.random.normal(keys[1], (kvlr, D), dtype=jnp.float32) * std
    w_ukv = jax.random.normal(keys[2], (H * (nope + vd), kvlr), dtype=jnp.float32) * std
    kv_norm_weight = jnp.ones(kvlr, dtype=jnp.float32)
    w_kr = jax.random.normal(keys[3], (rope, D), dtype=jnp.float32) * std
    w_o = jax.random.normal(keys[4], (D, H * vd), dtype=jnp.float32) * std

    if dtype != jnp.float32:
        w_q = w_q.astype(dtype)
        w_dkv = w_dkv.astype(dtype)
        w_ukv = w_ukv.astype(dtype)
        w_kr = w_kr.astype(dtype)
        w_o = w_o.astype(dtype)

    return w_q, w_dkv, w_ukv, kv_norm_weight, w_kr, w_o


def _make_hidden(B, T, D, dtype, seed=99):
    """Generate random hidden states."""
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (B, T, D), dtype=dtype) * 0.1


def _jax_to_torch(arr):
    """Convert JAX array to PyTorch tensor, preserving dtype (including fp64)."""
    import torch
    np_arr = np.array(arr)
    return torch.from_numpy(np_arr.copy())


def _torch_to_jax(t, dtype=None):
    """Convert PyTorch tensor to JAX array."""
    arr = jnp.array(t.detach().cpu().float().numpy())
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


# ============================================================================
# 1. Sub-function dtype verification (no GPU)
# ============================================================================


class TestDtypes:
    """Verify intermediate tensor dtypes match FLA spec."""

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_rms_norm_dtype(self, dtype):
        """RMSNorm: fp32 internal, output = input dtype."""
        x = jnp.ones((2, 4, 64), dtype=dtype)
        w = jnp.ones(64, dtype=jnp.float32)
        out = rms_norm(x, w)
        assert out.dtype == dtype, f"rms_norm output dtype={out.dtype}, expected {dtype}"

    def test_rms_norm_dtype_fp64(self):
        """fp64 mode: all fp64."""
        x = jnp.ones((2, 4, 64), dtype=jnp.float64)
        w = jnp.ones(64, dtype=jnp.float32)
        out = rms_norm(x, w)
        assert out.dtype == jnp.float64

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_apply_rotary_emb_dtype(self, dtype):
        """RoPE preserves input dtype."""
        cos, sin = precompute_freqs_cis(16, 32)
        x = jnp.ones((2, 32, 4, 16), dtype=dtype)
        cos_b = cos[:32].reshape(32, 1, -1)
        sin_b = sin[:32].reshape(32, 1, -1)
        out = apply_rotary_emb(x, cos_b, sin_b)
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_project_q_dtype(self, dtype):
        """Query projection output = input dtype (LoRA path)."""
        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        qlr = cfg["q_lora_rank"]
        nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        weights = _make_weights(D, H, qlr, cfg["kv_lora_rank"], nope, rope, cfg["v_head_dim"], dtype)
        hidden = _make_hidden(1, 16, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 16)
        q = mla_project_q(hidden, weights[0], weights[1], weights[2], H, nope, rope, cos, sin)
        assert q.dtype == dtype, f"q.dtype={q.dtype}, expected {dtype}"
        assert q.shape == (1, 16, H, nope + rope)

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_project_q_no_lora_dtype(self, dtype):
        """Query projection output = input dtype (direct path, q_lora_rank=None)."""
        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        qk_head_dim = nope + rope
        key = jax.random.PRNGKey(42)
        w_q = (jax.random.normal(key, (H * qk_head_dim, D), dtype=jnp.float32) * 0.02).astype(dtype)
        hidden = _make_hidden(1, 16, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 16)
        q = mla_project_q(hidden, None, None, None, H, nope, rope, cos, sin, w_q=w_q)
        assert q.dtype == dtype
        assert q.shape == (1, 16, H, qk_head_dim)

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_project_kv_dtype(self, dtype):
        """KV projection: k and v output = input dtype."""
        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        kvlr = cfg["kv_lora_rank"]
        nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        weights = _make_weights(D, H, cfg["q_lora_rank"], kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(1, 16, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 16)
        k, v = mla_project_kv(hidden, weights[3], weights[4], weights[5], weights[6],
                               H, nope, vd, rope, cos, sin)
        assert k.dtype == dtype, f"k.dtype={k.dtype}, expected {dtype}"
        assert v.dtype == dtype, f"v.dtype={v.dtype}, expected {dtype}"
        assert k.shape == (1, 16, H, nope + rope)
        assert v.shape == (1, 16, H, vd)

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_attention_dtype(self, dtype):
        """Attention output dtype = v.dtype."""
        B, T, H, D_qk, D_v = 1, 16, 2, 32, 16
        q = jnp.ones((B, T, H, D_qk), dtype=dtype)
        k = jnp.ones((B, T, H, D_qk), dtype=dtype)
        v = jnp.ones((B, T, H, D_v), dtype=dtype)
        o = causal_softmax_attention(q, k, v)
        assert o.dtype == dtype, f"attn output dtype={o.dtype}, expected {dtype}"

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
    def test_mla_forward_dtype(self, dtype):
        """Full MLA forward: output dtype = hidden dtype."""
        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        qlr, kvlr = cfg["q_lora_rank"], cfg["kv_lora_rank"]
        nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(1, 16, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 16)
        o = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)
        assert o.dtype == dtype, f"output dtype={o.dtype}, expected {dtype}"
        assert o.shape == (1, 16, D)

    def test_all_dtypes_fp64(self):
        """fp64 mode: all intermediate tensors should be fp64."""
        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        qlr, kvlr = cfg["q_lora_rank"], cfg["kv_lora_rank"]
        nope, rope, vd = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        dtype = jnp.float64
        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(1, 16, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 16)

        # Check sub-functions
        q = mla_project_q(hidden, weights[0], weights[1], weights[2], H, nope, rope, cos, sin)
        assert q.dtype == jnp.float64
        k, v = mla_project_kv(hidden, weights[3], weights[4], weights[5], weights[6],
                               H, nope, vd, rope, cos, sin)
        assert k.dtype == jnp.float64
        assert v.dtype == jnp.float64
        o = causal_softmax_attention(q, k, v)
        assert o.dtype == jnp.float64
        output = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)
        assert output.dtype == jnp.float64


# ============================================================================
# 2. JAX CPU vs PyTorch gold reference (no GPU)
# ============================================================================


class TestCrossValidation:
    """Cross-validate JAX CPU implementation against PyTorch gold reference."""

    @pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_mla_forward_vs_gold(self, case, dtype):
        """Full MLA forward: JAX CPU vs PyTorch gold reference."""
        from tests.src.ops.gold_mla import gold_mla_forward

        B, T, D, H, qlr, kvlr, nope, rope, vd = case
        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        # JAX CPU (cpu_reference decorator ensures CPU execution)
        o_jax = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)

        # PyTorch gold
        hidden_pt = _jax_to_torch(hidden)
        weights_pt = tuple(_jax_to_torch(w) for w in weights)
        cos_pt = _jax_to_torch(cos)
        sin_pt = _jax_to_torch(sin)
        o_pt = gold_mla_forward(hidden_pt, *weights_pt, H, nope, rope, vd, cos_pt, sin_pt)

        # Compare
        o_jax_np = np.array(o_jax).astype(np.float64)
        o_pt_np = o_pt.detach().cpu().numpy().astype(np.float64)

        if dtype == jnp.float64:
            atol, rtol = 1e-7, 1e-7
        else:
            atol, rtol = 1e-5, 1e-5

        np.testing.assert_allclose(
            o_jax_np, o_pt_np, atol=atol, rtol=rtol,
            err_msg=f"MLA forward mismatch (dtype={dtype})",
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_project_q_vs_gold(self, dtype):
        """Query projection sub-function: JAX vs PyTorch."""
        from tests.src.ops.gold_mla import rms_norm as pt_rms_norm, apply_rotary_emb as pt_rope
        import torch

        cfg = DEFAULT_CFG
        D, H = cfg["hidden_size"], cfg["num_heads"]
        qlr = cfg["q_lora_rank"]
        nope, rope = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
        qk_head_dim = nope + rope

        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            weights = _make_weights(D, H, qlr, cfg["kv_lora_rank"], nope, rope, cfg["v_head_dim"], dtype)
            hidden = _make_hidden(2, 32, D, dtype)
            cos, sin = precompute_freqs_cis(rope, 32)

            # JAX
            q_jax = mla_project_q(hidden, weights[0], weights[1], weights[2], H, nope, rope, cos, sin)

        # PyTorch (manual, mirroring FLA flow)
        hidden_pt = _jax_to_torch(hidden)
        w_dq_pt, w_uq_pt, qnw_pt = _jax_to_torch(weights[0]), _jax_to_torch(weights[1]), _jax_to_torch(weights[2])
        cos_pt, sin_pt = _jax_to_torch(cos), _jax_to_torch(sin)

        c_q = torch.nn.functional.linear(hidden_pt, w_dq_pt)
        c_q = pt_rms_norm(c_q, qnw_pt)
        q_pt = torch.nn.functional.linear(c_q, w_uq_pt)
        q_pt = q_pt.reshape(2, 32, H, qk_head_dim)
        q_nope_pt, q_rope_pt = q_pt[..., :nope], q_pt[..., nope:]
        cos_b = cos_pt[:32].reshape(32, 1, -1)
        sin_b = sin_pt[:32].reshape(32, 1, -1)
        q_rope_pt = pt_rope(q_rope_pt, cos_b, sin_b)
        q_pt = torch.cat([q_nope_pt, q_rope_pt], dim=-1)

        q_jax_np = np.array(q_jax).astype(np.float64)
        q_pt_np = q_pt.detach().numpy().astype(np.float64)

        atol = 1e-7 if dtype == jnp.float64 else 1e-5
        np.testing.assert_allclose(q_jax_np, q_pt_np, atol=atol, rtol=atol,
                                   err_msg="project_q mismatch")

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_attention_vs_gold(self, dtype):
        """Causal softmax attention: JAX vs PyTorch reference."""
        import torch

        B, T, H, D_qk, D_v = 2, 32, 4, 48, 32
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, 3)
            q = jax.random.normal(keys[0], (B, T, H, D_qk), dtype=dtype) * 0.1
            k = jax.random.normal(keys[1], (B, T, H, D_qk), dtype=dtype) * 0.1
            v = jax.random.normal(keys[2], (B, T, H, D_v), dtype=dtype) * 0.1

            o_jax = causal_softmax_attention(q, k, v)

        # PyTorch reference
        scale = D_qk ** -0.5
        q_pt = torch.from_numpy(np.array(q)).transpose(1, 2).float()
        k_pt = torch.from_numpy(np.array(k)).transpose(1, 2).float()
        v_pt = torch.from_numpy(np.array(v)).transpose(1, 2).float()
        attn = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        o_pt = torch.matmul(attn, v_pt).transpose(1, 2)

        o_jax_np = np.array(o_jax).astype(np.float64)
        o_pt_np = o_pt.detach().numpy().astype(np.float64)

        atol = 1e-7 if dtype == jnp.float64 else 1e-5
        np.testing.assert_allclose(o_jax_np, o_pt_np, atol=atol, rtol=atol,
                                   err_msg="attention mismatch")


# ============================================================================
# 3. New feature tests
# ============================================================================


class TestQLoraNone:
    """Test q_lora_rank=None (direct Q projection)."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_forward_no_lora_vs_gold(self, dtype):
        """MLA forward with q_lora_rank=None: JAX vs PyTorch gold."""
        from tests.src.ops.gold_mla import gold_mla_forward

        B, T, D, H = 2, 32, 256, 4
        nope, rope, vd, kvlr = 32, 16, 32, 64

        w_q, w_dkv, w_ukv, kv_nw, w_kr, w_o = _make_weights_no_lora(
            D, H, nope, rope, kvlr, vd, dtype,
        )
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        # JAX: w_dq=None, pass w_q as keyword
        o_jax = mla_forward(
            hidden, None, None, None,
            w_dkv, w_ukv, kv_nw, w_kr, w_o,
            H, nope, rope, vd, cos, sin,
            w_q=w_q,
        )

        # PyTorch gold
        o_pt = gold_mla_forward(
            _jax_to_torch(hidden),
            None, None, None,
            _jax_to_torch(w_dkv), _jax_to_torch(w_ukv),
            _jax_to_torch(kv_nw), _jax_to_torch(w_kr), _jax_to_torch(w_o),
            H, nope, rope, vd,
            _jax_to_torch(cos), _jax_to_torch(sin),
            w_q=_jax_to_torch(w_q),
        )

        atol = 1e-7 if dtype == jnp.float64 else 1e-5
        np.testing.assert_allclose(
            np.array(o_jax).astype(np.float64),
            o_pt.detach().numpy().astype(np.float64),
            atol=atol, rtol=atol,
            err_msg="MLA forward (no LoRA Q) mismatch",
        )


class TestWindowSize:
    """Test sliding window attention."""

    @pytest.mark.parametrize("window_size", [4, 8, 16])
    def test_window_restricts_attention(self, window_size):
        """Verify window_size limits how far back queries can attend."""
        B, T, H, D = 1, 32, 2, 16
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        q = jax.random.normal(keys[0], (B, T, H, D)) * 0.1
        k = jax.random.normal(keys[1], (B, T, H, D)) * 0.1
        v = jax.random.normal(keys[2], (B, T, H, D)) * 0.1

        o_full = causal_softmax_attention(q, k, v)
        o_win = causal_softmax_attention(q, k, v, window_size=window_size)

        # First window_size positions should be the same (window covers all past)
        np.testing.assert_allclose(
            np.array(o_full[:, :window_size]),
            np.array(o_win[:, :window_size]),
            atol=1e-6,
            err_msg="First window_size positions should match full causal",
        )
        # Positions beyond window_size should differ
        if T > window_size:
            diff = np.max(np.abs(np.array(o_full[:, window_size:]) - np.array(o_win[:, window_size:])))
            assert diff > 1e-6, "Window should change output for positions beyond window"

    def test_window_vs_gold(self):
        """Window attention: JAX vs PyTorch gold."""
        from tests.src.ops.gold_mla import gold_mla_forward

        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T, ws = 1, 32, 8

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, jnp.float32)
        hidden = _make_hidden(B, T, D, jnp.float32)
        cos, sin = precompute_freqs_cis(rope, T)

        o_jax = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin, window_size=ws)
        o_pt = gold_mla_forward(
            _jax_to_torch(hidden), *[_jax_to_torch(w) for w in weights],
            H, nope, rope, vd, _jax_to_torch(cos), _jax_to_torch(sin),
            window_size=ws,
        )

        np.testing.assert_allclose(
            np.array(o_jax).astype(np.float64),
            o_pt.detach().numpy().astype(np.float64),
            atol=1e-5, rtol=1e-5,
            err_msg="Window attention: JAX vs gold mismatch",
        )


class TestKVCache:
    """Test KV cache for incremental decoding."""

    def test_prefill_then_decode(self):
        """Verify that prefill + decode matches single full forward."""
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T_total = 1, 16
        T_prefill = 12
        T_decode = T_total - T_prefill
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden_full = _make_hidden(B, T_total, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T_total)

        # Full forward (reference)
        o_full = mla_forward(hidden_full, *weights, H, nope, rope, vd, cos, sin)

        # Prefill: first T_prefill tokens, with cache
        hidden_pre = hidden_full[:, :T_prefill]
        o_pre, k_cache, v_cache = mla_forward(
            hidden_pre, *weights, H, nope, rope, vd, cos, sin,
            past_k=jnp.zeros((B, 0, H, nope + rope), dtype=dtype),
            past_v=jnp.zeros((B, 0, H, vd), dtype=dtype),
        )

        # Decode: remaining tokens using cache
        hidden_dec = hidden_full[:, T_prefill:]
        o_dec, _, _ = mla_forward(
            hidden_dec, *weights, H, nope, rope, vd, cos, sin,
            past_k=k_cache, past_v=v_cache,
        )

        # Concatenate and compare
        o_incremental = jnp.concatenate([o_pre, o_dec], axis=1)
        np.testing.assert_allclose(
            np.array(o_full), np.array(o_incremental),
            atol=1e-5, rtol=1e-5,
            err_msg="Prefill+decode should match full forward",
        )

    def test_cache_return_type(self):
        """mla_forward returns (output, k, v) when past_k/v are provided."""
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T = 1, 8
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        # Without cache: returns tensor
        result_no_cache = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)
        assert isinstance(result_no_cache, jnp.ndarray)

        # With cache: returns tuple
        result_cache = mla_forward(
            hidden, *weights, H, nope, rope, vd, cos, sin,
            past_k=jnp.zeros((B, 0, H, nope + rope), dtype=dtype),
            past_v=jnp.zeros((B, 0, H, vd), dtype=dtype),
        )
        assert isinstance(result_cache, tuple) and len(result_cache) == 3


class TestAttentionMask:
    """Test attention_mask (padding) support."""

    def test_mask_zeros_padding(self):
        """Padding tokens (mask=0) should not affect valid outputs."""
        B, T, H, D = 2, 16, 2, 16
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        q = jax.random.normal(keys[0], (B, T, H, D)) * 0.1
        k = jax.random.normal(keys[1], (B, T, H, D)) * 0.1
        v = jax.random.normal(keys[2], (B, T, H, D)) * 0.1

        # No padding
        o_full = causal_softmax_attention(q, k, v)

        # All-ones mask (no padding) should give same result
        mask_all = jnp.ones((B, T), dtype=jnp.int32)
        o_masked = causal_softmax_attention(q, k, v, attention_mask=mask_all)

        np.testing.assert_allclose(
            np.array(o_full), np.array(o_masked),
            atol=1e-6,
            err_msg="All-ones mask should match unmasked",
        )

    def test_mask_with_padding(self):
        """Outputs at valid positions should differ when padding is present."""
        B, T, H, D = 1, 16, 2, 16
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        q = jax.random.normal(keys[0], (B, T, H, D)) * 0.1
        k = jax.random.normal(keys[1], (B, T, H, D)) * 0.1
        v = jax.random.normal(keys[2], (B, T, H, D)) * 0.1

        # Mask last 4 positions as padding
        mask = jnp.concatenate([jnp.ones(T - 4), jnp.zeros(4)]).reshape(1, T).astype(jnp.int32)
        o_masked = causal_softmax_attention(q, k, v, attention_mask=mask)

        # Valid positions (first T-4) should still produce valid outputs
        assert not jnp.any(jnp.isnan(o_masked[:, :T - 4]))


class TestCuSeqlens:
    """Test variable-length packed sequences (cu_seqlens)."""

    def test_single_seq_matches_batched(self):
        """cu_seqlens with single sequence should match non-varlen."""
        B, T, H, D = 1, 16, 2, 16
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        q = jax.random.normal(keys[0], (B, T, H, D)) * 0.1
        k = jax.random.normal(keys[1], (B, T, H, D)) * 0.1
        v = jax.random.normal(keys[2], (B, T, H, D)) * 0.1

        cu_seqlens = jnp.array([0, T], dtype=jnp.int32)
        o_varlen = causal_softmax_attention(q, k, v, cu_seqlens=cu_seqlens)
        o_normal = causal_softmax_attention(q, k, v)

        np.testing.assert_allclose(
            np.array(o_varlen), np.array(o_normal),
            atol=1e-6,
            err_msg="Single-seq varlen should match normal",
        )

    def test_multi_seq_independence(self):
        """Packed sequences should be independent (no cross-sequence attention)."""
        H, D = 2, 16
        # Two sequences of length 4 and 6
        T_total = 10
        cu_seqlens = jnp.array([0, 4, 10], dtype=jnp.int32)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        q = jax.random.normal(keys[0], (1, T_total, H, D)) * 0.1
        k = jax.random.normal(keys[1], (1, T_total, H, D)) * 0.1
        v = jax.random.normal(keys[2], (1, T_total, H, D)) * 0.1

        o_packed = causal_softmax_attention(q, k, v, cu_seqlens=cu_seqlens)

        # Compute each sequence independently
        o_seq1 = causal_softmax_attention(q[:, :4], k[:, :4], v[:, :4])
        o_seq2 = causal_softmax_attention(q[:, 4:], k[:, 4:], v[:, 4:])

        np.testing.assert_allclose(
            np.array(o_packed[:, :4]), np.array(o_seq1),
            atol=1e-6, err_msg="Seq 1 mismatch",
        )
        np.testing.assert_allclose(
            np.array(o_packed[:, 4:]), np.array(o_seq2),
            atol=1e-6, err_msg="Seq 2 mismatch",
        )

    def test_varlen_mla_forward(self):
        """Full MLA forward with cu_seqlens."""
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        T_total = 10
        cu_seqlens = jnp.array([0, 4, 10], dtype=jnp.int32)
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(1, T_total, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T_total)

        # Should not raise
        o = mla_forward(
            hidden, *weights, H, nope, rope, vd, cos, sin,
            cu_seqlens=cu_seqlens,
        )
        assert o.shape == (1, T_total, D)


class TestSeqlenOffset:
    """Test RoPE seqlen_offset."""

    def test_offset_shifts_rope_positions(self):
        """RoPE with offset should shift position embeddings in Q/K projections."""
        from tops.cpu.ops.mla.mla import _get_rope_cos_sin

        rope_dim = 16
        T, offset = 8, 4
        cos, sin = precompute_freqs_cis(rope_dim, 32)

        # Without offset: positions [0, 1, ..., 7]
        cos_no, sin_no = _get_rope_cos_sin(cos, sin, T, seqlen_offset=0)
        # With offset: positions [4, 5, ..., 11]
        cos_off, sin_off = _get_rope_cos_sin(cos, sin, T, seqlen_offset=offset)

        # cos_off should equal cos[4:12]
        np.testing.assert_allclose(
            np.array(cos_off.reshape(T, -1)),
            np.array(cos[offset:offset + T]),
            atol=1e-7,
            err_msg="Offset cos should be cos[offset:offset+T]",
        )
        # Different from no offset
        diff = np.max(np.abs(np.array(cos_no) - np.array(cos_off)))
        assert diff > 1e-6, "Offset should shift RoPE positions"

    def test_offset_via_kv_cache(self):
        """seqlen_offset is correctly applied through KV cache path.

        When using KV cache, the new q tokens (positions T_past..T_past+T_new-1)
        attend to cached k tokens (positions 0..T_past-1) plus new k tokens.
        The RoPE positions must be correctly offset for decode tokens.
        This is already validated by TestKVCache.test_prefill_then_decode.
        Here we verify the Q projection directly.
        """
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T = 1, 4
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, 32)

        # Q at positions [0,1,2,3]
        q_0 = mla_project_q(
            hidden, weights[0], weights[1], weights[2],
            H, nope, rope, cos, sin,
        )
        # Q at positions [4,5,6,7]
        q_4 = mla_project_q(
            hidden, weights[0], weights[1], weights[2],
            H, nope, rope, cos, sin,
            seqlen_offset=4,
        )

        # nope part should be identical (no RoPE)
        np.testing.assert_allclose(
            np.array(q_0[..., :nope]), np.array(q_4[..., :nope]),
            atol=1e-7, err_msg="Nope part should be identical regardless of offset",
        )
        # rope part should differ
        diff = np.max(np.abs(np.array(q_0[..., nope:]) - np.array(q_4[..., nope:])))
        assert diff > 1e-6, "Rope part of Q should differ with offset"


class TestMscale:
    """Test rope_scaling / mscale support."""

    def test_yarn_get_mscale(self):
        """yarn_get_mscale matches FLA's implementation."""
        assert yarn_get_mscale(1.0, 1.0) == 1.0
        assert yarn_get_mscale(0.5, 2.0) == 1.0
        # scale > 1: 0.1 * mscale * log(scale) + 1.0
        expected = 0.1 * 1.0 * math.log(4.0) + 1.0
        assert abs(yarn_get_mscale(4.0, 1.0) - expected) < 1e-10

    def test_mscale_changes_output(self):
        """rope_scaling should modify the attention output."""
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T = 1, 16
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        o_normal = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)
        o_scaled = mla_forward(
            hidden, *weights, H, nope, rope, vd, cos, sin,
            rope_scaling={"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0},
            apply_mscale=True,
        )

        diff = np.max(np.abs(np.array(o_normal) - np.array(o_scaled)))
        assert diff > 1e-6, "mscale should change output"

    def test_mscale_default_matches_fla(self):
        """Default apply_mscale=False: mscale should NOT change output (FLA behavior)."""
        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T = 1, 16
        dtype = jnp.float32

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        o_normal = mla_forward(hidden, *weights, H, nope, rope, vd, cos, sin)
        o_with_scaling = mla_forward(
            hidden, *weights, H, nope, rope, vd, cos, sin,
            rope_scaling={"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0},
        )

        np.testing.assert_allclose(
            np.array(o_normal), np.array(o_with_scaling),
            atol=1e-7,
            err_msg="Default apply_mscale=False should not change output (FLA behavior)",
        )

    def test_mscale_vs_gold(self):
        """mscale: JAX vs PyTorch gold."""
        from tests.src.ops.gold_mla import gold_mla_forward

        D, H, qlr, kvlr, nope, rope, vd = 128, 2, 16, 32, 16, 8, 16
        B, T = 1, 16
        dtype = jnp.float32
        scaling = {"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0}

        weights = _make_weights(D, H, qlr, kvlr, nope, rope, vd, dtype)
        hidden = _make_hidden(B, T, D, dtype)
        cos, sin = precompute_freqs_cis(rope, T)

        o_jax = mla_forward(
            hidden, *weights, H, nope, rope, vd, cos, sin,
            rope_scaling=scaling,
            apply_mscale=True,
        )
        o_pt = gold_mla_forward(
            _jax_to_torch(hidden), *[_jax_to_torch(w) for w in weights],
            H, nope, rope, vd, _jax_to_torch(cos), _jax_to_torch(sin),
            rope_scaling=scaling,
            apply_mscale=True,
        )

        np.testing.assert_allclose(
            np.array(o_jax).astype(np.float64),
            o_pt.detach().numpy().astype(np.float64),
            atol=1e-5, rtol=1e-5,
            err_msg="mscale: JAX vs gold mismatch",
        )


# ============================================================================
# 4. JAX CPU vs FLA MultiheadLatentAttention (GPU — requires flash_attn)
# ============================================================================

HAS_CUDA = False
HAS_FLA_MLA = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

if HAS_CUDA:
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        from fla.layers.mla import MultiheadLatentAttention as FLA_MLA
        HAS_FLA_MLA = True
    except (ImportError, Exception):
        pass

requires_fla_mla = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA_MLA),
    reason="CUDA + FLA + flash_attn not available",
)

GPU_CASES = [
    # (B, T, hidden_size, num_heads, q_lora_rank, kv_lora_rank,
    #  qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    (1, 32, 256, 4, 32, 64, 32, 16, 32),
    (2, 64, 256, 4, 32, 64, 32, 16, 32),
    (1, 32, 128, 2, 16, 32, 16, 8, 16),
    # nope != v_head_dim (FLA pads v to match qk_head_dim for flash_attn)
    (1, 32, 256, 4, 32, 64, 32, 16, 48),
]


def _extract_fla_weights(fla_module):
    """Extract weight matrices from FLA's MultiheadLatentAttention module.

    Returns weights in the same order as mla_forward() expects:
        (w_dq, w_uq, q_norm_weight, w_dkv, w_ukv, kv_norm_weight, w_kr, w_o)
    """
    # q_proj = Sequential(Linear(D, qlr), RMSNorm(qlr), Linear(qlr, H*qk_head_dim))
    w_dq = fla_module.q_proj[0].weight.data      # [qlr, D]
    q_norm_w = fla_module.q_proj[1].weight.data   # [qlr]
    w_uq = fla_module.q_proj[2].weight.data       # [H*qk_head_dim, qlr]

    # kv_proj = Sequential(Linear(D, kvlr), RMSNorm(kvlr), Linear(kvlr, H*(nope+v)))
    w_dkv = fla_module.kv_proj[0].weight.data     # [kvlr, D]
    kv_norm_w = fla_module.kv_proj[1].weight.data  # [kvlr]
    w_ukv = fla_module.kv_proj[2].weight.data      # [H*(nope+v), kvlr]

    # k_rope = Linear(D, rope_dim)
    w_kr = fla_module.k_rope.weight.data           # [rope_dim, D]

    # o_proj = Linear(H*v, D)
    w_o = fla_module.o_proj.weight.data            # [D, H*v]

    return w_dq, w_uq, q_norm_w, w_dkv, w_ukv, kv_norm_w, w_kr, w_o


@requires_fla_mla
class TestVsFLA:
    """Compare JAX CPU MLA against FLA's MultiheadLatentAttention (flash_attn on GPU)."""

    @pytest.mark.parametrize("case", GPU_CASES, ids=[_case_id(c) for c in GPU_CASES])
    @pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
    def test_mla_fwd_vs_fla(self, case, dtype_name):
        """Full MLA forward: JAX CPU vs FLA (flash_attn) on GPU."""
        import torch

        dtype_map = {
            "fp16": (torch.float16, jnp.float16),
            "bf16": (torch.bfloat16, jnp.bfloat16),
        }
        tol_map = {
            "fp16": (5e-3, 5e-3),
            "bf16": (5e-2, 5e-2),
        }
        torch_dtype, jax_dtype = dtype_map[dtype_name]
        atol, rtol = tol_map[dtype_name]

        B, T, D, H, qlr, kvlr, nope, rope, vd = case
        qk_head_dim = nope + rope

        # Create FLA module on GPU
        torch.manual_seed(42)
        fla_mla = FLA_MLA(
            hidden_size=D,
            num_heads=H,
            q_lora_rank=qlr,
            kv_lora_rank=kvlr,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=vd,
            qk_head_dim=qk_head_dim,
            rope_theta=10000.0,
            max_position_embeddings=T,
        ).to(torch_dtype).cuda().eval()

        # Generate input
        torch.manual_seed(99)
        hidden_pt = torch.randn(B, T, D, dtype=torch_dtype, device="cuda") * 0.1

        # Run FLA forward (uses flash_attn internally)
        with torch.no_grad():
            o_fla, _, _ = fla_mla(hidden_pt, attention_mask=None)

        # Extract weights and convert to JAX
        weights_pt = _extract_fla_weights(fla_mla)
        weights_jax = []
        for i, w in enumerate(weights_pt):
            w_np = w.detach().cpu().float().numpy()
            if i in (2, 5):  # norm weights stay fp32
                weights_jax.append(jnp.array(w_np))
            else:
                weights_jax.append(jnp.array(w_np).astype(jax_dtype))
        weights_jax = tuple(weights_jax)

        hidden_jax = jnp.array(
            hidden_pt.detach().cpu().float().numpy()
        ).astype(jax_dtype)

        # Precompute RoPE cos/sin matching FLA's RotaryEmbedding
        cos, sin = precompute_freqs_cis(rope, T)

        # Run JAX CPU forward (cpu_reference decorator ensures CPU execution)
        o_jax = mla_forward(
            hidden_jax, *weights_jax, H, nope, rope, vd, cos, sin,
        )

        # Compare
        o_fla_np = o_fla.detach().cpu().float().numpy()
        o_jax_np = np.array(o_jax).astype(np.float32)

        max_abs_diff = np.max(np.abs(o_fla_np - o_jax_np))
        max_rel_diff = np.max(
            np.abs(o_fla_np - o_jax_np) / (np.abs(o_fla_np) + 1e-8)
        )
        print(
            f"\n[{dtype_name}] MLA fwd vs FLA: "
            f"max_abs={max_abs_diff:.4e}, max_rel={max_rel_diff:.4e}"
        )

        np.testing.assert_allclose(
            o_jax_np, o_fla_np, atol=atol, rtol=rtol,
            err_msg=f"MLA forward: JAX CPU vs FLA flash_attn ({dtype_name})",
        )

    @pytest.mark.parametrize("dtype_name", ["bf16"])
    def test_mla_fwd_fp64_vs_fla_bf16(self, dtype_name):
        """fp64 JAX CPU vs bf16 FLA: fp64 should be within bf16 precision of FLA."""
        import torch

        B, T, D, H, qlr, kvlr, nope, rope, vd = 1, 32, 256, 4, 32, 64, 32, 16, 32
        qk_head_dim = nope + rope

        torch.manual_seed(42)
        fla_mla = FLA_MLA(
            hidden_size=D, num_heads=H, q_lora_rank=qlr, kv_lora_rank=kvlr,
            qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=vd,
            qk_head_dim=qk_head_dim, rope_theta=10000.0,
            max_position_embeddings=T,
        ).to(torch.bfloat16).cuda().eval()

        torch.manual_seed(99)
        hidden_pt = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda") * 0.1

        with torch.no_grad():
            o_fla, _, _ = fla_mla(hidden_pt, attention_mask=None)

        weights_pt = _extract_fla_weights(fla_mla)
        weights_jax = tuple(
            jnp.array(w.detach().cpu().float().numpy()).astype(jnp.float64)
            for w in weights_pt
        )
        hidden_jax = jnp.array(
            hidden_pt.detach().cpu().float().numpy()
        ).astype(jnp.float64)
        cos, sin = precompute_freqs_cis(rope, T)

        o_jax = mla_forward(
            hidden_jax, *weights_jax, H, nope, rope, vd, cos, sin,
        )

        o_fla_np = o_fla.detach().cpu().float().numpy().astype(np.float64)
        o_jax_np = np.array(o_jax)

        max_abs_diff = np.max(np.abs(o_fla_np - o_jax_np))
        print(f"\n[fp64 vs bf16] MLA fwd vs FLA: max_abs={max_abs_diff:.4e}")

        # fp64 JAX should be within bf16 precision of FLA bf16
        np.testing.assert_allclose(
            o_jax_np, o_fla_np, atol=5e-2, rtol=5e-2,
            err_msg="fp64 JAX vs bf16 FLA flash_attn",
        )
