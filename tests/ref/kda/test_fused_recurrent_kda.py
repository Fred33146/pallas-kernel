"""fused_recurrent_kda(+_fwd) / kda_gate_*: JAX CPU ref (tops.cpu.ops.kda) tests.

Tests:
  1. Gate formula verification (no GPU)
  2. Gate chunk cumsum verification (no GPU, varlen)
  3. Dtype verification (no GPU)
  4. Cross-validation: fused_recurrent vs naive_recurrent (no GPU)
  5. Wrapper equivalence: fused_recurrent_kda_fwd vs fused_recurrent_kda (no GPU)
  6. Varlen: cu_seqlens vs per-segment calls (no GPU)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
import jax.numpy as jnp

from tops.cpu.ops.common import chunk_local_cumsum
from tops.cpu.ops.kda import (
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    kda_gate_chunk_cumsum,
    naive_recurrent_kda,
)
from tests.utils import torch_to_jax
from tests.utils import compare_tensor

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:
    torch = None

try:
    from fla.ops.kda import fused_recurrent_kda as triton_fused_recurrent_kda

    HAS_FLA = True
except ImportError:
    HAS_FLA = False

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_FLA),
    reason="Requires CUDA device and flash-linear-attention",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_inputs(B, T, H, K, V, dtype, seed=42, *, h0=False):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=dtype)
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=dtype)
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=dtype)
    g_raw = jax.random.normal(keys[3], (B, T, H, K), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, H), dtype=dtype)).astype(dtype)
    A_log = jnp.log(jax.random.uniform(keys[5], (H,), minval=1.0, maxval=4.0)).astype(jnp.float32)
    dt_bias = jax.random.normal(keys[6], (H * K,), dtype=jnp.float32)
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    h0_arr = jax.random.normal(keys[7], (B, H, K, V), dtype=acc) if h0 else None
    return q, k, v, g_raw, beta, A_log, dt_bias, h0_arr


def _segment_reference(fn, q, k, v, g, beta, cu_seqlens, initial_state=None, **kwargs):
    T = q.shape[1]
    H, K, V = q.shape[2], q.shape[3], v.shape[3]
    out = jnp.zeros((1, T, H, V), dtype=q.dtype)
    states = []
    N = len(cu_seqlens) - 1
    for i in range(N):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        h0 = None if initial_state is None else initial_state[i:i + 1]
        o_seg, ht_seg = fn(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            g[:, bos:eos],
            beta[:, bos:eos],
            initial_state=h0,
            output_final_state=True,
            **kwargs,
        )
        out = out.at[:, bos:eos].set(o_seg)
        states.append(ht_seg[0])
    return out, jnp.stack(states, axis=0)


# ============================================================================
# Shape configs
# ============================================================================


ALL_DTYPES = [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64]

_CROSS_SHAPES = [
    dict(B=2, T=33, H=4, K=16, V=12, seed=2, h0=True),
    dict(B=2, T=17, H=3, K=8, V=10, seed=10, h0=False),
]


def _shape_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    if c.get("h0"):
        parts.append("h0")
    return "-".join(parts)


# ============================================================================
# 1. Gate formula verification (no GPU needed)
# ============================================================================


def test_fused_kda_gate_formula_fp64():
    B, T, H, K = 2, 17, 3, 8
    dtype = jnp.float64
    q, k, v, g_raw, beta, A_log, dt_bias, _ = _make_inputs(B, T, H, K, 4, dtype, seed=0)
    del q, k, v, beta

    gate = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias, output_dtype=jnp.float64)
    manual = -jnp.exp(A_log.astype(jnp.float64)).reshape(H, 1) * jax.nn.softplus(
        g_raw + dt_bias.astype(jnp.float64).reshape(H, K)
    )
    assert compare_tensor("kda_gate", manual, gate, atol=1e-12, rtol=1e-12, compare_dtype=np.float64)

    gate_lb = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias, lower_bound=-5.0, output_dtype=jnp.float64)
    manual_lb = -5.0 * jax.nn.sigmoid(
        jnp.exp(A_log.astype(jnp.float64)).reshape(H, 1)
        * (g_raw + dt_bias.astype(jnp.float64).reshape(H, K))
    )
    assert compare_tensor("kda_gate_lower_bound", manual_lb, gate_lb, atol=1e-12, rtol=1e-12, compare_dtype=np.float64)


# ============================================================================
# 2. Gate chunk cumsum verification (no GPU needed)
# ============================================================================


def test_kda_gate_chunk_cumsum_matches_manual():
    B, T, H, K = 1, 19, 2, 8
    dtype = jnp.float32
    q, k, v, g_raw, beta, A_log, dt_bias, _ = _make_inputs(B, T, H, K, 4, dtype, seed=1)
    del q, k, v, beta
    cu_seqlens = jnp.array([0, 7, 19], dtype=jnp.int32)

    gated = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias, output_dtype=jnp.float32)
    expected = chunk_local_cumsum(gated, 8, cu_seqlens=cu_seqlens)
    actual = kda_gate_chunk_cumsum(
        g_raw,
        A_log,
        chunk_size=8,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        output_dtype=jnp.float32,
    )
    assert compare_tensor("kda_gate_chunk_cumsum", expected, actual, atol=1e-6, rtol=1e-6, compare_dtype=np.float64)


# ============================================================================
# 3. Dtype verification (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_fused_recurrent_kda_dtypes(dtype):
    B, T, H, K, V = 2, 17, 3, 8, 10
    q, k, v, g_raw, beta, A_log, dt_bias, h0 = _make_inputs(B, T, H, K, V, dtype, seed=123, h0=True)
    o, ht = fused_recurrent_kda(
        q, k, v, g_raw, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
    )
    acc = jnp.float64 if dtype == jnp.float64 else jnp.float32
    assert o.dtype == v.dtype
    assert ht.dtype == acc


def test_fused_recurrent_kda_output_dtype_follows_v_or_out():
    B, T, H, K, V = 1, 5, 2, 8, 6
    q = jax.random.normal(jax.random.PRNGKey(101), (B, T, H, K), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(102), (B, T, H, K), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(103), (B, T, H, V), dtype=jnp.float32)
    g = jax.random.normal(jax.random.PRNGKey(104), (B, T, H, K), dtype=jnp.bfloat16)
    beta = jax.nn.sigmoid(
        jax.random.normal(jax.random.PRNGKey(105), (B, T, H), dtype=jnp.bfloat16)
    ).astype(jnp.bfloat16)
    A_log = jnp.zeros((H,), dtype=jnp.float32)
    dt_bias = jnp.zeros((H * K,), dtype=jnp.float32)

    o_v, _ = fused_recurrent_kda_fwd(
        q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        use_gate_in_kernel=True,
        inplace_final_state=False,
    )
    assert o_v.dtype == v.dtype

    out = jnp.zeros_like(v, dtype=jnp.bfloat16)
    o_out, _ = fused_recurrent_kda_fwd(
        q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        use_gate_in_kernel=True,
        out=out,
        inplace_final_state=False,
    )
    assert o_out.dtype == out.dtype


# ============================================================================
# 4. Cross-validation: fused_recurrent vs naive_recurrent (no GPU needed)
# ============================================================================


@pytest.mark.parametrize("cfg", _CROSS_SHAPES, ids=[_shape_id(c) for c in _CROSS_SHAPES])
def test_fused_recurrent_kda_matches_naive_fp64(cfg):
    q, k, v, g_raw, beta, A_log, dt_bias, h0 = _make_inputs(
        cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"],
        jnp.float64, seed=cfg["seed"], h0=cfg.get("h0", False),
    )
    g = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias, output_dtype=jnp.float64)
    o_ref, s_ref = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    o, s = fused_recurrent_kda(
        q, k, v, g_raw, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
    )
    assert compare_tensor("fused_recurrent_kda_o", o_ref, o, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("fused_recurrent_kda_s", s_ref, s, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


# ============================================================================
# 5. Wrapper equivalence: fused_recurrent_kda_fwd vs fused_recurrent_kda
# ============================================================================


def test_fused_recurrent_kda_fwd_matches_public_api():
    q, k, v, g_raw, beta, A_log, dt_bias, h0 = _make_inputs(2, 33, 4, 16, 12, jnp.float64, seed=22, h0=True)
    o1, s1 = fused_recurrent_kda_fwd(
        q, k, v, g_raw, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
        inplace_final_state=False,
    )
    o2, s2 = fused_recurrent_kda(
        q, k, v, g_raw, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
    )
    assert compare_tensor("fused_recurrent_kda_fwd_o", o1, o2, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("fused_recurrent_kda_fwd_s", s1, s2, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


# ============================================================================
# 6. Varlen: cu_seqlens vs segmented calls (no GPU needed)
# ============================================================================


def test_fused_recurrent_kda_varlen_matches_segmented():
    q, k, v, g_raw, beta, A_log, dt_bias, _ = _make_inputs(1, 15, 3, 8, 6, jnp.float64, seed=3, h0=False)
    cu_seqlens = jnp.array([0, 4, 9, 15], dtype=jnp.int32)
    h0 = jax.random.normal(jax.random.PRNGKey(99), (3, 3, 8, 6), dtype=jnp.float64)

    g = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias, output_dtype=jnp.float64)
    o_ref, s_ref = _segment_reference(
        naive_recurrent_kda,
        q, k, v, g, beta,
        cu_seqlens,
        initial_state=h0,
    )
    o, s = fused_recurrent_kda(
        q, k, v, g_raw, beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_gate_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    assert compare_tensor("fused_recurrent_kda_varlen_o", o_ref, o, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)
    assert compare_tensor("fused_recurrent_kda_varlen_s", s_ref, s, atol=1e-10, rtol=1e-10, compare_dtype=np.float64)


# ============================================================================
# 7. CPU ref vs FLA Triton (GPU, when available)
# ============================================================================


@requires_triton
@pytest.mark.parametrize(
    "torch_dtype,jax_dtype",
    [
        (torch.float16, jnp.float16),
        (torch.bfloat16, jnp.bfloat16),
        (torch.float32, jnp.float32),
        # Triton kernel computes in fp32 internally; CPU ref runs fp64 when inputs are fp64.
        # This makes fp64 a "higher-precision reference" rather than a kernel-equivalence test.
        (torch.float64, jnp.float64),
    ],
    ids=["float16", "bfloat16", "float32", "float64"],
)
def test_cpu_ref_vs_triton_fused_recurrent(torch_dtype, jax_dtype):
    """Compare JAX CPU ref vs FLA Triton fused_recurrent_kda (forward only)."""
    assert torch is not None

    B, T, H, K, V = 2, 64, 4, 32, 64
    HV = H
    dtype = torch_dtype

    if dtype is torch.float64:
        pytest.skip("float64 is a higher-precision CPU reference; Triton kernel uses fp32 math internally.")

    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)

    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype, generator=gen)
    v = torch.randn(B, T, HV, V, device="cuda", dtype=dtype, generator=gen)
    g_raw = torch.randn(B, T, HV, K, device="cuda", dtype=dtype, generator=gen)
    beta = torch.rand(B, T, HV, device="cuda", dtype=dtype, generator=gen).sigmoid()
    A_log = torch.log(torch.empty(H, device="cuda", dtype=torch.float32).uniform_(1.0, 4.0, generator=gen))
    dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32, generator=gen)
    h0 = torch.randn(B, HV, K, V, device="cuda", dtype=torch.float32, generator=gen)

    # Pre-compute gate via the CPU ref helper, then convert to torch.
    # The FLA Triton fused_recurrent_kda does NOT support use_gate_in_kernel
    # (it ignores A_log/dt_bias via **kwargs), so we must pre-compute the gate
    # and pass it directly to both implementations.
    g_raw_j = torch_to_jax(g_raw, dtype=jnp.float32)
    A_log_j = torch_to_jax(A_log, dtype=jnp.float32)
    dt_bias_j = torch_to_jax(dt_bias, dtype=jnp.float32)
    gate_j = fused_kda_gate(g_raw_j, A_log_j, dt_bias=dt_bias_j, output_dtype=jnp.float32)
    gate_torch = torch.tensor(np.array(gate_j), device="cuda", dtype=dtype)

    # Triton path — g is the pre-computed log-space gate
    o_triton, ht_triton = triton_fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gate_torch,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    # JAX CPU ref path — also pass pre-computed gate with use_gate_in_kernel=False
    q_j = torch_to_jax(q, dtype=jax_dtype)
    k_j = torch_to_jax(k, dtype=jax_dtype)
    v_j = torch_to_jax(v, dtype=jax_dtype)
    gate_j_typed = gate_j.astype(jax_dtype)
    beta_j = torch_to_jax(beta, dtype=jax_dtype)
    h0_j = torch_to_jax(h0, dtype=jnp.float32)

    o_cpu, ht_cpu = fused_recurrent_kda(
        q_j,
        k_j,
        v_j,
        gate_j_typed,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )

    # Compare in fp64 space (compare_tensor handles dtype/ulp tolerances)
    if dtype is torch.float32:
        # fp32 accumulation over T=64 recurrence steps can drift; allow modest slack.
        atol = 5e-4
        rtol = 5e-4
    else:
        # bf16/fp16: allow a bit more numerical drift vs fp32 CPU ref
        atol = 5e-2
        rtol = 5e-2
    assert compare_tensor(
        "o", torch_to_jax(o_triton), o_cpu,
        atol=atol, rtol=rtol, compare_dtype=np.float64,
    )
    assert compare_tensor(
        "ht", torch_to_jax(ht_triton), ht_cpu,
        atol=atol, rtol=rtol, compare_dtype=np.float64,
    )
