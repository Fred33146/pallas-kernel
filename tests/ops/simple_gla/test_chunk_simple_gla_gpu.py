"""chunk_simple_gla: forward (Triton vs JAX) & backward (Triton autograd vs JAX Pallas)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.chunk import chunk_simple_gla_fwd, chunk_simple_gla_bwd
from tests.utils import compare_tensor



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "This test requires CUDA. Please run on a machine with an NVIDIA GPU."

from fla.ops.simple_gla.chunk import chunk_simple_gla as triton_chunk_simple_gla

fp32_atol = 1e-4
fp32_rtol = 1e-4
bf16_atol = 1e-3
bf16_rtol = 1e-3

# ============================================================================
# Test configs
#
# Constraints for JAX chunk path:
#   - K, V must be multiples of 128 (chunk_fwd_h_kernel Pallas requirement)
#   - T must be a multiple of chunk_size (default 64)
#   - gate: only "g_gamma" or "none" (JAX chunk_fwd_h does not support scalar g yet)
# ============================================================================

CHUNK_SIZE = 64

CASES = [
    # ── standard shapes ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7),
    # ── with h0 ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, h0=True),
    dict(B=2, T=128, H=4, K=128, V=128, seed=13, h0=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=7, h0=True),
    # ── single head ──
    dict(B=2, T=64, H=1, K=128, V=128, seed=10),
    # ── K != V (both multiples of 128) ──
    dict(B=2, T=64, H=4, K=128, V=256, seed=20),
    dict(B=2, T=64, H=4, K=256, V=128, seed=21),
    # ── minimal T = chunk_size ──
    dict(B=2, T=64, H=2, K=128, V=128, seed=30),
    # ── large batch ──
    dict(B=8, T=64, H=4, K=128, V=128, seed=50),
    # ── many heads ──
    dict(B=1, T=128, H=16, K=128, V=128, seed=60),
    # ── no gate ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=100, gate="none"),
    dict(B=2, T=128, H=4, K=128, V=128, seed=101, gate="none"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=102, gate="none", h0=True),
    # ── g_gamma only ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=150, gate="g_gamma"),
    dict(B=2, T=128, H=4, K=128, V=128, seed=151, gate="g_gamma"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=152, gate="g_gamma", h0=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=153, gate="g_gamma"),
    dict(B=1, T=64, H=2, K=128, V=256, seed=154, gate="g_gamma"),
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=200, scale=0.1),
    dict(B=2, T=64, H=4, K=128, V=128, seed=201, scale=0.1, h0=True),
    # ── longer sequences ──
    dict(B=1, T=512, H=2, K=128, V=128, seed=300),
    dict(B=1, T=512, H=2, K=128, V=128, seed=301, h0=True),
    dict(B=1, T=512, H=2, K=128, V=128, seed=302, gate="g_gamma"),
    dict(B=1, T=512, H=2, K=128, V=128, seed=303, gate="g_gamma", h0=True),
    # ── multi-batch + long ──
    dict(B=4, T=256, H=2, K=128, V=128, seed=360),
    dict(B=2, T=256, H=4, K=128, V=128, seed=361, gate="g_gamma"),
    dict(B=2, T=4096, H=16, K=128, V=128, seed=151, gate="g_gamma", h0=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "none")
    if gate != "none":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Convert torch tensor to JAX array, preserving bfloat16 dtype."""
    # cpu_device = jax.devices("cpu")[0]
    np_arr = t.detach().cpu().float().numpy()
    jax_arr = jnp.array(np_arr)
    if t.dtype == torch.bfloat16:
        jax_arr = jax_arr.astype(jnp.bfloat16)
    # return jax.device_put(jax_arr, cpu_device)
    return jax_arr


def _run_triton(q, k, v, *, g=None, g_gamma=None, h0=None, scale=None):
    assert DEVICE == "cuda", "Triton implementation requires CUDA"
    kwargs = dict(output_final_state=True)
    if g is not None:
        kwargs["g"] = g.to(DEVICE)
    if g_gamma is not None:
        kwargs["g_gamma"] = g_gamma.to(DEVICE)
    if h0 is not None:
        kwargs["initial_state"] = h0.to(DEVICE)
    if scale is not None:
        kwargs["scale"] = scale
    o, ht = triton_chunk_simple_gla(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), **kwargs
    )
    return o.cpu(), ht.cpu() if ht is not None else None


def _run_triton_bwd(q, k, v, do, *, g_gamma=None, h0=None, dht=None, scale=None):
    """Run Triton forward + autograd backward to get gold gradients."""
    q_g = q.clone().to(DEVICE).requires_grad_()
    k_g = k.clone().to(DEVICE).requires_grad_()
    v_g = v.clone().to(DEVICE).requires_grad_()
    g_gamma_g = g_gamma.clone().to(DEVICE) if g_gamma is not None else None
    h0_g = h0.clone().to(DEVICE).requires_grad_() if h0 is not None else None
    do_g = do.clone().to(DEVICE)
    dht_g = dht.clone().to(DEVICE) if dht is not None else None

    output_final_state = dht is not None
    o_g, ht_g = triton_chunk_simple_gla(
        q_g, k_g, v_g, g=None, g_gamma=g_gamma_g, scale=scale,
        initial_state=h0_g, output_final_state=output_final_state,
    )

    loss = (o_g * do_g).sum()
    if dht_g is not None and ht_g is not None:
        loss = loss + (ht_g * dht_g).sum()
    loss.backward()

    dq_gold = q_g.grad.cpu()
    dk_gold = k_g.grad.cpu()
    dv_gold = v_g.grad.cpu()
    dh0_gold = h0_g.grad.cpu() if h0_g is not None else None
    return dq_gold, dk_gold, dv_gold, dh0_gold


def _run_jax_chunk(q, k, v, *, g_gamma=None, h0=None, scale=None,
                   chunk_size=CHUNK_SIZE):
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    g_gamma_j = _torch_to_jax(g_gamma) if g_gamma is not None else None
    h0_j = _torch_to_jax(h0) if h0 is not None else None

    # Use interpret=True on CPU (Pallas kernels require it)
    interpret = jax.default_backend() == "cpu"
    o, ht = chunk_simple_gla_fwd(
        q_j, k_j, v_j,
        g=None,
        g_gamma=g_gamma_j,
        scale=scale,
        h0=h0_j,
        use_ht=True,
        chunk_size=chunk_size,
    )
    return o, ht


# ============================================================================
# Parametrized test — Triton chunk (gold) vs JAX chunk
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_triton_chunk_vs_jax_chunk(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    # bf16 matmul accumulation order differs between CUDA (Triton) and JAX;
    # errors compound across chunks, so scale atol with NT.
    NT = T // CHUNK_SIZE
    atol = cfg.get("atol", min(5e-2, 5e-3 * NT))
    rtol = cfg.get("rtol", 1e-2)
    max_ulp = 2
    gate = cfg.get("gate", "none")
    scale = cfg.get("scale", None)

    dtype = torch.bfloat16
    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)

    g_gamma = None
    if gate == "g_gamma":
        g_gamma = -torch.nn.functional.softplus(torch.rand(H).abs() * 0.5)  # negative log-decay

    N = B
    h0 = torch.randn(N, H, K, V, dtype=dtype) if cfg.get("h0") else None

    o_tri, ht_tri = _run_triton(q, k, v, g=None, g_gamma=g_gamma, h0=h0, scale=scale)
    o_jax, ht_jax = _run_jax_chunk(q, k, v, g_gamma=g_gamma, h0=h0, scale=scale)

    assert compare_tensor("output", o_tri, o_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if ht_tri is not None and ht_jax is not None:
        assert compare_tensor("final_state", ht_tri, ht_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)


# ============================================================================
# Backward: JAX Pallas vs Triton autograd (g_gamma only)
# ============================================================================

BWD_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7),
    dict(B=2, T=64, H=4, K=128, V=128, seed=13, h0=True),
    dict(B=2, T=64, H=1, K=128, V=128, seed=10),
    dict(B=2, T=64, H=4, K=128, V=128, seed=20),
    dict(B=2, T=64, H=4, K=128, V=128, seed=21),
    dict(B=2, T=128, H=4, K=128, V=128, seed=400),
    dict(B=1, T=64, H=2, K=128, V=128, seed=41),
    dict(B=1, T=256, H=2, K=128, V=128, seed=300),
    dict(B=2, T=64, H=4, K=128, V=128, seed=99),
    dict(B=2, T=128, H=4, K=128, V=128, seed=503, chunk_size=64),
    # ── with dht (terminal state gradient) ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=600, dht=True),
    dict(B=1, T=128, H=2, K=128, V=128, seed=601, dht=True),
    dict(B=2, T=64, H=4, K=128, V=128, seed=602, h0=True, dht=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=603, h0=True, dht=True),
]


def _bwd_case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_bwd_case_id(c) for c in BWD_CASES])
def test_simple_gla_bwd_gamma(cfg):
    """JAX chunk_simple_gla_bwd should match Triton autograd backward (g_gamma only)."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = cfg.get("chunk_size", 64)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    g_gamma = -torch.nn.functional.softplus(torch.rand(H).abs() * 0.5)  # negative log-decay

    N = B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(N, H, K, V) if cfg.get("dht") else None

    # Triton gold (autograd)
    dq_gold, dk_gold, dv_gold, dh0_gold = _run_triton_bwd(
        q, k, v, do,
        g_gamma=g_gamma,
        h0=h0,
        dht=dht,
        scale=scale,
    )

    # JAX Pallas
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    do_j = _torch_to_jax(do)
    g_gamma_j = _torch_to_jax(g_gamma)
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    dht_j = _torch_to_jax(dht) if dht is not None else None

    dq_jax, dk_jax, dv_jax, dh0_jax = chunk_simple_gla_bwd(
        q_j, k_j, v_j, do_j,
        g_gamma=g_gamma_j,
        scale=scale,
        h0=h0_j,
        dht=dht_j,
        chunk_size=C,
    )

    NT = T // C
    atol = cfg.get("atol", min(5e-2, 5e-2 * max(NT, 1)))
    rtol = cfg.get("rtol", 5e-2)
    # bf16 cross-platform (Triton GPU vs JAX interpret) backward comparison:
    # accumulation order differs fundamentally, dht cases amplify dh magnitude.
    max_ulp = cfg.get("max_ulp", 40)
    assert compare_tensor("dq", dq_gold, dq_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dk", dk_gold, dk_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dv", dv_gold, dv_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)
    if dh0_gold is not None and dh0_jax is not None:
        assert compare_tensor("dh0", dh0_gold, dh0_jax, atol=atol, rtol=rtol, max_ulp=max_ulp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
