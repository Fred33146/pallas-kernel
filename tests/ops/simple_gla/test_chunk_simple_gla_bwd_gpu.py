"""chunk_simple_gla bwd: FLA Triton GPU (gold via autograd) vs JAX Pallas kernel.

Tests simple_gla with three gate modes:
  - g only: per-head scalar gate [B, T, H]
  - g_gamma only: fixed per-head log-decay [H]
  - g + g_gamma: both combined
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

os.environ["TRITON_F32_DEFAULT"] = "ieee"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import jax
import jax.numpy as jnp

from tops.ops.simple_gla.chunk import chunk_simple_gla_bwd as pallas_bwd
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from fla.ops.simple_gla.chunk import chunk_simple_gla as triton_simple_gla
assert DEVICE == "cuda", "This test requires CUDA. Please run on a machine with an NVIDIA GPU."

# ============================================================================
# Test configs
#
# Constraints for JAX Pallas backward:
#   - K, V must be multiples of 128
#   - T must be a multiple of chunk_size (default 64)
#   - gate: only "gamma" (chunk_simple_gla_bwd does not support g)
# ============================================================================

CASES = [
    # ── g only ──
    # dict(B=2, T=64, H=4, K=32, V=64, seed=42, gate="g"),
    # dict(B=1, T=128, H=2, K=64, V=128, seed=7, gate="g"),
    # dict(B=2, T=64, H=1, K=32, V=64, seed=10, gate="g"),
    # dict(B=2, T=64, H=4, K=16, V=128, seed=20, gate="g"),
    # dict(B=2, T=128, H=4, K=16, V=32, seed=40, gate="g"),
    # dict(B=1, T=256, H=2, K=32, V=64, seed=300, gate="g"),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=13, gate="g", h0=True),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=14, gate="g", dht=True),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=15, gate="g", h0=True, dht=True),
    # dict(B=2, T=100, H=4, K=32, V=64, seed=400, gate="g"),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=500, gate="g", chunk_size=16),
    # ── g_gamma only ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42, gate="gamma"),
    dict(B=1, T=128, H=2, K=128, V=128, seed=7, gate="gamma"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=13, gate="gamma", h0=True),
    dict(B=2, T=64, H=4, K=128, V=128, seed=15, gate="gamma", h0=True, dht=True),
    dict(B=2, T=128, H=4, K=128, V=128, seed=400, gate="gamma"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=600, gate="gamma", dht=True),
    dict(B=1, T=256, H=2, K=128, V=128, seed=601, gate="gamma", h0=True, dht=True),
    # ── larger shapes (keep memory reasonable for GPU autograd) ──
    dict(B=1, T=512, H=4, K=128, V=128, seed=500, gate="gamma"),
    dict(B=1, T=512, H=4, K=128, V=128, seed=501, gate="gamma", h0=True),
    dict(B=1, T=512, H=4, K=128, V=128, seed=502, gate="gamma", h0=True, dht=True),
    # ── both g + g_gamma ──
    # dict(B=2, T=64, H=4, K=32, V=64, seed=42, gate="both"),
    # dict(B=1, T=128, H=2, K=64, V=128, seed=7, gate="both"),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=15, gate="both", h0=True, dht=True),
    # dict(B=1, T=256, H=2, K=32, V=64, seed=300, gate="both"),
    # dict(B=2, T=100, H=4, K=32, V=64, seed=400, gate="both"),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=500, gate="both", chunk_size=16),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}_{c['gate']}"]
    cs = c.get("chunk_size", 64)
    if cs != 64:
        parts.append(f"C{cs}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _torch_to_jax(t: torch.Tensor, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    """Convert torch tensor to JAX array (float32)."""
    np_arr = t.detach().cpu().float().numpy()
    return jnp.array(np_arr, dtype=dtype)


# ============================================================================
# Main test
# ============================================================================


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_triton_vs_pallas_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)
    gate_mode = cfg["gate"]

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    do = torch.randn(B, T, H, V, dtype=torch.bfloat16)

    # Gates
    g_gamma = -torch.rand(H).abs() * 0.5 if gate_mode in ("gamma", "both") else None

    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # ── Triton gold (autograd) ──
    q_g = q.clone().to(DEVICE).requires_grad_()
    k_g = k.clone().to(DEVICE).requires_grad_()
    v_g = v.clone().to(DEVICE).requires_grad_()
    g_gamma_g = g_gamma.clone().to(DEVICE) if g_gamma is not None else None
    h0_g = h0.clone().to(DEVICE).requires_grad_() if h0 is not None else None
    do_g = do.clone().to(DEVICE)
    dht_g = dht.clone().to(DEVICE) if dht is not None else None

    output_final_state = dht is not None
    o_g, ht_g = triton_simple_gla(
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

    # ── Pallas backward ──
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    do_j = _torch_to_jax(do)
    g_gamma_j = _torch_to_jax(g_gamma, dtype=jnp.float32) if g_gamma is not None else None
    h0_j = _torch_to_jax(h0, dtype=jnp.float32) if h0 is not None else None
    dht_j = _torch_to_jax(dht, dtype=jnp.float32) if dht is not None else None

    dq_pl, dk_pl, dv_pl, dh0_pl = pallas_bwd(
        q_j, k_j, v_j, do_j,
        g_gamma=g_gamma_j,
        scale=scale,
        h0=h0_j,
        dht=dht_j,
        chunk_size=chunk_size,
    )

    # ── Compare ──
    # Both sides do bf16×bf16 matmul with f32 accumulation, but differ in
    # accumulation order (GPU tensor cores vs JAX interpret-mode sequential).
    # dht cases amplify dh magnitude, increasing cross-platform bf16 difference.
    NT = T // chunk_size
    atol = cfg.get("atol", min(5e-1, 5e-2 * max(NT, 1)))
    rtol = cfg.get("rtol", 5e-2)
    max_ulp = cfg.get("max_ulp", 40)
    assert compare_tensor("dq", dq_gold, dq_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dk", dk_gold, dk_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)
    assert compare_tensor("dv", dv_gold, dv_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)

    if dh0_gold is not None and dh0_pl is not None:
        assert compare_tensor("dh0", dh0_gold, dh0_pl, atol=atol, rtol=rtol, max_ulp=max_ulp)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
