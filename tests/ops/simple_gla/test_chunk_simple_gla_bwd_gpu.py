"""chunk_simple_gla bwd: FLA Triton GPU (gold via autograd) vs Torch CPU reference.

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
import torch.nn.functional as F

from tests.src.ops.simple_gla.chunk import chunk_simple_gla_bwd as cpu_bwd
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

triton_imports_available = False
try:
    from fla.ops.simple_gla.chunk import chunk_simple_gla as triton_simple_gla
    triton_imports_available = True
except ImportError:
    pass

print(f"Testing on device: {DEVICE}")
print(f"Triton imports available: {triton_imports_available}")
requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Test configs
# ============================================================================

# gate_mode: "g", "gamma", "both"
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
    dict(B=2, T=64, H=4, K=32, V=64, seed=42, gate="gamma"),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7, gate="gamma"),
    dict(B=2, T=64, H=4, K=32, V=64, seed=13, gate="gamma", h0=True),
    dict(B=2, T=64, H=4, K=32, V=64, seed=15, gate="gamma", h0=True, dht=True),
    dict(B=2, T=100, H=4, K=32, V=64, seed=400, gate="gamma"),
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
# Main test
# ============================================================================


@requires_triton
@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_gold_vs_cpu(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    chunk_size = cfg.get("chunk_size", 64)
    gate_mode = cfg["gate"]

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do = torch.randn(B, T, H, V)

    # Gates
    g_raw = F.logsigmoid(torch.randn(B, T, H)) if gate_mode in ("g", "both") else None
    g_gamma = -torch.rand(H).abs() * 0.5 if gate_mode in ("gamma", "both") else None

    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V) if cfg.get("dht") else None

    # ── Triton gold (autograd) ──
    q_g = q.clone().to(DEVICE).requires_grad_()
    k_g = k.clone().to(DEVICE).requires_grad_()
    v_g = v.clone().to(DEVICE).requires_grad_()
    g_g = g_raw.clone().to(DEVICE).requires_grad_() if g_raw is not None else None
    g_gamma_g = g_gamma.clone().to(DEVICE) if g_gamma is not None else None
    h0_g = h0.clone().to(DEVICE).requires_grad_() if h0 is not None else None
    do_g = do.clone().to(DEVICE)
    dht_g = dht.clone().to(DEVICE) if dht is not None else None

    output_final_state = dht is not None
    o_g, ht_g = triton_simple_gla(
        q_g, k_g, v_g, g=g_g, g_gamma=g_gamma_g, scale=scale,
        initial_state=h0_g, output_final_state=output_final_state,
    )

    loss = (o_g * do_g).sum()
    if dht_g is not None and ht_g is not None:
        loss = loss + (ht_g * dht_g).sum()
    loss.backward()

    dq_gold = q_g.grad.cpu()
    dk_gold = k_g.grad.cpu()
    dv_gold = v_g.grad.cpu()
    dg_gold = g_g.grad.cpu() if g_g is not None else None
    dh0_gold = h0_g.grad.cpu() if h0_g is not None else None

    # ── CPU reference ──
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = cpu_bwd(
        q.float(), k.float(), v.float(),
        g_raw.float() if g_raw is not None else None,
        g_gamma.float() if g_gamma is not None else None,
        scale, h0, do.float(), dht,
        chunk_size=chunk_size,
    )

    # ── Compare ──
    atol, rtol = 5e-5, 5e-5
    assert compare_tensor("dq", dq_gold, dq_cpu, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_gold, dk_cpu, atol=atol, rtol=rtol)
    assert compare_tensor("dv", dv_gold, dv_cpu, atol=atol, rtol=rtol)

    if dg_gold is not None:
        assert compare_tensor("dg", dg_gold, dg_cpu, atol=atol, rtol=rtol)

    if cfg.get("h0"):
        assert compare_tensor("dh0", dh0_gold, dh0_cpu, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])