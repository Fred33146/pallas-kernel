"""chunk_simple_gla_bwd: Pallas JAX vs Torch CPU reference (g_gamma only)."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import torch
import jax
import jax.numpy as jnp

from tests.src.ops.simple_gla.chunk import chunk_simple_gla_bwd as cpu_bwd
from tops.ops.simple_gla.chunk import chunk_simple_gla_bwd as jax_bwd
from tests.utils import compare_tensor


CASES = [
    dict(B=2, T=64, H=4, K=32, V=64, seed=42),
    dict(B=1, T=128, H=2, K=64, V=128, seed=7),
    dict(B=2, T=64, H=4, K=32, V=64, seed=13, h0=True),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=14, dht=True),
    # dict(B=2, T=64, H=4, K=32, V=64, seed=15, h0=True, dht=True),
    dict(B=2, T=64, H=1, K=32, V=64, seed=10),
    dict(B=2, T=64, H=4, K=16, V=128, seed=20),
    dict(B=2, T=64, H=4, K=128, V=16, seed=21),
    # odd T (needs padding)
    dict(B=2, T=100, H=4, K=32, V=64, seed=400),
    dict(B=1, T=50, H=2, K=32, V=64, seed=41),
    # larger
    dict(B=1, T=256, H=2, K=32, V=64, seed=300),
    # dict(B=1, T=256, H=2, K=32, V=64, seed=303, h0=True, dht=True),
    dict(B=4, T=64, H=8, K=32, V=64, seed=99),
    # non-default chunk_size
    dict(B=2, T=128, H=4, K=32, V=64, seed=502, chunk_size=32),
    dict(B=2, T=128, H=4, K=32, V=64, seed=503, chunk_size=64),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    cs = c.get("chunk_size", 16)
    if cs != 16:
        parts.append(f"C{cs}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    return "-".join(parts)


def _torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().to(torch.float32).numpy())


@pytest.mark.parametrize("cfg", CASES, ids=[_case_id(c) for c in CASES])
def test_simple_gla_bwd_gamma(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K**-0.5)
    C = cfg.get("chunk_size", 16)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do = torch.randn(B, T, H, V)
    g_gamma = -torch.rand(H).abs() * 0.5  # negative log-decay

    N = B
    h0 = torch.randn(N, H, K, V) if cfg.get("h0") else None
    dht = torch.randn(N, H, K, V) if cfg.get("dht") else None

    # Torch CPU reference (g_gamma only, no g)
    dq_cpu, dk_cpu, dv_cpu, dg_cpu, dh0_cpu = cpu_bwd(
        q.float(), k.float(), v.float(),
        g=None,
        g_gamma=g_gamma.float(),
        scale=scale,
        initial_state=h0,
        do=do.float(),
        dht=dht,
        chunk_size=C,
    )

    # JAX Pallas
    q_j = _torch_to_jax(q)
    k_j = _torch_to_jax(k)
    v_j = _torch_to_jax(v)
    do_j = _torch_to_jax(do)
    g_gamma_j = _torch_to_jax(g_gamma)
    h0_j = _torch_to_jax(h0) if h0 is not None else None
    dht_j = _torch_to_jax(dht) if dht is not None else None

    dq_jax, dk_jax, dv_jax, dh0_jax = jax_bwd(
        q_j, k_j, v_j, g_gamma_j,
        scale=scale,
        initial_state=h0_j,
        do=do_j,
        dht=dht_j,
        chunk_size=C,
    )

    atol, rtol = 2e-5, 1e-5
    assert compare_tensor("dq", dq_cpu, dq_jax, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_cpu, dk_jax, atol=atol, rtol=rtol)
    assert compare_tensor("dv", dv_cpu, dv_jax, atol=atol, rtol=rtol)
    if dh0_cpu is not None:
        assert compare_tensor("dh0", dh0_cpu, dh0_jax, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
