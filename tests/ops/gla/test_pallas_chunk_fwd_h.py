"""chunk_fwd_h: Pallas kernel vs Jax TPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from src.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_fwd_h_ref
from tests.utils import compare_tensor


PALLAS_CASES = [
    dict(
        B=1,
        T=1024,
        H=4,
        K=128,
        chunk_size=64,
        cu_seqlens=[0, 128, 256, 512, 1024],
        seed=11,
    ),
    dict(
        B=1,
        T=1024,
        H=4,
        K=256,
        chunk_size=64,
        cu_seqlens=[0, 128, 256, 512, 1024],
        seed=11,
    ),
    dict(
        B=4,
        T=512,
        H=4,
        K=256,
        chunk_size=64,
        seed=11,
    ),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}"]
    if c.get("cu_seqlens"):
        parts.append(f"segs{len(c['cu_seqlens']) - 1}")
    if c.get("chunk_size") is not None:
        parts.append(f"chunk{c['chunk_size']}")
    return "-".join(parts)


# ============================================================================
# Helpers
# ============================================================================


def _to_jax_cu_seqlens(cu_seqlens) -> jax.Array | None:
    if cu_seqlens is None:
        return None
    return jnp.asarray(cu_seqlens, dtype=jnp.int32)


def _run_tpu(
    k,
    v,
    gk=None,
    h0=None,
    chunk_size=64,
    *,
    cu_seqlens=None,
):

    cu = _to_jax_cu_seqlens(cu_seqlens)
    h, ht = chunk_fwd_h_ref(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens_cpu=cu,
        output_final_state=True,
    )

    return h, ht


def _run_pallas(
    k,
    v,
    gk=None,
    h0=None,
    chunk_size=64,
    *,
    cu_seqlens=None,
):
    cu = _to_jax_cu_seqlens(cu_seqlens)
    h, ht = chunk_fwd_h_kernel(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens=cu,
        output_final_state=True,
    )
    if cu is None:
        h = h.reshape(k.shape[0], -1, k.shape[2], k.shape[3], v.shape[-1])
    return h, ht


# ============================================================================
# Parametrized test — native vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_native_tpu_vs_pallas(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    atol = cfg.get("atol", 1e-6)
    rtol = cfg.get("rtol", 1e-6)
    chunk_size = cfg.get("chunk_size", 64)
    cu = cfg.get("cu_seqlens", None)
    N = 1
    if cu is not None:
        N = len(cu) - 1
    else:
        N = B
    key = jax.random.PRNGKey(1)
    k = jax.random.normal(key, (B, T, H, K))
    v = jax.random.normal(key, (B, T, H, K))
    gk = jax.random.normal(key, (B, T, H, K))
    h0 = jax.random.normal(key, (N, H, K, K))

    h, ht = _run_tpu(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )

    pallas_h, pallas_ht = _run_pallas(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )
    assert compare_tensor("h", h, pallas_h, atol=atol, rtol=rtol)
    assert compare_tensor("ht", ht, pallas_ht, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
