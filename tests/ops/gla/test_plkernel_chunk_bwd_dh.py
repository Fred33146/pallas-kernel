"""chunk_bwd_dh: Pallas kernel vs Jax TPU reference tests."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import jax
import jax.numpy as jnp

from tops.ops.common.chunk_h import chunk_bwd_dh_kernel, chunk_bwd_dh_ref
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
    q,
    k,
    v,
    do,
    gk=None,
    dht=None,
    scale=1.0,
    chunk_size=64,
    *,
    cu_seqlens=None,
):

    cu = _to_jax_cu_seqlens(cu_seqlens)
    dh, dh0 = chunk_bwd_dh_ref(
        q,
        k,
        v,
        gk=gk,
        do=do,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens_cpu=cu,
    )

    return dh, dh0


def _run_pallas(
    q,
    k,
    v,
    do,
    gk=None,
    dht=None,
    scale=1.0,
    chunk_size=64,
    *,
    cu_seqlens=None,
):
    cu = _to_jax_cu_seqlens(cu_seqlens)
    dh, dh0 = chunk_bwd_dh_kernel(
        q,
        k,
        v,
        gk=gk,
        do=do,
        dht=dht,
        scale=scale,
        output_dh0=True,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )
    if cu is None:
        dh = dh.reshape(q.shape[0], -1, q.shape[2], q.shape[3], do.shape[-1])
    return dh, dh0


# ============================================================================
# Parametrized test — native vs Pallas
# ============================================================================


@pytest.mark.parametrize("cfg", PALLAS_CASES, ids=[_case_id(c) for c in PALLAS_CASES])
def test_native_tpu_vs_pallas(cfg):
    B, T, H, K = cfg["B"], cfg["T"], cfg["H"], cfg["K"]
    V = K # In these tests V == K
    atol = cfg.get("atol", 1e-5)
    rtol = cfg.get("rtol", 1e-5)
    chunk_size = cfg.get("chunk_size", 64)
    cu = cfg.get("cu_seqlens", None)
    N = 1
    if cu is not None:
        N = len(cu) - 1
    else:
        N = B
    key = jax.random.PRNGKey(cfg.get("seed", 11))
    
    q = jax.random.normal(key, (B, T, H, K))
    k = jax.random.normal(key, (B, T, H, K))
    v = jax.random.normal(key, (B, T, H, V))
    do = jax.random.normal(key, (B, T, H, V))
    gk = jax.random.normal(key, (B, T, H, K))
    dht = jax.random.normal(key, (N, H, K, V))
    scale = K ** -0.5

    dh_ref, dh0_ref = _run_tpu(
        q,
        k,
        v,
        do,
        gk=gk,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )

    pallas_dh, pallas_dh0 = _run_pallas(
        q,
        k,
        v,
        do,
        gk=gk,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )
    assert compare_tensor("dh", dh_ref, pallas_dh, atol=atol, rtol=rtol)
    assert compare_tensor("dh0", dh0_ref, pallas_dh0, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
