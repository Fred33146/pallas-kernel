"""Tests for chunk_local_cumsum with cu_seqlens (chunked layout input)."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tops.cpu.ops.common.chunk_o import chunk_local_cumsum
from tops.cpu.ops.common.utils import gather_chunks
from tops.utils import prepare_chunk_indices


def test_cumsum_chunked_layout_independent():
    """Each chunk's cumsum is independent in chunked layout."""
    C = 16
    # Two segments: ones and twos
    g = jnp.concatenate([jnp.ones((1, 16, 2, 8)), jnp.ones((1, 16, 2, 8)) * 2], axis=1)
    cu = jnp.array([0, 16, 32])
    ci = prepare_chunk_indices(cu, C)
    g_c, _ = gather_chunks(g, cu, ci, C)
    # g_c: [2, 16, 2, 8]

    result = chunk_local_cumsum(g_c, C, cu_seqlens=cu)
    # Chunk 0: cumsum of ones = [1, 2, ..., 16]
    assert jnp.allclose(result[0, :, 0, 0], jnp.arange(1, 17, dtype=jnp.float32))
    # Chunk 1: cumsum of twos = [2, 4, ..., 32]
    assert jnp.allclose(result[1, :, 0, 0], jnp.arange(1, 17, dtype=jnp.float32) * 2)


def test_cumsum_chunked_matches_per_chunk():
    """Chunked layout cumsum matches doing cumsum on each chunk independently."""
    C = 16
    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (1, 48, 2, 8))
    cu = jnp.array([0, 16, 48])
    ci = prepare_chunk_indices(cu, C)
    g_c, _ = gather_chunks(g, cu, ci, C)  # [3, 16, 2, 8]

    result = chunk_local_cumsum(g_c, C, cu_seqlens=cu)

    # Each chunk independently
    for i in range(3):
        expected = jnp.cumsum(g_c[i], axis=0)
        assert jnp.allclose(result[i], expected)


def test_cumsum_chunked_3d():
    """Works for 3D chunked layout [total_NT, C, H]."""
    C = 16
    g = jnp.ones((1, 32, 4))
    cu = jnp.array([0, 16, 32])
    ci = prepare_chunk_indices(cu, C)
    g_c, _ = gather_chunks(g, cu, ci, C)  # [2, 16, 4]
    result = chunk_local_cumsum(g_c, C, cu_seqlens=cu)
    assert result.shape == (2, 16, 4)
    assert jnp.allclose(result[0, 0, 0], 1.0)


def test_cumsum_packed_unchanged():
    """cu_seqlens=None with packed layout behaves identically to original."""
    C = 16
    key = jax.random.PRNGKey(99)
    g = jax.random.normal(key, (2, 32, 4, 8))
    r1 = chunk_local_cumsum(g, C, cu_seqlens=None)
    r2 = chunk_local_cumsum(g, C)
    assert jnp.allclose(r1, r2)
