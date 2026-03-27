"""Tests for gather_chunks / scatter_chunks."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tops.cpu.ops.common.utils import gather_chunks, scatter_chunks
from tops.utils import prepare_chunk_indices


def test_gather_aligned():
    """Chunk-aligned segments: gather all chunks correctly."""
    q = jnp.arange(32).reshape(1, 32, 1, 1).astype(jnp.float32)
    cu = jnp.array([0, 16, 32])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(q, cu, ci, 16)
    assert chunks.shape == (2, 16, 1, 1)
    assert jnp.all(vlens == 16)
    assert jnp.allclose(chunks[0, :, 0, 0], jnp.arange(16, dtype=jnp.float32))
    assert jnp.allclose(chunks[1, :, 0, 0], jnp.arange(16, 32, dtype=jnp.float32))


def test_gather_unaligned():
    """Non-aligned last chunk: valid positions filled, rest zero."""
    q = jnp.ones((1, 30, 2, 8))
    cu = jnp.array([0, 10, 30])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(q, cu, ci, 16)
    assert chunks.shape == (3, 16, 2, 8)
    assert int(vlens[0]) == 10
    assert jnp.all(chunks[0, :10] == 1)
    assert jnp.all(chunks[0, 10:] == 0)
    assert int(vlens[1]) == 16
    assert int(vlens[2]) == 4
    assert jnp.all(chunks[2, 4:] == 0)


def test_gather_3d():
    """Works for 3D [1, T, H] tensors (Simple GLA g)."""
    g = jnp.ones((1, 30, 4))
    cu = jnp.array([0, 10, 30])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(g, cu, ci, 16)
    assert chunks.shape == (3, 16, 4)


def test_scatter_roundtrip():
    """gather then scatter recovers original (for valid positions)."""
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (1, 30, 2, 8))
    cu = jnp.array([0, 10, 30])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(q, cu, ci, 16)
    buf = jnp.zeros_like(q)
    recovered = scatter_chunks(buf, chunks, cu, ci, 16, vlens)
    assert jnp.allclose(recovered, q)


def test_scatter_3d():
    """scatter works for 3D tensors."""
    key = jax.random.PRNGKey(1)
    g = jax.random.normal(key, (1, 30, 4))
    cu = jnp.array([0, 10, 30])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(g, cu, ci, 16)
    buf = jnp.zeros_like(g)
    recovered = scatter_chunks(buf, chunks, cu, ci, 16, vlens)
    assert jnp.allclose(recovered, g)


def test_single_token_segment():
    """Segment of length 1: one chunk with valid_len=1."""
    q = jnp.ones((1, 17, 2, 8))
    cu = jnp.array([0, 1, 17])
    ci = prepare_chunk_indices(cu, 16)
    chunks, vlens = gather_chunks(q, cu, ci, 16)
    assert int(vlens[0]) == 1
    assert jnp.all(chunks[0, 1:] == 0)
