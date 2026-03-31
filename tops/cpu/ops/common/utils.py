"""Low-level utilities shared across CPU reference kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def cdiv(x: int, y: int) -> int:
    """Ceiling division of x by y."""
    return (x + y - 1) // y


def pad_to_multiple(x: jax.Array, multiple: int, axis: int) -> jax.Array:
    """Zero-pad array along the given axis so its length becomes a multiple of `multiple`.

    Args:
        x: Input array to pad.
        multiple: Target multiple for the axis length.
        axis: Axis along which to pad.

    Returns:
        Padded array (or original if already aligned).
    """
    length = x.shape[axis]
    remainder = length % multiple
    if remainder == 0:
        return x
    pad_len = multiple - remainder
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, pad_len)
    return jnp.pad(x, pad_widths)


def gather_chunks(
    tensor: jax.Array,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Read all chunks from a packed tensor into chunked layout.

    Uses vmap(dynamic_slice) for jit-friendly static-shape extraction.
    Out-of-bounds positions (partial chunks at segment ends) are zeroed.

    Args:
        tensor: [1, T_total, ...] packed tensor (any trailing dims).
        cu_seqlens: [N+1] cumulative sequence lengths.
        chunk_indices: [total_NT, 2] from prepare_chunk_indices.
        chunk_size: block size C.

    Returns:
        (chunks, valid_lens):
            chunks: [total_NT, C, ...] — zeroed at invalid positions.
            valid_lens: [total_NT] — number of valid tokens per chunk.
    """
    C = chunk_size
    T_total = tensor.shape[1]
    trailing_shape = tensor.shape[2:]

    # Pad tensor to T_total + C - 1 so that dynamic_slice(start, size=C) is
    # in-bounds for every valid chunk start (which can be as large as T_total-1).
    T_padded = T_total + C - 1
    pad_widths = [(0, 0)] * tensor.ndim
    pad_widths[1] = (0, T_padded - T_total)
    tensor_padded = jnp.pad(tensor, pad_widths)

    def read_one(idx):
        seq_id, local_t = idx[0], idx[1]
        bos = cu_seqlens[seq_id]
        eos = cu_seqlens[seq_id + 1]
        start = bos + local_t * C
        valid_len = jnp.minimum(C, eos - start)
        slices = (0, start) + (0,) * len(trailing_shape)
        sizes = (1, C) + trailing_shape
        chunk = lax.dynamic_slice(tensor_padded, slices, sizes)[0]
        # Mask invalid positions
        mask = jnp.arange(C) < valid_len
        mask_shape = (C,) + (1,) * len(trailing_shape)
        chunk = jnp.where(mask.reshape(mask_shape), chunk, 0)
        return chunk, valid_len

    chunks, valid_lens = jax.vmap(read_one)(chunk_indices)
    return chunks, valid_lens


def scatter_chunks(
    output_buf: jax.Array,
    chunks: jax.Array,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array,
    chunk_size: int,
    valid_lens: jax.Array,
) -> jax.Array:
    """Write chunks from chunked layout back to packed tensor.

    Uses lax.scan with dynamic_update_slice. Only writes valid positions.
    The output buffer is temporarily padded to a multiple of chunk_size so
    that dynamic_update_slice never silently clamps partial-last-chunk writes.

    Args:
        output_buf: [1, T_total, ...] pre-allocated buffer (typically zeros).
        chunks: [total_NT, C, ...] chunked data.
        cu_seqlens: [N+1] cumulative sequence lengths.
        chunk_indices: [total_NT, 2] from prepare_chunk_indices.
        chunk_size: block size C.
        valid_lens: [total_NT] valid lengths per chunk.

    Returns:
        [1, T_total, ...] with chunk data written to correct positions.
    """
    C = chunk_size
    T_total = output_buf.shape[1]
    trailing_shape = output_buf.shape[2:]

    # Pad the buffer to T_total + C - 1 so dynamic_update_slice never clamps
    # a partial last-chunk write (start can be as large as T_total-1, update size=C).
    T_padded = T_total + C - 1
    pad_widths = [(0, 0)] * output_buf.ndim
    pad_widths[1] = (0, T_padded - T_total)
    buf_padded = jnp.pad(output_buf, pad_widths)

    def write_one(buf, args):
        chunk, idx, vlen = args
        seq_id, local_t = idx[0], idx[1]
        start = cu_seqlens[seq_id] + local_t * C
        # Mask out invalid positions before writing
        mask = jnp.arange(C) < vlen
        mask_shape = (C,) + (1,) * len(trailing_shape)
        chunk = jnp.where(mask.reshape(mask_shape), chunk, 0)
        update = chunk[None]  # [1, C, ...]
        indices = (0, start) + (0,) * len(trailing_shape)
        buf = lax.dynamic_update_slice(buf, update, indices)
        return buf, None

    result_padded, _ = lax.scan(write_one, buf_padded, (chunks, chunk_indices, valid_lens))
    # Unpad back to original size
    return result_padded[:, :T_total]


def read_chunk(
    tensor: jax.Array,
    start: int,
    valid_len: int,
    chunk_size: int,
) -> jax.Array:
    """Read a single chunk from a packed tensor, zero-padding past valid_len.

    Args:
        tensor: [1, T, ...] packed tensor.
        start: token-level start position along axis 1.
        valid_len: number of valid tokens to read (may be < chunk_size for
            the last chunk of a segment).
        chunk_size: target chunk length C.

    Returns:
        [1, C, ...] chunk with positions [0, valid_len) from the tensor and
        positions [valid_len, C) zero-padded.
    """
    chunk = tensor[:, start:start + valid_len]
    if valid_len < chunk_size:
        pad_widths = [(0, 0)] * tensor.ndim
        pad_widths[1] = (0, chunk_size - valid_len)
        chunk = jnp.pad(chunk, pad_widths)
    return chunk


def write_chunk(
    buf: jax.Array,
    chunk: jax.Array,
    start: int,
    valid_len: int,
) -> jax.Array:
    """Write valid positions of a chunk into the output buffer.

    Args:
        buf: [1, T, ...] output buffer.
        chunk: [1, C, ...] chunk data.
        start: token-level start position along axis 1.
        valid_len: number of valid positions to write.

    Returns:
        Updated buf with chunk[:, :valid_len] written at start.
    """
    return buf.at[:, start:start + valid_len].set(chunk[:, :valid_len])


def acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


def dot(subscripts: str, a: jax.Array, b: jax.Array, acc: jnp.dtype) -> jax.Array:
    """Einsum simulating Triton's tl.dot with fp32 accumulation.

    XLA CPU's DotThunk does not support bf16×bf16→fp32 for certain einsum
    contraction patterns with asymmetric non-contracting dims (e.g.
    nchk,nhkv->nchv). Pre-casting half-precision inputs to fp32 is
    numerically equivalent to Triton tensor core behavior:
    bf16→fp32 cast is exact, and the product of two bf16 values fits in fp32
    (8+8=16 < 24 mantissa bits). fp16 is also covered (11+11=22 < 24).
    """
    if a.dtype in (jnp.bfloat16, jnp.float16) or b.dtype in (jnp.bfloat16, jnp.float16):
        a, b = a.astype(acc), b.astype(acc)
    return jnp.einsum(subscripts, a, b,
                      precision=lax.Precision.HIGHEST,
                      preferred_element_type=acc)
