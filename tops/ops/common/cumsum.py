"""Chunk-local cumulative sum with multiple backends.

Fixed-length inputs dispatch to JAX cumsum or matmul; variable-length inputs
use a Pallas kernel with vectorised Hillis-Steele prefix scan.
"""

import functools
import math

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

from tops.ops.utils import is_tpu_runtime
from tops.utils import export_public, prepare_chunk_indices

_TRIU_PRECISION = jax.lax.Precision.HIGHEST
_VMEM_HW_LIMIT_BYTES = 30 * 1024 * 1024  # 30 MB (TPUv6e VMEM ~32 MB)


# =============================================================================
# Fixed-length: jnp.cumsum (optimal for head_first=False)
# =============================================================================


def _chunk_local_cumsum_origin(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumsum via ``jnp.cumsum`` (best for ``head_first=False``).

  Args:
      g:            [B, T, H, S] or [B, H, T, S] — input gates.
      chunk_size:   block size (must be power of 2).
      reverse:      if True, compute reverse (suffix) cumsum.
      scale:        optional multiplicative scale applied to the output.
      head_first:   if True, ``g`` is [B, H, T, S]; otherwise [B, T, H, S].
      output_dtype: dtype of the output tensor (default float32).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum.
  """
  BT = chunk_size
  out_dtype = output_dtype or g.dtype

  if head_first:
    B, H, T, S = g.shape
  else:
    B, T, H, S = g.shape

  NT = (T + BT - 1) // BT
  T_padded = NT * BT
  pad_t = T_padded - T

  if head_first:
    g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, H, NT, BT, S).astype(jnp.float32)
    cum_axis = 3
  else:
    g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)
    cum_axis = 2

  if reverse:
    o_chunked = jnp.flip(
      jnp.cumsum(jnp.flip(g_chunked, axis=cum_axis), axis=cum_axis),
      axis=cum_axis,
    )
  else:
    o_chunked = jnp.cumsum(g_chunked, axis=cum_axis)

  if head_first:
    o = o_chunked.reshape(B, H, T_padded, S)[:, :, :T, :]
  else:
    o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

  if scale is not None:
    o = o * scale

  return o.astype(out_dtype)


# =============================================================================
# Fixed-length: matmul with tril/triu mask (optimal for head_first=True)
# =============================================================================


def _chunk_local_cumsum_matmul(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
  """Chunk-local cumsum via triangular matmul (best for ``head_first=True``).

  Args:
      g:            [B, T, H, S] or [B, H, T, S] — input gates.
      chunk_size:   block size (must be power of 2).
      reverse:      if True, compute reverse (suffix) cumsum.
      scale:        optional multiplicative scale applied to the output.
      head_first:   if True, ``g`` is [B, H, T, S]; otherwise [B, T, H, S].
      output_dtype: dtype of the output tensor (default float32).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum.
  """
  BT = chunk_size
  out_dtype = output_dtype or g.dtype

  if head_first:
    B, H, T, S = g.shape
  else:
    B, T, H, S = g.shape

  NT = (T + BT - 1) // BT
  T_padded = NT * BT
  pad_t = T_padded - T

  if head_first:
    g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, H, NT, BT, S).astype(jnp.float32)
  else:
    g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
    g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)

  if reverse:
    cum_mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.float32))
  else:
    cum_mask = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))

  if head_first:
    o_chunked = jnp.einsum(
      "ij,bhnjs->bhnis",
      cum_mask,
      g_chunked,
      precision=jax.lax.Precision.HIGHEST,
    )
    o = o_chunked.reshape(B, H, T_padded, S)[:, :, :T, :]
  else:
    o_chunked = jnp.einsum(
      "ij,bnjhs->bnihs",
      cum_mask,
      g_chunked,
      precision=jax.lax.Precision.HIGHEST,
    )
    o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

  if scale is not None:
    o = o * scale

  return o.astype(out_dtype)


# =============================================================================
# Full cumsum: recursive hierarchical triu matmul
# =============================================================================


def _triu_dot(a, b, contracting_a, contracting_b):
  """``dot_general`` with highest precision for triu matmul cumsum."""
  return jax.lax.dot_general(
    a, b,
    dimension_numbers=((contracting_a, contracting_b), ((), ())),
    precision=_TRIU_PRECISION,
  )


def _recursive_cumsum_2d(x_2d, chunk_size):
  """Recursive triu matmul cumsum on ``(B, L)`` along axis 1.

  Splits the sequence into ``chunk_size``-wide blocks, computes local
  cumsum via upper-triangular matmul, then recursively prefix-sums the
  chunk totals to obtain inter-chunk offsets.

  Recursion depth: ``ceil(log_{chunk_size}(L))``, typically <= 3.

  Args:
      x_2d:       [B, L] — flattened input.
      chunk_size:  block size for each recursion level.

  Returns:
      [B, L] — inclusive cumulative sum.
  """
  B, L = x_2d.shape

  # Base case: L fits in a single block
  if L <= chunk_size:
    triu = jnp.triu(jnp.ones((L, L), dtype=x_2d.dtype))
    return _triu_dot(x_2d, triu, (1,), (0,))

  # Pad to multiple of chunk_size
  seq_pad = (chunk_size - L % chunk_size) % chunk_size
  if seq_pad > 0:
    x_2d = jnp.pad(x_2d, [(0, 0), (0, seq_pad)])
  L_padded = L + seq_pad
  num_chunks = L_padded // chunk_size

  x_3d = x_2d.reshape(B, num_chunks, chunk_size)

  # Local cumsum within each chunk
  triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=x_3d.dtype))
  local_cs = _triu_dot(x_3d, triu, (2,), (0,))  # (B, num_chunks, chunk_size)

  # Chunk totals → recursive inclusive prefix sum
  chunk_totals = local_cs[:, :, -1]  # (B, num_chunks)
  totals_cumsum = _recursive_cumsum_2d(chunk_totals, chunk_size)

  # Exclusive offsets = inclusive_cumsum - own_total
  offsets = totals_cumsum - chunk_totals  # (B, num_chunks)
  result = local_cs + offsets[:, :, None]

  return result.reshape(B, L_padded)[:, :L]


def cumsum_triu_recursive(
  x: jax.Array,
  axis: int = -1,
  chunk_size: int = 128,
) -> jax.Array:
  """Full cumulative sum via recursive hierarchical triu matmul.

  Splits the target axis into ``chunk_size``-wide blocks and computes
  local cumsums via upper-triangular matrix multiplication, then
  recursively prefix-sums the chunk totals to propagate inter-chunk
  offsets.  All matmuls use ``Precision.HIGHEST`` for float32 accuracy.

  Equivalent to ``jnp.cumsum(x, axis=axis)`` but uses O(chunk_size^2)
  triu matrices instead of O(L) sequential scan, making it more
  TPU-friendly for long sequences.

  Args:
      x:          input tensor of any shape.
      axis:       axis along which to compute cumsum (default -1).
      chunk_size: block size per recursion level (default 128, aligned
                  to TPU MXU).

  Returns:
      Tensor of same shape and dtype as ``x`` — inclusive cumulative sum.
  """
  ndim = x.ndim
  assert ndim >= 1, f"x must have at least 1 dimension, got {ndim}"

  axis = axis % ndim
  L = x.shape[axis]

  # Flatten to (B, L)
  x_work = jnp.moveaxis(x, axis, -1)
  batch_shape = x_work.shape[:-1]
  B = math.prod(batch_shape) if batch_shape else 1
  x_2d = x_work.reshape(B, L)

  result = _recursive_cumsum_2d(x_2d, chunk_size)

  # Restore original shape
  result = result.reshape(*batch_shape, L) if batch_shape else result.reshape(L)
  return jnp.moveaxis(result, -1, axis)


# =============================================================================
# Pallas kernel: vectorised Hillis-Steele prefix scan (varlen)
# =============================================================================



def _chunk_cumsum_kernel_varlen(
  cu_seqlens_ref,
  chunk_indices_ref,
  s_ref,
  o_ref,
  *,
  BT: int,
  NT: int,
  REVERSE: bool,
  HAS_SCALE: bool,
  scale: float,
):
  """Pallas varlen kernel — BlockSpec tiles BH and S, loops over chunks.

  Grid: (NS, NBH).  BlockSpec loads ``(BB, T_alloc, BS)`` tiles, covering
  the full T extent.  The kernel iterates over ``NT`` chunks inside
  ``fori_loop``, using ``dslice`` on the T dimension with dynamic offsets
  derived from ``cu_seqlens``.

  This avoids both the VMEM blow-up of loading the entire tensor and
  the gather/scatter HBM overhead of pre-chunking.

  Args:
      cu_seqlens_ref:     [N+1] scalar-prefetched — cumulative sequence lengths.
      chunk_indices_ref:  [NT, 2] scalar-prefetched — (seq_idx, local_chunk_idx).
      s_ref:              [BB, T_alloc, BS] BlockSpec tile — input gates.
      o_ref:              [BB, T_alloc, BS] BlockSpec tile — output cumsum.
      BT:   chunk size along T (power of 2).
      NT:   total number of chunks across all sequences.
      REVERSE:  if True, compute reverse (suffix) cumsum.
      HAS_SCALE: whether to apply a multiplicative scale.
      scale: scale value (ignored if HAS_SCALE is False).
  """
  num_steps = int(math.log2(BT))

  def body(i_t, _):
    i_n = chunk_indices_ref[i_t, 0]
    local_i_t = chunk_indices_ref[i_t, 1]
    bos = cu_seqlens_ref[i_n]
    eos = cu_seqlens_ref[i_n + 1]
    start_t = bos + local_i_t * BT

    s = s_ref[:, dslice(start_t, BT), :]

    T_seq = eos - bos
    valid_len = T_seq - local_i_t * BT
    valid_mask = (jnp.arange(BT) < valid_len).astype(jnp.float32)[None, :, None]
    s = s.astype(jnp.float32) * valid_mask

    if REVERSE:
      for d in range(num_steps):
        stride = 1 << d
        top = s[:, : BT - stride, :] + s[:, stride:, :]
        bot = s[:, BT - stride :, :]
        s = jnp.concatenate([top, bot], axis=1)
    else:
      for d in range(num_steps):
        stride = 1 << d
        top = s[:, :stride, :]
        bot = s[:, stride:, :] + s[:, :-stride, :]
        s = jnp.concatenate([top, bot], axis=1)

    if HAS_SCALE:
      s = s * scale

    o_ref[:, dslice(start_t, BT), :] = s.astype(o_ref.dtype)
    return 0

  jax.lax.fori_loop(0, NT, body, 0)


# =============================================================================
# Pallas launcher for varlen mode
# =============================================================================


def _chunk_local_cumsum_pallas(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  cu_seqlens: jax.Array | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
  chunk_indices: jax.Array | None = None,
) -> jax.Array:
  """Pallas-based chunk-local cumsum for variable-length inputs.

  Uses BlockSpec to tile BH and S dimensions, with ``fori_loop`` +
  ``dslice`` to iterate over variable-length chunks along T inside the
  kernel.  BB is dynamically shrunk to fit tiles within hardware VMEM.

  Args:
      g:              [B, T, H, S] or [B, H, T, S] — input gates.
      chunk_size:     block size along T (must be power of 2).
      reverse:        if True, compute reverse (suffix) cumsum.
      scale:          optional multiplicative scale applied to the output.
      cu_seqlens:     [N+1] — cumulative sequence lengths (int32).
      head_first:     if True, ``g`` is [B, H, T, S]; otherwise [B, T, H, S].
      output_dtype:   dtype of the output tensor (default float32).
      chunk_indices:  [NT, 2] — precomputed chunk-to-sequence mapping (optional).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum.
  """
  BT = chunk_size
  BS = 128
  BB = 8

  if head_first:
    B, H, T, S = g.shape
    g_flat = g.reshape(B * H, T, S)
  else:
    B, T, H, S = g.shape
    g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

  BH = B * H
  out_dtype = output_dtype or g.dtype
  HAS_SCALE = scale is not None
  scale_val = scale if scale is not None else 1.0

  interpret = not is_tpu_runtime()

  # Pad S dimension to multiple of BS
  pad_S = (BS - (S % BS)) % BS
  if pad_S > 0:
    g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))
  S_padded = S + pad_S
  NS = S_padded // BS

  # Pad BH dimension to multiple of BB
  pad_BH = (BB - (BH % BB)) % BB
  if pad_BH > 0:
    g_flat = jnp.pad(g_flat, ((0, pad_BH), (0, 0), (0, 0)))
  BH_padded = BH + pad_BH

  if chunk_indices is None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
  NT = len(chunk_indices)

  # Pad T so the last chunk of each sequence can read BT elements safely
  g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))
  T_alloc = T + BT

  # Dynamically shrink BB so that the compiler's double-buffered tiles
  # (4 × BB × T_alloc × BS × 4 bytes) fit within hardware VMEM.
  elem_bytes = 4  # float32
  while BB > 1 and 4 * BB * T_alloc * BS * elem_bytes > _VMEM_HW_LIMIT_BYTES:
    BB //= 2
  NBH = BH_padded // BB

  grid = (NS, NBH)

  kernel = functools.partial(
    _chunk_cumsum_kernel_varlen,
    BT=BT, NT=NT,
    REVERSE=reverse, HAS_SCALE=HAS_SCALE, scale=scale_val,
  )

  def _index_map(i_s, i_bb, *_):
    return (i_bb, 0, i_s)

  o_flat = pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,
      grid=grid,
      in_specs=[
        pl.BlockSpec(
          block_shape=(BB, T_alloc, BS),
          index_map=_index_map,
        ),
      ],
      out_specs=pl.BlockSpec(
        block_shape=(BB, T_alloc, BS),
        index_map=_index_map,
      ),
    ),
    out_shape=jax.ShapeDtypeStruct(g_flat.shape, out_dtype),
    interpret=interpret,
    compiler_params=pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel"),
    ),
  )(cu_seqlens, chunk_indices, g_flat)

  # Remove padding
  o_flat = o_flat[:BH, :T, :S]

  if head_first:
    return o_flat.reshape(B, H, T, S)
  else:
    return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# =============================================================================
# Public API
# =============================================================================


def chunk_local_cumsum_vector(
  g: jax.Array,
  chunk_size: int,
  reverse: bool = False,
  scale: float | None = None,
  cu_seqlens: jax.Array | None = None,
  head_first: bool = False,
  output_dtype: jnp.dtype | None = jnp.float32,
  chunk_indices: jax.Array | None = None,
) -> jax.Array:
  """Chunk-local cumulative sum of gates with automatic backend dispatch.

  For fixed-length inputs (``cu_seqlens is None``):
    * ``head_first=True``  → triangular matmul path (``_chunk_local_cumsum_matmul``)
    * ``head_first=False`` → ``jnp.cumsum`` path (``_chunk_local_cumsum_origin``)

  For variable-length inputs (``cu_seqlens`` provided):
    * Vectorised Hillis-Steele Pallas kernel (``_chunk_local_cumsum_pallas``)

  Args:
      g:              [B, T, H, S] or [B, H, T, S] — input gates.
      chunk_size:     block size along T (must be power of 2).
      reverse:        if True, compute reverse (suffix) cumsum within each chunk.
      scale:          optional multiplicative scale applied to the output.
      cu_seqlens:     [N+1] int32 — cumulative sequence lengths for varlen mode.
      head_first:     if True, ``g`` layout is [B, H, T, S]; otherwise [B, T, H, S].
      output_dtype:   dtype of the output tensor (default float32).
      chunk_indices:  [NT, 2] — precomputed ``prepare_chunk_indices`` result (optional).

  Returns:
      o: same shape as ``g`` — chunk-local cumsum of the input gates.
  """
  # =================== assert kernel requirements start ===================
  assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
  assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
    "chunk_size must be power of 2"
  )
  # =================== assert kernel requirements done ====================

  if cu_seqlens is None:
    if head_first:
      return _chunk_local_cumsum_matmul(
        g,
        chunk_size,
        reverse,
        scale,
        head_first,
        output_dtype,
      )
    else:
      return _chunk_local_cumsum_origin(
        g,
        chunk_size,
        reverse,
        scale,
        head_first,
        output_dtype,
      )
  else:
    return _chunk_local_cumsum_pallas(
      g,
      chunk_size,
      reverse,
      scale,
      cu_seqlens,
      head_first,
      output_dtype,
      chunk_indices,
    )


__all__ = export_public(globals())

