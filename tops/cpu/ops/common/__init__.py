"""Shared utilities for JAX CPU reference kernels."""

from tops.cpu.ops.common.utils import (
  acc_dtype,
  cdiv,
  dot,
  gather_chunks,
  pad_to_multiple,
  scatter_chunks,
)
from tops.cpu.ops.common.chunk_h import chunk_fwd_h, chunk_bwd_dh
from tops.cpu.ops.common.chunk_delta_h import (
  chunk_gated_delta_rule_bwd_dhu,
  chunk_gated_delta_rule_fwd_h,
)
from tops.cpu.ops.common.chunk_o import chunk_fwd_o, chunk_local_cumsum

__all__ = [
  "acc_dtype",
  "cdiv",
  "dot",
  "pad_to_multiple",
  "gather_chunks",
  "scatter_chunks",
  "chunk_fwd_h",
  "chunk_bwd_dh",
  "chunk_fwd_o",
  "chunk_local_cumsum",
  "chunk_gated_delta_rule_fwd_h",
  "chunk_gated_delta_rule_bwd_dhu",
]
