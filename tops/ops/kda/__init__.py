"""KDA (Kimi Delta Attention) Pallas TPU kernels.

Exports:
  chunk_kda: Chunked KDA forward (+ backward via custom_vjp).
  chunk_kda_fwd: Forward-only chunked KDA.
  fused_recurrent_kda: Step-by-step recurrent KDA via lax.scan.
  naive_kda: Naive step-by-step KDA recurrence (CPU reference).
"""

from .chunk import chunk_kda, chunk_kda_fwd
from .fused_recurrent import fused_recurrent_kda
from .chunk_intra import chunk_kda_bwd_intra
from .naive import naive_kda

__all__ = [
    "chunk_kda",
    "chunk_kda_fwd",
    "fused_recurrent_kda",
    "naive_kda",
    "chunk_kda_bwd_intra",
]
