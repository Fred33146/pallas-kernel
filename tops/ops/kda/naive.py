"""KDA naive recurrent — re-exports from CPU reference.

This module provides the naive step-by-step KDA recurrence as a
convenience re-export from tops.cpu.ops.kda.naive.
"""

from tops.cpu.ops.kda.naive import naive_recurrent_kda as naive_kda

__all__ = ["naive_kda"]
