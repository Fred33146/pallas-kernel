"""Public API for tops.ops.

All public interfaces are exported exclusively via this file.
Any interface not re-exported here is considered an internal implementation
detail with **no API stability guarantee**.
"""

from .simple_gla import simple_gla

__all__ = [
  "simple_gla",
]
