from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure `tests/` parent (project root) is importable so that
# ``from tests.src.…`` style imports work regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

HAS_CUDA = torch.cuda.is_available()

try:
  import jax
  _JAX_BACKEND = jax.default_backend().lower()
except Exception:
  _JAX_BACKEND = "cpu"

HAS_TPU = _JAX_BACKEND == "tpu"

DEVICE = "cuda" if HAS_CUDA else "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed():
  torch.manual_seed(42)


@pytest.fixture
def device():
  return DEVICE


# ---------------------------------------------------------------------------
# Auto-skip tests based on hardware markers
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
  for item in items:
    if "gpu_only" in item.keywords and not HAS_CUDA:
      item.add_marker(pytest.mark.skip(reason="gpu_only: no CUDA available"))
    if "tpu_only" in item.keywords and not HAS_TPU:
      item.add_marker(pytest.mark.skip(reason="tpu_only: no TPU available"))
    if "cpu_only" in item.keywords and (HAS_CUDA or HAS_TPU):
      item.add_marker(pytest.mark.skip(reason="cpu_only: running on accelerator"))
