---
name: Kernel Design Proposal
about: Propose a new kernel or major kernel refactor (required before implementation)
title: "[Design] "
labels: design-proposal, kernel
assignees: ''
---

## Overview

What kernel is being proposed? What does it compute?

## Motivation

Why is this kernel needed? What use cases does it serve?

## 1. Algorithm and Code Logic

### Core Mathematical Formulas

```
# e.g., recurrence relations, attention computations
```

### Implementation Strategy

Describe the chosen approach and trade-offs with alternatives.

## 2. Data Flow and Tiling Logic

### Tiling / Blocking Strategy

- Block sizes:
- Rationale (how they map to hardware VMEM/SMEM):

### Data Movement

- DMA pipeline / double buffering strategy:
- Grid dimension partitioning (parallel vs. arbitrary):

### Tensor Layout

- Input shapes and semantics:
- Output shapes and semantics:

## 3. Performance Target and Roofline Analysis

- [ ] Arithmetic intensity calculated
- [ ] Bound type identified: Compute-bound / Memory bandwidth-bound
- [ ] 80% target defined

| Metric | Value |
|--------|-------|
| Arithmetic intensity | |
| Hardware peak (FLOPS or GB/s) | |
| 80% target | |
| Target hardware | |

## 4. Precision and Error Analysis

- Error metrics (atol, rtol, max ULP):
- Input distributions for testing:
- Shape configurations:
- [ ] Error bound constraint: TPU error ≤ GPU error

## Implementation Plan

1. [ ] CPU reference implementation in `tops/cpu/ops/<domain>/`
2. [ ] CPU reference review (all core developers)
3. [ ] GPU/TPU kernel in `tops/ops/<domain>/`
4. [ ] Comparison tests in `tests/ops/<domain>/`
5. [ ] Public API export via `tops/ops/__init__.py`
6. [ ] Docstrings and assertions
7. [ ] Benchmark and trace analysis

## References

Links to papers, related implementations, or prior art.

## Willingness to Contribute

- [ ] I am willing to submit a PR for this kernel.
