---
name: Performance Issue
about: Report a performance regression or optimization opportunity
title: "[Perf] "
labels: performance
assignees: ''
---

## Summary

Brief description of the performance issue.

## Type

- [ ] Performance regression (was faster before)
- [ ] Below expected performance target (not meeting 80% roofline)
- [ ] Optimization opportunity

## Kernel / Operation

Which kernel or operation is affected?

## Observed Performance

```
Paste benchmark results here (e.g., TFLOPS, GB/s, latency).
```

## Expected Performance

What performance do you expect? Include roofline analysis if available.

- Arithmetic intensity:
- Bound type: Compute-bound / Memory bandwidth-bound
- Hardware theoretical peak:
- 80% target:

## Environment

- Python version:
- JAX version:
- Hardware: CPU / GPU (model) / TPU (version)
- OS:

## Reproduction

Minimal code or script to reproduce the measurement:

```python
# Your benchmark code here
```

## Trace / Profile Data

If available, attach trace viewer screenshots (Perfetto / TensorBoard Profiler) or link to trace files.

## Comparison

If this is a regression, what was the previous performance and which commit/version introduced it?

- Previous performance:
- Last known good commit/version:

## Additional Context

Any other context, related issues, or potential root causes.
