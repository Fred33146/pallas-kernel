# Contributing to tops

Thank you for your interest in contributing to tops! This project implements high-performance JAX/Pallas TPU and GPU kernels for modern neural network architectures.

For a detailed understanding of the system architecture, layer design, and coding standards, please read [ARCHITECTURE.md](ARCHITECTURE.md).

## Getting Started

### Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/primatrix/pallas-kernel.git
cd pallas-kernel
```

2. Install dependencies:

```bash
# Base install (CPU only)
uv sync

# With GPU support (CUDA 12)
uv sync --extra gpu

# With TPU support
uv sync --extra tpu
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

4. Run tests to verify your setup:

```bash
uv run pytest tests/ -v
```

## Development Workflow

### Branching

1. Create your own development branch in this repository (e.g., `dev/<your-name>/my-feature`, `fix/issue-123`).
2. Make your changes and push to the branch.
3. CI will automatically run checks to ensure code passes lint, formatting, and tests.
4. Once CI passes, open a Pull Request against the `main` branch for code review.

### Commit Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>
```

**Types:**

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code refactoring (no behavior change) |
| `docs` | Documentation only |
| `ci` | CI/CD changes |
| `test` | Adding or updating tests |
| `perf` | Performance improvement |

**Scope** is optional and indicates the affected module: `feat(gla):`, `fix(pallas):`, etc.

**Examples from project history:**

- `ci: fix TPU job premature failure due to log disconnect`
- `fix(gla): relax g_gamma non-positive assertion`
- `feat: fused chunk simple gla vjp`

### Pre-Submit Checks

Before submitting a PR, ensure the following pass locally:

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Tests
uv run pytest tests/ -v
```

Pre-commit hooks will run `ruff check --fix`, `ruff format`, `trailing-whitespace`, and `check-added-large-files` automatically on each commit.

## Issue Management

### Title Convention

All issue titles must start with a type prefix:

| Prefix | Usage | Label |
|--------|-------|-------|
| `[Bug]` | Bug report | `bug` |
| `[Feature]` | Feature request | `enhancement` |
| `[Epic]` | Large milestone or roadmap | `epic` |
| `[Perf]` | Performance issue or optimization | `performance` |
| `[Design]` | Kernel design proposal | `design-proposal` |

Issues without a prefix should have one added during triage.

### Priority

| Priority | Meaning | Response Time |
|----------|---------|---------------|
| `P0` | Blocking — affects core functionality or CI | Immediate |
| `P1` | Important — must complete in current iteration | Start within a week |
| `P2` | Improvement — can be scheduled for later | Planned |

All new issues must be tagged with a priority label. Untagged issues default to `P2`.

### Labels

**Type labels:** `bug`, `enhancement`, `documentation`, `question`
**Priority labels:** `P0`, `P1`, `P2`
**Special types:** `epic`, `performance`, `design-proposal`, `cpu-ref`
**Community labels:** `good first issue`, `help wanted`

The `cpu-ref` label must be added to any issue or PR that modifies code under `tops/cpu/`.

### Issue Lifecycle

```
Created → [Optional] Triage/Assign → In Development → PR linked (Closes #N) → Auto-closed on merge
```

- Use the appropriate [issue template](.github/ISSUE_TEMPLATE/) when creating an issue.
- Include the title prefix and a priority label.
- Assignees default to unset; assigned later by a maintainer or automation.
- PRs should use `Closes #N` to link the related issue; the issue is auto-closed on merge.

### Epic & Sub-issues

- Epic issues use the `[Epic]` prefix and `epic` label.
- Parent-child relationships are established via the GitHub Sub-issues API.
- The Epic body should include a `- [ ] #N` checklist for progress tracking.
- Each sub-issue should be independently completable with a single PR.
- Epic issues are manually closed after all sub-issues are resolved.

### Issue Types (Supplementary)

The organization has configured GitHub Issue Types (Bug / Feature / Goal / Task / SubTask). Setting the appropriate type when creating an issue is optional. **Title prefixes and labels are mandatory** — Issue Types do not replace them.

## Code Review Process

### General Changes

Features, bug fixes, documentation, CI, and tests require **1 core maintainer approval** to merge.

### Reference Implementation Changes

Changes to code under `tops/cpu/` (the correctness baseline for all GPU/TPU kernel validation) require review by **all core developers**. This review must be conducted via:

- Video conference discussion, OR
- Mailing list review

PRs modifying `tops/cpu/` must be labeled with `cpu-ref` to notify all core developers.

**Rationale:** `tops/cpu/` serves as the Golden reference for all kernel correctness validation. Changes here impact the entire test and validation system.

### Review Process

A complete PR review follows these steps:

1. **Author self-check**: Ensure all CI checks pass (lint, format, tests). PR description clearly explains what changed and why.
2. **Reviewer examination**:
   - **Correctness**: Is the logic correct? Are edge cases covered?
   - **Coding standards**: Do docstrings, assertions, and naming follow project standards?
   - **Test coverage**: Are tests added at the appropriate layer with sufficient coverage?
   - **Performance impact**: Does this introduce unnecessary performance regressions? (For kernel PRs, check roofline analysis results.)
3. **Iterate**: After reviewer feedback, the author addresses comments and pushes fixes until all comments are resolved.
4. **Merge**: Once the required number of approvals is obtained, the reviewer or author merges into `main`.

## Coding Standards

Key rules (see [ARCHITECTURE.md Section 2](ARCHITECTURE.md#2-agent-readability) for full details):

- **Docstrings**: All public functions must have comprehensive docstrings that explicitly describe tensor shapes, dimension semantics, and business logic.
- **Input assertions**: All public functions must enforce strict shape/type constraints using `assert` or `assert_shape_or_none` from `tops.utils`.
- **No-crash guarantee**: Kernels must never crash after passing entry-point assertions.

## Testing Guidelines

The project uses a layered testing structure (see [ARCHITECTURE.md Section 3](ARCHITECTURE.md#3-testing--validation-boundaries) for full details):

| Directory | Purpose |
|-----------|---------|
| `tests/ops/` | Kernel correctness tests (GPU/TPU vs CPU reference) |
| `tests/modules/` | Module unit tests |
| `tests/layers/` | Layer integration tests |
| `tests/ref/` | Reference implementation comparison tests |

Key requirements:

- New kernels must have CPU reference comparison tests.
- Error bound constraint: TPU error must not exceed GPU error ($\epsilon_{\text{TPU}} \leq \epsilon_{\text{GPU}}$).
- Modifications at a specific layer must be verified by tests at that same layer.

## Kernel Design Requirements

Every new or refactored Kernel **must** have a design document in `docs/design-docs/ops/` covering the following:

### 1. Algorithm and Code Logic

- Core mathematical formulas (e.g., recurrence relations, attention computations).
- Implementation strategy: why this approach was chosen and trade-offs with alternatives.

### 2. Data Flow and Tiling Logic

- Tiling / Blocking strategy: rationale for block sizes, how they map to hardware (TPU VMEM / GPU SMEM).
- Data movement between HBM and VMEM/SMEM (DMA pipeline, double buffering, etc.).
- Grid dimension partitioning: which dimensions are parallel vs. arbitrary, and why.

### 3. Performance Target and Roofline Analysis

The performance target for all Kernels is **80% of hardware theoretical peak**:

- If the Kernel is **memory bandwidth-bound**: target is 80% of hardware memory bandwidth peak.
- If the Kernel is **compute-bound**: target is 80% of hardware compute peak (FLOPS).

The design document must include:

- **Roofline analysis**: arithmetic intensity of the Kernel and whether it is bandwidth-bound or compute-bound.
- **Local benchmark results**: measured throughput on target hardware (e.g., TFLOPS, GB/s).
- **Trace viewer analysis**: include screenshots or links from a trace viewer (e.g., Perfetto / TensorBoard Profiler) showing the Kernel's actual execution timeline — compute, memory transfer, and synchronization phases with their time distribution — to verify pipeline effectiveness and identify bubbles or unnecessary stalls.
- **Whether the 80% target is met**. If not, the reason must be clearly stated:
  - **Hardware limitation**: unsupported instructions, memory alignment constraints, known hardware bottlenecks, etc.
  - **Code logic limitation**: tiling strategy constraints, pipeline bubbles, synchronization overhead, etc., along with future optimization directions.

### 4. Precision and Error Analysis

- **Error metrics**: error between Kernel output and `tops/cpu/` high-precision reference (atol, rtol, max ULP).
- **Gradient error** (if applicable): backward pass error metrics.
- **Test methodology**: input distributions used, shape configurations, hardware tested on.
- **Error bound constraint**: TPU error must not exceed GPU error ($\epsilon_{\text{TPU}} \leq \epsilon_{\text{GPU}}$).

## Adding New Kernels

1. Write a design document in `docs/design-docs/ops/` first.
2. Implement CPU reference in `tops/cpu/ops/<domain>/`.
3. Submit CPU reference for **all-core-developer review** (see [Code Review Process](#reference-implementation-changes)).
4. Implement GPU/TPU kernel in `tops/ops/<domain>/`.
5. Add comparison tests in `tests/ops/<domain>/`.
6. Export public API via `tops/ops/__init__.py`.
7. Ensure docstrings and assertions meet coding standards.
8. Submit non-`tops/cpu/` code (steps 4-7), which requires **1 core maintainer approval** to merge.

> **Note:** Due to the current small number of core developers, the tiered review policy above is enforced by convention only — it is not yet backed by GitHub branch protection rules or CODEOWNERS. Automated enforcement will be added as the team grows.

## Release Process

The project uses trunk-based development with release branches (see [ARCHITECTURE.md Section 4](ARCHITECTURE.md#4-release--versioning-policy) for full details):

1. RC tag created on `main` when ready for delivery (`vX.Y.Z-rc.N`).
2. Release branch `release/vX.Y` pulled from RC tag.
3. Integration testing; fixes cherry-picked back to `main`.
4. Final tag (`vX.Y.Z`) when stable.

## Profiling Guide

### Installation

Install profiling dependencies:

```bash
uv sync --extra profile
```

This installs [xprof](https://github.com/google/xprof) (TensorBoard Profiler) for collecting and analyzing kernel execution traces on TPU/GPU.

### Collecting Traces

Use `jax.profiler` to capture traces for a specific code region:

```python
import jax

# Start profiler server
jax.profiler.start_server(port=9012)

# Or manually mark a trace region
jax.profiler.start_trace("/tmp/my_trace")
# ... run your kernel ...
jax.profiler.stop_trace()
```

You can also connect to the profiler server remotely via TensorBoard for interactive trace capture.

### Analyzing Traces

```bash
# Launch TensorBoard to view traces
tensorboard --logdir /tmp/my_trace
```

In the TensorBoard **Profile** tab, focus on:

- **Trace Viewer**: kernel execution timeline — check time distribution across compute, memory transfer, and synchronization phases.
- **Pipeline effectiveness**: look for bubbles, unnecessary waits, or synchronization overhead.
- **Op-level timing**: identify hotspot operations.

Trace analysis results must be included in the [Kernel Design document](#3-performance-target-and-roofline-analysis).

## Hardware Access

The project provides [SkyPilot](https://skypilot.readthedocs.io/)-based cluster launch scripts:

```bash
# Launch GPU cluster
./scripts/launch_gpu.sh L4 my-cluster

# Launch TPU cluster
./scripts/launch_tpu.sh tpu-v6e-1 my-cluster
```

See the resource configuration files in `scripts/` (`gpu_resource.sky.yaml`, `tpu_resource.sky.yaml`) for details. You need a configured cloud account and SkyPilot environment before use.

## Questions?

If you have questions, open an issue on GitHub or reach out to the maintainers.