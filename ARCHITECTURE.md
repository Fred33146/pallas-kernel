# Architecture Overview (ARCHITECTURE)

This document is the top-level navigation map for this code repository and serves as the **System of Record** for all developers (both humans and agents) to understand the system design principles and module layering.
Based on the "Agent-First World" engineering philosophy, all architecture rules, dependency directions, and domain boundaries are explicitly defined and enforced to eliminate misunderstandings and prevent architecture drift.

## 1. Core Layers & Dependencies

The codebase is structured around a layered architectural model. We recommend a unidirectional dependency rule; cross-layer (reverse) dependencies are currently treated as **Warnings rather than strict prohibitions**. Although permitted, there must be a compelling reason for reverse invocations.

*   **Low-level should avoid depending on high-level**: Ideally, bottom-layer code should not reverse-reference upper-level business logic or structures.
*   **Explicit Domain Boundaries**: Each core domain has limited responsibilities.

The current package-level dependency tree (ordered bottom-up):

1.  **`tops/utils.py`**
    *   **Responsibility**: Provides the most fundamental and general utility functions (e.g., shape assertions like `assert_shape_or_none`, alignment logic, etc.).
    *   **Dependency**: Generally not allowed to depend on any other packages in the repository.
2.  **`tops/ops/`**
    *   **Responsibility**: Core operators and low-level compute kernel implementations (low-level hardware interactions at the Pallas TPU/GPU level), such as GLA and simple GLA variant kernels.
    *   **Rules**: Highly optimized. Strictly forbidden from containing any logic related to business metrics (model training states, loss evaluation). Must be strongly associated with specific hardware features.
    *   **Dependency**: Recommended to only depend on `utils` or pure math/hardware primitives.
3.  **`tops/modules/`**
    *   **Responsibility**: Building Blocks. Provides specific standalone network components (RMSNorm, LayerNorm, Convolutions, etc., for single-layer capabilities).
    *   **Dependency**: Mainly depends on `tops/ops/` and `tops/utils.py`.
4.  **`tops/layers/`**
    *   **Responsibility**: High-level neural network layers (based on Flax NNX, etc.), representing complex model structures formed by cascading components from lower layers.
    *   **Dependency**: Allowed to call `tops/modules/`, `tops/ops/`, and the general `utils.py`.

**(Baseline Dependency Flow: `utils` → `ops` → `modules` → `layers`. Deviating from this convention will trigger warnings.)**

---

## 2. Agent Readability

**Only content kept under version control in the repository is visible and traceable to agents and humans.**

*   **Progressive Disclosure**: It is unnecessary for every agent or new member to read the entire repository upon onboarding. They should start from specific focal points (e.g., `CLAUDE.md` or this document).
*   **Planning & Design System (System of Record)**:
    *   Temporary changes should be managed via tickets / lightweight plans.
    *   Complex refactoring or modifications should be documented in the `docs/exec-plans/` directory.
    *   **Kernel Design Constraints**: For core Kernels (especially complex Pallas operators), **it is strictly required that all Kernels (whether legacy or newly added) must have corresponding design documents added to or supplemented in `docs/design-docs/ops/` (including but not limited to formulas, memory layouts, Tiling strategies, and hardware abstractions).** These act as the sole source of truth for development and refactoring. Any maintenance on a Kernel must be based on a pre-existing design document.
*   **Taste & Review Rules (Golden Rules)**:
    1.  **Exhaustive Business & Dimensional Constraints**: All public APIs **must** clearly define tensor dimensional semantics and data flow patterns.
    2.  **Strict Fail-Fast & Assertion Mechanisms**: We encourage as many rigorous tensor and condition assertions as possible. As an absolute baseline constraint, **all public compute interfaces/operator functions must not be in a "Zero Asserts" state**. At least one basic shape or condition check (using `assert` or `assert_shape_or_none`, etc.) must exist. We will enforce this baseline through static code analysis.

---

## 3. Testing & Validation Boundaries

*   Validation is also subject to strict layer divisions and physical isolation. **Modifications at a specific layer must be verified by tests corresponding to that same layer.**
*   Test types within the `tests/` directory are strictly restricted to the following two types of reference comparisons:
    1.  **CPU Reference Tests (vs JAX-CPU)**: Output and gradients from Pallas kernels are checked for tolerance against reference implementations written in pure `jax.numpy`. All these pure-CPU reference primitives must be centralized under **`tops/cpu/`** and its corresponding layered subdirectories as the standard answers (Goldens).
    2.  **GPU Reference Tests (vs Torch-GPU/Triton)**: Aligning the computation results of Pallas kernels against known-correct, cross-framework computation libraries (such as those based on PyTorch or existing high-priority components like FlashAttention) under identical hardware conditions.
*   **`tests/ops/`**: Modifications to low-level operators (e.g., scheduling optimizations of Pallas kernels) must use the two comparison test categories above to verify results or gradient tolerances. It is strictly prohibited to overstep and rely on high-level tests (such as `test_gla.py` in the layers tier) as a workaround for validation.
*   **`tests/modules/` & `tests/layers/`**: Modifications at the network component or layer levels must include corresponding integration encapsulation and data flow validation tests.

*   **`tops/cpu/` is the default Gold/Reference**: Implementations under `tops/cpu/` are treated as canonical reference implementations. Any new Pallas/JAX implementation added under `tops/ops/` must, by default, align against `tops/cpu/` through reference-comparison tests.
*   **How reference correctness is established**: The correctness of `tops/cpu/` must be continuously validated against `torch_gpu/torch_cpu` comparisons, so it remains a trustworthy baseline.
*   **Default comparator and GPU-specific exception**: The default comparator is `tops/cpu/`. If GPU-based comparison is required (e.g., Torch/Triton), create a separate test file and append the `_gpu` suffix to its filename.

---

If a new feature needs to be added within an existing layer, find the level that best matches its granularity and adhere to that level's restrictions (and be sure to add corresponding tests). For new components whose classification is ambiguous, start by initiating a technical design brainstorm via `docs/design-docs/`.
