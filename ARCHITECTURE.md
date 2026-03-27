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
    1.  **Strict Fail-Fast & Assertion Mechanisms**: We encourage as many rigorous tensor and condition assertions as possible. As an absolute baseline constraint, **all public compute interfaces/operator functions must not be in a "Zero Asserts" state**. At least one basic shape or condition check (using `assert` or `assert_shape_or_none`, etc.) must exist. We will enforce this baseline through static code analysis.
    2.  **Mandatory external public API documentation**: All externally exposed public APIs must include complete comments/docstrings that explicitly define:
        *   Tensor sizes and dimension semantics (input/output item-by-item);
        *   Function business semantics and behavioral boundaries;
        *   Data flow (how inputs transform into outputs and key intermediate state semantics).
    3.  **Kernel No-Crash-After-Assert Guarantee**: Once a Kernel function passes its entry-point assertions, **it must never produce runtime errors or coredumps by design**. If the input is valid (not rejected by asserts), the Kernel must guarantee safe execution to completion with correct results under all valid input conditions. Assertions are the only permitted rejection gate — passing them constitutes a contract of execution safety.
    4.  **Mandatory Edge-Case Testing**: All Kernel test suites **must include coverage of boundary and extreme input scenarios**, such as: very large batch sizes, very large sequence lengths (tokens), minimal dimensions, non-aligned sizes, etc. The expected behavior for these edge cases must be one of: **(a) being explicitly caught by entry-point asserts and raising `AssertionError`**, thereby validating that the Kernel's defensive assertions are complete and effective; or **(b) the Kernel correctly handles the input and returns correct results**. If an out-of-design-range input is neither caught by asserts nor handled correctly (e.g., resulting in a coredump, silent error, or numerical anomaly), it is treated as a defect that must be fixed.

---

## 3. Testing & Validation Boundaries

*   Validation is also subject to strict layer divisions and physical isolation. **Modifications at a specific layer must be verified by tests corresponding to that same layer.**

### 3.1 Reference Implementations & Tests

*   **`tops/cpu/`** is the sole location for canonical reference implementations (Gold/Reference). All reference implementations are written in pure JAX, executed on CPU, and serve as the standard answers (Goldens). The directory is organized by operator domain:

    ```
    tops/cpu/
    └── ops/
        ├── common/      # Shared chunk utility reference implementations
        ├── gla/          # GLA operator reference implementations
        ├── simple_gla/   # Simple GLA operator reference implementations
        └── mla/          # MLA operator reference implementations
    ```

*   **`tests/ref/`** is the unified location for reference comparison tests, mirroring the directory structure of `tops/cpu/ops/`. These tests compare `tops/cpu/` reference implementations against other known-correct implementations (e.g., PyTorch CPU/GPU) to continuously validate the correctness of the reference implementations themselves:

    ```
    tests/ref/
    ├── common/       # Reference tests for shared utilities
    ├── gla/          # Reference tests for GLA implementations
    ├── simple_gla/   # Reference tests for Simple GLA implementations
    └── mla/          # Reference tests for MLA implementations
    ```

*   **Alignment requirement**: All new Pallas/JAX implementations added to `tops/ops/` must align with `tops/cpu/` and pass reference comparison tests.
*   **How reference correctness is established**: The correctness of `tops/cpu/` must be continuously validated through tests in `tests/ref/` against `torch_gpu/torch_cpu` comparisons, so it remains a trustworthy baseline.

### 3.2 Operator & Layer Tests

*   Test types within the `tests/` directory are strictly restricted to the following two types of reference comparisons:
    1.  **CPU Reference Tests (vs JAX-CPU)**: Output and gradients from Pallas kernels are checked for tolerance against reference implementations in `tops/cpu/` written in pure JAX.
    2.  **GPU Reference Tests (vs Torch-GPU/Triton)**: Aligning the computation results of Pallas kernels against known-correct, cross-framework computation libraries (such as those based on PyTorch or existing high-priority components like FlashAttention) under identical hardware conditions.
*   **`tests/ops/`**: Modifications to low-level operators (e.g., scheduling optimizations of Pallas kernels) must use the two comparison test categories above to verify results or gradient tolerances. It is strictly prohibited to overstep and rely on high-level tests (such as `test_gla.py` in the layers tier) as a workaround for validation.
*   **`tests/modules/` & `tests/layers/`**: Modifications at the network component or layer levels must include corresponding integration encapsulation and data flow validation tests.
*   **Default comparator and GPU-specific exception**: The default comparator is `tops/cpu/`. If GPU-based comparison is required (e.g., Torch/Triton), create a separate test file and append the `_gpu` suffix to its filename.

---

## 4. Release & Versioning Policy

The project uses an agile release strategy that combines "Trunk-based Development" and "Release Branches" to balance development efficiency with the stability of downstream dependencies (e.g., `ant-pretrain`).

### 4.1 Release Flow

1.  **Periodic Cut-off (RC Tag)**: When functionality on the `main` branch is ready for delivery, a `vX.Y.Z-rc.N` tag is created on `main` (e.g., `v1.0.0-rc.1`).
2.  **Pull Release Branch**: A release branch `release/vX.Y` (e.g., `release/v1.0`) is created based on the RC Tag. This branch serves as a "protective umbrella" for bug fixes during integration testing.
3.  **Integration Testing & Fixing**: Downstream projects (like `ant-pretrain`) depend on the RC version for testing. If bugs are found:
    *   Fixes are submitted to the `release/vX.Y` branch.
    *   **Mandatory**: Fixes must be cherry-picked back to the `main` branch to prevent regressions in future versions.
    *   A new RC Tag (e.g., `v1.0.0-rc.2`) is created on the release branch for further verification.
4.  **Final Promotion (Final Tag)**: Once downstream verification is complete and an RC version is confirmed stable, a final tag without a suffix (e.g., `v1.0.0`) is created on the release branch.

### 4.2 Versioning & Installation Rules

*   **External Source-Installation**: External repositories must use the `release/vX.Y` branch or its corresponding stable Tag as the installation source.
*   **Version Bump Rule**: Every release update must synchronously bump the project version (e.g., the version field in `pyproject.toml`). Releasing changed content without a version bump is strictly prohibited.
*   **Use of RC Suffix**: The `-rc.N` suffix is mandatory during the testing phase before a formal release to clearly indicate that the version has not been fully system-tested.

---

If a new feature needs to be added within an existing layer, find the level that best matches its granularity and adhere to that level's restrictions (and be sure to add corresponding tests). For new components whose classification is ambiguous, start by initiating a technical design brainstorm via `docs/design-docs/`.
