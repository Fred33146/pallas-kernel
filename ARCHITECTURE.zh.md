# 架构总览 (ARCHITECTURE)

本文档是本代码仓库的顶级导航地图，也是全体开发者（不论人类还是智能体）理解系统设计原则与模块分层的**唯一事实来源（System of Record）**。
基于“智能体优先的世界（Agent-First World）”工程理念，所有架构规范、依赖方向以及领域边界都被显式定义并强制执行，以消除理解偏差、防止架构漂移。

## 1. 核心分层与依赖原则 (Core Layers & Dependencies)

代码库围绕一个架构模型构建。我们推荐单向依赖规则，跨层（逆向）依赖目前被视为**警告级别（Warned）但非绝对禁止**。尽管允许，但在逆向调用时需具有充分的理由。

*   **底层尽量不依赖高层**：理想状况下，底层代码不应反向引用上层的业务逻辑或结构。
*   **明确的领域边界**：每个核心域具有有限的职能。

当前的包级抽象树（按自底向上依赖顺序排列）：

1.  **`tops/utils.py`**
    *   **职能**：提供最基础的通用辅助功能（如形状断言 `assert_shape_or_none`、对齐逻辑等）。
    *   **依赖**：一般不允许依赖仓库内任何其他包。
2.  **`tops/ops/`**
    *   **职能**：核心算子与底层计算内核实现（Pallas TPU/GPU 级别的低层次硬件交互），如 GLA 和简单 GLA 的变体内核。
    *   **规范**：高度优化，严禁包含任何与业务（模型训练状态、损失评估）相关的逻辑。需要与对应硬件特性强关联。
    *   **依赖**：推荐只依赖 `utils` 或纯数学/硬件原语包。
3.  **`tops/modules/`**
    *   **职能**：构建基石（Building Blocks）。提供具体的独立网络小构件（RMSNorm, Layernorm, Convolutions 等单层模型能力）。
    *   **依赖**：主要依赖 `tops/ops/` 以及 `tops/utils.py`。
4.  **`tops/layers/`**
    *   **职能**：高阶神经网络层（基于 Flax NNX 等），由下层的小构件级联和组合而成的复杂模型结构层。
    *   **依赖**：支持调用 `tops/modules/`, `tops/ops/` 和通用 `utils.py`。

**(基准依赖流：`utils` → `ops` → `modules` → `layers`，偏离此约定将触发警告)**

---

## 2. 智能体的可读性 (Agent Readability)

**只有在代码仓库里以版本控制留存的内容，对智能体和人类才是可见和可溯源的。**

*   **渐进式披露**：无需让每个智能体或新成员一上来就通读全库。从特定的核心点开始跳转（如 `CLAUDE.md` 或者本文档）。
*   **规划与设计系统 (System of Record)**：
    *   临时变动通过工单/轻量计划管理；
    *   复杂改造写入 `docs/exec-plans/` 目录；
    *   **Kernel 设计约束**：针对核心 Kernel（特别是复杂的 Pallas 算子），**强制要求所有 Kernel（无论是历史遗留还是后续新增）必须在 `docs/design-docs/ops/` 中添加或补齐相应的设计文档（包括但不限于公式、内存排布、Tiling 策略和硬件抽象等），作为开发和重构的唯一事实来源**。任何对 Kernel 的维护都必须建立在有设计文档的基础上。
*   **品味与审查规范**（Golden Rules）：
    1.  **强制的快速失败与断言机制**：我们鼓励尽可能丰富且严谨的张量及条件断言。作为底线约束，**所有公有计算接口/算子函数内不得呈现“零断言（Zero Asserts）”的状态**。至少必须存在一处基础形状或条件校验（`assert` 或 `assert_shape_or_none` 等）。我们将通过静态代码分析确保这一底线。
    2.  **对外公共 API 注释强制要求**：所有对外暴露的公共 API 必须提供完整注释/Docstring，且必须明确描述：
        *   Tensor 形状与维度含义（输入/输出逐项说明）；
        *   函数业务语义与行为边界；
        *   数据流（输入如何变换为输出，关键中间状态的语义）。
    3.  **Kernel 零崩溃保证（No-Crash-After-Assert）**：所有 Kernel 函数在通过入口断言（assert）校验之后，**设计上不允许出现任何运行时报错或 coredump**。即：若输入合法（未被 assert 拦截），则 Kernel 必须保证在任何合法输入条件下安全执行完毕并返回正确结果。断言是唯一允许的拒绝入口——通过了断言，就意味着承诺了执行安全。
    4.  **强制边界用例覆盖（Mandatory Edge-Case Testing）**：所有 Kernel 对应的测试用例，**必须包含边界与极端输入场景的覆盖**，例如：超大 batch、超大 sequence length (token)、极小维度、非对齐尺寸等。这类边界用例的预期行为是以下两者之一：**（a）被入口 assert 明确拦截并抛出 `AssertionError`**，以此验证 Kernel 的防御性断言完整且有效；或者 **（b）Kernel 能够正确处理该输入并返回正确结果**。若某个超出设计范围的输入既未被 assert 捕获、也未被正确处理（例如产生 coredump、静默错误或数值异常），则视为缺陷，需修复。

## 3. 测试与验证 (Validation Boundaries)

*   验证同样受到严格的层次划分和物理隔离，**对应层级的修改，必须有对应层级的测试进行确认**。

### 3.1 参考实现与参考测试 (Reference Implementations & Tests)

*   **`tops/cpu/`** 是标准参考实现（Gold/Reference）的唯一存放位置。所有参考实现使用纯 JAX 编写，在 CPU 上运行，作为标准答案输出（Goldens）。目录结构按算子域组织：

    ```
    tops/cpu/
    └── ops/
        ├── common/      # 共享的 chunk 工具函数参考实现
        ├── gla/          # GLA 算子参考实现
        ├── simple_gla/   # Simple GLA 算子参考实现
        └── mla/          # MLA 算子参考实现
    ```

*   **`tests/ref/`** 是参考对比测试的统一存放位置，采用与 `tops/cpu/ops/` 镜像的目录结构。这些测试将 `tops/cpu/` 的参考实现与其他已知正确的实现（如 PyTorch CPU/GPU）进行对比，以持续校验参考实现自身的正确性：

    ```
    tests/ref/
    ├── common/       # 共享工具函数的参考测试
    ├── gla/          # GLA 参考实现的对比测试
    ├── simple_gla/   # Simple GLA 参考实现的对比测试
    └── mla/          # MLA 参考实现的对比测试
    ```

*   **对齐要求**：所有新增到 `tops/ops/` 的 Pallas/JAX 实现，必须与 `tops/cpu/` 对齐并通过对照测试。
*   **Reference 正确性来源**：`tops/cpu/` 的正确性应通过 `tests/ref/` 中与 `torch_gpu/torch_cpu` 的对比持续校验，确保其可作为可信基线。

### 3.2 算子与层级测试 (Operator & Layer Tests)

*   在 `tests/` 目录下的测试类型被严格限制为以下两类标准的对比（Reference）：
    1.  **CPU 参考对比测试 (vs JAX-CPU)**：将 Pallas 内核的输出和梯度，与 `tops/cpu/` 中纯 JAX 编写的参考实现做容差（Tolerances）检查。
    2.  **GPU 参考对比测试 (vs Torch-GPU/Triton)**：将 Pallas 内核与已知正确、跨框架计算库（如基于 PyTorch 编写或者 FlashAttention 等现有高优组件）在同等硬件条件下的计算结果进行对齐。
*   **`tests/ops/`**：底层算子的修改（如 Pallas 内核的调度优化），必须采用上述两类对照测试验证结果或梯度容差，绝不可越权依赖高层级（如层的 `test_gla.py`）来变相验证。
*   **`tests/modules/` & `tests/layers/`**：网络单元层级的修改必须有对应的集成封装和数据流验证测试。
*   **默认对照对象与 GPU 例外规则**：默认测试与 `tops/cpu/` 参考实现比较；若需要做 GPU 对照（如 Torch/Triton），应单独新开测试文件，并在文件名增加 `_gpu` 后缀。

---

## 4. 发布与版本策略 (Release & Versioning Policy)

项目采用基于“主干开发（Trunk-based Development）”与“发布分支（Release Branch）”结合的敏捷发布策略，以兼顾开发效率与下游依赖（如 `ant-pretrain`）的稳定性。

### 4.1 发布流程 (Release Flow)

1.  **阶段性截断 (RC Tag)**：当 `main` 分支功能开发到可交付阶段，在 `main` 上打入 `vX.Y.Z-rc.N` 标签（例如 `v1.0.0-rc.1`）。
2.  **拉取发布分支 (Release Branch)**：基于该 RC Tag 拉取对应的发布分支 `release/vX.Y`（例如 `release/v1.0`）。此分支作为“保护伞”，用于后续集成测试期间的 Bug 修复。
3.  **集成测试与修复**：下游项目（如 `ant-pretrain`）依赖该 RC 版本进行测试。若发现 Bug：
    *   在 `release/X.Y` 分支上进行修复并提交。
    *   **必须**将修复代码同步（Cherry-pick）回 `main` 分支，防止 Bug 在未来版本中复现。
    *   在发布分支上打出新的 RC Tag（如 `v1.0.0-rc.2`）供下游继续验证。
4.  **正式转正 (Final Tag)**：当下游验证通过，确认某个 RC 版本稳定后，在发布分支的当前 Commit 上打出正式不带后缀的 Tag（如 `v1.0.0`）。

### 4.2 版本号与安装规则

*   **外部仓库源码安装入口**：外部仓库需要从本仓库源码安装时，必须以 `release/vX.Y` 分支或对应的稳定 Tag 作为安装来源。
*   **版本号更新规则**：每次发布内容更新都必须同步提升版本号（例如 `pyproject.toml` 中的版本字段），严禁“内容变更但版本不变”。
*   **RC 后缀的使用**：在正式发布前的测试阶段强制使用 `-rc.N` 后缀，以明确标识该版本未经完全系统测试，避免其他开发者误用。

---

如果需要在现有层内新增某项特性，寻找最符合其粒度的层级并遵循该层级的限制（且务必添加相应配套测试）。对于难以分辨的新组件，应先通过 `docs/design-docs/` 发起技术设计脑暴。
