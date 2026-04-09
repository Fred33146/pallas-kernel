# fused_recurrent_kda_fwd design

参考文献

https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf

本设计文档记录了 Kimi Delta Attention (KDA / Gated Delta Rule) 的 Fused Recurrent 前向内核的设计与实现逻辑。

### 核心思想与背景

Delta Rule 注意力的核心思想是"**误差驱动的记忆更新**"。与传统注意力无脑累加所有历史 $`K, V`$ 不同，系统维护一个关联键值的隐藏状态矩阵 $`\mathbf{S}_{t}`$。在写入新知识前，模型会先尝试用当前 $`\mathbf{k}_{t}`$ 检索已有记忆，如果记忆中已经能完美预测出 $`\mathbf{v}_{t}`$，则误差为 0，不进行更新；否则仅将残差（Delta）通过学习率/门控 $`\beta`$ 写入系统。

KDA 公式为：

$$\mathbf{S}_{t} = (\mathbf{I} - \beta_{t} \mathbf{k}_{t} \mathbf{k}_{t}^{\top} ) \text{Diag}(\boldsymbol{\alpha}_{t}) \mathbf{S}_{t-1} + \beta_{t} \mathbf{k}_{t} \mathbf{v}_{t}^{\top}$$

其中，$`\boldsymbol{\alpha}_{t} \in [0,1]^{K}`$ 是**通道级**衰减门向量（细粒度门控），$`\text{Diag}(\boldsymbol{\alpha}_{t})`$ 对状态做逐通道衰减；$`(\mathbf{I} - \beta_{t} \mathbf{k}_{t} \mathbf{k}_{t}^{\top})`$ 起到**选择性擦除 (Selective Erasure)** 的作用，为新记忆腾出空间。计算顺序为：先衰减，再删除。

然而，直接计算 $`\mathbf{k}_{t} \mathbf{k}_{t}^{\top} \mathbf{S}_{t-1}`$ 需要 $`O(K^{2}V)`$ 的稠密矩阵计算。本设计利用数学等价代数变换，将其转化为**残差更新形式**，仅需计算向量内积和外积，将计算复杂度完美降至 $`O(KV)`$，这是在 SRAM 中实现高效单步循环（Recurrent Loop）的核心机密。

---

### 符号定义

- $`t`$: 时间步，$`t = 1, \dots, T`$。
- $`\mathbf{q}_{t}, \mathbf{k}_{t} \in \mathbb{R}^{K}`$: Query 和 Key 向量。
- $`\mathbf{v}_{t} \in \mathbb{R}^{V}`$: Value 向量。
- $`\mathbf{S}_{t} \in \mathbb{R}^{K \times V}`$: 时刻 $`t`$ 的隐藏状态缓存矩阵（缓存在 SRAM/VMEM中）。
- $`g_{t} \in \mathbb{R}`$: 标量记忆衰减门控 (Scalar Decay)。
- $`\mathbf{gk}_{t} \in \mathbb{R}^{K}, \mathbf{gv}_{t} \in \mathbb{R}^{V}`$: Key/Value 维度的向量细粒度衰减。在 KDA 中，衰减门 $`\boldsymbol{\alpha}_{t} = \exp(\mathbf{gk}_{t}) \in \mathbb{R}^{K}`$ 是通道级向量。
- $`\beta_{t} \in \mathbb{R}`$ (或 $`\in \mathbb{R}^{V}`$): 写入保留率门控 (Write Gating / Learning Rate)。
- $`s`$: 缩放因子 (`scale`，通常为 $`1/\sqrt{K}`$)。

---

### 计算流程 (Recurrent Step)

#### 公式等价变换 (Mathematical Transformation)

直接计算 Delta Rule 的原始状态更新公式效率极低：

$$\mathbf{S}_{t} = (\mathbf{I} - \beta_{t} \mathbf{k}_{t} \mathbf{k}_{t}^{\top} ) \text{Diag}(\boldsymbol{\alpha}_{t}) \mathbf{S}_{t-1} + \beta_{t} \mathbf{k}_{t} \mathbf{v}_{t}^{\top}$$

因为 $`\mathbf{k}_{t} \mathbf{k}_{t}^{\top}`$ 会产生一个 $`K \times K`$ 的稠密矩阵，再与 $`K \times V`$ 的 $`\mathbf{S}_{t-1}`$ 相乘，计算复杂度达到了矩阵级的 $`O(K^{2} V)`$。

为了在单步循环中实现极致的硬件效率，我们对其进行代数展开和结合律变换：

1. **吸收衰减项**：定义衰减向量 $`\boldsymbol{\alpha}_{t} = \exp(\mathbf{g}_{t}) \in \mathbb{R}^{K}`$，令 $`\mathbf{S}_{t-1}^{\prime} = \text{Diag}(\boldsymbol{\alpha}_{t}) \mathbf{S}_{t-1}`$（即对每一行乘以对应的 $`\alpha_{t,k}`$），代入原式可得中间状态表示：

$$\mathbf{S}_{t} = (\mathbf{I} - \beta_{t} \mathbf{k}_{t} \mathbf{k}_{t}^{\top} ) \mathbf{S}_{t-1}^{\prime} + \beta_{t} \mathbf{k}_{t} \mathbf{v}_{t}^{\top}$$

2. **展开括号**：

$$\mathbf{S}_{t} = \mathbf{S}_{t-1}^{\prime} - \beta_{t} \mathbf{k}_{t} \mathbf{k}_{t}^{\top} \mathbf{S}_{t-1}^{\prime} + \beta_{t} \mathbf{k}_{t} \mathbf{v}_{t}^{\top}$$

3. **提取公因式 $`\mathbf{k}_{t}`$**：

$$\mathbf{S}_{t} = \mathbf{S}_{t-1}^{\prime} + \mathbf{k}_{t} \Big[ \beta_{t} ( \mathbf{v}_{t}^{\top} - \mathbf{k}_{t}^{\top} \mathbf{S}_{t-1}^{\prime} ) \Big]$$

经过变换后，括号内部的 $`\mathbf{k}_{t}^{\top} \mathbf{S}_{t-1}^{\prime}`$ 退化为了两个向量的内积（或者针对 $`\mathbf{S}`$ 列向量的点积），得到一个维度为 $`V`$ 的小向量。这就将原本的 $`O(K^{2} V)`$ 恶梦化解为了只需 $`O(KV)`$ 的纯向量运算。它也获得了明确的物理意义：**误差驱动更新**（Error-Driven Update）。

基于上述变换，在每一个时间步 $`t`$ 中，内核在隐藏状态 $`\mathbf{S}`$ 上的具体操作细分为以下 4 步：

#### 1. 记忆衰减 (Memory Decay)

应用基于时间/空间的门控对上一时刻状态 $`\mathbf{S}_{t-1}`$ 进行指数衰减：

$$\mathbf{S}_{t-1}^{\prime} = \mathbf{S}_{t-1} \odot \exp(g_{t} + \mathbf{gk}_{t} \mathbf{1}^{\top} + \mathbf{1}\mathbf{gv}_{t}^{\top})$$

#### 2. 回忆预测与残差计算 (Prediction & Delta Error)

用当前上下文 $`\mathbf{k}_{t}`$ 去"查询"衰减后的历史记忆，得到系统当前的预测值 $`\hat{\mathbf{v}}_{t}`$：

$$\hat{\mathbf{v}}_{t} = \mathbf{S}_{t-1}^{\prime\top} \mathbf{k}_{t}$$

计算真实的待存价值 $`\mathbf{v}_{t}`$ 与预测值之间的残差（Delta）：

$$\Delta\mathbf{v}_{t} = \mathbf{v}_{t} - \hat{\mathbf{v}}_{t}$$

#### 3. 门控调制与状态更新 (Beta Gating & State Update)

用写入门控 $`\beta_{t}`$ 调制残差，决定多少新信息需要被硬记入突触：

$$\tilde{\mathbf{v}}_{t} = \beta_{t} \odot \Delta\mathbf{v}_{t}$$

用外积 (Outer Product) 更新系统状态（实现 Rank-1 更新）：

$$\mathbf{S}_{t} = \mathbf{S}_{t-1}^{\prime} + \mathbf{k}_{t} \tilde{\mathbf{v}}_{t}^{\top}$$

#### 4. 模型输出计算 (Output Computation)

计算当前步的模型输出：

$$\mathbf{o}_{t} = s \cdot \mathbf{S}_{t}^{\top} \mathbf{q}_{t}$$

#### 5. FLA 参考代码映射 (Triton Implementation Mapping)

以上推导步骤在核心循环中的 Triton 原型实现（摘自 `fla` 库 `fused_recurrent_gated_delta_rule_fwd_kernel`）：

```python
# 1. 记忆衰减 (Memory Decay) -> S'_{t-1}
if USE_G:
    b_g = tl.load(p_g).to(tl.float32)
    b_h *= exp(b_g)
if USE_GK:
    b_gk = tl.load(p_gk).to(tl.float32)
    b_h *= exp(b_gk[:, None])
if USE_GV:
    b_gv = tl.load(p_gv).to(tl.float32)
    b_h *= exp(b_gv[None, :])

# 2 & 3. 预测、求残差、门控调制 (Predict, Delta Error & Beta Gating) -> \tilde{v}_t
# tl.sum(b_h * b_k[:, None], 0) 计算预测 \hat{v}_t = S^T k
b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))

# 3(续). 用外积更新记忆矩阵 (State Update)
b_h += b_k[:, None] * b_v

# 4. 模型输出计算 (Output Computation) -> o_t
b_o = tl.sum(b_h * b_q[:, None], 0)
tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
```

---

### Pallas 数据流与调度网格设计 (Grid Design)

#### 维度依赖分析与网格切分

在构建 Pallas 数据流时，核心原则是**将无依赖的维度切分发给 Grid，有依赖的维度留在 Block 内处理**：

- **无依赖维度**：样本 Batch 维度 `N` (或 `B`)、Query/Key 头维度 `H` 以及 Value 头维度 `HV` 彼此之间是完全独立的，没有任何数据依赖。
- **有依赖维度**：时间序列维度 `T` 存在严格的 RNN 隐状态依赖（$`S_{t}`$ 依赖 $`S_{t-1}`$）。
- **假设模型**：在 Pallas Kernel 设计中，我们**假设整个序列 `T`、或者通过 Chunk 技术切分后的 `T_chunk` 能够完全塞载进一块 TPU 的 VMEM 中**。

基于以上分析，为了最大化 GPU/TPU 并行效率，我们平铺所有独立无依赖的维度：

#### 数据排布与对齐要求 (Tensors Layout & Alignment)

按照 TPU Pallas 的对齐约束 `(block_dims[-1] % 128 == 0)`，以及优化内存 DMA 连续读取的考量。我们强制要求：

1. **特征维度对齐**：在 Host 端将 $`K`$ 和 $`V`$ 维度 `align_up` 补齐至 128 的倍数（即 `K_align = align_up(K, 128)`, `V_align = align_up(V, 128)`）。
2. **头维度外置 (Transpose)**：考虑到变长序列（Varlen）支持的情况下，样本 Batch `B` 和时序 `T` 通常会被 packing（打包）重塑为一个平铺的 `total_T` 维度（两者不可分隔）。因此我们要**直接将无依赖的 `H` / `HV` 维度 Transpose 到最外层/最高维度**，以保证剩余维度的绝对时序连续性。

预处理后的物理 Tensor Shape 如下：

- `Q`, `K`, `GK`: `[H, B, T, K_align]` _(变长场景下为 `[H, total_T, K_align]`)_
- `V`, `GV`: `[HV, B, T, V_align]` _(变长场景下为 `[HV, total_T, V_align]`)_
- `g`, `beta`: `[HV, B, T]` _(变长场景下为 `[HV, total_T]`)_
- `S_state` (SRAM): `[BK, BV]` 寄存器矩阵

#### Grid 调度

- $`K`$ 的规模通常较小（例如 64 或 128），直接令 `BK = K_align` 完全装入 SRAM 中。
- $`V`$ 通常也是 128，**默认情况下无需在 V 维度做切片，我们将完整的 V 维度全部放进块内（即 `BV = V_align`）**。一块 `[128, 128]` 或者 `[64, 128]` 规模的 `S_state` 寄存器状态矩阵完全可以轻松塞进 TPU VMEM 中。
- 除非模型结构极其特殊导致 $`V`$ 极大塞载不下，我们才退而求其次对 $`V`$ 进一步切块切片。
- 最终网格设计：默认打平调度为 `grid = [HV, B]` （变长场景下为 `[HV, N]`）。仅在开启大 V 退化切分兜底时，才向外拓展出第三维度变为 `grid = [CeilDiv(V_align, BV), HV, N]`。

这使得每一个 Block 独立负责**某一个特定的 Value Head**在完整的特征维度 $`V`$ 上的序列 `T` 历程。
通过 `i_h = i_hv // (HV // H)` 映射可以简单地在 SRAM 中跨 Head 获取对应共享的 Query/Key Head 数据。

### Varlen (变长序列支持)

对于 Packing 后的变长序列，其输入是展平后的连续 Token 流。

- **全载入假设**：在变长情况下，我们默认所有 batch 的总长度 $`T`$（即 $`\sum T_{i}`$）限制在一定大小以内，这样整个 batch 对应的连续时间步数据都可以完全塞入一块 TPU 的 VMEM 中进行处理。
- 在应用了"头维度外置"后，输入经过 Transpose 与 Flatten 后为：[H, $`\sum T_{i}`$, K]。
- 传入累积长度数组 `cu_seqlens` : `[N + 1]` (其中 $`N`$ 为变长 Packing 前的有效序列条数)。
- 网格中分配 $`N`$ 个并行实例给序列轴，默认调度即 `grid = [HV, N]`（若 V 需切分则拓展为三维）。
- 由于 $`H`$ 在最外层，各个 Head 对应的数据块完全连续独立，块内通过 `cu_seqlens(i_n)` 和 `cu_seqlens(i_n + 1)` 加载各自独属的变长子序列起始位置 `bos` 和结束位置 `eos`。
- 在 `bos` 处如果需要支持连续对话扩展/KV Cache 恢复，可增加从 `h0` 按需初始化的逻辑。

### Pallas Kernel Demo (伪代码)

基于上述"全载入假设"与维度映射，这里展示一段 KDA Fused Recurrent Forward 极简版 Pallas Kernel 伪代码。
（注：此处以处理某一条时间步长度为 `seq_len` 的序列为例。为突显核心的 $`O(KV)`$ 计算以及内存对应关系，忽略了 DMA 的 BlockSpec 复杂控制、流水线（Pipelining）和双缓冲（Double Buffering）细节）。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kda_fused_recurrent_fwd_kernel(
    q_ref, k_ref, v_ref, alpha_ref, beta_ref, o_ref,
    # 通过 Pallas 编译器参数 / block_size 传入的标量：
    seq_len: int, K: int, V: int
):
    # 1. 获取 Grid 坐标 (以 grid = [HV, N] 为例)
    hv_id = pl.program_id(0) # 执行所在的 Head / SubHead
    n_id  = pl.program_id(1) # 执行所在的 Batch ID / Varlen Sequence ID

    # (如果 Q/K 未随 HV 切分，可能需要：h_id = hv_id // (HV // H))

    # 2. S 状态寄存器初始化
    # S_state (SRAM / Register): [K, V]
    # 在 T 的遍历中复用，需要使用 float32 全精度防止严重舍入误差
    S = jnp.zeros((K, V), dtype=jnp.float32)

    # 3. 沿时间步 T 依次进行 Recurrence (T 次循环)
    # 对于变长 Packing 的情况，循环起止就是 cu_seqlens[n_id] 和 cu_seqlens[n_id+1]
    def time_step_fn(t, state):
        S_tm1 = state

        # --- 数据提取 (Data Fetch) ---
        # 假设数据已经全部由 BlockSpec 切入 TPU 的 VMEM
        # 对该 head 该样本在该时间步的特征进行索引
        q_t = q_ref[hv_id, n_id, t, :]     # [K]
        k_t = k_ref[hv_id, n_id, t, :]     # [K]
        v_t = v_ref[hv_id, n_id, t, :]     # [V]
        g_t = alpha_ref[hv_id, n_id, t, :] # [K] 通道级衰减门（对数空间）
        b_t = beta_ref[hv_id, n_id, t]     # [scalar]

        # --- 状态更新 (Error-driven State Update) ---
        # KDA 原始公式: S_t = (I - b_t * k_t * k_t^T) * Diag(exp(g_t)) * S_{t-1} + b_t * (k_t @ v_t^T)

        # 优化: 巧妙使用向量乘法优先策略，避免大矩阵乘。O(K^2 V) -> O(KV)
        # 3.1: 先做通道级衰减 Diag(α_t) * S_{t-1}
        S_decayed = S_tm1 * jnp.exp(g_t)[:, None]  # [K, V]

        # 3.2: 计算 k_t 与衰减后状态的点积 -> 预测值 shape 为 [V]
        hk_scalar = jnp.dot(k_t, S_decayed)

        # 3.3: Error-driven 残差更新
        # jnp.outer(k_t, ...) 可以被硬件转化为一系列 vector scaling & addition
        S_next = S_decayed + b_t * jnp.outer(k_t, v_t - hk_scalar)

        # --- 输出计算 (Output Generation) ---
        # o_t = q_t^T @ S_next
        o_t = jnp.dot(q_t, S_next)         # [V]

        # 写回引用 (随后通过 BlockSpec 同步回 HBM)
        o_ref[hv_id, n_id, t, :] = o_t

        return S_next

    # 驱动在 TPU VMEM 内的循环
    final_S = jax.lax.fori_loop(0, seq_len, time_step_fn, S)
```
