# chunk_kda_bwd_wy_dqkg_fused design

参考文献

- https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf
- FLA Triton 源码: `fla.ops.kda.chunk_bwd.chunk_kda_bwd_kernel_wy_dqkg_fused`

本设计文档记录了 Kimi Delta Attention (KDA) Chunk 反向传播中 **WY 表示下 dq/dk/dg 融合内核** 的设计与实现逻辑。该内核是 KDA Chunk Backward 中比较重要的一环，负责利用前两个阶段（dAv 和 dhu）产生的中间梯度 $`\mathbf{dh}`$ 和 $`\mathbf{dv}`$，一次性融合计算出 $`\mathbf{dq}, \mathbf{dk}, \mathbf{dv}_{\text{out}}, \mathbf{d\beta}, \mathbf{dg}, \mathbf{dA}`$ 六个梯度张量。

---

### 核心思想与背景

KDA 的 Chunk 反向传播采用三阶段分解策略：

1. **Stage 1 — dAv**: 计算注意力矩阵梯度 $`\mathbf{dA}_{qk}`$ 和值的初步梯度 $`\mathbf{dv}`$。
2. **Stage 2 — dhu**: 利用 chunk 间的隐状态递推关系，计算每个 chunk 的隐状态梯度 $`\mathbf{dh}`$，并累加更新 $`\mathbf{dv}`$。
3. **Stage 3 — WY dqkg fused (本内核)**: 融合计算所有剩余梯度。

本内核之所以称为 "WY fused"，是因为 KDA 的 Chunk 前向过程中使用了 **WY 表示法 (WY Representation)** 来高效实现 chunk 内的 Delta Rule 更新。WY 表示将 chunk 内连续的 rank-1 更新压缩为一个下三角矩阵 $`\mathbf{A}_{kk}`$（Akk inverse）和变换后的值 $`\mathbf{v}_{\text{new}}`$，使得 chunk 内操作可以高度并行化。反向传播时，需要对这些 WY 中间量求导。

#### Chunk 反向中本内核的定位

在前向 Chunk 流程中，每个 chunk 内部的计算可以概括为：

$$\mathbf{o}_{c} = s \cdot (\mathbf{q}_{c} \odot 2^{\mathbf{g}_{n}}) \cdot \mathbf{h}_{c} + (\mathbf{A}_{qk} \odot \mathbf{M}) \cdot \mathbf{v}_{\text{new},c}$$

其中 $`\mathbf{h}_{c} \in \mathbb{R}^{K \times V}`$ 是 chunk $`c`$ 起始时的隐状态，$`\mathbf{A}_{qk}`$ 是 chunk 内 query-key 的门控注意力矩阵。本内核接收来自前两个阶段的 $`\mathbf{dh}`$ 和 $`\mathbf{dv}`$，反向计算 chunk 内所有参数的梯度。

---

### 符号定义

| 符号 | 形状 | 含义 |
|------|------|------|
| $`B`$ | 标量 | Batch size |
| $`T`$ | 标量 | 序列总长度 |
| $`H`$ | 标量 | Head 数量 |
| $`K`$ | 标量 | Key / Query 维度 |
| $`V`$ | 标量 | Value 维度 |
| $`\text{BT}`$ | 标量 | Chunk 大小（默认 64），$`T`$ 必须是 $`\text{BT}`$ 的整数倍 |
| $`\text{NT}`$ | 标量 | Chunk 数量，$`\text{NT} = T / \text{BT}`$ |
| $`\text{BK}`$ | 标量 | K 维度的 tile 大小，$`K`$ 必须是 $`\text{BK}`$ 的整数倍 |
| $`\text{BV}`$ | 标量 | V 维度的 tile 大小，$`V`$ 必须是 $`\text{BV}`$ 的整数倍 |
| $`\mathbf{q}`$ | $`[B, T, H, K]`$ | Query 张量 |
| $`\mathbf{k}`$ | $`[B, T, H, K]`$ | Key 张量 |
| $`\mathbf{v}`$ | $`[B, T, H, V]`$ | 原始 Value 张量 |
| $`\mathbf{v}_{\text{new}}`$ | $`[B, T, H, V]`$ | WY 变换后的 Value 张量 |
| $`\mathbf{g}`$ | $`[B, T, H, K]`$ | 累积衰减门（log-space，base-2 缩放） |
| $`\beta`$ | $`[B, T, H]`$ | WY beta 系数（写入门控） |
| $`\mathbf{A}`$ | $`[B, T, H, \text{BT}]`$ | $`\mathbf{A}_{kk}`$ 逆矩阵（chunk 内下三角） |
| $`\mathbf{h}`$ | $`[B, \text{NT}, H, K, V]`$ | 每个 chunk 起始的隐状态 |
| $`\mathbf{do}`$ | $`[B, T, H, V]`$ | 输出梯度 |
| $`\mathbf{dh}`$ | $`[B, \text{NT}, H, K, V]`$ | 隐状态梯度（来自 Stage 2） |
| $`\mathbf{dv}`$ | $`[B, T, H, V]`$ | 值梯度（来自 Stage 1 & 2 的累加） |
| $`s`$ | 标量 | 缩放因子，通常为 $`1/\sqrt{K}`$ |

---

### 输出梯度

| 输出 | 形状 | 含义 |
|------|------|------|
| $`\mathbf{dq}`$ | $`[B, T, H, K]`$ | Query 梯度 (float32) |
| $`\mathbf{dk}`$ | $`[B, T, H, K]`$ | Key 梯度 (float32) |
| $`\mathbf{dv}_{\text{out}}`$ | $`[B, T, H, V]`$ | Value 梯度输出 (float32) |
| $`\mathbf{d\beta}`$ | $`[B, T, H]`$ | Beta 梯度 (float32) |
| $`\mathbf{dg}`$ | $`[B, T, H, K]`$ | 门控梯度 (float32) |
| $`\mathbf{dA}`$ | $`[B, T, H, \text{BT}]`$ | $`\mathbf{A}_{kk}^{-1}`$ 梯度 (float32)，严格下三角 |

---

### 计算流程 (Per-Chunk Backward)

对于每个 chunk $`c`$，取对应的 chunk-local 切片。以下公式中省略 chunk 下标，所有张量均为 chunk 内的局部切片。$`K`$ 和 $`V`$ 维度分别被切分为 $`\text{NK} = K / \text{BK}`$ 和 $`\text{NV} = V / \text{BV}`$ 个 tile，通过嵌套的 K-V 二重循环处理。

#### 1. 内层 V-Loop 累加（对固定的 K-tile $`i_k`$）

对每个 V-tile $`i_v`$，从 $`\mathbf{h}`$、$`\mathbf{dh}`$、$`\mathbf{do}`$、$`\mathbf{dv}`$、$`\mathbf{v}_{\text{new}}`$ 中切出 $`[\text{BK}, \text{BV}]`$ 或 $`[\text{BT}, \text{BV}]`$ 的子块，累加以下梯度：

$$\mathbf{b\_dgk} \mathrel{+}= \sum_{v} \mathbf{h}_{[i_k, i_v]} \odot \mathbf{dh}_{[i_k, i_v]} \quad \in \mathbb{R}^{\text{BK}}$$

$$\mathbf{b\_dq}_k \mathrel{+}= \mathbf{do}_{[:, i_v]} \cdot \mathbf{h}_{[i_k, i_v]}^{\top} \quad \in \mathbb{R}^{\text{BT} \times \text{BK}}$$

$$\mathbf{b\_dk}_k \mathrel{+}= \mathbf{v\_new}_{[:, i_v]} \cdot \mathbf{dh}_{[i_k, i_v]}^{\top} \quad \in \mathbb{R}^{\text{BT} \times \text{BK}}$$

$$\mathbf{b\_dw}_k \mathrel{+}= \mathbf{dv}_{[:, i_v]} \cdot \mathbf{h}_{[i_k, i_v]}^{\top} \quad \in \mathbb{R}^{\text{BT} \times \text{BK}}$$

**仅当 $`i_k = 0`$ 时**（避免重复计算）：

$$\mathbf{dA}_{\text{acc}} \mathrel{+}= \mathbf{dv}_{[:, i_v]} \cdot \mathbf{v}_{[:, i_v]}^{\top} \quad \in \mathbb{R}^{\text{BT} \times \text{BT}}$$

$$\mathbf{dv}_{\text{out},[:, i_v]} = (\mathbf{A}^{\top} \cdot \mathbf{dv}_{[:, i_v]}) \odot \beta_{[:, \text{None}]}$$

$$\mathbf{d\beta}_{\text{acc}} \mathrel{+}= \sum_{v} (\mathbf{A}^{\top} \cdot \mathbf{dv}_{[:, i_v]}) \odot \mathbf{v}_{[:, i_v]}$$

#### 2. 外层 K-Loop 门控应用

V-loop 结束后，对当前 K-tile 应用门控指数变换：

$$\mathbf{gk\_exp} = 2^{\mathbf{g}_{[:, i_k]}} \quad \in \mathbb{R}^{\text{BT} \times \text{BK}}$$

$$\mathbf{b\_dq}_k = \mathbf{b\_dq}_k \odot \mathbf{gk\_exp} \cdot s$$

$$\mathbf{b\_dk}_k = \mathbf{b\_dk}_k \odot 2^{(\mathbf{g}_n - \mathbf{g}_{[:, i_k]})}$$

$$\mathbf{b\_dgk} = \mathbf{b\_dgk} \odot 2^{\mathbf{g}_n}$$

其中 $`\mathbf{g}_n = \mathbf{g}[\text{BT}-1, i_k]`$ 是 chunk 内最后一个时间步在当前 K-tile 上的门控值。

#### 3. dA 和 dk 的 WY 修正

$$\mathbf{kg} = \mathbf{k}_{[:, i_k]} \odot \mathbf{gk\_exp}$$

$$\mathbf{dA}_{\text{acc}} \mathrel{+}= (-\mathbf{b\_dw}_k) \cdot \mathbf{kg}^{\top}$$

$$\mathbf{dkgb} = \mathbf{A}^{\top} \cdot (-\mathbf{b\_dw}_k)$$

$$\mathbf{d\beta}_{\text{acc}} \mathrel{+}= \sum_{k} \mathbf{dkgb} \odot \mathbf{kg}$$

$$\mathbf{b\_dk}_k \mathrel{+}= \mathbf{dkgb} \odot \mathbf{gk\_exp} \odot \beta_{[:, \text{None}]}$$

#### 4. 门控梯度 dg 计算

$$\mathbf{kdk} = \mathbf{k}_{[:, i_k]} \odot \mathbf{b\_dk}_k$$

$$\mathbf{b\_dgk\_sum} = \mathbf{b\_dgk} + \sum_{t} \mathbf{kdk}$$

$$\mathbf{dg}_{[:, i_k]} = \mathbf{q}_{[:, i_k]} \odot \mathbf{b\_dq}_k - \mathbf{kdk} + \mathbf{m}_{\text{last}} \odot \mathbf{b\_dgk\_sum} + \mathbf{kg} \odot \mathbf{dkgb} \odot \beta_{[:, \text{None}]}$$

其中 $`\mathbf{m}_{\text{last}}`$ 是一个 one-hot 向量，仅在 chunk 最后一个时间步为 1。

#### 5. dA 后处理（矩阵逆梯度）

K-loop 结束后，对累积的 $`\mathbf{dA}`$ 进行矩阵逆梯度的链式法则展开：

$$\mathbf{dA} = \text{lower}(\mathbf{dA} \odot \beta_{[\text{None}, :]})$$

$$\mathbf{dA} = \mathbf{dA} \cdot \mathbf{A}^{\top}$$

$$\mathbf{dA} = \mathbf{A}^{\top} \cdot \mathbf{dA}$$

$$\mathbf{dA} = \text{lower}(-\mathbf{dA})$$

其中 $`\text{lower}(\cdot)`$ 表示仅保留严格下三角部分（对角线及以上置零）。这保证了 $`\mathbf{dA}`$ 维持与 $`\mathbf{A}_{kk}`$ 相同的严格下三角结构。

---

### FLA 参考代码映射 (Triton Implementation Mapping)

本内核对应 FLA 库中的 `chunk_kda_bwd_kernel_wy_dqkg_fused`。以下是核心计算逻辑的 Triton 源码对应关系：

```python
# V-loop 内层累加
b_dgk += tl.sum(b_h * b_dh, axis=1)       # dgk: sum(h * dh) over V
b_dq += tl.dot(b_do, tl.trans(b_h))       # dq += do @ h^T
b_dk += tl.dot(b_v_new, tl.trans(b_dh))   # dk += v_new @ dh^T
b_dw += tl.dot(b_dv, tl.trans(b_h))       # dw += dv @ h^T

# K-loop 门控应用
b_dq *= tl.exp2(b_g) * scale               # dq *= exp2(g) * scale
b_dk *= tl.exp2(b_gn[None, :] - b_g)       # dk *= exp2(gn - g)

# WY 修正
b_dA += tl.dot((-b_dw), tl.trans(b_kg))    # dA += -dw @ kg^T
b_dkgb = tl.dot(b_A, (-b_dw))              # dkgb = A^T @ (-dw)

# dA 后处理（矩阵逆梯度）
b_dA = tl.where(m_lower, b_dA * b_beta[None, :], 0)
b_dA = tl.dot(b_dA, b_A)                   # dA = dA @ A^T
b_dA = tl.dot(b_A, b_dA)                   # dA = A^T @ dA
b_dA = tl.where(m_lower, -b_dA, 0)
```

---

### 实现方案

本次实现提供了两个版本：

#### 1. CPU 参考实现 (`tops/cpu/ops/kda/chunk_bwd.py`)

- 使用 `@cpu_reference` 装饰器标记为 CPU 参考实现。
- 通过 `jax.vmap` 在 (batch, head) 维度上并行化，chunk 维度同样通过 `jax.vmap` 并行化。
- 内部使用 `jax.lax.fori_loop` 实现 K-tile 和 V-tile 的双重循环。
- 使用 `jax.lax.cond` 实现 $`i_k = 0`$ 的条件分支（dA/dv/db 仅在第一个 K-tile 计算）。
- 输入自动 padding 至 chunk_size 的整数倍。

#### 2. Pallas Kernel 实现 (`tops/ops/kda/chunk_bwd.py`)

- 使用 `pl.pallas_call` 封装，grid 调度为 `(BH * NT,)` 即所有 (batch, head, chunk) 组合的一维展平。
- 内部同样使用 `jax.lax.fori_loop` 实现 K-V 双重 tile 循环。
- 条件分支 $`i_k = 0`$ 使用 `jnp.where` 替代 `jax.lax.cond`，以兼容 Pallas trace 限制。
- 支持 `interpret` 模式（CPU 模拟）和 TPU 原生模式。

---

### 测试策略

提供两套测试，均使用 Triton forward 前向的真实中间数据作为输入：

| 测试文件 | 被测目标 | 说明 |
|----------|----------|------|
| `tests/ref/kda/test_chunk_bwd.py` | CPU 参考实现 | JAX CPU ref vs Triton，验证算法正确性 |
| `tests/ref/kda/test_chunk_bwd_pallas.py` | Pallas kernel | JAX Pallas vs Triton，验证 kernel 正确性 |

#### 测试参数覆盖

```python
@pytest.mark.parametrize("B,T,H,K,V", [
    (1, 64, 1, 64, 64),      # 最小 single-chunk
    (2, 128, 2, 64, 64),     # multi-batch multi-head multi-chunk
    (1, 192, 4, 32, 32),     # 非对称 K/V，3 chunks
    (2, 64, 2, 32, 64),      # K < V
    (2, 2048, 8, 128, 128),  # 大规模压力测试
])
```

#### 验证项目

1. **数值正确性**: 6 个输出梯度（dq, dk, dv, db, dg, dA）均与 Triton 参考实现逐元素比对，默认容差 `atol=1e-5, rtol=1e-5`。
2. **形状正确性**: 验证所有输出张量的 shape 符合预期。
3. **dA 结构正确性**: 验证 dA 在每个 chunk 内的上三角部分（含对角线）严格为零，保持下三角结构不变性。

---

### 文件结构

```
tops/
├── ops/
│   ├── kda/
│   │   └── chunk_bwd.py          # Pallas kernel 实现
│   └── utils.py                   # 新增 exp2()
├── cpu/
│   └── ops/
│       └── kda/
│           └── chunk_bwd.py      # CPU 参考实现
tests/
└── ref/
    └── kda/
        ├── test_chunk_bwd.py         # CPU ref vs Triton 测试
        └── test_chunk_bwd_pallas.py  # Pallas vs Triton 测试
```
