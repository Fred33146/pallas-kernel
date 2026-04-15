# chunk_kda_bwd design

参考文献

https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf

本设计文档记录了 Kimi Delta Attention (KDA) 的 Chunk-Parallel **反向**内核的设计与实现逻辑。阅读本文档前，请先阅读 [chunk_kda_fwd design](./chunk-fwd.md)。

### 核心思想与背景

反向传播的目标：给定损失函数对输出的梯度 $`\frac{\partial \mathcal{L}}{\partial \mathbf{o}}`$（记为 $`d\mathbf{o}`$），计算损失函数对所有输入 $`\mathbf{q}, \mathbf{k}, \mathbf{v}, \mathbf{g}, \boldsymbol{\beta}`$ 的梯度，以及对初始状态 $`\mathbf{h}_0`$ 的梯度。

反向传播按照**前向计算图的逆序**执行。回顾前向的 4 步流水线：

```
Step 0: g = cumsum(g_raw)                                           [chunk_local_cumsum]
Step 1: w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(q, k, v, g, β)   [chunk_kda_fwd_intra]
Step 2: h, v_new = chunk_gated_delta_rule_fwd_h(kg, w, u, gk=g)    [chunk_gated_delta_rule_fwd_h]
Step 3: o = chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h)                [chunk_gla_fwd_o_gk]
```

反向按 Step 3 → 2 → 1 → 0 的顺序执行。但 FLA 的实际实现为了减少显存、融合计算，将多个反向步骤进行了**重组和融合**。

---

### 符号定义

继承 [chunk-fwd.md](./chunk-fwd.md) 的所有符号，新增以下梯度符号：

- $`d\mathbf{o} \in \mathbb{R}^{C \times V}`$: 损失对输出 $`\mathbf{o}`$ 的梯度（上游传入）。
- $`d\mathbf{h} \in \mathbb{R}^{K \times V}`$: 损失对每个 chunk 起始状态 $`\mathbf{h}^{[t]}`$ 的梯度。
- $`d\mathbf{h}_0 \in \mathbb{R}^{K \times V}`$: 损失对初始状态的梯度。
- $`d\mathbf{A}_{qk} \in \mathbb{R}^{C \times C}`$: 损失对查询-键注意力矩阵的梯度。
- $`d\mathbf{A}_{kk} \in \mathbb{R}^{C \times C}`$: 损失对键-键交互矩阵（逆）的梯度。
- $`d\mathbf{v}_{\text{new}} \in \mathbb{R}^{C \times V}`$: 损失对 delta rule 修正后值的梯度。
- $`d\mathbf{w} \in \mathbb{R}^{C \times K}`$: 损失对有效键的梯度。
- $`d\mathbf{u} \in \mathbb{R}^{C \times V}`$: 损失对有效值的梯度。
- $`d\mathbf{q}, d\mathbf{k} \in \mathbb{R}^{C \times K}`$, $`d\mathbf{v} \in \mathbb{R}^{C \times V}`$, $`d\mathbf{g} \in \mathbb{R}^{C \times K}`$, $`d\boldsymbol{\beta} \in \mathbb{R}^{C}`$: 损失对原始输入的梯度。

---

### 计算流程 (Chunk-Parallel Backward)

#### 整体流水线概览

FLA 的 chunk_kda_bwd 由 6 个阶段组成：

```
recompute_w_u_fwd ──→ chunk_gated_delta_rule_fwd_h ──→ chunk_kda_bwd_dAv
    (重算前向)              (重算前向)                    (Step 3 反向)
                                                            │
                                                            ▼
chunk_kda_bwd_intra ◀── chunk_kda_bwd_wy_dqkg_fused ◀── chunk_gated_delta_rule_bwd_dhu
   (Step 1 反向-B)          (Step 1 反向-A)                (Step 2 反向)
```

| 阶段 | FLA 函数 | 文件路径 | 功能 |
|---|---|---|---|
| Recomp 1 | `recompute_w_u_fwd` | `fla/ops/kda/wy_fast.py` | 从保存的 $`\mathbf{A}_{kk}`$ 重算 $`\mathbf{w}, \mathbf{u}, \text{qg}, \text{kg}`$ |
| Recomp 2 | `chunk_gated_delta_rule_fwd_h` | `fla/ops/common/chunk_delta_h.py` | 重算 $`\mathbf{h}, \mathbf{v}_{\text{new}}`$ |
| Bwd Step 3 | `chunk_kda_bwd_dAv` | `fla/ops/kda/chunk_bwd.py` | $`d\mathbf{o} \to d\mathbf{A}_{qk}, d\mathbf{v}_{\text{new}}`$ |
| Bwd Step 2 | `chunk_gated_delta_rule_bwd_dhu` | `fla/ops/common/chunk_delta_h.py` | $`d\mathbf{o}, d\mathbf{v}_{\text{new}} \to d\mathbf{h}, d\mathbf{h}_0, d\mathbf{v}_{\text{new}}'`$ |
| Bwd Step 1-A | `chunk_kda_bwd_wy_dqkg_fused` | `fla/ops/kda/chunk_bwd.py` | $`d\mathbf{h}, d\mathbf{v}_{\text{new}}' \to d\mathbf{q}, d\mathbf{k}, d\mathbf{v}, d\boldsymbol{\beta}, d\mathbf{g}, d\mathbf{A}_{kk}`$ |
| Bwd Step 1-B | `chunk_kda_bwd_intra` | `fla/ops/kda/chunk_intra.py` | $`d\mathbf{A}_{qk}, d\mathbf{A}_{kk} \to d\mathbf{q}, d\mathbf{k}, d\boldsymbol{\beta}, d\mathbf{g}`$（累加） |

_注：Step 0 (cumsum) 的反向是 reverse cumsum，在 Bwd Step 1-B 结束后对 $`d\mathbf{g}`$ 执行。_

下面逐步详细说明每个阶段的数学推导与代码映射。

---

#### 阶段 0. 前向重算 (Recomputation)

反向传播需要前向中间量 $`\mathbf{w}, \mathbf{u}, \text{kg}, \text{qg}, \mathbf{h}, \mathbf{v}_{\text{new}}`$，但为节省显存，前向时仅保存 $`\mathbf{A}_{kk}, \mathbf{A}_{qk}`$，在反向开始时**重新计算**这些中间量。

##### FLA 代码映射 (`chunk_kda_bwd`)

```python
# 从保存的 Akk 重算 w, u, qg, kg
w, u, qg, kg = recompute_w_u_fwd(
    q=q, k=k, v=v, beta=beta, A=Akk, gk=g, ...
)

# 重算块间状态 h 和修正值 v_new
h, v_new, _ = chunk_gated_delta_rule_fwd_h(
    k=kg, w=w, u=u, gk=g, initial_state=initial_state, ...
)
```

---

#### 阶段 1. 输出层反向 (Bwd Step 3 → chunk_kda_bwd_dAv)

##### 1.1 数学推导

回顾前向 Step 3 的输出公式（每个 chunk 内）：

$$\mathbf{o} = \underbrace{s \cdot (\mathbf{q} \odot \exp(\mathbf{g})) \cdot \mathbf{h}}_{\text{inter-chunk 项}} + \underbrace{\text{tril}(\mathbf{A}_{qk}) \cdot \mathbf{v}_{\text{new}}}_{\text{intra-chunk 项}}$$

这是两个独立项之和。反向传播时，$`d\mathbf{o}`$ 的梯度会分别流入这两项。阶段 1 只处理 **intra-chunk 项**的反向（inter-chunk 项的反向在阶段 2 和 3 中处理）。

**第一步：识别矩阵乘法的梯度规则**

Intra-chunk 项的形式是 $`\mathbf{Y} = \mathbf{A} \cdot \mathbf{X}`$，其中 $`\mathbf{A} = \text{tril}(\mathbf{A}_{qk}) \in \mathbb{R}^{C \times C}`$，$`\mathbf{X} = \mathbf{v}_{\text{new}} \in \mathbb{R}^{C \times V}`$。

对于矩阵乘法 $`\mathbf{Y} = \mathbf{A} \cdot \mathbf{X}`$，给定上游梯度 $`d\mathbf{Y}`$，链式法则给出：

$$d\mathbf{A} = d\mathbf{Y} \cdot \mathbf{X}^\top, \quad d\mathbf{X} = \mathbf{A}^\top \cdot d\mathbf{Y}$$

这两个公式的直觉：$`d\mathbf{A}`$ 需要把 $`d\mathbf{Y}`$ 的每一行与 $`\mathbf{X}`$ 的对应行做外积求和（即矩阵乘法 $`d\mathbf{Y} \cdot \mathbf{X}^\top`$）；$`d\mathbf{X}`$ 需要把 $`\mathbf{A}`$ 的每一列的权重作用到 $`d\mathbf{Y}`$ 上（即 $`\mathbf{A}^\top \cdot d\mathbf{Y}`$）。

**第二步：求 $`d\mathbf{A}_{qk}`$**

直接代入规则，得到未 mask 的梯度：

$$d\mathbf{A}_{qk}^{\text{raw}} = d\mathbf{o} \cdot \mathbf{v}_{\text{new}}^\top \in \mathbb{R}^{C \times C}$$

但前向只使用了 $`\mathbf{A}_{qk}`$ 的**下三角部分**（$`r \geq j`$），上三角部分被 mask 为 0，对输出没有贡献。因此上三角位置的梯度也为 0（不存在的通路没有梯度流过）。同时乘以 scale $`s`$：

$$d\mathbf{A}_{qk}(r, j) = \begin{cases} s \cdot \sum_{v=1}^{V} d\mathbf{o}_{r,v} \cdot \mathbf{v}_{\text{new},j,v} & \text{if } r \geq j \\ 0 & \text{if } r < j \end{cases}$$

逐元素展开来看：$`d\mathbf{A}_{qk}(r, j)`$ 就是 $`d\mathbf{o}`$ 的第 $`r`$ 行（$`V`$ 维向量）与 $`\mathbf{v}_{\text{new}}`$ 的第 $`j`$ 行（$`V`$ 维向量）的**内积**，再乘以 $`s`$。这度量的是：位置 $`j`$ 的修正值 $`\mathbf{v}_{\text{new},j}`$ 对位置 $`r`$ 的输出梯度 $`d\mathbf{o}_r`$ 的贡献。

**第三步：求 $`d\mathbf{v}_{\text{new}}`$**

代入规则 $`d\mathbf{X} = \mathbf{A}^\top \cdot d\mathbf{Y}`$：

$$d\mathbf{v}_{\text{new}} = \text{tril}(\mathbf{A}_{qk})^\top \cdot d\mathbf{o}$$

转置的原因：前向中 $`\mathbf{A}_{qk}`$ 的第 $`r`$ 行权重了 $`\mathbf{v}_{\text{new}}`$ 的各行来生成 $`\mathbf{o}_r`$。反向时，$`d\mathbf{o}_r`$ 需要按相同的权重**反向分配**回 $`\mathbf{v}_{\text{new}}`$ 的各行——这恰好是矩阵乘法的转置。

逐元素来看：

$$d\mathbf{v}_{\text{new},j} = \sum_{r \geq j} A_{qk}(r, j) \cdot d\mathbf{o}_r$$

即位置 $`j`$ 的修正值接收来自所有 $`r \geq j`$ 的输出梯度，权重为 $`A_{qk}(r, j)`$——这正是因果掩码的转置方向（从"读者"反传到"被读者"）。

##### FLA 代码映射 (`chunk_kda_bwd`)

```python
dAqk, dv = chunk_kda_bwd_dAv(
    q=q, k=k, v=v_new, do=do, A=Aqk, scale=scale, ...
)
```

##### Triton Kernel 核心计算 (`chunk_kda_bwd_kernel_dAv`)

```python
# 循环遍历 V 维度块
for i_v in range(cdiv(V, BV)):
    b_do = tl.load(p_do)           # [BT, BV]
    b_v = tl.load(p_v)             # [BT, BV]

    # dAqk += do @ v_new^T    (累积多个 V 块)
    b_dA += tl.dot(b_do, tl.trans(b_v))    # [BT, BT]

    # dv_new = Aqk^T @ do     (每个 V 块独立存回)
    b_dv = tl.dot(tl.trans(b_A), b_do)     # [BT, BV]
    tl.store(p_dv, b_dv)

# 下三角掩码 + scale
b_dA = tl.where(row >= col, b_dA * scale, 0)
```

---

#### 阶段 2. 块间递推反向 (Bwd Step 2 → chunk_gated_delta_rule_bwd_dhu)

##### 2.1 数学推导

回顾前向 Step 2 中涉及 $`\mathbf{h}^{[t]}`$ 的三个计算（每个 chunk 内）：

$$\text{(I) 输出路径: } \mathbf{o}_{\text{inter}}^{[t]} = s \cdot (\mathbf{q} \odot \exp(\mathbf{g}))^\top \cdot \mathbf{h}^{[t]}$$

$$\text{(II) Delta Rule 修正: } \mathbf{v}_{\text{new}}^{[t]} = \mathbf{u} - \mathbf{w} \cdot \mathbf{h}^{[t]}$$

$$\text{(III) 状态递推: } \mathbf{h}^{[t+1]} = \exp(\mathbf{g}_C) \odot \mathbf{h}^{[t]} + \text{kg}^\top \cdot \mathbf{v}_{\text{new}}^{[t]}$$

$`\mathbf{h}^{[t]}`$ 同时参与这三个计算，因此 $`d\mathbf{h}^{[t]}`$ 是三条路径梯度之和。下面逐一推导。

**路径 1：来自输出的贡献 $`d\mathbf{h}_{\text{out}}^{[t]}`$**

前向公式 (I) 中，定义 $`\text{qg} = \mathbf{q} \odot \exp(\mathbf{g}) \in \mathbb{R}^{C \times K}`$，则 $`\mathbf{o}_{\text{inter}} = s \cdot \text{qg} \cdot \mathbf{h}`$。

这是矩阵乘法 $`\mathbf{Y} = s \cdot \mathbf{A} \cdot \mathbf{X}`$ 的形式，其中 $`\mathbf{A} = \text{qg} \in \mathbb{R}^{C \times K}`$，$`\mathbf{X} = \mathbf{h} \in \mathbb{R}^{K \times V}`$。

应用矩阵乘法梯度规则 $`d\mathbf{X} = \mathbf{A}^\top \cdot d\mathbf{Y}`$：

$$d\mathbf{h}_{\text{out}}^{[t]} = s \cdot \text{qg}^\top \cdot d\mathbf{o} = s \cdot (\mathbf{q} \odot \exp(\mathbf{g}))^\top \cdot d\mathbf{o} \in \mathbb{R}^{K \times V}$$

直觉：$`\text{qg}^\top \cdot d\mathbf{o}`$ 把 $`d\mathbf{o}`$ 的每一行（$`V`$ 维）按 $`\text{qg}`$ 的权重累加到 $`K`$ 个通道上——相当于"哪些通道的状态对输出贡献大，梯度就大"。

**路径 2：来自 Delta Rule 修正的贡献 $`d\mathbf{h}_{\text{delta}}^{[t]}`$**

前向公式 (II) 中，$`\mathbf{v}_{\text{new}} = \mathbf{u} - \mathbf{w} \cdot \mathbf{h}`$。$`\mathbf{h}`$ 通过 $`-\mathbf{w} \cdot \mathbf{h}`$ 项影响 $`\mathbf{v}_{\text{new}}`$。

这是 $`\mathbf{Y} = -\mathbf{A} \cdot \mathbf{X}`$ 的形式（$`\mathbf{A} = \mathbf{w} \in \mathbb{R}^{C \times K}`$，$`\mathbf{X} = \mathbf{h} \in \mathbb{R}^{K \times V}`$），梯度为：

$$d\mathbf{h}_{\text{delta}}^{[t]} = -\mathbf{w}^\top \cdot d\mathbf{v}_{\text{new}} \in \mathbb{R}^{K \times V}$$

负号来自前向公式中 $`\mathbf{h}`$ 前面的减号：状态中已有的信息通过 $`\mathbf{w}`$ 被**减去**（delta rule 修正），所以梯度也带负号。

**路径 3：来自状态递推的贡献 $`d\mathbf{h}_{\text{recur}}^{[t]}`$**

前向公式 (III) 中，$`\mathbf{h}^{[t+1]} = \exp(\mathbf{g}_C) \odot \mathbf{h}^{[t]} + \text{kg}^\top \cdot \mathbf{v}_{\text{new}}`$。$`\mathbf{h}^{[t]}`$ 通过逐元素乘 $`\exp(\mathbf{g}_C) \odot \mathbf{h}^{[t]}`$ 传递到下一个 chunk。

逐元素乘 $`\mathbf{Y} = \mathbf{a} \odot \mathbf{X}`$（$`\mathbf{a}`$ 为常数向量，逐行广播）的梯度是 $`d\mathbf{X} = \mathbf{a} \odot d\mathbf{Y}`$，因此：

$$d\mathbf{h}_{\text{recur}}^{[t]} = \exp(\mathbf{g}_C) \odot d\mathbf{h}^{[t+1]}$$

这就是**反向时间递推**的来源：chunk $`t`$ 的状态梯度包含了来自 chunk $`t+1`$ 的状态梯度经衰减后的贡献。从最后一个 chunk 开始，每一步都把 $`d\mathbf{h}^{[t+1]}`$ 乘以衰减 $`\exp(\mathbf{g}_C)`$ 后传递给 $`d\mathbf{h}^{[t]}`$——这与前向递推的方向恰好相反。

**三条路径合并**：

$$d\mathbf{h}^{[t]} = \underbrace{s \cdot (\mathbf{q} \odot \exp(\mathbf{g}))^\top \cdot d\mathbf{o}}_{\text{路径 1: 输出}} \underbrace{- \mathbf{w}^\top \cdot d\mathbf{v}_{\text{new}}}_{\text{路径 2: delta rule}} + \underbrace{\exp(\mathbf{g}_C) \odot d\mathbf{h}^{[t+1]}}_{\text{路径 3: 递推}}$$

**$`d\mathbf{v}_{\text{new}}`$ 的更新**

前向公式 (III) 中，$`\mathbf{v}_{\text{new}}`$ 还通过 $`\text{kg}^\top \cdot \mathbf{v}_{\text{new}}`$ 参与状态更新，这条路径也会产生对 $`\mathbf{v}_{\text{new}}`$ 的梯度：

$$d\mathbf{v}_{\text{new，from\_h}}^{[t]} = \text{kg} \cdot d\mathbf{h}^{[t+1]}$$

这里用 $`d\mathbf{h}^{[t+1]}`$ 而非 $`d\mathbf{h}^{[t]}`$，因为 $`\mathbf{v}_{\text{new}}^{[t]}`$ 在前向中影响的是**下一个** chunk 的状态 $`\mathbf{h}^{[t+1]}`$。

合并阶段 1 传来的 $`d\mathbf{v}_{\text{new}}`$（来自 $`\mathbf{A}_{qk}`$ 路径），得到更新后的总梯度：

$$d\mathbf{v}_{\text{new}}' = \underbrace{d\mathbf{v}_{\text{new}}}_{\text{来自阶段 1 (intra-chunk)}} + \underbrace{\text{kg} \cdot d\mathbf{h}^{[t+1]}}_{\text{来自状态更新 (inter-chunk)}}$$

**关键实现细节**：在 Triton Kernel 中，先存储 $`d\mathbf{h}^{[t]}`$，然后用**更新前的** $`d\mathbf{h}`$（即 $`d\mathbf{h}^{[t+1]}`$ 经衰减和累加后的值）来计算 $`d\mathbf{v}_{\text{new}}'`$。这要求递推循环的顺序和时序必须严格正确。

##### FLA 代码映射 (`chunk_kda_bwd`)

```python
dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
    q=qg,                    # qg = q * exp(g)，前向重算时已算好
    k=kg,                    # 门控键
    w=w,                     # 有效键
    gk=g,                    # 衰减门
    h0=initial_state,
    dht=dht,                 # 最终状态的梯度（上游传入）
    do=do,
    dv=dv,                   # 来自阶段 1 的 dv_new
    scale=scale,
    use_exp2=True,
)
```

输出：

- `dh`: `[B, NT, H, K, V]` — 每个 chunk 起始状态的梯度
- `dh0`: `[B, H, K, V]` — 初始状态的梯度
- `dv`: `[B, T, H, V]` — 更新后的 $`d\mathbf{v}_{\text{new}}'`$

##### Triton Kernel 核心计算 (`chunk_gated_delta_rule_bwd_kernel_dhu`)

```python
# 反向时间循环：从最后一个 chunk 向前
for i_t in range(NT-1, -1, -1):
    # 1. 存储当前 chunk 的 dh
    tl.store(p_dh, b_dh)

    # 2. 更新 dv_new: dv += kg @ dh（inter-chunk 贡献）
    b_dv = tl.dot(b_k, b_dh)              # kg @ dh: [BT, BV]
    b_dv = tl.load(p_dv) + b_dv           # 加上阶段 1 传来的 dv
    tl.store(p_dv2, b_dv)

    # 3. 递推 dh: dh = exp(gk_last) * dh + qg @ do * scale - w @ dv
    b_dh *= exp2(b_gk_last)[:, None]      # 衰减
    b_dh += tl.dot(b_q, b_do) * scale     # 输出路径
    b_dh -= tl.dot(b_w, b_dv)             # delta rule 路径（取负）
```

---

#### 阶段 3. WY 表示反向 + q/k/g 梯度融合 (Bwd Step 1-A → chunk_kda_bwd_wy_dqkg_fused)

这是最复杂的 Kernel，将多条梯度路径融合在一起。

##### 3.1 数学推导

这个 Kernel 融合了 6 条梯度路径（A-F），涉及前向中多个计算步骤的反向传播。下面逐一推导每条路径。

**路径 A：从输出 inter-chunk 部分反传到 $`d\mathbf{q}`$**

前向：$`\mathbf{o}_{\text{inter}} = s \cdot \text{qg} \cdot \mathbf{h}`$，其中 $`\text{qg} = \mathbf{q} \odot \exp(\mathbf{g}) \in \mathbb{R}^{C \times K}`$。

这是 $`\mathbf{Y} = s \cdot \mathbf{A} \cdot \mathbf{X}`$ 的形式。对 $`\mathbf{A} = \text{qg}`$ 求梯度（规则 $`d\mathbf{A} = d\mathbf{Y} \cdot \mathbf{X}^\top`$）：

$$d\text{qg} = s \cdot d\mathbf{o} \cdot \mathbf{h}^\top \in \mathbb{R}^{C \times K}$$

逐元素来看：$`d\text{qg}_{r,d} = s \cdot \sum_{v} d\mathbf{o}_{r,v} \cdot \mathbf{h}_{d,v}`$，即位置 $`r`$ 的 qg 梯度是 $`d\mathbf{o}_r`$ 与 $`\mathbf{h}`$ 第 $`d`$ 行的内积。

然后从 $`\text{qg} = \mathbf{q} \odot \exp(\mathbf{g})`$ 反传到 $`\mathbf{q}`$。逐元素乘 $`\mathbf{y} = \mathbf{a} \odot \mathbf{b}`$ 的梯度为 $`d\mathbf{a} = d\mathbf{y} \odot \mathbf{b}`$，因此：

$$d\mathbf{q}_{\text{inter}} = d\text{qg} \odot \exp(\mathbf{g}) = s \cdot d\mathbf{o} \cdot \mathbf{h}^\top \odot \exp(\mathbf{g})$$

**路径 B：从状态更新反传到 $`d\mathbf{k}`$**

前向：$`\mathbf{h}^{[t+1]} += \text{kg}^\top \cdot \mathbf{v}_{\text{new}}`$，其中 $`\text{kg} = \mathbf{k} \odot \exp(\mathbf{g}_C - \mathbf{g}) \in \mathbb{R}^{C \times K}`$。

这是 $`\mathbf{Y} += \mathbf{A}^\top \cdot \mathbf{X}`$ 的形式（$`\mathbf{A} = \text{kg}`$，$`\mathbf{X} = \mathbf{v}_{\text{new}}`$）。对 $`\mathbf{A}`$ 求梯度需要用规则：若 $`\mathbf{Y} = \mathbf{A}^\top \cdot \mathbf{X}`$，则 $`d\mathbf{A} = \mathbf{X} \cdot d\mathbf{Y}^\top`$。

这里 $`d\mathbf{Y} = d\mathbf{h}^{[t+1]}`$（注意是下一个 chunk 的状态梯度），所以：

$$d\text{kg} = \mathbf{v}_{\text{new}} \cdot d\mathbf{h}^{[t+1]\top} \in \mathbb{R}^{C \times K}$$

等价地写成 $`d\text{kg} = d\mathbf{h}^{[t+1]} \cdot \mathbf{v}_{\text{new}}^\top`$ 的转置形式。在实现中用的是 $`d\mathbf{k} = \mathbf{v}_{\text{new}} \cdot d\mathbf{h}^\top`$（因为 dh 已在阶段 2 中存储了每个 chunk 对应的梯度）。

然后从 $`\text{kg} = \mathbf{k} \odot \exp(\mathbf{g}_C - \mathbf{g})`$ 反传到 $`\mathbf{k}`$：

$$d\mathbf{k}_{\text{inter}} = d\text{kg} \odot \exp(\mathbf{g}_C - \mathbf{g})$$

**路径 C：从 Delta Rule 修正经 $`\mathbf{w}`$ 反传到 $`d\mathbf{A}_{kk}`$ 和 $`d\mathbf{k}`$**

这条路径有两步：先从 $`\mathbf{v}_{\text{new}} = \mathbf{u} - \mathbf{w} \cdot \mathbf{h}`$ 得到 $`d\mathbf{w}`$，再从 $`\mathbf{w} = \mathbf{A}_{kk} \cdot (\mathbf{k} \odot \beta \odot \exp(\mathbf{g}))`$ 反传到 $`d\mathbf{A}_{kk}`$ 和 $`d\mathbf{k}`$。

**C.1：求 $`d\mathbf{w}`$**

$`\mathbf{v}_{\text{new}} = \mathbf{u} - \mathbf{w} \cdot \mathbf{h}`$ 中，$`\mathbf{w}`$ 通过 $`-\mathbf{w} \cdot \mathbf{h}`$ 影响 $`\mathbf{v}_{\text{new}}`$。这是 $`\mathbf{Y} = -\mathbf{A} \cdot \mathbf{X}`$（$`\mathbf{A} = \mathbf{w} \in \mathbb{R}^{C \times K}`$，$`\mathbf{X} = \mathbf{h} \in \mathbb{R}^{K \times V}`$），梯度为：

$$d\mathbf{w} = -d\mathbf{v}_{\text{new}} \cdot \mathbf{h}^\top \in \mathbb{R}^{C \times K}$$

**C.2：从 $`d\mathbf{w}`$ 反传到 $`d\mathbf{A}_{kk}`$**

前向：$`\mathbf{w} = \mathbf{A}_{kk} \cdot \underbrace{(\mathbf{k} \odot \boldsymbol{\beta} \odot \exp(\mathbf{g}))}_{\text{记为 } \mathbf{p} \in \mathbb{R}^{C \times K}}`$。这是矩阵乘法，对 $`\mathbf{A}_{kk}`$ 求梯度：

$$d\mathbf{A}_{kk} \mathrel{+}= d\mathbf{w} \cdot \mathbf{p}^\top = d\mathbf{w} \cdot (\mathbf{k} \odot \boldsymbol{\beta} \odot \exp(\mathbf{g}))^\top$$

**C.3：从 $`d\mathbf{w}`$ 反传到 $`d\mathbf{k}`$**

对 $`\mathbf{p}`$ 求梯度：$`d\mathbf{p} = \mathbf{A}_{kk}^\top \cdot d\mathbf{w}`$，然后反传到 $`\mathbf{k}`$（$`\mathbf{p} = \mathbf{k} \odot \boldsymbol{\beta} \odot \exp(\mathbf{g})`$，逐元素乘的梯度是对方）：

$$d\mathbf{k}_{\text{w}} = \mathbf{A}_{kk}^\top \cdot d\mathbf{w} \odot \boldsymbol{\beta} \odot \exp(\mathbf{g})$$

**路径 D：从 Delta Rule 修正经 $`\mathbf{u}`$ 反传到 $`d\mathbf{A}_{kk}`$、$`d\mathbf{v}`$ 和 $`d\boldsymbol{\beta}`$**

前向：$`\mathbf{v}_{\text{new}} = \mathbf{u} - \mathbf{w} \cdot \mathbf{h}`$ 中 $`\mathbf{u}`$ 的系数是 $`+1`$，所以 $`d\mathbf{u} = d\mathbf{v}_{\text{new}}`$。

然后从 $`\mathbf{u} = \mathbf{A}_{kk} \cdot (\mathbf{v} \odot \boldsymbol{\beta})`$ 反传：

**D.1：反传到 $`d\mathbf{A}_{kk}`$**（矩阵乘法梯度 $`d\mathbf{A} = d\mathbf{Y} \cdot \mathbf{X}^\top`$）：

$$d\mathbf{A}_{kk} \mathrel{+}= d\mathbf{u} \cdot (\mathbf{v} \odot \boldsymbol{\beta})^\top = d\mathbf{v}_{\text{new}} \cdot (\mathbf{v} \odot \boldsymbol{\beta})^\top$$

**D.2：反传到 $`d\mathbf{v}`$ 和 $`d\boldsymbol{\beta}`$**

$`d(\mathbf{v} \odot \boldsymbol{\beta}) = \mathbf{A}_{kk}^\top \cdot d\mathbf{u} = \mathbf{A}_{kk}^\top \cdot d\mathbf{v}_{\text{new}}`$

从逐元素乘 $`\mathbf{v} \odot \boldsymbol{\beta}`$ 反传（$`\boldsymbol{\beta} \in \mathbb{R}^C`$ 逐行广播）：

$$d\mathbf{v} = \mathbf{A}_{kk}^\top \cdot d\mathbf{v}_{\text{new}} \odot \boldsymbol{\beta}$$

$$d\boldsymbol{\beta} \mathrel{+}= \text{行求和}\left(\mathbf{A}_{kk}^\top \cdot d\mathbf{v}_{\text{new}} \odot \mathbf{v}\right)$$

其中 $`d\beta_r = \sum_{d} [\mathbf{A}_{kk}^\top \cdot d\mathbf{v}_{\text{new}}]_{r,d} \cdot v_{r,d}`$，即第 $`r`$ 行的逐元素乘积再求和为标量。

**路径 E：矩阵逆的梯度**

这是最关键的数学推导。前向中 $`\mathbf{A}_{kk} = \mathbf{M}^{-1}`$，其中 $`\mathbf{M} = \mathbf{I} + \mathbf{A}_{\text{raw}}`$ 是单位下三角矩阵。

**推导过程**：从恒等式 $`\mathbf{M} \cdot \mathbf{M}^{-1} = \mathbf{I}`$ 两边取微分：

$$d\mathbf{M} \cdot \mathbf{M}^{-1} + \mathbf{M} \cdot d(\mathbf{M}^{-1}) = \mathbf{0}$$

移项得矩阵逆的微分：

$$d(\mathbf{M}^{-1}) = -\mathbf{M}^{-1} \cdot d\mathbf{M} \cdot \mathbf{M}^{-1}$$

在自动微分中，我们有上游梯度 $`d\mathbf{A}_{kk}`$（损失对 $`\mathbf{M}^{-1}`$ 的梯度），需要求 $`d\mathbf{M}`$（损失对 $`\mathbf{M}`$ 的梯度）。利用链式法则的迹公式 $`d\mathcal{L} = \text{tr}(d\mathbf{A}_{kk}^\top \cdot d(\mathbf{M}^{-1}))`$，代入上式：

$$d\mathcal{L} = \text{tr}\left(d\mathbf{A}_{kk}^\top \cdot (-\mathbf{M}^{-1} \cdot d\mathbf{M} \cdot \mathbf{M}^{-1})\right) = \text{tr}\left((-\mathbf{M}^{-\top} \cdot d\mathbf{A}_{kk} \cdot \mathbf{M}^{-\top})^\top \cdot d\mathbf{M}\right)$$

因此 $`d\mathbf{M}`$ 为：

$$d\mathbf{M} = -\mathbf{M}^{-\top} \cdot d\mathbf{A}_{kk} \cdot \mathbf{M}^{-\top} = -\mathbf{A}_{kk}^\top \cdot d\mathbf{A}_{kk} \cdot \mathbf{A}_{kk}^\top$$

但我们真正需要的是 $`d\mathbf{A}_{\text{raw}}`$（$`\mathbf{M} = \mathbf{I} + \mathbf{A}_{\text{raw}}`$，$`\mathbf{I}`$ 是常数），而 $`\mathbf{A}_{\text{raw}}`$ 是**严格下三角**的，所以只需保留 $`d\mathbf{M}`$ 的严格下三角部分。

**实现中的分解**：上面的三矩阵连乘在实现中被拆为两步，并在每步之间施加严格下三角 mask：

$$\text{Step 1: } d\mathbf{A}_{kk} \leftarrow \text{tril}(d\mathbf{A}_{kk}, \text{diag}=-1) \cdot \mathbf{A}_{kk} \quad \text{（右乘逆矩阵）}$$

$$\text{Step 2: } d\mathbf{A}_{kk} \leftarrow \mathbf{A}_{kk} \cdot d\mathbf{A}_{kk} \quad \text{（左乘逆矩阵的转置... 注意 A_kk 非对称，此处利用了下三角结构）}$$

$$\text{Step 3: } d\mathbf{A}_{kk} \leftarrow -\text{tril}(d\mathbf{A}_{kk}, \text{diag}=-1) \quad \text{（取负 + 保留严格下三角）}$$

**路径 F：$`\mathbf{g}`$ 的梯度汇总**

$`\mathbf{g}`$ 出现在前向的多个地方，每处都会产生梯度贡献。总梯度是所有贡献之和：

**F.1：来自 $`\text{qg} = \mathbf{q} \odot \exp(\mathbf{g})`$（路径 A）**

$`\exp(\mathbf{g})`$ 对 $`\mathbf{g}`$ 的导数是 $`\exp(\mathbf{g})`$ 自身（$`\frac{\partial e^x}{\partial x} = e^x`$）。结合路径 A 中 $`d\text{qg} = s \cdot d\mathbf{o} \cdot \mathbf{h}^\top`$：

$$d\mathbf{g}_{\text{from\_A}} = d\text{qg} \odot \mathbf{q} \odot \exp(\mathbf{g}) = d\mathbf{q}_{\text{inter}} \odot \mathbf{q}$$

直觉：$`d\mathbf{g} \propto \mathbf{q} \odot d\mathbf{q}`$——$`\mathbf{g}`$ 的梯度等于 $`\mathbf{q}`$ 与 $`d\mathbf{q}`$ 的逐元素乘积（因为 $`\mathbf{g}`$ 和 $`\mathbf{q}`$ 在 $`\text{qg} = \mathbf{q} \odot \exp(\mathbf{g})`$ 中对称出现）。

**F.2：来自 $`\text{kg} = \mathbf{k} \odot \exp(\mathbf{g}_C - \mathbf{g})`$（路径 B）**

$`\exp(\mathbf{g}_C - \mathbf{g})`$ 对 $`\mathbf{g}`$ 的导数是 $`-\exp(\mathbf{g}_C - \mathbf{g})`$（负号来自 $`-\mathbf{g}`$）。因此：

$$d\mathbf{g}_{\text{from\_B}} = -d\text{kg} \odot \mathbf{k} \odot \exp(\mathbf{g}_C - \mathbf{g}) = -d\mathbf{k}_{\text{inter}} \odot \mathbf{k}$$

**F.3：来自状态衰减 $`\exp(\mathbf{g}_C) \odot \mathbf{h}`$**

前向中 $`\mathbf{h}^{[t+1]} = \exp(\mathbf{g}_C) \odot \mathbf{h}^{[t]} + \ldots`$，$`\mathbf{g}_C`$（chunk 末尾位置的累积 $`\mathbf{g}`$）影响状态衰减。

$$d\mathbf{g}_{\text{from\_h}} = \sum_{v} \mathbf{h}_{d,v} \cdot d\mathbf{h}_{d,v} = \text{行求和}(\mathbf{h} \odot d\mathbf{h})$$

（在 V 维度上对 $`\mathbf{h} \odot d\mathbf{h}`$ 求和，得到 $`K`$ 维向量）

**汇总**：

$$d\mathbf{g} = d\mathbf{q}_{\text{inter}} \odot \mathbf{q} - d\mathbf{k}_{\text{inter}} \odot \mathbf{k} + \text{行求和}(\mathbf{h} \odot d\mathbf{h}) + \ldots$$

其中 "$`\ldots`$" 包含来自路径 C/D 中 $`\exp(\mathbf{g})`$ 的贡献（在阶段 4 中进一步累加）。

##### FLA 代码映射 (`chunk_kda_bwd`)

```python
dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
    q=q, k=k, v=v, v_new=v_new,
    g=g, beta=beta, A=Akk,
    h=h, do=do, dh=dh, dv=dv,
    scale=scale, ...
)
```

##### Triton Kernel 核心计算 (`chunk_kda_bwd_kernel_wy_dqkg_fused`)

```python
# 循环遍历 V 维度块
for i_v in range(cdiv(V, BV)):
    b_h  = tl.load(p_h)              # [BK, BV]
    b_dh = tl.load(p_dh)             # [BK, BV]
    b_do = tl.load(p_do)             # [BT, BV]
    b_v_new = tl.load(p_v_new)       # [BT, BV]

    # 路径 F 的 g 梯度贡献：dgk += sum(h * dh, axis=V)
    b_dgk += tl.sum(b_h * b_dh, axis=1)     # [BK]

    # 路径 A: dq += do @ h^T
    b_dq += tl.dot(b_do, tl.trans(b_h))     # [BT, BK]

    # 路径 B: dk += v_new @ dh^T
    b_dk += tl.dot(b_v_new, tl.trans(b_dh)) # [BT, BK]

    # 路径 C: dw = -dv_new @ h^T
    b_dw += tl.dot(b_dv, tl.trans(b_h))     # [BT, BK] (取负在后面)

    # 路径 D (仅 i_k == 0 时): dA += dv_new @ v^T, dv2 = A^T @ dv_new * beta
    if i_k == 0:
        b_dA += tl.dot(b_dv, tl.trans(b_v))
        b_dv2 = tl.dot(tl.trans(b_A), b_dv) * b_beta[:, None]

# 门控缩放
b_dq = b_dq * exp2(b_g) * scale             # 路径 A 的 exp(g)
b_dk = b_dk * exp2(b_gn - b_g)              # 路径 B 的 exp(gn - g)

# 路径 C 的 dA 贡献: dA += -dw @ (k * exp(g))^T
b_dA += tl.dot(-b_dw, tl.trans(b_k * exp2(b_g)))

# 路径 E: 矩阵逆梯度
b_dA = tl.where(row > col, b_dA, 0)         # 严格下三角
b_dA = tl.dot(b_dA, b_A)                    # 右乘 A
b_dA = tl.dot(b_A, b_dA)                    # 左乘 A
b_dA = tl.where(row > col, -b_dA, 0)        # 取负 + 重新掩码

# 路径 F: g 梯度汇总
b_dg = b_q * b_dq - b_k * b_dk + ...
```

---

#### 阶段 4. 注意力矩阵反向 (Bwd Step 1-B → chunk_kda_bwd_intra)

##### 4.1 数学推导

回顾前向中 $`\mathbf{A}_{qk}`$ 和 $`\mathbf{A}_{kk}`$ 的构建公式：

$$A_{qk}(r, j) = s \cdot \mathbf{q}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j], \quad r \geq j$$

$$A_{kk}(r, j) = \beta_r \cdot \mathbf{k}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j], \quad r > j$$

两者结构相同：$`\text{左侧向量}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j]`$，仅左侧向量不同（$`s \cdot \mathbf{q}_r`$ vs $`\beta_r \cdot \mathbf{k}_r`$）。

现在给定上游梯度 $`d\mathbf{A}_{qk}`$（来自阶段 1）和 $`d\mathbf{A}_{kk}`$（来自阶段 3），需要反传到 $`\mathbf{q}, \mathbf{k}, \boldsymbol{\beta}, \mathbf{g}`$。

**$`\mathbf{k}_j`$ 在公式中出现了两次**——既作为"左侧向量"（$`\mathbf{A}_{kk}`$ 中的 $`\mathbf{k}_r`$），又作为"右侧向量"（$`\mathbf{A}_{qk}`$ 和 $`\mathbf{A}_{kk}`$ 中的 $`\mathbf{k}_j`$），所以 $`d\mathbf{k}`$ 有两个方向的贡献——对应实现中的 **Section 1** 和 **Section 2**。

---

**$`d\mathbf{q}`$ 的推导**（来自 $`\mathbf{A}_{qk}`$，仅 Section 1 方向）

$`A_{qk}(r, j)`$ 对 $`\mathbf{q}_r`$ 求偏导（$`\mathbf{q}_r`$ 只出现在第 $`r`$ 行）：

$$\frac{\partial A_{qk}(r, j)}{\partial \mathbf{q}_r} = s \cdot [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j] \in \mathbb{R}^K$$

利用链式法则，$`d\mathbf{q}_r`$ 是对所有 $`j \leq r`$ 的贡献求和：

$$d\mathbf{q}_r = \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot \frac{\partial A_{qk}(r, j)}{\partial \mathbf{q}_r} = s \cdot \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j]$$

逐元素来看，第 $`d`$ 个分量为：

$$dq_{r,d} = s \cdot \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot \exp(g_{r,d} - g_{j,d}) \cdot k_{j,d}$$

矩阵形式中，可以将 $`\exp(\mathbf{g}_r)`$ 提到求和外面（因为它不依赖 $`j`$）：

$$d\mathbf{q}_r = s \cdot \exp(\mathbf{g}_r) \odot \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot [\exp(-\mathbf{g}_j) \odot \mathbf{k}_j]$$

即 $`d\mathbf{q} = s \cdot \exp(\mathbf{g}) \odot [d\mathbf{A}_{qk} \cdot (\exp(-\mathbf{g}) \odot \mathbf{k})]`$，实现中进一步优化了数值稳定性。

---

**$`d\mathbf{k}`$ 的推导 — Section 1（作为"左侧向量" $`\mathbf{k}_r`$）**

$`A_{kk}(r, j)`$ 对 $`\mathbf{k}_r`$（左侧向量）求偏导：

$$\frac{\partial A_{kk}(r, j)}{\partial \mathbf{k}_r} = \beta_r \cdot [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j]$$

对所有 $`j < r`$ 求和：

$$d\mathbf{k}_r^{\text{sec1}} = \beta_r \sum_{j < r} d\mathbf{A}_{kk}(r, j) \cdot [\exp(\mathbf{g}_r - \mathbf{g}_j) \odot \mathbf{k}_j]$$

结构与 $`d\mathbf{q}_r`$ 完全对称——只是用 $`d\mathbf{A}_{kk}`$ 和 $`\beta_r`$ 代替了 $`d\mathbf{A}_{qk}`$ 和 $`s`$。

实现中的 **$`d\boldsymbol{\beta}`$ 推导**也在此处完成：

$$d\beta_r = \sum_{j < r} d\mathbf{A}_{kk}(r, j) \cdot a_{r,j} = \sum_{d} dk_{r,d}^{\text{sec1（除以 }\beta_r\text{前）}} \cdot k_{r,d}$$

即 $`d\beta_r`$ 等于 Section 1 中 $`d\mathbf{k}_r`$（$`\beta_r`$ 还没乘进去时）与 $`\mathbf{k}_r`$ 的内积。代码中先算 $`d\mathbf{k}`$（不含 $`\beta`$），提取 $`d\beta = \text{sum}(d\mathbf{k} \odot \mathbf{k})`$，再乘以 $`\beta`$。

---

**$`d\mathbf{k}`$ 的推导 — Section 2（作为"右侧向量" $`\mathbf{k}_j`$）**

$`\mathbf{k}_j`$ 作为右侧向量，同时出现在 $`\mathbf{A}_{qk}`$ 和 $`\mathbf{A}_{kk}`$ 中。对 $`\mathbf{k}_j`$ 求偏导：

来自 $`\mathbf{A}_{qk}`$：

$$\frac{\partial A_{qk}(r, j)}{\partial \mathbf{k}_j} = s \cdot \mathbf{q}_r \odot \exp(\mathbf{g}_r - \mathbf{g}_j)$$

来自 $`\mathbf{A}_{kk}`$：

$$\frac{\partial A_{kk}(r, j)}{\partial \mathbf{k}_j} = \beta_r \cdot \mathbf{k}_r \odot \exp(\mathbf{g}_r - \mathbf{g}_j)$$

对所有 $`r > j`$（梯度从"下方"的行流入）求和：

$$d\mathbf{k}_j^{\text{sec2}} = \sum_{r > j} \left[d\mathbf{A}_{qk}(r, j) \cdot s \cdot \mathbf{q}_r + d\mathbf{A}_{kk}(r, j) \cdot \beta_r \cdot \mathbf{k}_r\right] \odot \exp(\mathbf{g}_r - \mathbf{g}_j)$$

这就是**转置方向**：Section 1 中梯度从行端（$`r`$）出发向右下方的列（$`j`$）索取权重；Section 2 中梯度从列端（$`j`$）出发，**收集**所有 $`r > j`$ 的行对它的贡献。在实现中对应遍历"更晚的子块 $`i_j > i_i`$"。

矩阵形式：$`d\mathbf{k}^{\text{sec2}} = d\mathbf{A}_{qk}^\top \cdot (s \cdot \mathbf{q} \odot \exp(\mathbf{g})) + d\mathbf{A}_{kk}^\top \cdot (\boldsymbol{\beta} \cdot \mathbf{k} \odot \exp(\mathbf{g}))`$（需注意衰减方向）。

**最终 $`d\mathbf{k}`$** = Section 1 + Section 2 + 阶段 3 传来的累积值：

$$d\mathbf{k} = d\mathbf{k}^{\text{sec1}} + d\mathbf{k}^{\text{sec2}} + d\mathbf{k}_{\text{from\_stage3}}$$

---

**$`d\mathbf{g}`$ 的推导与 reverse cumsum**

$`\mathbf{g}`$ 在 $`A_{qk}(r,j)`$ 和 $`A_{kk}(r,j)`$ 中通过 $`\exp(\mathbf{g}_r - \mathbf{g}_j)`$ 出现。对 $`\mathbf{g}_r`$ 和 $`\mathbf{g}_j`$ 分别求偏导：

$$\frac{\partial}{\partial \mathbf{g}_r}\exp(\mathbf{g}_r - \mathbf{g}_j) = +\exp(\mathbf{g}_r - \mathbf{g}_j) \quad \text{（$`\mathbf{g}_r`$ 以正号出现在指数中）}$$

$$\frac{\partial}{\partial \mathbf{g}_j}\exp(\mathbf{g}_r - \mathbf{g}_j) = -\exp(\mathbf{g}_r - \mathbf{g}_j) \quad \text{（$`\mathbf{g}_j`$ 以负号出现在指数中）}$$

从 Section 1 方向得到的 $`d\mathbf{g}`$ 贡献为正（$`d\mathbf{k}^{\text{sec1}} \odot \mathbf{k}`$ 方向），从 Section 2 方向得到的 $`d\mathbf{g}`$ 贡献为负（$`-d\mathbf{k}^{\text{sec2}} \odot \mathbf{k}`$ 方向），两者组合后：

$$d\mathbf{g}_{\text{from\_intra}} = (d\mathbf{k}^{\text{sec1}} - d\mathbf{k}^{\text{sec2}}) \odot \mathbf{k} + \ldots$$

**最后一步：reverse cumsum**

回顾前向 Step 0 做了 $`\mathbf{g} = \text{cumsum}(\mathbf{g}_{\text{raw}})`$，即 $`g_r = \sum_{i=1}^{r} g_i^{\text{raw}}`$。

cumsum 的反向是 **reverse cumsum**（反向前缀和）：

$$dg_r^{\text{raw}} = \sum_{j=r}^{C} dg_j$$

直觉：前向中 $`g_r^{\text{raw}}`$ 影响了 $`g_r, g_{r+1}, \ldots, g_C`$ 所有位置（因为 cumsum 是前缀和），所以反向时需要把 $`r`$ 及之后所有位置的梯度求和。

实现中调用 `chunk_local_cumsum(dg, reverse=True)` 完成。

##### FLA 代码映射 (`chunk_kda_bwd`)

```python
dq, dk, db, dg = chunk_kda_bwd_intra(
    q=q, k=k, g=g, beta=beta,
    dAqk=dAqk, dAkk=dAkk,
    dq=dq, dk=dk, db=db, dg=dg,    # 累加到阶段 3 的结果上
    ...
)
```

_注：最后对 $`d\mathbf{g}`$ 执行 `chunk_local_cumsum(dg, reverse=True)` 完成 Step 0 的反向。_

##### Triton Kernel 核心计算 (`chunk_kda_bwd_kernel_intra`)

```python
# ===== Section 1: 左侧方向 (dq 和 dk 作为"行端") =====
# 对于子块 i_i，遍历所有更早的子块 i_j < i_i
for i_j in range(i_i):
    b_kg = k_j * exp2(g_n_i - g_j)              # 门控键 [BC, BK]
    b_dq += tl.dot(b_dAqk[i_i, i_j], b_kg)      # dq += dAqk @ kg
    b_dk += tl.dot(b_dAkk[i_i, i_j], b_kg)      # dk += dAkk @ kg
b_dq *= exp2(g_i - g_n_i)                        # 还原门控
b_dk *= exp2(g_i - g_n_i)

# 对角块：逐列循环
for j in range(BC):
    b_kgj = k_j * exp2(g_i - g_j)
    b_dq += where(row >= j, dAqk[row, j]) * b_kgj
    b_dk += where(row >= j, dAkk[row, j]) * b_kgj

b_db = sum(b_dk * k, dim=-1)                     # beta 梯度
b_dk *= beta[:, None]

# ===== Section 2: 转置方向 (dk 作为"列端") =====
# 对于子块 i_i，遍历所有更晚的子块 i_j > i_i
for i_j in range(i_i + 1, NC):
    b_qg = q_j * exp2(g_j - g_n_i)              # 门控查询
    b_kbg = k_j * beta_j * exp2(g_j - g_n_i)    # 门控键*beta
    b_dkt += tl.dot(dAqk^T[i_i, i_j], b_qg)     # 转置方向
    b_dkt += tl.dot(dAkk^T[i_i, i_j], b_kbg)
b_dkt *= exp2(g_n_i - g_i)

# 梯度汇总
b_dg = (b_dk_section1 - b_dkt) * k + dg_incoming
b_dk = b_dk_section1 + dk_incoming + b_dkt
```

---

### 整体梯度流向总结

```
do ──→ chunk_kda_bwd_dAv ──→ dAqk, dv_new
          │
          ▼
       chunk_gated_delta_rule_bwd_dhu ──→ dh, dh0, dv_new'
          │
          ▼
       chunk_kda_bwd_wy_dqkg_fused ──→ dq, dk, dv, dβ, dg, dAkk
          │
          ▼
       chunk_kda_bwd_intra ──→ dq, dk, dβ, dg (累加)
          │
          ▼
       chunk_local_cumsum(dg, reverse=True) ──→ dg_raw
```

梯度的主要流向：
1. $`d\mathbf{o}`$ 经过 $`\mathbf{A}_{qk}`$ 路径产生 $`d\mathbf{A}_{qk}`$ 和 $`d\mathbf{v}_{\text{new}}`$
2. $`d\mathbf{v}_{\text{new}}`$ 和 $`d\mathbf{o}`$ 经过**反向时间递推**产生每个 chunk 的 $`d\mathbf{h}`$
3. $`d\mathbf{h}`$ 和 $`d\mathbf{v}_{\text{new}}`$ 经过 WY 表示反向产生 $`d\mathbf{q}, d\mathbf{k}, d\mathbf{v}, d\mathbf{A}_{kk}`$
4. $`d\mathbf{A}_{qk}`$ 和 $`d\mathbf{A}_{kk}`$ 经过注意力矩阵反向**累加**到 $`d\mathbf{q}, d\mathbf{k}`$

---

### 各张量 Shape 汇总

上游梯度：

| 张量 | Shape | 说明 |
|---|---|---|
| `do` | `[B, T, H, V]` | 损失对输出的梯度 |
| `dht` | `[B, H, K, V]` | 损失对最终状态的梯度（可选） |

阶段 1 输出：

| 张量 | Shape | 说明 |
|---|---|---|
| `dAqk` | `[B, T, H, C]` | $`d\mathbf{o} \cdot \mathbf{v}_{\text{new}}^\top`$（下三角） |
| `dv` | `[B, T, H, V]` | $`\mathbf{A}_{qk}^\top \cdot d\mathbf{o}`$ |

阶段 2 输出：

| 张量 | Shape | 说明 |
|---|---|---|
| `dh` | `[B, NT, H, K, V]` | 每个 chunk 起始状态的梯度 |
| `dh0` | `[B, H, K, V]` | 初始状态的梯度（可选） |
| `dv` | `[B, T, H, V]` | 更新后的 $`d\mathbf{v}_{\text{new}}'`$ |

阶段 3 输出：

| 张量 | Shape | 说明 |
|---|---|---|
| `dq` | `[B, T, H, K]` | q 梯度（inter-chunk + WY） |
| `dk` | `[B, T, H, K]` | k 梯度（inter-chunk + WY） |
| `dv` | `[B, T, H, V]` | v 梯度 |
| `db` | `[B, T, H]` | beta 梯度 |
| `dg` | `[B, T, H, K]` | g 梯度（WY 部分） |
| `dAkk` | `[B, T, H, C]` | 键-键矩阵逆的梯度 |

阶段 4 输出（累加到阶段 3 结果上）：

| 张量 | Shape | 说明 |
|---|---|---|
| `dq` | `[B, T, H, K]` | 累加 $`d\mathbf{A}_{qk}`$ 的贡献 |
| `dk` | `[B, T, H, K]` | 累加 $`d\mathbf{A}_{qk} + d\mathbf{A}_{kk}`$ 的贡献 |
| `db` | `[B, T, H]` | 累加 $`d\mathbf{A}_{kk}`$ 的贡献 |
| `dg` | `[B, T, H, K]` | reverse cumsum 后的最终 g 梯度 |

最终输出：

| 张量 | Shape | 说明 |
|---|---|---|
| `dq` | `[B, T, H, K]` | 最终 q 梯度 |
| `dk` | `[B, T, H, K]` | 最终 k 梯度 |
| `dv` | `[B, T, H, V]` | 最终 v 梯度 |
| `db` | `[B, T, H]` | 最终 beta 梯度 |
| `dg` | `[B, T, H, K]` | 最终 g 梯度 |
| `dh0` | `[B, H, K, V]` | 初始状态梯度（可选） |
