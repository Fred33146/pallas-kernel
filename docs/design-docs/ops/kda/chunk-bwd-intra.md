# chunk_kda_bwd_intra design

参考文献

https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf

本设计文档记录了 KDA Chunk-Parallel 反向传播的**阶段 4（Bwd Step 1-B）**——`chunk_kda_bwd_intra` kernel 的数学推导与实现细节。阅读本文档前，请先阅读 [chunk-fwd.md](./chunk-fwd.md) 和 [chunk-bwd.md](./chunk-bwd.md)。

---

### 定位与职责

在整个反向流水线中，`chunk_kda_bwd_intra` 是最后一个 kernel，负责将注意力矩阵的梯度 $`d\mathbf{A}_{qk}`$ 和 $`d\mathbf{A}_{kk}`$ 反传到原始输入 $`\mathbf{q}, \mathbf{k}, \boldsymbol{\beta}, \mathbf{g}`$，并**累加**到阶段 3（`chunk_kda_bwd_wy_dqkg_fused`）已计算的 inter-chunk 梯度上。

```
chunk_kda_bwd_wy_dqkg_fused ──→ dq_inter, dk_inter, dg_inter, db_inter, dAkk
                                     │
chunk_kda_bwd_dAv ──→ dAqk           │
                                     ▼
                    chunk_kda_bwd_intra
                         │
                         ▼
              dq = dq_inter + dq_intra
              dk = dk_inter + dk_intra
              db = db_inter + db_intra
              dg = reverse_cumsum(dg_inter + dg_intra)
```

---

### 符号定义

继承 [chunk-fwd.md](./chunk-fwd.md) 和 [chunk-bwd.md](./chunk-bwd.md) 的所有符号。以下为本文档新增或重点使用的符号：

- $`C`$: chunk 大小（BT），通常为 64。
- $`\text{BC}`$: 子块大小（sub-chunk），通常为 16。每个 chunk 被分成 $`\text{NC} = C / \text{BC}`$ 个子块。
- $`\mathbf{A}_{qk}(r, j)`$: query-key 注意力矩阵元素，下三角（$`r \geq j`$）。
- $`\mathbf{A}_{kk}(r, j)`$: key-key 自注意力矩阵元素，严格下三角（$`r > j`$）。
- $`d\mathbf{A}_{qk}, d\mathbf{A}_{kk}`$: 上游传入的注意力矩阵梯度。
- $`\tilde{\mathbf{k}}_j = \mathbf{k}_j \odot 2^{-\mathbf{g}_j}`$: decayed key。
- $`\hat{\mathbf{q}}_r = s \cdot \mathbf{q}_r \odot 2^{\mathbf{g}_r}`$: anti-decayed query (含 scale)。
- $`\hat{\mathbf{k}}_r = \beta_r \cdot \mathbf{k}_r \odot 2^{\mathbf{g}_r}`$: anti-decayed beta·key。

---

### 前向回顾

chunk 内注意力矩阵的定义（省略 batch/head 维度，$`r, j`$ 为 chunk 内的时间索引）：

$$\mathbf{A}_{qk}(r, j) = s \cdot \sum_{d=1}^{K} q_{r,d} \cdot k_{j,d} \cdot 2^{g_{r,d} - g_{j,d}}, \quad r \geq j$$

$$\mathbf{A}_{kk}(r, j) = \beta_r \cdot \sum_{d=1}^{K} k_{r,d} \cdot k_{j,d} \cdot 2^{g_{r,d} - g_{j,d}}, \quad r > j$$

两者的结构完全对称，仅"左侧向量"不同：

$$\mathbf{A}(r, j) = \underbrace{\text{left}_r}_{\substack{s \cdot \mathbf{q}_r \\ \text{或} \\ \beta_r \cdot \mathbf{k}_r}} \cdot \underbrace{\left[\mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}\right]}_{\text{right}_j}$$

利用 decay 的可分解性 $`2^{g_r - g_j} = 2^{g_r} \cdot 2^{-g_j}`$，可以把 $`\mathbf{k}_j \odot 2^{-\mathbf{g}_j}`$ 提出来作为"decayed key" $`\tilde{\mathbf{k}}_j`$。

---

### 数学推导

$`\mathbf{k}_j`$ 在 $`\mathbf{A}_{qk}`$ 和 $`\mathbf{A}_{kk}`$ 中**同时出现两次**——一次作为"左侧向量"中 $`\mathbf{k}_r`$（行端），一次作为"右侧向量"中 $`\mathbf{k}_j`$（列端）。因此 $`d\mathbf{k}`$ 有**两个方向**的梯度贡献，对应实现中的 **Section 1（行方向）** 和 **Section 2（列方向）**。

#### 1. Section 1：行方向梯度（$`r`$ 作为查询端）

##### 1.1 $`d\mathbf{q}`$

$`\mathbf{A}_{qk}(r, j)`$ 对 $`\mathbf{q}_r`$ 的偏导：

$$\frac{\partial A_{qk}(r, j)}{\partial \mathbf{q}_r} = s \cdot \mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

对所有 $`j \leq r`$ 求和（链式法则）：

$$d\mathbf{q}_r = \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot s \cdot \mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

提出不依赖 $`j`$ 的因子 $`2^{\mathbf{g}_r}`$：

$$\boxed{d\mathbf{q}_r = s \cdot 2^{\mathbf{g}_r} \odot \sum_{j \leq r} d\mathbf{A}_{qk}(r, j) \cdot \tilde{\mathbf{k}}_j}$$

矩阵形式：$`d\mathbf{q} = s \cdot 2^{\mathbf{g}} \odot \left[d\mathbf{A}_{qk} \cdot \tilde{\mathbf{k}}\right]`$，其中 $`d\mathbf{A}_{qk} \cdot \tilde{\mathbf{k}}`$ 是 $`[C, C] \times [C, K] = [C, K]`$ 的矩阵乘法（下三角 mask 已隐含在 $`d\mathbf{A}_{qk}`$ 中）。

##### 1.2 $`d\mathbf{k}_r^{\text{sec1}}`$（$`\mathbf{k}_r`$ 作为左侧向量）

$`\mathbf{A}_{kk}(r, j)`$ 对 $`\mathbf{k}_r`$（左侧向量）的偏导与 $`d\mathbf{q}_r`$ 完全对称：

$$d\mathbf{k}_r^{\text{sec1}} = \sum_{j < r} d\mathbf{A}_{kk}(r, j) \cdot \mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

$$\boxed{d\mathbf{k}_r^{\text{sec1,pre}} = 2^{\mathbf{g}_r} \odot \sum_{j < r} d\mathbf{A}_{kk}(r, j) \cdot \tilde{\mathbf{k}}_j}$$

矩阵形式：$`d\mathbf{k}^{\text{sec1,pre}} = 2^{\mathbf{g}} \odot \left[d\mathbf{A}_{kk} \cdot \tilde{\mathbf{k}}\right]`$。

注意 $`d\mathbf{k}_r^{\text{sec1,pre}}`$ 还没有乘 $`\beta_r`$（因为 $`\mathbf{A}_{kk}(r,j) = \beta_r \cdot \mathbf{k}_r \cdot \ldots`$，$`\beta_r`$ 的梯度需要先提取出来）。

##### 1.3 $`d\boldsymbol{\beta}`$

$`\beta_r`$ 作为标量因子出现在 $`\mathbf{A}_{kk}(r, j) = \beta_r \cdot \mathbf{k}_r^T \ldots`$ 中。对 $`\beta_r`$ 求偏导：

$$\frac{\partial A_{kk}(r, j)}{\partial \beta_r} = \mathbf{k}_r^T \cdot [\mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}]$$

对所有 $`j < r`$ 求和：

$$d\beta_r = \sum_{j < r} d\mathbf{A}_{kk}(r, j) \cdot \mathbf{k}_r^T [\mathbf{k}_j \odot 2^{\mathbf{g}_r - \mathbf{g}_j}]$$

注意到求和项正好是 $`d\mathbf{k}_r^{\text{sec1,pre}}`$ 的各分量被 $`\mathbf{k}_r`$ 加权求和：

$$\boxed{d\beta_r = \sum_{d=1}^{K} dk_{r,d}^{\text{sec1,pre}} \cdot k_{r,d} = \langle d\mathbf{k}_r^{\text{sec1,pre}}, \mathbf{k}_r \rangle}$$

这就是实现中 `b_db = tl.sum(b_dk2 * b_k, 1)` 的来源。先算 $`d\beta`$，再令 $`d\mathbf{k}_r^{\text{sec1}} = d\mathbf{k}_r^{\text{sec1,pre}} \cdot \beta_r`$。

---

#### 2. Section 2：列方向梯度（$`j`$ 作为被查询端）

##### 2.1 $`d\mathbf{k}_j^{\text{sec2}}`$（$`\mathbf{k}_j`$ 作为右侧向量）

$`\mathbf{k}_j`$ 作为右侧向量，同时出现在 $`\mathbf{A}_{qk}(r, j)`$ 和 $`\mathbf{A}_{kk}(r, j)`$ 中。

来自 $`\mathbf{A}_{qk}`$：

$$\frac{\partial A_{qk}(r, j)}{\partial \mathbf{k}_j} = s \cdot \mathbf{q}_r \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

来自 $`\mathbf{A}_{kk}`$：

$$\frac{\partial A_{kk}(r, j)}{\partial \mathbf{k}_j} = \beta_r \cdot \mathbf{k}_r \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

对所有 $`r > j`$（梯度从"下方"行流入）求和：

$$d\mathbf{k}_j^{\text{sec2}} = \sum_{r > j} \left[d\mathbf{A}_{qk}(r, j) \cdot s \cdot \mathbf{q}_r + d\mathbf{A}_{kk}(r, j) \cdot \beta_r \cdot \mathbf{k}_r\right] \odot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

提出不依赖 $`r`$ 的因子 $`2^{-\mathbf{g}_j}`$：

$$\boxed{d\mathbf{k}_j^{\text{sec2}} = 2^{-\mathbf{g}_j} \odot \sum_{r > j} \left[d\mathbf{A}_{qk}(r, j) \cdot \hat{\mathbf{q}}_r + d\mathbf{A}_{kk}(r, j) \cdot \hat{\mathbf{k}}_r\right]}$$

矩阵形式：$`d\mathbf{k}^{\text{sec2}} = 2^{-\mathbf{g}} \odot \left[d\mathbf{A}_{qk}^T \cdot \hat{\mathbf{q}} + d\mathbf{A}_{kk}^T \cdot \hat{\mathbf{k}}\right]`$。

关键区别：Section 1 读取 $`d\mathbf{A}`$ 的**行**，是 $`r`$ 向更早的 $`j`$ 查询；Section 2 读取 $`d\mathbf{A}`$ 的**列**（转置），是 $`j`$ 收集来自更晚 $`r`$ 的贡献。

---

#### 3. $`d\mathbf{g}`$ 的推导

$`\mathbf{g}`$ 通过 $`2^{\mathbf{g}_r - \mathbf{g}_j}`$ 出现在 $`\mathbf{A}_{qk}`$ 和 $`\mathbf{A}_{kk}`$ 中。对 $`\mathbf{g}_r`$ 和 $`\mathbf{g}_j`$ 分别求偏导：

$$\frac{\partial}{\partial \mathbf{g}_r} 2^{\mathbf{g}_r - \mathbf{g}_j} = +\ln 2 \cdot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

$$\frac{\partial}{\partial \mathbf{g}_j} 2^{\mathbf{g}_r - \mathbf{g}_j} = -\ln 2 \cdot 2^{\mathbf{g}_r - \mathbf{g}_j}$$

（实现中使用 $`2^x`$ 而非 $`e^x`$，因此 $`\ln 2`$ 因子被吸收进 exp2 的基底中。以下推导中省略 $`\ln 2`$ 因子。）

**从 Section 1 方向**（$`\mathbf{g}_r`$ 以正号出现）：

$`\mathbf{g}_r`$ 影响 $`\mathbf{A}_{qk}(r, \cdot)`$ 和 $`\mathbf{A}_{kk}(r, \cdot)`$ 中的所有列。链式法则给出：

$$d\mathbf{g}_r^{\text{sec1}} = \mathbf{q}_r \odot d\mathbf{q}_r^{\text{intra}} + \mathbf{k}_r \odot d\mathbf{k}_r^{\text{sec1}}$$

其中 $`d\mathbf{q}_r^{\text{intra}}`$ 和 $`d\mathbf{k}_r^{\text{sec1}}`$ 已在 Section 1 中计算。

直觉：$`\mathbf{g}_r`$ 与左侧向量（$`\mathbf{q}_r`$ 或 $`\mathbf{k}_r`$）通过 $`\text{left}_r \odot 2^{\mathbf{g}_r}`$ 形式相乘，因此 $`d\mathbf{g}_r \propto \text{left}_r \odot d(\text{left}_r)`$。

**从 Section 2 方向**（$`\mathbf{g}_j`$ 以负号出现）：

$`\mathbf{g}_j`$ 以 $`-\mathbf{g}_j`$ 出现在 $`2^{\mathbf{g}_r - \mathbf{g}_j}`$ 中，因此梯度带负号：

$$d\mathbf{g}_j^{\text{sec2}} = -\mathbf{k}_j \odot d\mathbf{k}_j^{\text{sec2}}$$

**合并**：

$$\boxed{d\mathbf{g}_r = \underbrace{\mathbf{q}_r \odot d\mathbf{q}_r^{\text{intra}}}_{A_{qk} \text{ 行端}} + \underbrace{\mathbf{k}_r \odot (d\mathbf{k}_r^{\text{sec1}} - d\mathbf{k}_r^{\text{sec2}})}_{\text{行端 - 列端}}}$$

实现中对应 `b_dg2 = b_q * b_dq2`（$`\mathbf{A}_{qk}`$ 行端贡献）和 `b_dg2 += (b_dk2 - b_dkt) * b_k`（$`\mathbf{k}`$ 的行端减列端贡献）。

---

#### 4. Reverse cumsum

$`\mathbf{g}`$ 是前向 Step 0 中 $`\mathbf{g}_{\text{raw}}`$ 经过 cumsum 得到的：$`g_r = \sum_{i=1}^{r} g_i^{\text{raw}}`$。

cumsum 的 Jacobian 是下三角全 1 矩阵，其转置即 reverse cumsum：

$$dg_r^{\text{raw}} = \sum_{j=r}^{C} dg_j$$

实现中在 kernel 外通过 `chunk_local_cumsum(dg, reverse=True)` 完成。这一步将 chunk 内的 $`d\mathbf{g}`$ 转换为对原始 $`\mathbf{g}_{\text{raw}}`$ 的梯度。

---

### 实现细节

#### Grid 与分块策略

```
Grid: (NK * NC, NT, B * H)
  NK = ceil(K / BK)    head_dim 方向的分块数
  NC = BT / BC          每个 chunk 内的子块数（通常 64/16 = 4）
  NT                    chunk 数量
  B * H                 batch × head
```

每个 thread block 处理一个 **(子块 $`i_i`$, K 分块 $`i_k`$, chunk $`i_t`$, batch-head $`i_{bh}`$)** 的组合。在 K 维度上分块是因为 $`K`$ 可能大于寄存器容量（BK 通常为 32）。

#### 数值稳定性：参考点归一化

直接计算 $`2^{\mathbf{g}_r - \mathbf{g}_j}`$ 可能溢出（$`\mathbf{g}`$ 是累积和，绝对值可能很大）。实现中使用**参考点归一化**：

**Section 1**（行方向）：选取当前子块起始位置的 $`\mathbf{g}_{n}`$ 作为参考点。

$$2^{\mathbf{g}_r - \mathbf{g}_j} = \underbrace{2^{\mathbf{g}_r - \mathbf{g}_n}}_{\text{后乘}} \cdot \underbrace{2^{\mathbf{g}_n - \mathbf{g}_j}}_{\text{先算}}$$

- 先计算 $`\tilde{\mathbf{k}}_j' = \mathbf{k}_j \cdot 2^{\mathbf{g}_n - \mathbf{g}_j}`$（循环中累加）
- 循环结束后统一乘 $`2^{\mathbf{g}_r - \mathbf{g}_n}`$

**Section 2**（列方向）：选取当前子块末尾位置的 $`\mathbf{g}_{n}`$ 作为参考点。

$$2^{\mathbf{g}_r - \mathbf{g}_j} = \underbrace{2^{\mathbf{g}_r - \mathbf{g}_n}}_{\text{先算}} \cdot \underbrace{2^{\mathbf{g}_n - \mathbf{g}_j}}_{\text{后乘}}$$

- 先计算 $`\hat{\mathbf{q}}_r' = \mathbf{q}_r \cdot 2^{\mathbf{g}_r - \mathbf{g}_n}`$（循环中累加）
- 循环结束后统一乘 $`2^{\mathbf{g}_n - \mathbf{g}_j}`$

#### 对角块的特殊处理

当子块 $`i_i`$ 与自身交互时（$`r, j`$ 在同一个子块内），需要**逐列扫描**而非子块级矩阵乘法，因为因果 mask（$`r \geq j`$ 或 $`r > j`$）切割了矩阵乘法的矩形结构。

```python
# 对角块：逐列 j 遍历
for j in range(BC):
    b_kgj = k[j] * exp2(g[row] - g[j])
    # Section 1: dq[row] += where(row >= j, dAqk[row, j]) * kgj
    b_dq += where(row >= j, dAqk[row, j] * kgj, 0)
    # Section 2: dkt[row] += where(row <= j, dAqk[j, row] * q[j] + ...) * exp2(g[j]-g[row])
    b_dkt += where(row <= j, (dAkk[j, row] * kb[j] + dAqk[j, row] * q[j]) * exp2(g[j]-g[row]), 0)
```

#### db 的跨 K 分块累加

$`d\beta_r = \sum_d dk_{r,d}^{\text{sec1,pre}} \cdot k_{r,d}`$ 需要对**整个 K 维度**求和，但 kernel 沿 K 维度分块（每块 BK）。因此 db 的 shape 为 `[NK, B, T, H]`，每个 K 分块独立写入，kernel 结束后在 wrapper 中 `db2.sum(0)` 合并。

---

### 完整计算流程

```python
# === Section 1: 行方向（当前子块 i_i 作为 query 端）===

dq2 = zeros([BC, BK])
dk2 = zeros([BC, BK])

# 1a. 跨子块累加（i_j < i_i）
g_ref = g[i_i * BC]                      # 参考点
for i_j in range(0, i_i):
    kg_j = k[i_j] * exp2(g_ref - g[i_j])  # [BC, BK]
    dq2 += dAqk[i_i, i_j] @ kg_j          # [BC, BC] @ [BC, BK]
    dk2 += dAkk[i_i, i_j] @ kg_j
dq2 *= exp2(g[i_i] - g_ref)               # 还原衰减
dk2 *= exp2(g[i_i] - g_ref)

# 1b. 对角块（逐列）
for j in range(BC):
    kgj = k[j] * exp2(g - g[j])           # [BC, BK]
    dq2 += where(row >= j, dAqk[row, j] * kgj, 0)
    dk2 += where(row >= j, dAkk[row, j] * kgj, 0)

# 1c. 提取 db，应用 beta
db = sum(dk2 * k, dim=-1)                 # [BC]
dk2 *= beta[:, None]                       # [BC, BK]

# 1d. dg 行方向贡献
dg = q * dq2                              # [BC, BK]

# 1e. 累加 inter-chunk 梯度
dq2 += dq_inter
store(dq2)
store(db)

# === Section 2: 列方向（当前子块 i_i 作为 key 端）===

dkt = zeros([BC, BK])

# 2a. 跨子块累加（i_j > i_i）
g_ref = g[(i_i+1)*BC - 1]                 # 参考点（子块末尾）
for i_j in range(i_i + 1, NC):
    qg_j = q[i_j] * exp2(g[i_j] - g_ref)  # [BC, BK]
    kbg_j = k[i_j] * beta[i_j] * exp2(g[i_j] - g_ref)
    dkt += dAqk[i_j, i_i]^T @ qg_j + dAkk[i_j, i_i]^T @ kbg_j
dkt *= exp2(g_ref - g[i_i])

# 2b. 对角块（逐行）
for j in range(BC):
    gkq = exp2(g[j] - g)                  # [BC, BK]
    dkt += where(row <= j, (dAkk[j, row] * kb[j] + dAqk[j, row] * q[j]) * gkq, 0)

# === 合并 ===
dg += (dk2 - dkt) * k + dg_inter          # 行端 - 列端
dk = dk_inter + dk2 + dkt
store(dk)
store(dg)
```

---

### Triton Kernel 参数与 I/O

#### 输入

| 参数 | Shape | 说明 |
|------|-------|------|
| `q` | `[B, T, H, K]` | query |
| `k` | `[B, T, H, K]` | key |
| `g` | `[B, T, H, K]` | cumsum-ed gate (log2) |
| `beta` | `[B, T, H]` | 标量混合系数 |
| `dAqk` | `[B, T, H, BT]` | query-key 注意力矩阵梯度（下三角） |
| `dAkk` | `[B, T, H, BT]` | key-key 注意力矩阵梯度（严格下三角） |
| `dq` | `[B, T, H, K]` | inter-chunk 的 dq（输入，待累加） |
| `dk` | `[B, T, H, K]` | inter-chunk 的 dk（输入，待累加） |
| `dg` | `[B, T, H, K]` | inter-chunk 的 dg（输入，待累加） |
| `db` | `[B, T, H]` | inter-chunk 的 db（输入，待累加） |

#### 输出

| 参数 | Shape | 说明 |
|------|-------|------|
| `dq2` | `[B, T, H, K]` | 最终 dq = dq_inter + dq_intra |
| `dk2` | `[B, T, H, K]` | 最终 dk = dk_inter + dk_intra |
| `dg2` | `[B, T, H, K]` | dg（cumsum 前），后续需 reverse_cumsum |
| `db` | `[NK, B, T, H]` | db 按 K 分块累加，后续需 sum over NK 再加 db_inter |

#### Wrapper 后处理

```python
dq = dq2
dk = dk2
db = db2.sum(0) + db_inter              # 合并 K 分块 + inter-chunk
dg = chunk_local_cumsum(dg2, reverse=True)  # reverse cumsum
```

---

### 梯度公式速查

| 梯度 | 公式 | 方向 |
|------|------|------|
| $`d\mathbf{q}`$ | $`s \cdot 2^{\mathbf{g}} \odot [d\mathbf{A}_{qk} \cdot \tilde{\mathbf{k}}]`$ | Section 1（行） |
| $`d\mathbf{k}^{\text{sec1}}`$ | $`\beta \cdot 2^{\mathbf{g}} \odot [d\mathbf{A}_{kk} \cdot \tilde{\mathbf{k}}]`$ | Section 1（行） |
| $`d\mathbf{k}^{\text{sec2}}`$ | $`2^{-\mathbf{g}} \odot [d\mathbf{A}_{qk}^T \hat{\mathbf{q}} + d\mathbf{A}_{kk}^T \hat{\mathbf{k}}]`$ | Section 2（列） |
| $`d\boldsymbol{\beta}`$ | $`\langle d\mathbf{k}^{\text{sec1,pre}}, \mathbf{k} \rangle`$ | Section 1 的副产品 |
| $`d\mathbf{g}`$ | $`\mathbf{q} \odot d\mathbf{q}^{\text{intra}} + \mathbf{k} \odot (d\mathbf{k}^{\text{sec1}} - d\mathbf{k}^{\text{sec2}})`$ | 行端 - 列端 |

最终 $`d\mathbf{k} = d\mathbf{k}^{\text{sec1}} + d\mathbf{k}^{\text{sec2}} + d\mathbf{k}_{\text{inter}}`$。

---

### 与 PyTorch 矩阵形式的对照

在 `tops/cpu/ops/kda/chunk_bwd_intra_ref.py` 中，上述逐子块计算被简化为 4 个矩阵乘法：

```python
k̃  = k * exp(-g)           # decayed key
q̃  = q * exp(g)            # anti-decayed query
β̃k = beta * k * exp(g)     # anti-decayed beta·key

# Section 1 → 2 个 matmul
Mqk = dAqk  @ k̃            # dq_pre = exp(g) * Mqk
Mkk = dAkk  @ k̃            # dk_sec1_pre = exp(g) * Mkk

# Section 2 → 2 个 matmul
N   = dAqk.mT @ q̃ + dAkk.mT @ β̃k   # dkt = exp(-g) * N

# 梯度组装
dq = exp(g) * Mqk * scale
dk = beta * exp(g) * Mkk + exp(-g) * N
dβ = sum(k * exp(g) * Mkk, dim=-1)
dg = q̃ * Mqk + β̃k * Mkk − k̃ * N
```
