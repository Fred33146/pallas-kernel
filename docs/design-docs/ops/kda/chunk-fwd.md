# chunk_kda_fwd design

参考文献

https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf

https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py

本设计文档记录了 Kimi Delta Attention (KDA) 的 Chunk-Parallel 前向内核的设计与实现逻辑。

### 核心思想与背景

在 [fused_recurrent_kda_fwd](./fused-recurrent-fwd.md) 中，我们已经推导了 KDA 的单步循环形式：每一个时间步 $`t`$ 都严格依赖 $`\mathbf{S}_{t-1}`$，因此在 $`T`$ 维度上是**完全串行**的。当序列很长时，这种 $`O(T)`$ 的串行深度成为硬件利用率的瓶颈。

**Chunk-Parallel 的核心思想**是：将长度为 $`T`$ 的序列切分为 $`\text{NT} = T / C`$ 个大小为 $`C`$ 的块（chunk），在**块间串行**传递状态、**块内并行**计算输出。这与对 cumsum 做并行扫描的思路类似。

但 KDA 的块内并行比 GLA 要复杂：

- **GLA**：$`\mathbf{H}_t = \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top`$，块内每步写入的 $`\mathbf{k}_t \mathbf{v}_t^\top`$ **不依赖当前状态**，因此可以直接并行
- **KDA**：$`\mathbf{S}_t = (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)\text{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top`$，块内每步的写入**依赖当前衰减后的状态** $`\mathbf{S}_{t-1}'`$

这意味着 KDA 的块内并行需要**先求解一个下三角线性系统**来解耦块内的依赖关系，然后才能像 GLA 一样进行并行矩阵运算。本文档将详细推导这一过程。

---

### 符号定义

- $`t`$: 时间步，$`t = 1, \dots, T`$。
- $`C`$: chunk 大小（块大小），通常为 64。
- $`\text{NT} = T / C`$: chunk 数量。
- $`[t]`$: 第 $`t`$ 个 chunk 的索引。$`r = 1, \dots, C`$: chunk 内的局部位置。
- $`\mathbf{q}_t, \mathbf{k}_t \in \mathbb{R}^{K}`$: Query 和 Key 向量。
- $`\mathbf{v}_t \in \mathbb{R}^{V}`$: Value 向量。
- $`\mathbf{S}_t \in \mathbb{R}^{K \times V}`$: 时刻 $`t`$ 的隐藏状态缓存矩阵。
- $`\mathbf{S}_0`$: 每个 chunk 的**起始状态**（前一个 chunk 的终态）。
- $`\mathbf{g}_t \in \mathbb{R}^{K}`$: 对数空间的通道级衰减门。$`\boldsymbol{\alpha}_t = \exp(\mathbf{g}_t)`$。
- $`\beta_t \in \mathbb{R}`$: 写入门控 / 学习率 (Learning Rate)。
- $`\boldsymbol{\delta}_t \in \mathbb{R}^{V}`$: 残差向量。$`\boldsymbol{\delta}_t = \beta_t(\mathbf{v}_t - \mathbf{k}_t^\top \mathbf{S}_t')`$。
- $`s`$: 缩放因子 (`scale`，通常为 $`1/\sqrt{K}`$)。
- $`a_{r,i} \in \mathbb{R}`$: 键-键对齐分数。$`a_{r,i} = \mathbf{k}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i]`$。
- $`\mathbf{A} \in \mathbb{R}^{C \times C}`$: 交互矩阵（求解后包含 $`(\mathbf{I} + \mathbf{A}_{\text{raw}})^{-1} \cdot \text{diag}(\boldsymbol{\beta})`$）。
- $`\mathbf{w} \in \mathbb{R}^{C \times K}`$: 有效键。$`\mathbf{w} = \mathbf{A} \cdot (\exp(\mathbf{g}) \odot \mathbf{k})`$。
- $`\mathbf{u} \in \mathbb{R}^{C \times V}`$: 有效值。$`\mathbf{u} = \mathbf{A} \cdot \mathbf{v}`$。

---

### 计算流程 (Chunk-Parallel Forward)

#### 公式等价变换 (Mathematical Transformation)

##### 1. 块内递推展开

在每个 chunk 内部，利用 [fused_recurrent 设计文档](./fused-recurrent-fwd.md) 中已推导的残差更新形式（$`O(KV)`$ 等价变换）逐步展开。

回顾单步残差更新形式：

$$\mathbf{S}_r' = \text{Diag}(\boldsymbol{\alpha}_r) \mathbf{S}_{r-1} = \exp(\mathbf{g}_r^{\text{raw}}) \odot \mathbf{S}_{r-1}$$

$$\mathbf{S}_r = \mathbf{S}_r' + \mathbf{k}_r \boldsymbol{\delta}_r^\top, \quad \text{其中 } \boldsymbol{\delta}_r = \beta_r(\mathbf{v}_r - \mathbf{k}_r^\top \mathbf{S}_r')$$

设 $`\mathbf{g}`$ 为 chunk 内的前缀累加和（cumsum），即 $`\mathbf{g}_r = \sum_{i=1}^{r} \mathbf{g}_i^{\text{raw}}`$。这样从位置 $`i`$ 到位置 $`r`$ 的累积衰减为 $`\exp(\mathbf{g}_r - \mathbf{g}_i)`$。

**$`r=1`$**：

$$\mathbf{S}_1' = \exp(\mathbf{g}_1) \odot \mathbf{S}_0$$

$$\boldsymbol{\delta}_1 = \beta_1(\mathbf{v}_1 - \mathbf{k}_1^\top \mathbf{S}_1')$$

$$\mathbf{S}_1 = \mathbf{S}_1' + \mathbf{k}_1 \boldsymbol{\delta}_1^\top$$

**$`r=2`$**：

$$\mathbf{S}_2' = \exp(\mathbf{g}_2^{\text{raw}}) \odot \mathbf{S}_1 = \exp(\mathbf{g}_2^{\text{raw}}) \odot (\mathbf{S}_1' + \mathbf{k}_1 \boldsymbol{\delta}_1^\top)$$

将 $`\mathbf{S}_1' = \exp(\mathbf{g}_1) \odot \mathbf{S}_0`$ 代入，并利用 $`\exp(\mathbf{g}_2^{\text{raw}}) \cdot \exp(\mathbf{g}_1) = \exp(\mathbf{g}_2)`$：

$$\mathbf{S}_2' = \exp(\mathbf{g}_2) \odot \mathbf{S}_0 + \exp(\mathbf{g}_2 - \mathbf{g}_1) \odot \mathbf{k}_1 \boldsymbol{\delta}_1^\top$$

$$\boldsymbol{\delta}_2 = \beta_2(\mathbf{v}_2 - \mathbf{k}_2^\top \mathbf{S}_2')$$

$$\mathbf{S}_2 = \mathbf{S}_2' + \mathbf{k}_2 \boldsymbol{\delta}_2^\top$$

**$`r=3`$**：

$$\mathbf{S}_3' = \exp(\mathbf{g}_3^{\text{raw}}) \odot \mathbf{S}_2 = \exp(\mathbf{g}_3^{\text{raw}}) \odot (\mathbf{S}_2' + \mathbf{k}_2 \boldsymbol{\delta}_2^\top)$$

展开 $`\mathbf{S}_2'`$：

$$\mathbf{S}_3' = \exp(\mathbf{g}_3) \odot \mathbf{S}_0 + \exp(\mathbf{g}_3 - \mathbf{g}_1) \odot \mathbf{k}_1 \boldsymbol{\delta}_1^\top + \exp(\mathbf{g}_3 - \mathbf{g}_2) \odot \mathbf{k}_2 \boldsymbol{\delta}_2^\top$$

**归纳一般形式**：观察规律，位置 $`r`$ 处衰减后的状态为：

$$\mathbf{S}_r' = \underbrace{\exp(\mathbf{g}_r) \odot \mathbf{S}_0}_{\text{外部状态经 r 步累积衰减}} + \underbrace{\sum_{i=1}^{r-1} \exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top}_{\text{块内前 r-1 步的写入，各自衰减到位置 r}}$$

其中每个 $`\boldsymbol{\delta}_i = \beta_i(\mathbf{v}_i - \mathbf{k}_i^\top \mathbf{S}_i') \in \mathbb{R}^V`$。

**关键观察**：$`\boldsymbol{\delta}_r`$ 依赖 $`\mathbf{S}_r'`$，而 $`\mathbf{S}_r'`$ 又包含了所有之前的 $`\boldsymbol{\delta}_1, \dots, \boldsymbol{\delta}_{r-1}`$——构成**顺序依赖**。这是 KDA 与 GLA 的本质区别：GLA 的写入项 $`\mathbf{k}_t \mathbf{v}_t^\top`$ 不依赖状态，因此无此耦合。

##### 2. 构建下三角线性系统

目标：把 $`\boldsymbol{\delta}_r`$ 之间的顺序依赖转化为一个可以求解的线性系统。

**第一步：代入 $`\mathbf{S}_r'`$ 的展开式**

回顾 $`\boldsymbol{\delta}_r`$ 的定义和 $`\mathbf{S}_r'`$ 的一般形式：

$$\boldsymbol{\delta}_r = \beta_r(\mathbf{v}_r - \mathbf{k}_r^\top \mathbf{S}_r')$$

$$\mathbf{S}_r' = \exp(\mathbf{g}_r) \odot \mathbf{S}_0 + \sum_{i=1}^{r-1} \exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top$$

将 $`\mathbf{S}_r'`$ 代入 $`\boldsymbol{\delta}_r`$：

$$\boldsymbol{\delta}_r = \beta_r \left(\mathbf{v}_r - \mathbf{k}_r^\top \left[\exp(\mathbf{g}_r) \odot \mathbf{S}_0 + \sum_{i=1}^{r-1} \exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top \right]\right)$$

**第二步：展开 $`\mathbf{k}_r^\top`$ 与各项的乘积**

$`\mathbf{k}_r^\top`$ 分别作用于两项：

$$\boldsymbol{\delta}_r = \beta_r \mathbf{v}_r - \beta_r \underbrace{\mathbf{k}_r^\top [\exp(\mathbf{g}_r) \odot \mathbf{S}_0]}_{\text{（甲）与外部状态相关}} - \beta_r \sum_{i=1}^{r-1} \underbrace{\mathbf{k}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i]}_{\text{（乙）标量 } a_{r,i}} \boldsymbol{\delta}_i$$

其中（乙）是关键：$`\mathbf{k}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i]`$ 是一个**标量**，因为 $`\mathbf{k}_r \in \mathbb{R}^K`$ 与 $`[\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i] \in \mathbb{R}^K`$ 做内积。

具体展开过程：首先 $`\odot`$（逐元素乘）将衰减逐分量乘进 $`\mathbf{k}_i`$，得到一个新的 $`K`$ 维向量：

$$\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i = \begin{pmatrix} \exp(g_{r,1} - g_{i,1}) \cdot k_{i,1} \\ \exp(g_{r,2} - g_{i,2}) \cdot k_{i,2} \\ \vdots \\ \exp(g_{r,K} - g_{i,K}) \cdot k_{i,K} \end{pmatrix}$$

然后 $`\mathbf{k}_r^\top`$ 与该向量做内积（对应分量相乘再求和）：

$$\mathbf{k}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i] = k_{r,1} \cdot \exp(g_{r,1} - g_{i,1}) \cdot k_{i,1} + k_{r,2} \cdot \exp(g_{r,2} - g_{i,2}) \cdot k_{i,2} + \cdots + k_{r,K} \cdot \exp(g_{r,K} - g_{i,K}) \cdot k_{i,K}$$

写成求和符号：

$$a_{r,i} = \sum_{d=1}^{K} k_{r,d} \cdot \exp(g_{r,d} - g_{i,d}) \cdot k_{i,d}$$

它度量的是：位置 $`i`$ 通过 $`\mathbf{k}_i`$ 写入状态的内容，经过 $`i \to r`$ 的衰减后，位置 $`r`$ 的 $`\mathbf{k}_r`$ 能检索到多少——即块内两个位置之间的**带衰减的键-键相似度**（耦合强度）。

**第三步：分离已知项和未知项**

定义右端项（仅依赖外部状态 $`\mathbf{S}_0`$，与 $`\boldsymbol{\delta}`$ 无关）：

$$\mathbf{b}_r = \beta_r \left(\mathbf{v}_r - \mathbf{k}_r^\top [\exp(\mathbf{g}_r) \odot \mathbf{S}_0]\right) \in \mathbb{R}^V$$

则 $`\boldsymbol{\delta}_r`$ 的表达式整理为：

$$\boldsymbol{\delta}_r = \mathbf{b}_r - \beta_r \sum_{i=1}^{r-1} a_{r,i} \cdot \boldsymbol{\delta}_i$$

移项：

$$\boldsymbol{\delta}_r + \beta_r \sum_{i=1}^{r-1} a_{r,i} \cdot \boldsymbol{\delta}_i = \mathbf{b}_r$$

**第四步：写出具体的逐行方程**

$$r=1: \quad \boldsymbol{\delta}_1 = \mathbf{b}_1$$

$$r=2: \quad \boldsymbol{\delta}_2 + \beta_2 \cdot a_{2,1} \cdot \boldsymbol{\delta}_1 = \mathbf{b}_2$$

$$r=3: \quad \boldsymbol{\delta}_3 + \beta_3 \cdot a_{3,1} \cdot \boldsymbol{\delta}_1 + \beta_3 \cdot a_{3,2} \cdot \boldsymbol{\delta}_2 = \mathbf{b}_3$$

$$\vdots$$

**第五步：组装为矩阵形式**

定义 $`\mathbf{A}_{\text{raw}} \in \mathbb{R}^{C \times C}`$，元素为：

$$(\mathbf{A}_{\text{raw}})_{r,i} = \begin{cases} \beta_r \cdot a_{r,i} & \text{if } r > i \\ 0 & \text{if } r \leq i \end{cases}$$

这是一个**严格下三角矩阵**（对角线和上三角全为 0）。上面的方程组可以统一写为：

$$(\mathbf{I} + \mathbf{A}_{\text{raw}}) \begin{pmatrix} \boldsymbol{\delta}_1 \\ \boldsymbol{\delta}_2 \\ \vdots \\ \boldsymbol{\delta}_C \end{pmatrix} = \begin{pmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_C \end{pmatrix}$$

$`\mathbf{I} + \mathbf{A}_{\text{raw}}`$ 对角线全为 1，是一个**单位下三角矩阵**，因此必定可逆，可以通过前代法求解。

##### 3. 前代法求解

$`\mathbf{I} + \mathbf{A}_{\text{raw}}`$ 是单位下三角矩阵，可通过原地前代法（forward substitution）求解其逆。求解后将 $`\beta`$ 整合进矩阵：$`\mathbf{A} = (\mathbf{I} + \mathbf{A}_{\text{raw}})^{-1} \cdot \text{diag}(\boldsymbol{\beta})`$。

##### 4. 有效键值与输出

用 $`\mathbf{A}`$ 变换原始 $`\mathbf{k}, \mathbf{v}`$ 为有效键值后，输出公式变为与 GLA 类似的两部分结构。下面给出完整的推导过程。

**第一步：展开 $`\mathbf{S}_r`$**

每个位置 $`r`$ 的输出为 $`\mathbf{o}_r = \mathbf{q}_r^\top \mathbf{S}_r`$。将 $`\mathbf{S}_r = \mathbf{S}_r' + \mathbf{k}_r \boldsymbol{\delta}_r^\top`$ 完全展开：

$$\mathbf{S}_r = \exp(\mathbf{g}_r) \odot \mathbf{S}_0 + \sum_{i=1}^{r-1} \exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top + \mathbf{k}_r \boldsymbol{\delta}_r^\top$$

注意最后一项 $`\mathbf{k}_r \boldsymbol{\delta}_r^\top = \exp(\mathbf{g}_r - \mathbf{g}_r) \odot \mathbf{k}_r \boldsymbol{\delta}_r^\top`$（因为 $`\exp(0) = 1`$），所以求和上界可以合并到 $`r`$：

$$\mathbf{S}_r = \exp(\mathbf{g}_r) \odot \mathbf{S}_0 + \sum_{i=1}^{r} \exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top$$

**第二步：左乘 $`\mathbf{q}_r^\top`$，分离 inter-chunk 和 intra-chunk**

$$\mathbf{o}_r = \mathbf{q}_r^\top \left[\exp(\mathbf{g}_r) \odot \mathbf{S}_0\right] + \sum_{i=1}^{r} \mathbf{q}_r^\top \left[\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top\right]$$

- **Inter-chunk 项**：逐元素乘可以移到 $`\mathbf{q}_r`$ 上：$`\mathbf{q}_r^\top [\exp(\mathbf{g}_r) \odot \mathbf{S}_0] = (\mathbf{q}_r \odot \exp(\mathbf{g}_r))^\top \mathbf{S}_0`$
- **Intra-chunk 项**：利用与推导 $`a_{r,i}`$ 时相同的技巧（$`\odot`$ 先吸收进 $`\mathbf{k}_i`$，再用结合律），每一项变为：

$$\mathbf{q}_r^\top [\exp(\mathbf{g}_r - \mathbf{g}_i) \odot \mathbf{k}_i \boldsymbol{\delta}_i^\top] = \underbrace{(\mathbf{q}_r \odot \exp(\mathbf{g}_r - \mathbf{g}_i))^\top \mathbf{k}_i}_{A_{qk}(r,i)} \cdot \boldsymbol{\delta}_i$$

这个标量 $`A_{qk}(r,i)`$ 与 $`A_{kk}(r,i)`$ 结构完全一样，只是把 $`\mathbf{k}_r`$ 换成了 $`\mathbf{q}_r`$。

组合得到：

$$\mathbf{o}_r = (\mathbf{q}_r \odot \exp(\mathbf{g}_r))^\top \mathbf{S}_0 + \sum_{i=1}^{r} A_{qk}(r,i) \cdot \boldsymbol{\delta}_i$$

**第三步：用有效键值 $`\mathbf{u}, \mathbf{w}`$ 替换 $`\boldsymbol{\delta}`$**

Intra-chunk 项中的 $`\boldsymbol{\delta}_i`$ 依赖 $`\mathbf{S}_0`$，不利于并行。利用三角系统求解后的表达式 $`\boldsymbol{\delta}_r = \sum_j A_{r,j} \cdot (\mathbf{v}_j - \mathbf{k}_j^\top [\exp(\mathbf{g}_j) \odot \mathbf{S}_0])`$，结合有效值和有效键的定义：

$$\mathbf{u}_r = \sum_j A_{r,j} \cdot \mathbf{v}_j, \quad \mathbf{w}_r = \sum_j A_{r,j} \cdot (\exp(\mathbf{g}_j) \odot \mathbf{k}_j)$$

可以得到：

$$\boldsymbol{\delta}_r = \mathbf{u}_r - \mathbf{w}_r^\top \mathbf{S}_0$$

代入第二步的结果，得到最终的 chunk-parallel 输出公式：

$$\mathbf{o}_r = \underbrace{(\mathbf{q}_r \odot \exp(\mathbf{g}_r))^\top \mathbf{S}_0}_{\text{inter-chunk}} + \underbrace{\sum_{j} \mathbf{A}_{qk}(r,j) \cdot (\mathbf{u}_j - \mathbf{w}_j \mathbf{S}_0)}_{\text{intra-chunk}}$$

这样的好处是：$`\mathbf{u}`$ 和 $`\mathbf{w}`$ 只依赖 $`\mathbf{k}, \mathbf{v}, \mathbf{g}, \boldsymbol{\beta}`$（不依赖 $`\mathbf{S}_0`$），可以在 Step 2-3 中并行预计算；只有最后合并输出时才需要 $`\mathbf{S}_0`$，而 $`\mathbf{S}_0`$ 通过块间串行递推获得。

基于上述变换，chunk-parallel 前向计算细分为以下 4 步：

#### Step 1. 块内 g 累积 (chunk_local_cumsum)

对每个 chunk 内的 $`\mathbf{g}`$ 做前缀累加和，使得后续任意两个位置 $`i, j`$ 间的衰减可通过 $`\exp(\mathbf{g}_j - \mathbf{g}_i)`$ 直接 $`O(1)`$ 查表：

$$\mathbf{g}_{r} \leftarrow \sum_{i=1}^{r} \mathbf{g}_i \quad (r = 1, \dots, C)$$

#### Step 2. 构建交互矩阵 A 并求解三角系统

这是 **KDA 区别于 GLA 的核心步骤**。GLA 没有 delta rule，不需要这一步。

**2.1 计算键-键对齐矩阵**：

$$a_{r,i} = \sum_{d=1}^{K} k_{r,d} \cdot \exp(g_{r,d} - g_{i,d}) \cdot k_{i,d}$$

**2.2 乘以 $`\beta`$、mask 上三角（含对角线）、取负**。得到严格下三角矩阵 $`\mathbf{A}_{\text{raw}}`$。

**2.3 前代法求逆**：对第 $`i`$ 行，把它对之前所有行的间接依赖累加进来。完成后下三角部分即为 $`(\mathbf{I} + \mathbf{A}_{\text{raw}})^{-1} - \mathbf{I}`$。

**2.4 加单位矩阵、右乘 $`\text{diag}(\boldsymbol{\beta})`$**。

#### Step 3. 计算有效键 w 和有效值 u

$$\mathbf{w} = \mathbf{A} \cdot (\exp(\mathbf{g}) \odot \mathbf{k}) \in \mathbb{R}^{C \times K}$$

$$\mathbf{u} = \mathbf{A} \cdot \mathbf{v} \in \mathbb{R}^{C \times V}$$

- $`\mathbf{w}_r`$：位置 $`r`$ 对外部状态 $`\mathbf{S}_0`$ 的有效查询权重
- $`\mathbf{u}_r`$：位置 $`r`$ 经三角系统修正后的有效值

#### Step 4. 块间递推 + 输出

对每个 chunk $`[t]`$，依次执行：

**4.1 构建 intra-chunk 注意力矩阵 $`\mathbf{A}_{qk}`$**：

$$A_{qk}(r, j) = (\mathbf{q}_r \odot \exp(\mathbf{g}_r - \mathbf{g}_j))^\top \mathbf{k}_j$$

下三角掩码（含对角线）。

**4.2 Delta Rule 修正**：

$$\mathbf{v}_{\text{corrected}} = \mathbf{u} - \mathbf{w} \, \mathbf{S}_0$$

**4.3 输出计算**：

$$\mathbf{o} = (\mathbf{q} \odot \exp(\mathbf{g}))^\top \mathbf{S}_0 + \mathbf{A}_{qk} \cdot \mathbf{v}_{\text{corrected}}$$

**4.4 状态更新**：

$$\mathbf{S}_0^{[t+1]} = \exp(\mathbf{g}_C) \odot \mathbf{S}_0^{[t]} + (\exp(\mathbf{g}_C - \mathbf{g}) \odot \mathbf{k})^\top \mathbf{v}_{\text{corrected}}$$

#### FLA 优化实现映射 (Optimized Implementation Mapping)

以上推导步骤在 FLA 库中被拆分为 **4 个 Triton Kernel** 的流水线（摘自 `fla/ops/kda/chunk.py`）：

```python
# ===== Step 0: chunk-local cumsum =====
g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)    # 对数空间前缀和，base-2

# ===== Step 1: 块内并行 — 构建 Akk, Aqk 并求解三角系统 =====
# chunk_kda_fwd_intra: 一个融合 Kernel 完成 Step 2-3 的所有工作
w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(q, k, v, gk=g, beta=beta, scale=scale)

# ===== Step 2: 块间递推 — 串行传播隐状态 =====
# chunk_gated_delta_rule_fwd_h: 以 chunk 为单位做 RNN 状态更新
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=kg, w=w, u=u, gk=g, initial_state=initial_state
)

# ===== Step 3: 融合输出 — 合并 inter-chunk 与 intra-chunk =====
# chunk_gla_fwd_o_gk: 计算最终输出 o
o = chunk_gla_fwd_o_gk(q, v=v_new, g=g, A=Aqk, h=h, scale=scale)
```

##### Kernel 1: `chunk_kda_fwd_intra` — 块内三角系统求解

对应推导中的 Step 2（构建 $`\mathbf{A}_{kk}`$ 并求逆）和 Step 3（计算 $`\mathbf{w}, \mathbf{u}`$）。

文件路径：`fla/ops/kda/chunk_intra.py` + `fla/ops/kda/wy_fast.py`

Grid: `(NT, B * H)`，每个 program 处理一个 chunk 的一个 batch-head。

核心策略：将 $`C=64`$ 的 chunk 分成 4 个 $`\text{BC}=16`$ 的子块，利用**分块下三角求逆**代替逐行前代法。

```python
# ===== 阶段 A: 构建 Akk 和 Aqk 的分块矩阵 =====
# 遍历 K 维度的分块，累积分块间的注意力分数
for i_k in range(cdiv(K, BK)):
    # 子块 1 vs 子块 0:
    b_gqn = exp2(g1 - gn1)                          # 门控归一化
    b_kgt = trans(k0 * exp2(gn1 - g0))              # 门控键（转置）
    Aqk10 += dot(q1 * b_gqn, b_kgt)                 # [BC, BC] 查询-键分数
    Akk10 += dot(k1 * b_gqn, b_kgt)                 # [BC, BC] 键-键分数
    # 子块 2 vs {0, 1}, 子块 3 vs {0, 1, 2}: 同理

# ===== 阶段 B: 对角块前代法求逆 =====
# 对每个 BC×BC 的对角块，逐行前代求逆
b_Akk_diag = -lower_triangular(Akkd)                 # 取负 + 严格下三角
for i in range(2, BC):
    b_Akk_diag[i, :] += sum(b_Akk_diag[i, :] * b_Akk_diag[:, :])   # 累积间接依赖
b_Akk_diag += I                                      # 加单位矩阵

# ===== 阶段 C: 分块合并为完整的逆矩阵 =====
# 利用分块下三角矩阵逆的递推公式
Ai10 = -Ai11 @ Akk10 @ Ai00
Ai20 = -Ai22 @ (Akk20 @ Ai00 + Akk21 @ Ai10)
Ai30 = -Ai33 @ (Akk30 @ Ai00 + Akk31 @ Ai10 + Akk32 @ Ai20)
# ... 其他分块类推
```

逆矩阵 $`\mathbf{A}_{kk}^{-1}`$ 存储后，由 `recompute_w_u_fwd` 计算有效键值：

```python
# fla/ops/kda/wy_fast.py — recompute_w_u_fwd Kernel
# Grid: (NT, B*H)
for i_v in range(cdiv(V, BV)):
    b_vb = v * beta[:, None]                         # [BT, BV]
    u = dot(Akk_inv, b_vb)                           # u = A @ (v * β)

for i_k in range(cdiv(K, BK)):
    b_kb = k * beta[:, None] * exp2(gk)              # [BT, BK]
    w = dot(Akk_inv, b_kb)                           # w = A @ (k * β * exp(g))
    kg = k * exp2(gn - gk)                           # kg = k * exp(g_C - g)
```

##### Kernel 2: `chunk_gated_delta_rule_fwd_h` — 块间状态递推

对应推导中的 Step 4.2（Delta Rule 修正）和 Step 4.4（状态更新）。

文件路径：`fla/ops/common/chunk_delta_h.py`

Grid: `(cdiv(V, BV), N * H)`，每个 program 负责一个 head 在完整 V 分片上的所有 chunk 递推。

```python
# 初始化状态
b_h = zeros([BK, BV])                               # 或从 h0 加载

# 串行遍历所有 chunk
for i_t in range(NT):
    # 1. 存储当前 chunk 的起始状态 h[i_t]
    store(h[i_t], b_h)

    # 2. Delta Rule 修正: v_new = u - w @ h
    b_v = zeros([BT, BV])
    for each K_strip:
        b_w = load(w[i_t, K_strip])                  # [BT, BK]
        b_v += dot(b_w, b_h_strip)                   # w @ h 的分片累积
    b_v = load(u[i_t]) - b_v                         # v_new = u - w @ h

    # 3. 门控衰减
    b_gk_last = gk[chunk 末尾位置]
    b_v *= exp2(b_gk_last - b_gk)                    # 对 v_new 应用相对衰减
    b_h *= exp2(b_gk_last)                           # 对状态应用 chunk 末尾衰减

    # 4. 状态更新: h += kg^T @ v_new
    for each K_strip:
        b_k = load(kg[K_strip])                      # [BK, BT] (转置)
        b_h_strip += dot(b_k, b_v)                   # 秩-C 更新
```

##### Kernel 3: `chunk_gla_fwd_o_gk` — 融合输出

对应推导中的 Step 4.1（构建 $`\mathbf{A}_{qk}`$）和 Step 4.3（输出计算）。$`\mathbf{A}_{qk}`$ 已在 Kernel 1 中预计算。

文件路径：`fla/ops/gla/chunk.py`（GLA/KDA 共用）

Grid: `(cdiv(V, BV), NT, B * H)`

```python
m_s = arange(BT)[:, None] >= arange(BT)[None, :]    # 下三角因果掩码

# Inter-chunk: o = scale * (q * exp(g)) @ h
b_o = zeros([BT, BV])
for i_k in range(cdiv(K, BK)):
    b_qg = q * exp2(g)                              # [BT, BK] 门控查询
    b_h = load(h[i_t, i_k*BK, i_v*BV])              # [BK, BV] 起始状态
    b_o += dot(b_qg, b_h)
b_o *= scale

# Intra-chunk: o += tril(Aqk) @ v_new
b_v = load(v_new[i_t])                              # [BT, BV]
b_A = load(Aqk[i_t])                                # [BT, BT]
b_A = where(m_s, b_A, 0)                            # 因果掩码
b_o += dot(b_A, b_v)
```

---

### 与 GLA chunk-fwd 的核心差异

| | GLA | KDA |
|---|---|---|
| 衰减门 | $`g \in \mathbb{R}`$ 标量或 $`\mathbb{R}^K`$ 向量 | $`\mathbf{g} \in \mathbb{R}^K`$ 通道级向量 |
| Delta Rule | 无 | 有（$`\beta_t`$ 学习率） |
| 块内依赖 | 无（$`\mathbf{k}_t \mathbf{v}_t^\top`$ 独立于 $`\mathbf{S}`$） | 有（$`\boldsymbol{\delta}_t`$ 依赖 $`\mathbf{S}_t'`$） |
| 块内并行方式 | 直接矩阵乘 | 需要先求解下三角系统 |
| 额外步骤 | 无 | Step 2（交互矩阵 A + 前代法） |
| 输出中的值 | 直接用 $`\mathbf{v}`$ | 修正后的 $`\mathbf{v}_{\text{corrected}} = \mathbf{u} - \mathbf{w} \mathbf{S}`$ |

核心差异：**GLA 的块内写入是独立的**，可以直接并行。**KDA 的块内写入是耦合的**（每步 $`\boldsymbol{\delta}_t`$ 依赖之前所有步的写入），必须先求解线性系统解耦后才能并行。

