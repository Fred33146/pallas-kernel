# fused_recurrent_gla_fwd design

参考文献

[https://arxiv.org/pdf/2312.06635](https://arxiv.org/pdf/2312.06635)

[https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26](https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26)

这个kernel 本质上没啥好说的, 采用单步循环

### 符号定义

*   $t$: 时间步，$t = 1, \dots, T$（若 `REVERSE=True`，则 $t = T, \dots, 1$）。

*   $\mathbf{q}_t, \mathbf{k}_t \in \mathbb{R}^K$: Query 和 Key 向量（列向量）。

*   $\mathbf{v}_t \in \mathbb{R}^V$: Value 向量（列向量）。

*   $\mathbf{H}_t \in \mathbb{R}^{K \times V}$: 时刻 $t$ 的隐藏状态矩阵。

*   $g_t \in \mathbb{R}$: 标量衰减门控（Scalar decay）。

*   $\gamma \in \mathbb{R}$: 全局衰减偏置（Global decay bias, 来自 `g_gamma`）。

*   $\mathbf{gk}_t \in \mathbb{R}^K$: Key 维度的向量衰减门控。

*   $\mathbf{gv}_t \in \mathbb{R}^V$: Value 维度的向量衰减门控。

*   $s$: 缩放因子 (`scale`)。


### 1. 衰减矩阵 (Decay Matrix)

在每个时间步，计算一个作用于隐藏状态 $\mathbf{H}_{t-1}$ 的逐元素衰减矩阵 $\mathbf{\Lambda}_t \in \mathbb{R}^{K \times V}$。根据启用的参数不同，其定义为：

$\mathbf{\Lambda}_{t} = \exp\Big(g_t + \gamma\Big) \cdot \left( \exp(\mathbf{gk}_t) \cdot \exp(\mathbf{gv}_t)^\top \right)$

_注：若对应的_ `USE_G`, `USE_GK` _等为 False，则对应项为 0（即乘数因子为 1）。_

### 2. 状态更新 (State Update)

隐藏状态首先经过衰减，加上当前时刻的 Key-Value 外积，得到当前时刻的状态 $\mathbf{H}_t$：

$\mathbf{H}_t = \mathbf{\Lambda}_t \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top$

_注：_$\odot$ _表示逐元素乘法（Hadamard product）。初始状态_ $\mathbf{H}_0$ _由_ `h0` _给定，默认为 0。_

### 3. 输出计算 (Output Computation)

输出 $\mathbf{o}_t \in \mathbb{R}^V$ 是查询向量 $\mathbf{q}_t$ 与更新后的当前状态 $\mathbf{H}_t$ 的乘积：

$\mathbf{o}_t = s \cdot \mathbf{q}_t^\top \mathbf{H}_t$

### 总结公式

将上述步骤合并，该 kernel 计算的是：

$\begin{aligned} \mathbf{H}_t &= \underbrace{\exp(g_t + \gamma + \mathbf{gk}_t \oplus \mathbf{gv}_t)}_{\text{Decay}} \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top \\ \mathbf{o}_t &= \text{scale} \cdot \mathbf{q}_t^\top \mathbf{H}_t \end{aligned}$

_(其中_ $\oplus$ _表示广播加法，构成_ $K \times V$ _的矩阵)_

伪代码:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A97eoWKOke/img/89677c13-533a-460b-9659-12e830fde7f6.png)

triton:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A97eoWKOke/img/7a2ab0f3-5076-4abb-a769-f33e86bd8a13.png)

数据流设计:

Q = [B, T, H, K]
K = [B, T, H, K]
V = [B, T, H, V]
GK = [B, T, H, K]
GV = [B, T, H, V]
H0 = [B, H, K, V]
HT = [B, H, K, V]

pallas 要求:
(block_dims[-1] % 128 == 0) or (block_dims[-1] == dims[-1])
(block_dims[-2] % 8 == 0) or (block_dims[-2] == dims[-2])

这种情况下, 我们优先做transpose, 并且把K/V align up到128的倍数, 然后block shape 切 [T, K/V]
```text
grid = [CeilDiv(K, BK), CeilDiv(V, BV), B * H]
BK = 128, BV = 128
K_align = align_up(K, BK)
V_align = align_up(V, BV)

Q = [B, T, H, K] -> [B, H, T, K] -> [B*H, T, K_align]
K = [B, T, H, K] -> [B, H, T, K] -> [B*H, T, K_align]
V = [B, T, H, V] -> [B, H, T, V] -> [B*H, T, V_align]
GK = [B, T, H, K] -> [B, H, T, K] -> [B*H, T, K_align]
GV = [B, T, H, V] -> [B, H, T, V] -> [B*H, T, V_align]
H0 = [B, H, K, V] -> [B*H, K, V_align]
HT = [B, H, K, V] -> [B*H, K, V_align]
O = [NK, B, H, T, V_align]

q_blockshape = [1, 1, T, BK]
k_blockshape = [1, 1, T, BK]
v_blockshape = [1, 1, T, BV]
h0_blockshape = [1, 1, 1, BK, BV]
ht_blockshape = [1, 1, 1, BK, BV]
o_blockshape = [1, 1, 1, BK, BV]
```
pallas demo:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65A97eoWKOke/img/c6dce695-6518-45f5-add2-07a170cab85c.png)

最终, 再补一个reduce sum, o[NK, B, H, T, V] -> [B, H, T V]

对于varlen的支持

对于varlen的支持, 需要作出如下改变.

1.  H transpose到最高维度, 例如 q = [B, T, H, K] -> [H, B, T, K], [B,T] reshape成 [total_t],

2.  grid 改成 [ceildiv(K, BK), ceildiv(V, BV), H]

3.  在内部每次更新idx_t的时候, check cu_seqlens, 是否在新的batch中, 如果是, 重新初始化h, 从 h0中加载.


note:

因为模型中, K往往是128的倍数, 所以对于 pallas_call的要求, 采用k alignup 128, 比把T transpose到最低纬度会好很多.
