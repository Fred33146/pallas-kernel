# chunk_gla_fwd design

参考文献

[https://arxiv.org/pdf/2312.06635](https://arxiv.org/pdf/2312.06635)

[https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26](https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26)

从文献中, 以及 fused_recurrent_gla_fwd 中, 可得知.

[《fused_recurrent_gla_fwd design》](https://alidocs.dingtalk.com/api/doc/transit?dentryUuid=14dA3GK8gjYp2qjRh7AEd6oDJ9ekBD76&queryString=utm_medium%3Ddingdoc_doc_plugin_card%26utm_source%3Ddingdoc_doc)

gla 这个整体的公式为

$\begin{aligned} \mathbf{H}_t &= \underbrace{\exp(g_t + \gamma + \mathbf{gk}_t \oplus \mathbf{gv}_t)}_{\text{Decay}} \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top \\[1ex] \mathbf{\Lambda}_{t} &= \exp(g_t + \gamma + \mathbf{gk}_t \oplus \mathbf{gv}_t)  \\[1ex] \mathbf{\Lambda}_{t} &= \exp(g_t + \gamma) \cdot \left( \exp(\mathbf{gk}_t) \exp(\mathbf{gv}_t)^\top \right) \\[1ex] \mathbf{H}_t &= \mathbf{\Lambda}_t \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top  \\[1ex] \mathbf{o}_t &= \text{scale} \cdot \mathbf{q}_t^\top \mathbf{H}_t \end{aligned}$

_注：_$\odot$ _表示逐元素乘法（Hadamard product）。_

但是这个公式有个问题, $\mathbf{H}_t$是递归计算得到的, 没有办法并行做, 所有出现了chunk-wise 并行, 这种思路有点像是对cumsum 做并行一样~~(回头有兴趣大致说下cumsum怎么做并行).~~

思路整体分为三部分

1.  计算chunk内部

2.  累加chunk之间的变化.

3.  更新并且输出o


核心部分就是把H的计算做并行化, 演变如下.

$\mathbf{H}_t = \mathbf{\Lambda} \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top$

然后

在一个大小为 $C$ 的 Chunk 内（例如从 $t=1$ 到 $t=C$），如果我们展开递归关系：


*   $t=1$: $\mathbf{H}\_1 = \mathbf{\Lambda}\_1 \mathbf{H}\_0 + \mathbf{k}\_1 \mathbf{v}\_1^\top$

*   $t=2$: $\mathbf{H}\_2 = \mathbf{\Lambda}\_2 \mathbf{H}\_1 + \mathbf{k}\_2 \mathbf{v}\_2^\top$

*              : $\mathbf{H}\_2 = \mathbf{\Lambda}\_2 (\mathbf{\Lambda}\_1 \mathbf{H}\_0 + \mathbf{k}\_1 \mathbf{v}\_1^\top) + \mathbf{k}\_2 \mathbf{v}\_2^\top$

*              : $\mathbf{H}\_2= (\mathbf{\Lambda}\_2 \mathbf{\Lambda}\_1) \mathbf{H}\_0 + \mathbf{\Lambda}\_2 \mathbf{k}\_1 \mathbf{v}\_1^\top + \mathbf{k}\_2 \mathbf{v}\_2^\top$

*   $t=3$: $\mathbf{H}\_3 = \mathbf{\Lambda}\_3 \mathbf{H}\_2 + \mathbf{k}\_3 \mathbf{v}\_3^\top$

*              : $\mathbf{H}\_3 = \mathbf{\Lambda}\_3 (\mathbf{\Lambda}\_2 \mathbf{H}\_1 + \mathbf{k}\_2 \mathbf{v}\_2^\top) + \mathbf{k}\_3 \mathbf{v}\_3^\top$

*              : $\mathbf{H}\_3 = (\mathbf{\Lambda}\_3 \mathbf{\Lambda}\_2 \mathbf{\Lambda}\_1) \mathbf{H}\_0 + (\mathbf{\Lambda}\_3 \mathbf{\Lambda}\_2) \mathbf{k}\_1 \mathbf{v}\_1^\top + \mathbf{\Lambda}\_3 \mathbf{k}\_2 \mathbf{v}\_2^\top + \mathbf{k}\_3 \mathbf{v}\_3^\top$

*   ...

*   $t=C$: $\mathbf{H}\_C = \left( \prod\_{i=1}^{C} {\Lambda}\_i \right) \mathbf{H}\_0 + \sum\_{j=1}^{C} \left( {\textstyle \prod\_{m=j+1}^{C}} {\Lambda}\_m \right) \mathbf{k}\_j \mathbf{v}\_j^\top$



_注：${\Lambda}$ 为[K, V] 矩阵,_ $\prod_{i=1}^{C}{\Lambda}_i$ _表示逐元素累乘_

_注：_在kernel中, 我们实际上会把${\Lambda}$变为对数空间内的计算

放到GLA kernel中, 按照上面三部曲的思路, 详细拆解成如下

1.  计算每个 chunk 内部从0 ~ chunk_size 内的g的累积.

    为了快速计算每个chunk 内部${\Lambda}$的衰减, 并且把后续kernel 计算衰减项的复杂度降为O(1) , 这里提前计算, 并且在对数空间内做cumsum. 最终, 我们得到了一个前缀和的表, 可快速计算出

    chunk内部, 任意 $\prod_{i=0}^{t} {\Lambda}_i$

    $t = chunksize \\ Decay(0→t)=exp(Gt −G0)$

2.  根据第一项的结果, 计算每个chunk内的增量$ΔH$, 并且以此计算所有chunk 内部的最终边界值$H_{start}$

    $\Delta \mathbf{H}_{\text{chunk}} = \sum_{t=1}^{C} (exp(Gt −G0) \cdot \mathbf{k}_t) \otimes \mathbf{v}_t$

    $H end =H start ⋅Decaytotal +ΔH chunk ​$

    其中 $\sum_{t=1}^{C}$主要是依靠矩阵运算

3.  根据第二项的结果, 计算每个chunk内的o,

    当我们拿到每个chunk 的 $H_{start}$的时候, 其实这里就已经解决了并行性的依赖了,  但是对于chunk mode来说, 公式还需要稍微变化下, 分成两部分, 并且在这部分解决chunk内的计算.

    $\mathbf{o}_t = \text{scale} \cdot \mathbf{q}_t^\top \mathbf{H}_t$

    $\mathbf{H}_t = \underbrace{\text{Decay}(t_{start} \to t) \odot \mathbf{H}_{start}}_{\text{继承的历史状态}} + \underbrace{\sum_{j=t_{start}}^{t} \text{Decay}(j \to t) \odot (\mathbf{k}_j \mathbf{v}_j^\top)}_{\text{当前块内新产生的状态}}$

    $\mathbf{o}_t = \underbrace{\text{scale} \cdot \left( \mathbf{q}_t^\top  (\mathbf{\Lambda}_{start \to t} \odot \mathbf{H}_{start}) \right)}_{\text{Part 1: Inter-chunk (历史贡献)}} + \underbrace{\sum_{j=t_{start}}^t \left( \text{scale} \cdot \mathbf{q}_t^\top (\mathbf{\Lambda}_{j \to t} \odot \mathbf{k}_j) \right) \mathbf{v}_j^\top}_{\text{Part 2: Intra-chunk (块内贡献)}}$

    在第三部分中, 我们计算每个chunk 内部的 $A = \underbrace{\text{scale} \cdot \mathbf{q}_t^\top (\mathbf{\Lambda}_{j \to t} \odot \mathbf{k}_j) }_{\text{Part 2: Intra-chunk (块内贡献)}}$

    其中$\sum_{j=t_{start}}^t A\mathbf{v}_j^\top$在第四部分使用矩阵进行计算

4.  最终更新输出o


当前kernel 负责 $\mathbf{o}_t = \underbrace{\text{scale} \cdot \left( \mathbf{q}_t^\top  (\mathbf{\Lambda}_{start \to t} \odot \mathbf{H}_{start}) \right)}_{\text{Part 1: Inter-chunk (历史贡献)}} + \sum_{j=t_{start}}^t A\mathbf{v}_j^\top$

下面我们大致列下 kernel内的参数以及初版切分策略

### 1.计算chunk 内部g的累积(triton kernel = chunk_local_cumsum)

g = [B, T, H, K] -> [B, H, T, K]

BK = 128

BT = chunk_size

NT = cdiv(T, BT)

grid = (cdiv(K, BK), NT, B * H)

g_blockshape = [BT, BK]

triton kernel:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Lk3lbmbGXR1QjOm9/img/d7e4ca1c-92db-4a50-8574-6c72ee82120f.png)

对于varlen的支持:

1.  限制每个batch的T 对齐到chunk_size 即可

2.  H transpose到最前面


### 2.计算每个chunk 内$ΔH$, 并且计算边界值（triton kernel = chunk_fwd_h）

K = [B, T, H, K] -> []

V = [B, T, H, V]

GK = [B, T, H, K]

BK = 128

BT = chunk_size

NT = cdiv(T, BT)

grid = (cdiv(K, BK), cdiv(V, BV), B * H)

k_blockshape = [BT, BK]

v_blockshape = [BT, BV]

gk_blockshape = [BT, BK]

triton_kernel:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Lk3lbmbGXR1QjOm9/img/9c0a18e7-6bdf-492d-a39f-c9cd265bb4bd.png)

对于varlen的支持:

1.  限制每个batch的T 对齐到chunk_size 即可

2.  H transpose到最前面

3.  每个chunk check是否是新的batch, 如果是新的batch, reset h

4.  grid改成(cdiv(K, BK), cdiv(V, BK), H)


### 3.根据第二项的结果, 计算每个chunk内的o (triton kernel = chunk_gla_fwd_intra_gk)

q = [B, T, H, K] -> [B, H, T, K]

k = [B, T, H, K] -> [B, H, T, K]

g = [B, T, H, K] -> [B, H, T, K]

chunk_size = 64

BT = chunk_size

BK = 128

NT = cdiv(T, BT)

按照公式来看 $A = \underbrace{\text{scale} \cdot \mathbf{q}_t^\top (\mathbf{\Lambda}_{j \to t} \odot \mathbf{k}_j) }_{\text{Part 2: Intra-chunk (块内贡献)}}$, 我们直接计算

q_blockshape = [BT, BK]

k_blockshape = [BT, BK]

g_blockshape = [BT, BK]

```python
qk = jnp.dot(q, k.T) * scale
decay = jnp.exp(g[..., None] - g[..., None, ...])
mask = jnp.tril(jnp.zeros(BT, BT))
A = qk * decay
A = A.masked_fill(~mask, 0)
```

note: varlen的支持

1.  要求对齐到chunk_size 即可


### 4.最终更新O (triton_kernel = chunk_gla_fwd_o_gk)

q = [B, T, H, K] -> [B, H, T, K]

v = [B, T, H, V] -> [B, H, T, V]

g = [B, T, H, K] -> [B, H, T, K]

A = [B, T, H, BT] -> [B, H, T, BT]

h = [B, NT, H, K, V] -> [B, NT, H, K, V]

BT = chunk_size

NT = cdiv(T, BT)

BV = 128

grid = (cdiv(V, BV), NT, B * H)

triton_kernel:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Lk3lbmbGXR1QjOm9/img/a52c7eda-25c3-494c-8c2e-1cd5703d9e1f.png)

对于varlen的支持:

1.  限制每个batch的T 对齐到chunk_size 即可

2.  H transpose到最前面

3.  每个chunk check是否是新的batch, 如果是新的batch, reset h

4.  kernel 内部改双重循环, 外面循环NT, 内部循环K

5.  grid改成(cdiv(V, BV), NT, H)
