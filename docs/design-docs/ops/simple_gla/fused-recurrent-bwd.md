# fused_recurrent_gla_bwd design

参考文献：

[《fused_recurrent_gla_fwd design》](https://alidocs.dingtalk.com/i/nodes/14dA3GK8gjYp2qjRh7AEd6oDJ9ekBD76)

[https://arxiv.org/pdf/2312.06635](https://arxiv.org/pdf/2312.06635)

[https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26](https://github.com/fla-org/flash-linear-attention/blob/f24317a6a4f513748cd7eb05818534ce66029957/fla/ops/common/fused_recurrent.py#L26)

这个kernel 同样不需要说太多

前向的公式总结为：

$\begin{aligned} \mathbf{H}_t &= \underbrace{\exp(g_t + \gamma + \mathbf{gk}_t \oplus \mathbf{gv}_t)}_{\text{Decay}} \odot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top \\ \mathbf{o}_t &= \text{scale} \cdot \mathbf{q}_t^\top \mathbf{H}_t \end{aligned}$

反向的公式为:

符号定义:

*   $\mathbf{G}_t = \exp(g_t + \gamma + \mathbf{gk}_t \oplus \mathbf{gv}_t)$

*   $\tilde{\mathbf{H}}_t = \mathbf{G}_t \odot \mathbf{H}_{t-1}$

*   $\mathbf{M}_t = \overline{\mathbf{H}}_t \odot \tilde{\mathbf{H}}_t$


_note_: 其中${\mathbf{H}}_t \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{H}_t}$为隐状态的伴随（adjoint），$\overline{\mathbf{o}}_t \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t}$

初始化 $(t = T)$:

$\overline{\mathbf{H}}_T = \frac{\partial \mathcal{L}}{\partial \mathbf{H}_T} \quad (\text{外部梯度，若无则为 } \mathbf{0})$

反向递推$（t=T,T−1,…,1t=T,T−1,…,1）$：

$\begin{aligned} \overline{\mathbf{H}}_t &\mathrel{+}= \text{scale} \cdot \mathbf{q}_t \, \overline{\mathbf{o}}_t^\top \\[6pt] \mathbf{dq}_t &= \text{scale} \cdot \mathbf{H}_t \, \overline{\mathbf{o}}_t \\[4pt] \mathbf{dk}_t &= \overline{\mathbf{H}}_t \, \mathbf{v}_t \\[4pt] \mathbf{dv}_t &= \overline{\mathbf{H}}_t^\top \, \mathbf{k}_t \\[6pt] dg_t &= \mathbf{1}^\top \mathbf{M}_t \, \mathbf{1} \\[4pt] d\mathbf{gk}_t &= \mathbf{M}_t \, \mathbf{1} \\[4pt] d\mathbf{gv}_t &= \mathbf{M}_t^\top \, \mathbf{1} \\[6pt] \overline{\mathbf{H}}_{t-1} &= \mathbf{G}_t \odot \overline{\mathbf{H}}_t \end{aligned}$

终止:

 $d\mathbf{H}_0 = \overline{\mathbf{H}}_0$

triton kernel 为了工程化，做了一些改动，计算顺序改为如下:

1.  重放 H并且计算$d\mathbf{q}$:

    ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/mxPOG5z4NZmxAnKa/img/8010dc28-5497-4052-8c86-d48783578ef3.png)

2.  反向循环，计算$d\mathbf{k}, d\mathbf{v}, d\mathbf{g}, d\mathbf{gk}, d\mathbf{gv}$

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/mxPOG5z4NZmxAnKa/img/0e871b98-6a4b-459e-a7b4-bb5f0a081566.png)

pallas的kernel 实现的时候，直接按照fwd 一样，grid 设置为[N*H] 应该就可以。

note:这里对于K/V不切，完整的传入，是因为默认假设T不会很大。

对于varlen的支持:

1.  H transpose到最高维度, 例如 q = [B, T, H, K] -> [H, B, T, K], [B,T] reshape成 [total_t],

2.  grid 改成 [H]

3.  在内部每次更新idx_t的时候, check cu_seqlens, 是否在新的batch中, 如果是, 重新初始化h, 从 h0中加载.
