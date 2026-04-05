# chunk gla bwd kernel

# 公式

定义 chunk-local cumsum：gc[i] = g[0] + g[1] + ... + g[i]，令 g_n =  gc[C-1]（整个 chunk 的 gate 总和）。

这里的0到i是chunk内索引

逐 chunk 的 forward 公式：

$h_{n+1} = h_n \odot e^{g_n} + \sum_{j=0}^{C-1} \underbrace{k_j \odot e^{g_n - gc_j}}_{k\text{-decay}_j}    \otimes v_j$

$o_i = \underbrace{\text{scale} \cdot (q_i \odot e^{gc_i})^\top h_n}_{\text{inter-chunk}} +                  \underbrace{\sum{j \le i} A_{ij} v_j}_{\text{intra-chunk}}$

其中$A_{ij} = \text{scale} \cdot (q_i \odot e^{gc_i}) \cdot (k_j \odot e^{-gc_j})$

这里的h是每个chunk一个, n是chunk index, gc[i] 是chunk内的local cumsum, i是chunk内的索引

各变量的梯度推导

1.  **dh_n（隐状态梯度，用于跨 chunk 反传）**

$h_n$出现在两处：当前 chunk 的所有 $o_i$（inter-chunk 路径），以及下一个 chunk 的 $h_{n+1}$。

$\boxed{dh_n = e^{g_n} \odot dh_{n+1} + \text{scale} \cdot   \underbrace{Q_{gated}^\top \cdot DO}_{[C,K]^\top [C,V] \to [K,V]}}$

其中 $Q_{gated}[i] = q_i \odot e^{gc_i}$，沿 chunk 内时步求和。这就是 chunk_bwd_dh_ref 做的事——从 chunk NT-1 倒序传到 0。

2.  **dA（intra-chunk attention 矩阵的梯度）**

    由 $o_i = \sum_{j \le i} A_{ij} v_j$ 对 $A_{ij}$ 求导：

    $\boxed{dA_{ij} = \text{scale} \cdot do_i \cdot v_j \qquad (j \le i,\   \text{下三角})}$

    矩阵形式：$dA = \text{tril}(DO \cdot V^\top) \times \text{scale}$，shape $[C, C]$。

3.  **dv[j]（两条路径之和）**

    Intra-chunk：$v_j$ 被所有 $i \ge j$ 的 $o_i$ 使用，通过 $A_{ij}$：

    $dv_j^{\text{intra}} = \sum_{i \ge j} A_{ij} \cdot do_i \quad \Rightarrow   \quad dV^{\text{intra}} = A^\top \cdot DO \quad [C,C]^\top [C,V] \to [C,V]$

    Inter-chunk：$v_j$ 参与构建了 $h_{n+1}$，贡献 $k_{\text{decay},j} \otimes v_j$：

    $dv_j^{\text{inter}} = k_{\text{decay},j}^\top \cdot dh_{n+1} \quad   \Rightarrow \quad dV^{\text{inter}} = K_{\text{decay}} \cdot dh_{n+1} \quad   [C,K][K,V] \to [C,V]$

    $\boxed{dv_j = dv_j^{\text{intra}} + dv_j^{\text{inter}}}$

4.  **dq[i]（两条路径之和）**

    Intra-chunk：通过 $A_{ij} = \text{scale} \cdot (q_i \odot e^{gc_i}) \cdot (k_j \odot e^{-gc_j})$ 对 $q_i$ 求导：

    $dq_i^{\text{intra}} = e^{gc_i} \cdot \sum_{j \le i} dA_{ij} \cdot (k_j \odot    e^{-gc_j}) = e^{gc_i} \cdot (dA_i \cdot K_{\text{neg}})$

    Inter-chunk：通过 $o_i^{\text{inter}} = \text{scale} \cdot (q_i \odot e^{gc_i})^\top h_n$ 对 $q_i$ 求导：

    $dq_i^{\text{inter}} = \text{scale} \cdot e^{gc_i} \cdot (h_n \cdot do_i)$

    $\boxed{dq_i = e^{gc_i} \odot \left[ \text{scale} \cdot h_n \cdot do_i +   \sum_{j \le i} dA_{ij} \cdot k_j e^{-gc_j} \right]}$

5.  **dk[j]（两条路径之和）**

    Intra-chunk：通过 $A_{ij}$ 对 $k_j \odot e^{-gc_j}$ 求导：

    $dk_j^{\text{intra}} = e^{-gc_j} \cdot \sum_{i \ge j} dA_{ij} \cdot (q_i   \odot e^{gc_i}) = e^{-gc_j} \cdot (dA_{:,j}^\top \cdot Q_{\text{pos}})$

    Inter-chunk：通过 $k_{\text{decay},j} = k_j \odot e^{g_n - gc_j}$ 进入$h_{n+1}$：

    $dk_j^{\text{inter}} = e^{g_n - gc_j} \odot (dh_{n+1} \cdot v_j) = e^{g_n -   gc_j} \odot (V \cdot dh^\top)_j$

    $\boxed{dk_j = e^{-gc_j} \odot (dA^\top \cdot Q_{\text{pos}})_j + e^{g_n -   gc_j} \odot (V \cdot dh^\top)_j}$

6.  **dg**

g 影响 L 的所有路径

先明确：raw gate g[s]（chunk 内第 s 个位置）通过 cumsum 影响 gc[i]（对所有 i >= s），所以要先找 gc[i] 的所有使用路径，再用 cumsum 的链式法则还原 dg[s]。

gc[i] 出现的地方（chunk 内位置 i）

gc[i] 出现在三处：

1.  q_gated[i] = q[i] * exp(+gc[i]) → 正号

2.  k_neg[i] = k[i] * exp(-gc[i]) → 负号（用于 intra-chunk A）

3.  k_decay[i] = k[i] * exp(g_n - gc[i]) → gc[i] 带负号

    g_n 出现的地方（chunk 级别）

    g_n = gc[C-1] = g[0] + ... + g[C-1]，还额外出现在：

4.  h_n * exp(+g_n) → 正号，衰减上一个隐状态

5.  k_decay[j] = k[j] * exp(+g_n - gc[j]) → g_n 带正号（对所有 j）


路径 1、2、3 都通过 gc[i]，而且每个位置 i 是独立的。我们对每个位置 i 计算：

$\frac{\partial L}{\partial gc[i]} = \underbrace{q[i] \odot dq[i]}_{\text{路径1，正号}} - \underbrace{k[i] \odot dk[i]}_{\text{路径2+3，负号}} =   \texttt{dg\_raw}[i]$

然后因为 g[s] 贡献给所有 i >= s 的 gc[i]，用链式法则：

$\frac{\partial L}{\partial g[s]} = \sum_{i \ge s} \frac{\partial L}{\partial gc[i]} = \texttt{reverse\_cumsum}(\texttt{dg\_raw})[s]$

路径 4、5 都通过 g_n（正号），而 g_n 被 chunk 内所有 g[s] 贡献（dg_n/dg[s] = 1），所以这部分梯度对 chunk 内每个位置 s 都相同：

$dgk_{inter} = \underbrace{e^{g_n} \odot \sum_V (h_n \odot dh_{n+1})}_{\texttt{路径4：h\_n 衰减}} + \underbrace{\sum{j \in \text{chunk}} dk_j^{\text{inter}}   \odot k_j}_{\texttt{路径5：k\_decay 正号}}$

这是一个 chunk 级常量（shape [K]），加到每个位置上：

$dg[s] = \texttt{reverse\_cumsum}(\texttt{dg\_raw})[s] + dgk_{inter}$

---

# 实现

1.  chunk_bwd_dh_kernel 计算inter-chunk的状态传递

    grid = (H, K/BK, V/BV)

2.  chunk_gla_bwd_fused_kernel 计算 dA, dq, dk, dv,dg, 所有的inter信息由前面计算的dh提供

grid = (H, total_NT)
