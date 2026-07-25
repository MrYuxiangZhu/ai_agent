# Transformer 整体架构详解

本文根据图中的 Transformer 编码器-解码器结构，按数据流详细说明每个环节的原理、公式推导和设计目的。经典 Transformer 来自论文 *Attention Is All You Need*，核心思想是用自注意力机制替代循环神经网络和卷积网络，使模型可以并行建模序列中任意位置之间的依赖关系。

## 1. 整体任务与数据流

Transformer 最初用于序列到序列任务，例如机器翻译：

- 输入序列：`x = (x_1, x_2, ..., x_n)`
- 输出序列：`y = (y_1, y_2, ..., y_m)`
- 目标：建模条件概率

$$
P(y \mid x) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, x)
$$

其中：

- 编码器负责把输入序列 `x` 编码成上下文表示。
- 解码器在已生成的目标 token `y_{<t}` 和编码器输出的条件下，预测下一个 token `y_t`。

整体流程可以概括为：

$$
\text{Input Tokens}
\rightarrow \text{Embedding}
\rightarrow \text{Positional Encoding}
\rightarrow \text{Encoder}
\rightarrow \text{Decoder}
\rightarrow \text{Linear}
\rightarrow \text{Softmax}
\rightarrow \text{Output Distribution}
$$

## 2. Token 与嵌入层

### 2.1 Tokenization

模型不能直接处理自然语言文本，需要先将文本切分为 token。例如：

```text
I love machine learning
```

可能被切分为：

```text
["I", "love", "machine", "learning"]
```

每个 token 会被映射成词表中的整数 ID：

$$
x_i \in \{1, 2, ..., V\}
$$

其中 `V` 是词表大小。

### 2.2 Input Embedding

嵌入层将离散 token ID 映射为连续向量：

$$
e_i = E[x_i]
$$

其中：

- `E \in \mathbb{R}^{V \times d_{model}}` 是可学习的嵌入矩阵。
- `d_{model}` 是模型隐藏维度。
- `e_i \in \mathbb{R}^{d_{model}}` 是第 `i` 个 token 的向量表示。

整个输入序列可表示为：

$$
X = [e_1, e_2, ..., e_n] \in \mathbb{R}^{n \times d_{model}}
$$

### 2.3 设计目的

嵌入层的目的有三个：

1. 将离散符号转成神经网络可处理的连续向量。
2. 让语义相近的 token 在向量空间中更接近。
3. 为后续注意力机制提供可学习的内容表示。

## 3. 位置编码

### 3.1 为什么需要位置编码

自注意力本身对顺序不敏感。对于一个序列，如果只看 token 向量集合：

$$
\{e_1, e_2, ..., e_n\}
$$

注意力机制无法天然知道哪个 token 在前、哪个 token 在后。

例如：

```text
狗 咬 人
人 咬 狗
```

二者 token 集合相同，但语义完全不同。因此 Transformer 必须额外注入位置信息。

### 3.2 正弦余弦位置编码

经典 Transformer 使用固定的正弦余弦位置编码：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- `pos` 表示 token 位置。
- `i` 表示向量维度索引。
- 偶数维使用正弦函数。
- 奇数维使用余弦函数。

最终输入编码为：

$$
Z_0 = X + PE
$$

### 3.3 公式直觉

不同维度使用不同频率的正弦波：

$$
\frac{1}{10000^{2i/d_{model}}}
$$

低维对应较高频率，适合表达局部位置差异；高维对应较低频率，适合表达长距离位置信息。

正弦余弦编码有一个重要性质：相对位置可以由线性变换表达。因为：

$$
\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)
$$

$$
\cos(pos + k) = \cos(pos)\cos(k) - \sin(pos)\sin(k)
$$

所以模型可以更容易学习“当前位置之后第 `k` 个 token”这类相对关系。

### 3.4 设计目的

位置编码的设计目的：

1. 弥补自注意力没有顺序感的问题。
2. 让模型同时感知绝对位置和相对位置。
3. 固定正弦编码可以外推到训练时未见过的更长序列。

现代模型也常使用可学习位置编码、RoPE 或 ALiBi，但图中展示的是经典 Transformer 的位置编码思想。

## 4. Scaled Dot-Product Attention

图中底部黄色模块是 Transformer 的核心：缩放点积注意力。

### 4.1 Query、Key、Value 的来源

给定输入矩阵：

$$
X \in \mathbb{R}^{n \times d_{model}}
$$

通过三个可学习线性变换得到：

$$
Q = XW^Q
$$

$$
K = XW^K
$$

$$
V = XW^V
$$

其中：

- `Q \in \mathbb{R}^{n \times d_k}`：Query，表示“我想找什么”。
- `K \in \mathbb{R}^{n \times d_k}`：Key，表示“我拥有什么特征”。
- `V \in \mathbb{R}^{n \times d_v}`：Value，表示“如果被关注，应输出什么信息”。
- `W^Q, W^K, W^V` 是可学习参数矩阵。

### 4.2 注意力分数

第 `i` 个 token 对第 `j` 个 token 的关注程度由点积计算：

$$
score_{ij} = q_i \cdot k_j
$$

矩阵形式为：

$$
S = QK^T
$$

其中：

$$
S \in \mathbb{R}^{n \times n}
$$

`S_{ij}` 表示第 `i` 个位置关注第 `j` 个位置的原始分数。

### 4.3 为什么除以 `\sqrt{d_k}`

如果 `q_i` 和 `k_j` 的各维度近似独立，均值为 0，方差为 1，则点积：

$$
q_i \cdot k_j = \sum_{\ell=1}^{d_k} q_{i\ell} k_{j\ell}
$$

它的方差近似为：

$$
Var(q_i \cdot k_j) = d_k
$$

当 `d_k` 很大时，点积分数会变得很大，进入 softmax 后容易饱和。

softmax 为：

$$
softmax(s_i) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

如果某个 `s_i` 远大于其他分数，softmax 会接近 one-hot，梯度变小，不利于训练。

因此使用缩放：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

使分数方差约为 1，训练更稳定。

### 4.4 Softmax 得到注意力权重

对每一行做 softmax：

$$
A = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

其中：

- `A \in \mathbb{R}^{n \times n}`
- `A_{ij}` 表示第 `i` 个 token 从第 `j` 个 token 聚合信息的权重。
- 每一行满足：

$$
\sum_{j=1}^{n} A_{ij} = 1
$$

### 4.5 加权求和得到输出

最终输出为：

$$
Attention(Q, K, V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

第 `i` 个位置的输出向量：

$$
o_i = \sum_{j=1}^{n} A_{ij}v_j
$$

也就是说，每个 token 的新表示是所有 token 的 value 向量的加权组合。

### 4.6 设计目的

缩放点积注意力的设计目的：

1. 让每个位置可以直接访问序列中任意位置的信息。
2. 根据内容相关性动态分配权重。
3. 避免 RNN 的长距离信息传递瓶颈。
4. 支持高度并行计算。
5. 通过缩放项稳定 softmax 和梯度。

## 5. Multi-Head Self-Attention

图中红色模块展示了多头自注意力。

### 5.1 为什么需要多头

单个注意力头只能在一个表示子空间中计算相关性。但语言中的依赖关系有多种类型，例如：

- 主谓关系
- 指代关系
- 修饰关系
- 局部短语关系
- 长距离语义关系

多头注意力允许模型在不同子空间中并行学习不同关系。

### 5.2 多头注意力公式

第 `h` 个注意力头：

$$
head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)
$$

将所有头拼接：

$$
H = Concat(head_1, head_2, ..., head_h)
$$

再经过输出线性变换：

$$
MultiHead(X) = H W^O
$$

完整写作：

$$
MultiHead(Q, K, V) =
Concat(head_1, ..., head_h)W^O
$$

$$
head_i =
Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

通常：

$$
d_k = d_v = \frac{d_{model}}{h}
$$

这样多头拼接后的维度仍然是 `d_{model}`。

### 5.3 Self-Attention 的含义

Self-Attention 指 `Q`、`K`、`V` 都来自同一个序列：

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$

编码器中的自注意力让输入序列中每个 token 都可以看见完整输入序列。

### 5.4 设计目的

多头自注意力的设计目的：

1. 在多个子空间同时建模关系。
2. 捕捉不同距离、不同语义类型的依赖。
3. 增强模型表达能力。
4. 保持整体维度不变，便于堆叠网络层。

## 6. Add & Norm

图中每个子层之后都有 Add & Norm。

### 6.1 残差连接

设子层函数为 `Sublayer(x)`，残差连接为：

$$
x + Sublayer(x)
$$

Transformer 中常见写法是：

$$
LayerNorm(x + Sublayer(x))
$$

这称为 Post-LN。现代大模型更常使用 Pre-LN：

$$
x + Sublayer(LayerNorm(x))
$$

图中的经典结构对应 Post-LN 思想。

### 6.2 Layer Normalization

对一个 token 的隐藏向量：

$$
x = (x_1, x_2, ..., x_d)
$$

计算均值：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

计算方差：

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

归一化：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

再加入可学习缩放和平移参数：

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

### 6.3 设计目的

残差连接的目的：

1. 缓解深层网络梯度消失。
2. 保留原始信息，避免子层破坏已有表示。
3. 让模型更容易学习增量修正。

LayerNorm 的目的：

1. 稳定每层激活分布。
2. 加速训练收敛。
3. 降低对初始化和学习率的敏感性。

## 7. Feed Forward Network

图中蓝色模块是前馈网络，位于每个 Encoder 层和 Decoder 层中。

### 7.1 公式

对每个位置独立应用同一个两层 MLP：

$$
FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

也可以写成：

$$
FFN(x) = W_2 \cdot ReLU(W_1x + b_1) + b_2
$$

经典 Transformer 中：

- 输入维度：`d_{model}`
- 中间维度：`d_{ff}`
- 输出维度：`d_{model}`

通常 `d_{ff}` 大于 `d_{model}`，例如：

$$
d_{model}=512,\quad d_{ff}=2048
$$

### 7.2 为什么注意力后还需要 FFN

注意力层主要做 token 之间的信息混合：

$$
\text{跨位置交互}
$$

而 FFN 主要做每个位置内部的非线性特征变换：

$$
\text{逐位置特征加工}
$$

注意力决定“从哪里取信息”，FFN 决定“如何加工取到的信息”。

### 7.3 设计目的

前馈网络的设计目的：

1. 引入非线性表达能力。
2. 对每个 token 的聚合信息做进一步变换。
3. 扩大隐藏维度后再压回原维度，提升特征组合能力。
4. 与注意力层分工明确：注意力负责信息路由，FFN 负责特征计算。

现代模型常用 GELU、SwiGLU 等激活函数替代 ReLU，但核心作用一致。

## 8. Encoder 编码器

图中左侧是编码器，由 `N` 个相同结构的 Encoder Layer 堆叠而成。

### 8.1 单层编码器结构

每个 Encoder Layer 包含：

1. Multi-Head Self-Attention
2. Add & Norm
3. Feed Forward
4. Add & Norm

设第 `l` 层输入为 `H^{l-1}`：

$$
\tilde{H}^{l} =
LayerNorm(H^{l-1} + MultiHeadSelfAttention(H^{l-1}))
$$

$$
H^{l} =
LayerNorm(\tilde{H}^{l} + FFN(\tilde{H}^{l}))
$$

经过 `N` 层后得到编码器输出：

$$
H^N = Encoder(X)
$$

### 8.2 编码器自注意力

编码器中：

$$
Q = H^{l-1}W^Q
$$

$$
K = H^{l-1}W^K
$$

$$
V = H^{l-1}W^V
$$

所有输入 token 可以互相注意，没有因果遮挡。

### 8.3 设计目的

编码器的目标是为每个输入 token 生成上下文化表示：

$$
h_i = f(x_i, x_1, x_2, ..., x_n)
$$

即第 `i` 个 token 的表示不仅包含它自己的语义，也融合了整句上下文。

例如在句子：

```text
The bank is near the river
```

`bank` 的表示会结合 `river` 判断其含义更接近“河岸”，而不是“银行”。

## 9. Decoder 解码器

图中右侧是解码器，也由 `N` 个相同结构的 Decoder Layer 堆叠而成。

### 9.1 单层解码器结构

每个 Decoder Layer 包含：

1. Masked Multi-Head Self-Attention
2. Add & Norm
3. Encoder-Decoder Attention
4. Add & Norm
5. Feed Forward
6. Add & Norm

### 9.2 输出嵌入与右移目标序列

训练时，解码器输入不是完整目标序列本身，而是右移后的目标序列：

```text
<sos>, y_1, y_2, ..., y_{m-1}
```

模型目标是预测：

```text
y_1, y_2, ..., y_m
```

也就是在位置 `t`，模型只能使用：

$$
y_{<t}
$$

来预测：

$$
y_t
$$

### 9.3 Masked Self-Attention

普通自注意力会让当前位置看到未来 token，这在生成任务中是不允许的。

例如预测 `y_3` 时，不能看到 `y_4`、`y_5`。

因此解码器自注意力加入因果 mask：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\\\
-\infty, & j > i
\end{cases}
$$

注意力变为：

$$
Attention(Q,K,V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

由于：

$$
e^{-\infty} = 0
$$

未来位置经过 softmax 后权重为 0。

### 9.4 设计目的

Masked Self-Attention 的目的：

1. 保证自回归生成的因果性。
2. 防止训练时信息泄露。
3. 允许训练阶段并行计算所有目标位置，而不需要像 RNN 一样逐步计算。

## 10. Encoder-Decoder Attention

图中解码器中间的模块是编码器-解码器注意力，也叫 Cross-Attention。

### 10.1 Q、K、V 的来源

在 Cross-Attention 中：

- `Q` 来自解码器当前隐状态。
- `K` 和 `V` 来自编码器输出。

设：

$$
S = \text{decoder hidden states}
$$

$$
H = \text{encoder outputs}
$$

则：

$$
Q = SW^Q
$$

$$
K = HW^K
$$

$$
V = HW^V
$$

注意力为：

$$
CrossAttention(S, H) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 10.2 直观理解

解码器每生成一个目标 token，都要问输入序列：

```text
为了生成当前词，我应该关注源句子的哪些位置？
```

例如翻译：

```text
I love you -> 我 爱 你
```

当解码器生成“你”时，Cross-Attention 可能重点关注输入中的 `you`。

### 10.3 设计目的

Encoder-Decoder Attention 的目的：

1. 把源序列信息注入目标序列生成过程。
2. 实现源 token 和目标 token 之间的软对齐。
3. 让每个目标位置动态选择相关输入位置。
4. 解决固定长度上下文向量的信息瓶颈。

## 11. Linear 与 Softmax 输出层

### 11.1 线性层

解码器最终输出：

$$
S \in \mathbb{R}^{m \times d_{model}}
$$

经过线性层映射到词表大小：

$$
Logits = SW_o + b_o
$$

其中：

- `W_o \in \mathbb{R}^{d_{model} \times V}`
- `b_o \in \mathbb{R}^{V}`
- `Logits \in \mathbb{R}^{m \times V}`

每个位置都会得到一个长度为 `V` 的向量，表示对词表中每个 token 的未归一化分数。

### 11.2 Softmax 概率分布

对 logits 做 softmax：

$$
P(y_t = v \mid y_{<t}, x)
=
\frac{\exp(z_{t,v})}{\sum_{u=1}^{V}\exp(z_{t,u})}
$$

其中：

- `z_{t,v}` 是位置 `t` 上词表 token `v` 的 logit。
- 输出是一个概率分布。

### 11.3 训练目标：交叉熵损失

若真实 token 是 `y_t`，则单位置损失：

$$
\mathcal{L}_t = -\log P(y_t \mid y_{<t}, x)
$$

整个序列损失：

$$
\mathcal{L}
=
-\sum_{t=1}^{m} \log P(y_t \mid y_{<t}, x)
$$

等价于最大化目标序列条件概率：

$$
\max \prod_{t=1}^{m} P(y_t \mid y_{<t}, x)
$$

取负对数后得到最小化交叉熵：

$$
\min -\sum_{t=1}^{m} \log P(y_t \mid y_{<t}, x)
$$

### 11.4 设计目的

输出层的设计目的：

1. 将隐藏状态投影到词表空间。
2. 将模型内部表示转成下一个 token 的概率。
3. 为训练提供可微分的最大似然目标。

## 12. 训练阶段与推理阶段

### 12.1 训练阶段

训练时使用 teacher forcing：

```text
Decoder input:  <sos>, y_1, y_2, ..., y_{m-1}
Target output:  y_1,   y_2, ..., y_m
```

虽然模型同时计算所有位置，但因果 mask 保证每个位置不能看到未来 token。

训练目标：

$$
\mathcal{L}
=
-\sum_{t=1}^{m}\log P(y_t \mid y_{<t}, x)
$$

### 12.2 推理阶段

推理时没有真实目标序列，需要自回归生成：

1. 输入 `<sos>`。
2. 模型预测第一个 token。
3. 将预测 token 拼回解码器输入。
4. 重复直到生成 `<eos>` 或达到最大长度。

形式化表示：

$$
\hat{y}_t = \arg\max_v P(y_t=v \mid \hat{y}_{<t}, x)
$$

也可以使用 beam search、top-k sampling、top-p sampling 等策略。

### 12.3 设计目的

训练阶段追求高效并行和稳定优化；推理阶段遵守自回归条件概率分解，逐步生成符合上下文的输出。

## 13. 为什么 Transformer 能替代 RNN

### 13.1 RNN 的问题

RNN 逐步处理序列：

$$
h_t = f(h_{t-1}, x_t)
$$

长距离信息必须经过很多步传递，容易出现：

1. 梯度消失或爆炸。
2. 难以并行。
3. 长距离依赖建模困难。

### 13.2 Transformer 的优势

Self-Attention 中任意两个位置只需要一次注意力计算即可交互：

$$
score_{ij} = q_i \cdot k_j
$$

路径长度从 RNN 的 `O(n)` 降为 `O(1)`。

并且所有位置的注意力矩阵可以并行计算：

$$
QK^T
$$

这非常适合 GPU/TPU 上的大规模矩阵运算。

## 14. 复杂度分析

设序列长度为 `n`，隐藏维度为 `d`。

Self-Attention 主要计算：

$$
QK^T \in \mathbb{R}^{n \times n}
$$

复杂度为：

$$
O(n^2d)
$$

空间复杂度主要来自注意力矩阵：

$$
O(n^2)
$$

这也是 Transformer 在长序列上成本高的主要原因。

FFN 的复杂度约为：

$$
O(ndd_{ff})
$$

通常在大模型中，FFN 参数量和计算量也非常大。

## 15. 各模块的核心分工

| 模块 | 作用 | 关键公式 | 设计目的 |
| --- | --- | --- | --- |
| Embedding | token 转向量 | `e_i = E[x_i]` | 提供连续语义表示 |
| Positional Encoding | 注入位置信息 | `X + PE` | 弥补注意力无顺序感 |
| Q/K/V 投影 | 构造查询、键、值 | `Q=XW^Q` | 为注意力匹配和信息聚合做准备 |
| Scaled Dot-Product Attention | 计算相关性并聚合 | `softmax(QK^T/sqrt(d_k))V` | 动态选择上下文信息 |
| Multi-Head Attention | 多子空间并行注意力 | `Concat(head_i)W^O` | 捕捉多类型关系 |
| Add & Norm | 残差与归一化 | `LayerNorm(x+Sublayer(x))` | 稳定深层训练 |
| FFN | 逐位置非线性变换 | `W_2 ReLU(W_1x+b_1)+b_2` | 增强特征表达 |
| Encoder | 编码输入上下文 | `H^N=Encoder(X)` | 得到源序列表示 |
| Masked Self-Attention | 遮挡未来信息 | `softmax((QK^T/sqrt(d_k))+M)V` | 保证生成因果性 |
| Cross-Attention | 连接编码器和解码器 | `Q` 来自解码器，`K,V` 来自编码器 | 对齐源序列和目标序列 |
| Linear + Softmax | 输出词表概率 | `softmax(SW_o+b_o)` | 预测下一个 token |

## 16. 一句话理解每个环节

Transformer 的每个环节可以这样理解：

1. Embedding：把词变成向量。
2. Positional Encoding：告诉模型词的位置。
3. Self-Attention：让每个词查看整句话中相关的词。
4. Multi-Head：从多个角度查看相关性。
5. Add & Norm：保证信息和梯度稳定流动。
6. FFN：对每个词的表示做更复杂的非线性加工。
7. Encoder：把输入句子变成上下文表示。
8. Masked Decoder Self-Attention：生成时只能看已经生成的词。
9. Cross-Attention：生成目标词时参考输入句子。
10. Linear + Softmax：把隐藏表示转成词表概率。

## 17. 总结

Transformer 的核心创新是用注意力机制直接建模任意 token 之间的依赖关系。它通过位置编码补足顺序信息，通过多头机制增强关系建模能力，通过残差连接和归一化稳定深层训练，通过前馈网络提升非线性表达，最终形成一个既能并行训练、又能强力捕捉长距离依赖的序列建模架构。

在编码器-解码器任务中，编码器负责理解输入，解码器负责在输入条件下自回归生成输出。图中的每个模块都服务于这个目标：让模型在每一步生成时都能知道“已经生成了什么”“输入中哪些部分最相关”“下一个 token 最可能是什么”。
