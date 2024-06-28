长期以来，隐变量模型存在着长期信息保存和短期输入缺失的问题。解决这一问题的最早方法之一是长短期存储器（Long Short-Term Memory，LSTM）。LSTM 有许多与门控循环单元（GRU）相似的属性。尽管 LSTM 的设计比 GRU 复杂一些，但它诞生的时间要早近 20 年。

## 一、门控记忆元

LSTM 的设计灵感来源于计算机的逻辑门。LSTM 引入了*记忆元*（memory cell），或简称为*单元*（cell）。有些文献认为记忆元是隐状态的一种特殊类型，它们与隐状态具有相同的形状，设计目的是用于记录附加的信息。为了控制记忆元，LSTM 需要许多门。一个门用来从单元中输出条目，称为*输出门*（output gate）。另一个门决定何时将数据读入单元，称为*输入门*（input gate）。还有一个门管理重置单元的内容，称为*遗忘门*（forget gate）。这种设计的动机与门控循环单元相同，通过专用机制决定什么时候记忆或忽略隐状态中的输入。

## 二、输入门、遗忘门和输出门

当前时间步的输入和前一个时间步的隐状态作为数据送入 LSTM 的门中。这些门由三个具有 sigmoid 激活函数的全连接层处理，以计算输入门、遗忘门和输出门的值。这三个门的值都在 (0, 1) 的范围内。

![../_images/lstm-0.svg](https://zh.d2l.ai/_images/lstm-0.svg)

我们来细化一下 LSTM 的数学表达。假设有 $h$ 个隐藏单元，批量大小为 $n$，输入数为 $d$。输入为 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$，前一时间步的隐状态为 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$。相应地，时间步 $t$ 的门定义如下：输入门是 $\mathbf{I}_t \in \mathbb{R}^{n \times h}$，遗忘门是 $\mathbf{F}_t \in \mathbb{R}^{n \times h}$，输出门是 $\mathbf{O}_t \in \mathbb{R}^{n \times h}$。它们的计算方法如下：

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

其中 $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ 和 $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ 是权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ 是偏置参数。

## 三、候选记忆元

LSTM 引入了*候选记忆元*（candidate memory cell）$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$。它的计算与上述三个门的计算类似，但使用 $\tanh$ 函数作为激活函数，函数值范围为 $(-1, 1)$。下面导出在时间步 $t$ 的方程：

$$
\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),
$$

其中 $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ 和 $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ 是权重参数，$\mathbf{b}_c \in \mathbb{R}^{1 \times h}$ 是偏置参数。

![../_images/lstm-1.svg](https://zh.d2l.ai/_images/lstm-1.svg)

## 四、记忆元

LSTM 中有两个门用于控制输入和遗忘（或跳过）：输入门 $\mathbf{I}_t$ 控制采用多少来自 $\tilde{\mathbf{C}}_t$ 的新数据，遗忘门 $\mathbf{F}_t$ 控制保留多少过去的记忆元 $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ 的内容。使用按元素乘法，得出：

$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.
$$

如果遗忘门始终为 1 且输入门始终为 0，则过去的记忆元 $\mathbf{C}_{t-1}$ 将随时间被保存并传递到当前时间步。引入这种设计是为了缓解梯度消失问题，并更好地捕获序列中的长距离依赖关系。

![../_images/lstm-2.svg](https://zh.d2l.ai/_images/lstm-2.svg)

## 五、隐状态

最后，我们需要定义如何计算隐状态 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$，这就是输出门发挥作用的地方。在 LSTM 中，隐状态是记忆元的 $\tanh$ 的门控版本。这样确保 $\mathbf{H}_t$ 的值始终在区间 $(-1, 1)$ 内：

$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).
$$

只要输出门接近 1，我们就能够有效地将所有记忆信息传递给预测部分；而对于输出门接近 0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

![../_images/lstm-3.svg](https://zh.d2l.ai/_images/lstm-3.svg)