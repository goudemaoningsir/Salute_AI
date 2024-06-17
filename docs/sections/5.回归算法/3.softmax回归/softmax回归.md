## 一、OVO 和 OVR

在多分类问题中，最常用的两种策略是 "一对一" (One-vs-One, OVO) 和 "一对多" (One-vs-Rest, OVR) 方法。这两种方法都是将多分类问题分解成多个二分类问题，进而应用二分类算法来解决。

### 1、一对多 (One-vs-Rest, OVR)

一对多方法将每个类别与所有其他类别进行对比，并为每个类别训练一个二分类器。

#### （1）步骤

**Step1：类别分解**：假设有$ K $ 个类别，则会训练$ K $ 个二分类器。每个分类器$ i $ 用于区分类别$ i $ 和所有其他类别。

**Step2：训练**：对于每个分类器$ i $，所有属于类别$ i $ 的样本被标记为正类 (1)，其他所有样本被标记为负类 (0)。然后，使用二分类算法（如逻辑回归、支持向量机等）训练模型。

**Step3：预测**：在预测阶段，给定一个新样本，所有$ K $ 个分类器都会对其进行预测，产生$ K $ 个得分或概率。最终，选择得分或概率最高的类别作为预测结果。

#### （2）优点

- 实现简单。
- 可扩展性好，适用于大型数据集。

#### （3）缺点

- 训练多个二分类器可能导致一些类别的样本不平衡问题。
- 每个分类器只关注一个类别，可能导致模型不够全局优化。

#### （4）实现

[OVR](code/ovr.py ':include :type=code ')

### 2、一对一 (One-vs-One, OVO)

一对一方法将每一对类别进行对比，并为每一对类别训练一个二分类器。

#### （1）步骤

**Step1：类别分解**：假设有$ K $ 个类别，则需要训练$ \frac{K(K-1)}{2} $ 个二分类器。每个分类器$ (i, j) $ 用于区分类别$ i $ 和类别$ j $。

**Step2：训练**：对于每个分类器$ (i, j) $，只使用类别$ i $ 和类别$ j $ 的样本进行训练，类别$ i $ 的样本标记为正类 (1)，类别$ j $ 的样本标记为负类 (0)。然后，使用二分类算法训练模型。

**Step3：预测**：在预测阶段，给定一个新样本，所有$ \frac{K(K-1)}{2} $ 个分类器都会对其进行预测。每个分类器会投票给一个类别。最终，选择得票数最高的类别作为预测结果。

#### （2）优点

- 每个分类器只处理两个类别的样本，通常效果较好。
- 适用于类别较多但样本量较少的情况。

#### （3）缺点

- 分类器数量随类别数量的增加呈平方级增长，训练和预测的计算复杂度较高。
- 在类别数量很多的情况下，训练和预测时间会显著增加。

#### （4）实现

[OVO](code/ovo.py ':include :type=code ')

### 3、OVO 和 OVR 的比较

- **计算复杂度**：
  - OVR：需要训练$ K $ 个分类器，训练和预测时间较少。
  - OVO：需要训练$ \frac{K(K-1)}{2} $ 个分类器，训练和预测时间较多。

- **处理样本不平衡**：
  - OVR：可能导致某些分类器的正负样本严重不平衡。
  - OVO：每个分类器只处理两个类别，样本不平衡问题较少。

- **模型性能**：
  - OVR：在某些情况下可能表现不佳，特别是当类别之间的边界不明显时。
  - OVO：通常能获得更高的分类精度，因为每个分类器只需区分两个类别，决策边界更明确。

### 4、实际应用中的选择

- 当类别数量较少（如 3-10 个）时，OVO 方法通常表现较好，因为其分类器较少且效果更好。
- 当类别数量较多时（如 10 个以上），OVR 方法更具优势，因为其计算复杂度较低，更易于扩展和实现。

## 二、Softmax

### 1、简介

Softmax函数是一种广泛应用于多类分类任务的激活函数，特别是在神经网络的输出层。给定一个输入向量 $\mathbf{x} \in \mathbb{R}^n$，Softmax回归模型的输出是一个类别概率分布。假设有 $K$ 个类别，模型的输出是一个 $K$ 维向量，其中每个元素表示输入属于该类别的概率。

模型的假设函数为：

$$
\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b} 
$$

其中：

- $\mathbf{W} \in \mathbb{R}^{K \times n}$ 是权重矩阵
- $\mathbf{b} \in \mathbb{R}^K$ 是偏置向量

Softmax函数将 $\mathbf{z}$ 转换为概率分布：

$$
\sigma(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$

其中 $ \sigma(\mathbf{z})_j $ 表示输入属于第 $j$ 类的概率。

### 2、损失函数

在Softmax回归中，损失函数用于度量模型预测与真实标签之间的差异。为了优化模型参数，使预测结果尽可能接近真实值，我们使用最大似然估计，最小化负对数似然。

#### （1）对数似然

假设我们有一个输入数据集 $\mathbf{X} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(n)}\}$ 和对应的标签集 $\mathbf{Y} = \{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}$，其中 $y^{(i)}$ 是 $\mathbf{x}^{(i)}$ 的标签。

Softmax回归模型输出类别 $k$ 的概率为：

$$
P(y = k | \mathbf{x}; \mathbf{W}, \mathbf{b}) = \sigma(\mathbf{z})_k = \frac{e^{\mathbf{w}_k^\top \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top \mathbf{x} + b_j}}
$$

为了度量预测效果，我们使用最大似然估计，目标是最大化样本的对数似然函数：

$$
\log P(\mathbf{Y}|\mathbf{X}; \mathbf{W}, \mathbf{b}) = \sum_{i=1}^n \log P(y^{(i)}|\mathbf{x}^{(i)}; \mathbf{W}, \mathbf{b})
$$

负对数似然函数（或称损失函数）为：

$$
-\log P(\mathbf{Y}|\mathbf{X}; \mathbf{W}, \mathbf{b}) = -\sum_{i=1}^n \log P(y^{(i)}|\mathbf{x}^{(i)}; \mathbf{W}, \mathbf{b})
$$

#### （2）交叉熵损失

负对数似然函数也被称为交叉熵损失。假设标签 $y^{(i)}$ 是独热编码（one-hot encoding），即如果样本属于类别 $k$，则 $y^{(i)}_k = 1$，否则 $y^{(i)}_k = 0$。交叉熵损失函数为：

$$
\mathcal{L} = -\sum_{i=1}^n \sum_{k=1}^K y^{(i)}_k \log \hat{y}^{(i)}_k
$$

其中 $\hat{y}^{(i)}_k = P(y = k | \mathbf{x}^{(i)}; \mathbf{W}, \mathbf{b})$ 是模型预测的第 $i$ 个样本属于类别 $k$ 的概率。



#### （3）数值稳定的softmax

##### 1）背景

Softmax函数用于将神经网络的输出转换为概率分布，其定义为：
$$
\hat{y}_j = \frac{\exp(o_j)}{\sum_{k} \exp(o_k)}
$$
其中，$\hat{y}_j$是第$j$个类别的预测概率，$o_j$是未规范化的预测得分。

##### 2）数值稳定性问题

直接计算softmax可能会导致数值稳定性问题，特别是在计算大指数值时。例如，如果某些$o_k$非常大，$\exp(o_k)$可能会超过计算机可以表示的最大值，导致上溢（overflow）。这会使分子或分母变成`inf`（无穷大），结果变成无意义的数值。

为了解决这个问题，可以在计算softmax之前，减去一个常数，这个常数通常选择为所有$o_k$中的最大值$\max(o_k)$。具体计算如下：

假设我们有一组未规范化的预测得分$\mathbf{o}$。我们定义新的变量：
$$
o'_j = o_j - \max(o_k)
$$
然后计算softmax：
$$
\hat{y}_j = \frac{\exp(o'_j)}{\sum_{k} \exp(o'_k)}
$$
因为减去一个常数$\max(o_k)$不会改变softmax的相对值，我们引入这一减法项并同时乘回$\exp(\max(o_k))$，所以新的softmax值$\hat{y}_j$和原来的计算结果是相同的。下面是详细推导：

$$
\begin{aligned}
\hat{y}_j & = \frac{\exp(o_j)}{\sum_{k} \exp(o_k)} \\
& = \frac{\exp(o_j - \max(o_k)) \exp(\max(o_k))}{\sum_{k} \exp(o_k - \max(o_k)) \exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k)) \exp(\max(o_k))}{\exp(\max(o_k)) \sum_{k} \exp(o_k - \max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_{k} \exp(o_k - \max(o_k))}
\end{aligned}
$$
通过这种方式，计算中的指数部分被减小，避免了可能的上溢问题。

##### 3）结合交叉熵损失的计算

在计算交叉熵损失时，我们可以进一步优化计算来避免下溢（underflow）问题。下溢问题发生在非常小的概率值接近零时。计算交叉熵损失时，我们实际上需要计算log(softmax)。通过将softmax和交叉熵结合，我们可以避免直接计算非常小的数值。

首先，交叉熵损失的定义是：
$$
\text{CrossEntropy} = -\sum_{j} y_j \log(\hat{y}_j)
$$
其中，$y_j$是真实标签的独热编码，$\hat{y}_j$是预测的概率分布。

通过前面的推导，softmax的对数可以写成：
$$
\begin{aligned}
\log(\hat{y}_j) & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_{k} \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))} - \log{\left( \sum_{k} \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) - \log{\left( \sum_{k} \exp(o_k - \max(o_k)) \right)}
\end{aligned}
$$
这样，我们可以直接计算softmax的对数值而不需要计算非常小的概率值，从而避免了下溢问题。

##### 4）实际应用中的实现

在实际应用中，为了确保数值稳定性，我们将未规范化的预测得分传递给交叉熵损失函数，同时计算softmax及其对数。这种方法避免了直接计算非常大的或非常小的数值，确保了数值稳定性。

例如，使用PyTorch的实现如下：

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

通过这种实现方式，我们既保留了传统的softmax函数用于概率评估，又通过数值稳定的计算方法确保了模型训练的稳定性。

### 3、梯度计算

为了优化模型，我们需要计算损失函数对模型参数（权重和偏置）的梯度。具体来说：

#### （1）权重梯度的计算

为了计算交叉熵损失对权重矩阵 $\mathbf{W}$ 的梯度，我们首先计算损失对每个输入样本的梯度。令 $\mathbf{W}_k$ 表示权重矩阵 $\mathbf{W}$ 的第 $k$ 行向量，即与类别 $k$ 对应的权重向量。

对权重 $\mathbf{W}_k$ 的梯度计算如下：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_k} = \sum_{i=1}^n \left( \hat{y}^{(i)}_k - y^{(i)}_k \right) \mathbf{x}^{(i)}
$$

其中 $\hat{y}^{(i)}_k$ 是第 $i$ 个样本被预测为类别 $k$ 的概率，$y^{(i)}_k$ 是第 $i$ 个样本的真实标签（独热编码）。

#### （2）偏置梯度的计算

对偏置 $\mathbf{b}_k$ 的梯度计算如下：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_k} = \sum_{i=1}^n \left( \hat{y}^{(i)}_k - y^{(i)}_k \right)
$$

#### （3）公式推导

我们从交叉熵损失函数出发：

$$
\mathcal{L} = -\sum_{i=1}^n \sum_{k=1}^K y^{(i)}_k \log \hat{y}^{(i)}_k
$$

将 $\hat{y}^{(i)}_k$ 代入：

$$
\hat{y}^{(i)}_k = \frac{e^{\mathbf{w}_k^\top \mathbf{x}^{(i)} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top \mathbf{x}^{(i)} + b_j}}
$$

交叉熵损失函数变为：

$$
\mathcal{L} = -\sum_{i=1}^n \sum_{k=1}^K y^{(i)}_k \log \left( \frac{e^{\mathbf{w}_k^\top \mathbf{x}^{(i)} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top \mathbf{x}^{(i)} + b_j}} \right)
$$

### 4、梯度推导

为了计算梯度，我们首先需要计算对 $\mathbf{z}_k^{(i)}$ 的导数，其中 $\mathbf{z}_k^{(i)} = \mathbf{w}_k^\top \mathbf{x}^{(i)} + b_k$。

对 $\mathbf{z}_k^{(i)}$ 的偏导数为：

$$
\frac{\partial \hat{y}^{(i)}_j}{\partial \mathbf{z}_k^{(i)}} =
\begin{cases} 
\hat{y}^{(i)}_j (1 - \hat{y}^{(i)}_j), & \text{如果 } j = k \\
-\hat{y}^{(i)}_j \hat{y}^{(i)}_k, & \text{如果 } j \neq k 
\end{cases}
$$

根据链式法则，我们可以计算交叉熵损失对 $\mathbf{z}_k^{(i)}$ 的偏导数：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_k^{(i)}} = \hat{y}^{(i)}_k - y^{(i)}_k
$$

接下来，我们利用以上结果计算对权重 $\mathbf{W}$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_k} = \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial \mathbf{z}_k^{(i)}} \frac{\partial \mathbf{z}_k^{(i)}}{\partial \mathbf{W}_k} = \sum_{i=1}^n (\hat{y}^{(i)}_k - y^{(i)}_k) \mathbf{x}^{(i)}
$$

同理，对偏置 $\mathbf{b}$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_k} = \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial \mathbf{z}_k^{(i)}} = \sum_{i=1}^n (\hat{y}^{(i)}_k - y^{(i)}_k)
$$

### 5、参数更新

使用梯度下降法来更新模型参数：

- 对于权重 $\mathbf{W}$：

$$
\mathbf{W} \leftarrow \mathbf{W} - \

alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}}
$$

- 对于偏置 $\mathbf{b}$：

$$
\mathbf{b} \leftarrow \mathbf{b} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{b}}
$$

其中 $\alpha$ 是学习率。

### 6、实现

下面是使用PyTorch在Fashion MNIST数据集上实现softmax分类的代码示例。我们会使用内置的`torchvision`模块来加载数据，并实现一个简单的神经网络模型来进行分类。

[softmax实现](code/softmax实现.py ':include :type=code ')
