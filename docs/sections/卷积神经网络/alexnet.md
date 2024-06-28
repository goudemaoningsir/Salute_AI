## 一、AlexNet 介绍

AlexNet 是 2012 年由 Alex Krizhevsky 等人提出的一种深度卷积神经网络，成功在 ImageNet 图像分类挑战中取得了显著的成绩。

![../_images/alexnet.svg](../../../img/55.svg)

AlexNet 的结构相对于 LeNet 更加复杂，包括更多的卷积层、更大的模型和更强的计算能力支持。AlexNet 的主要特点和创新点包括：

- **使用 ReLU 激活函数**：相比传统的 sigmoid 或 tanh 激活函数，ReLU 加快了模型的训练速度。
- **使用 Dropout 正则化**：在全连接层中使用 Dropout 以防止过拟合。
- **数据增强**：通过数据增强技术（如随机裁剪、水平翻转）提高模型的泛化能力。
- **重叠池化**：使用步幅为 2 的 3x3 池化层代替 2x2 池化层以提高模型的表现。
