DenseNet（Densely Connected Convolutional Networks）是一种深度神经网络结构，通过密集连接（Dense Connections）实现信息流动和梯度传播的有效性。DenseNet的主要特点是跨层连接，它的每一层接收所有前面层的输出作为其输入。这与ResNet中的残差连接（Residual Connections）有所不同，ResNet是将输入和输出相加，而DenseNet是在通道维上进行连结。

![img](https://raw.githubusercontent.com/goudemaoningsir/Salute_AI/main/img/63.svg)

图中将部分前后相邻的运算抽象为模块$A$和模块$B$。与ResNet的主要区别在于，DenseNet里模块$B$的输出不是像ResNet那样和模块$A$的输出相加，而是在通道维上连结。这样模块$A$的输出可以直接传入模块$B$后面的层。在这个设计里，模块$A$直接跟模块$B$后面的所有层连接在了一起。这也是它被称为“稠密连接”的原因。

DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。

## 一、主要模块

DenseNet的主要构建模块包括稠密块（Dense Block）和过渡层（Transition Layer）。

### 1、稠密块（Dense Block）

稠密块由多个卷积层组成，每个卷积层的输入是所有前面卷积层的输出的连结。在稠密块中，假设第 $l$ 层的输入为 $x_l$，输出为 $H_l(x)$，那么第 $l+1$ 层的输入为：

$$
x_{l+1} = [x_0, x_1, ..., x_l]
$$
其中 $[x_0, x_1, ..., x_l]$ 表示将所有前面层的输出在通道维上连结。

### 2、过渡层（Transition Layer）

过渡层的作用是控制模型的复杂度，它通过1×1卷积层减少通道数，并通过步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。

### 3、DenseNet 模型

DenseNet模型首先使用一个初始卷积层，然后依次堆叠多个稠密块和过渡层，最后使用全局平均池化层和全连接层来进行分类任务。DenseNet中的每个稠密块会增加通道数，所以在稠密块之间加入过渡层来减小通道数和特征图的大小。

## 二、主要构建步骤

1. **初始卷积层**：通常使用7×7卷积和3×3最大池化。
2. **稠密块**：每个稠密块由多个卷积层组成，每层的输入是前面所有层的输出的连结。
3. **过渡层**：在稠密块之间使用过渡层来控制通道数和特征图的大小。
4. **全局平均池化和全连接层**：最后通过全局平均池化层将特征图的高和宽缩减为1，然后使用全连接层进行分类。

## 三、示例代码

以下是使用PyTorch实现DenseNet的完整示例代码，并在Fashion-MNIST数据集上进行训练。

```python
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义稠密块
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    )
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上连结输入和输出
        return X


# 定义过渡层
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )
    return blk


# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(
        self, num_channels=64, growth_rate=32, num_convs_in_dense_blocks=[4, 4, 4, 4]
    ):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            self.net.add_module("DenseBlock_%d" % i, DB)
            num_channels = DB.out_channels
            if i != len(num_convs_in_dense_blocks) - 1:
                self.net.add_module(
                    "TransitionBlock_%d" % i,
                    transition_block(num_channels, num_channels // 2),
                )
                num_channels = num_channels // 2

        self.net.add_module("BN", nn.BatchNorm2d(num_channels))
        self.net.add_module("ReLU", nn.ReLU())
        self.net.add_module("GlobalAvgPool", nn.AdaptiveAvgPool2d((1, 1)))
        self.net.add_module("Flatten", nn.Flatten())
        self.net.add_module("FC", nn.Linear(num_channels, 10))

    def forward(self, X):
        return self.net(X)


# 下载并加载FashionMNIST数据集
def load_data_fashion_mnist(
    batch_size, resize=None, root="/workspace/codes/datasets/FashionMNIST"
):
    trans = [transforms.Resize(size=resize)] if resize else []
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    train_iter = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_iter, test_iter


# 训练模型
def train(net, train_loader, test_loader, num_epochs, lr, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, num_examples = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()
            num_examples += X.size(0)

        net.eval()
        test_acc, num_test_examples = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()
                num_test_examples += X.size(0)

        print(
            f"Epoch {epoch + 1}, Loss {train_loss / num_examples:.4f}, "
            f"Train Acc {train_acc / num_examples:.4f}, Test Acc {test_acc / num_test_examples:.4f}"
        )


# 设置参数并训练模型
batch_size = 32
train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=96)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DenseNet()
num_epochs, lr = 5, 0.001
train(net, train_loader, test_loader, num_epochs, lr, device)
```

- **稠密块和过渡层的定义**：我们定义了稠密块（DenseBlock）和过渡层（TransitionBlock），稠密块包含多个卷积层，每层的输入是所有前面层的输出的连结；过渡层通过1×1卷积层和平均池化层减小通道数和特征图的大小。
- **DenseNet模型**：我们使用多个稠密块和过渡层构建DenseNet模型，并在最后使用全局平均池化层和全连接层进行分类。
- **数据加载和预处理**：我们使用Fashion-MNIST数据集，并对图像进行缩放和标准化。
- **模型训练**：定义了训练函数，使用Adam优化器和交叉熵损失函数进行训练和评估。
