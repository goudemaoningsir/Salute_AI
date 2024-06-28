import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time

# 下载并加载FashionMNIST数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root="/workspace/codes/datasets/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="/workspace/codes/datasets/FashionMNIST",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)


# 加载数据函数
def load_data_fashion_mnist(batch_size):
    if sys.platform.startswith("win"):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_iter, test_iter


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 确定设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                1, 6, kernel_size=5, padding=2
            ),  # 输入通道数1，输出通道数6，卷积核大小5，padding 2
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化，池化核大小2，步幅2
            nn.Conv2d(6, 16, kernel_size=5),  # 输入通道数6，输出通道数16，卷积核大小5
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化，池化核大小2，步幅2
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 输入大小16*5*5，输出大小120
            nn.ReLU(),
            nn.Linear(120, 84),  # 输入大小120，输出大小84
            nn.ReLU(),
            nn.Linear(84, 10),  # 输入大小84，输出大小10
        )

    def forward(self, img):
        feature = self.conv(img)  # 通过卷积层
        output = self.fc(feature.view(img.shape[0], -1))  # 展平并通过全连接层
        return output


net = LeNet()


# 评估模型在测试集上的准确性
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in data_iter:
            net.eval()  # 评估模式，关闭dropout
            acc_sum += (
                (net(X.to(device)).argmax(dim=1) == y.to(device))
                .float()
                .sum()
                .cpu()
                .item()
            )
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


# 训练函数
def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)  # 将模型移到设备上
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)  # 前向传播
            l = loss(y_hat, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
            % (
                epoch + 1,
                train_l_sum / batch_count,
                train_acc_sum / n,
                test_acc,
                time.time() - start,
            )
        )


# 设置学习率和训练轮数
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

# 测试模型性能并可视化一些结果
import matplotlib.pyplot as plt

# 获取测试数据集中的一个批次
test_iter = iter(test_iter)
X, y = next(test_iter)

# 将模型切换到评估模式
net.eval()

# 对测试批次进行预测
with torch.no_grad():
    y_hat = net(X.to(device)).argmax(dim=1)


# 可视化部分测试图像及其预测结果
def show_images(images, labels, preds, n=8):
    plt.figure(figsize=(12, 12))
    for i in range(n):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f"True: {labels[i]}, Pred: {preds[i]}")
        plt.axis("off")
    plt.show()


# 显示前8个测试图像及其预测结果
show_images(X.cpu().numpy(), y.cpu().numpy(), y_hat.cpu().numpy(), n=8)
