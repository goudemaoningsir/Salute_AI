import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# 数据集相关 --------------------------------------------------------------------------------------------------
def load_data_fashion_mnist(batch_size, num_workers=0, root="./Datasets/FashionMNIST"):
    """
    加载Fashion MNIST数据集并返回训练和测试数据加载器
    """
    # 定义数据转换
    transform = transforms.ToTensor()
    # 下载并加载训练集
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    # 下载并加载测试集
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def get_fashion_mnist_labels(labels):
    """
    将数字标签转换为对应的文本标签
    """
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    """
    显示一组Fashion MNIST图像及其对应标签
    """
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy(), cmap="gray")
        f.set_title(lbl)
        f.axis("off")
    plt.show()


# 模型定义 --------------------------------------------------------------------------------------------------------
class FlattenLayer(nn.Module):
    """
    定义一个展平层，用于将输入张量展平成二维张量
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


def evaluate_accuracy(data_iter, net, device):
    """
    评估模型在给定数据迭代器上的准确率
    """
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train(net, train_loader, test_loader, loss, num_epochs, optimizer, device):
    """
    训练模型并评估在测试集上的准确率
    """
    net.to(device)  # 将模型迁移到设备（CPU或GPU）
    for epoch in range(num_epochs):
        net.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = net(X)  # 前向传播
            l = loss(outputs, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += l.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        test_acc = evaluate_accuracy(test_loader, net, device)  # 评估测试集上的准确率
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / total:.4f}, Train Accuracy: {correct / total:.3f}, Test Accuracy: {test_acc:.3f}"
        )


# 主程序 ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    num_inputs, num_outputs = 28 * 28, 10  # 输入和输出维度
    num_epochs, lr = 5, 0.1  # 训练的轮数和学习率
    batch_size = 256  # 批量大小
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 设备选择（GPU或CPU）

    # 加载数据
    train_loader, test_loader = load_data_fashion_mnist(batch_size, num_workers=4)

    # 定义模型
    net = nn.Sequential(
        OrderedDict(
            [
                ("flatten", FlattenLayer()),  # 展平成二维张量
                ("linear", nn.Linear(num_inputs, num_outputs)),  # 线性层
            ]
        )
    )

    # 初始化模型参数
    nn.init.normal_(net.linear.weight, mean=0, std=0.01)
    nn.init.constant_(net.linear.bias, 0)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练模型
    train(net, train_loader, test_loader, loss, num_epochs, optimizer, device)

    # 在测试集上进行预测并显示部分结果
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)
    true_labels = get_fashion_mnist_labels(y.cpu().numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).cpu().numpy())
    titles = [true + "\n" + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(X[:9].cpu(), titles[:9])
