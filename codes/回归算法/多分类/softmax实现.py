import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 加载Fashion MNIST数据集并返回数据加载器
def load_data_fashion_mnist(
    batch_size, num_workers=0, root="../../datasets/FashionMNIST"
):
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


# 将数字标签转换为对应的文本标签
def get_fashion_mnist_labels(labels):
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


# 显示一组Fashion MNIST图像及其对应标签
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy(), cmap="gray")
        f.set_title(lbl)
        f.axis("off")
    plt.show()


# 定义模型
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


# 训练模型并评估在测试集上的准确率
def train_and_evaluate(
    model, train_loader, test_loader, loss_fn, optimizer, num_epochs, device
):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        train_accuracy = correct / total

        test_accuracy = evaluate_accuracy(model, test_loader, device)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )


# 评估模型在给定数据加载器上的准确率
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


if __name__ == "__main__":
    # 参数设置
    num_inputs, num_outputs = 28 * 28, 10  # 输入和输出维度
    num_epochs, lr = 5, 0.1  # 训练的轮数和学习率
    batch_size = 256  # 批量大小
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 设备选择（GPU或CPU）

    # 加载数据
    train_loader, test_loader = load_data_fashion_mnist(batch_size, num_workers=4)

    # 初始化模型
    model = FashionMNISTModel()

    # 初始化模型参数
    nn.init.normal_(model.linear.weight, mean=0, std=0.01)
    nn.init.constant_(model.linear.bias, 0)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 训练模型并评估在测试集上的准确率
    train_and_evaluate(
        model, train_loader, test_loader, loss_fn, optimizer, num_epochs, device
    )

    # 在测试集上进行预测并显示部分结果
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)
    true_labels = get_fashion_mnist_labels(y.cpu().numpy())
    pred_labels = get_fashion_mnist_labels(model(X).argmax(dim=1).cpu().numpy())
    titles = [true + "\n" + pred for true, pred in zip(true_labels, pred_labels)]
    show_fashion_mnist(X[:9].cpu(), titles[:9])
