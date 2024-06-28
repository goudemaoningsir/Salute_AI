import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 确定设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# 下载并加载FashionMNIST数据集
def load_data_fashion_mnist(
    batch_size,
    resize=None,
    root="/workspace/codes/datasets/FashionMNIST",
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


# 评估准确率
def evaluate_accuracy(data_iter, net, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total


# 训练模型
def train(net, train_loader, test_loader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        running_loss, running_corrects, num_samples = 0.0, 0, 0
        start_time = time.time()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += (predicted == y).sum().item()
            num_samples += y.size(0)

        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects / num_samples
        test_acc = evaluate_accuracy(test_loader, net, device)
        elapsed_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {elapsed_time:.1f} sec"
        )


# 设置超参数
batch_size = 32
lr, num_epochs = 0.0001, 10

# 加载数据
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

# 初始化模型、优化器
net = AlexNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

# 训练模型
train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
