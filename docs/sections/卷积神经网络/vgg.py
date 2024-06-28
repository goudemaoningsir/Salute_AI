import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))  # 使用inplace=True可以节省内存
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# 定义VGG网络
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    layers = []
    in_channels = 1  # 输入通道数为1，适配FashionMNIST数据集
    for num_convs, in_channels, out_channels in conv_arch:
        layers.append(vgg_block(num_convs, in_channels, out_channels))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(fc_features, fc_hidden_units))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(fc_hidden_units, fc_hidden_units))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(fc_hidden_units, 10))  # 10类分类问题输出层
    return nn.Sequential(*layers)


# VGG网络配置
conv_arch = [(1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512)]
fc_features = 512 * 7 * 7  # 经过5个vgg_block后，特征图大小为7x7，通道数为512
fc_hidden_units = 4096

# 构建VGG模型
net = vgg(conv_arch, fc_features, fc_hidden_units)
print(net)


# 加载FashionMNIST数据集
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


# 设置超参数
batch_size = 3
lr, num_epochs = 0.001, 5

# 加载数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# 训练和评估模型
def train(net, train_loader, test_loader, optimizer, criterion, device, num_epochs):
    net = net.to(device)
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
train(net, train_iter, test_iter, optimizer, criterion, device, num_epochs)
