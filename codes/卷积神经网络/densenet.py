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
