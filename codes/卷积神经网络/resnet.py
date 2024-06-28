import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义Residual Block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


# 定义ResNet模块
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


# 数据预处理
transform = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])

# 加载数据
train_dataset = datasets.FashionMNIST(
    root="/workspace/codes/datasets/FashionMNIST",
    train=True,
    transform=transform,
    download=True,
)
test_dataset = datasets.FashionMNIST(
    root="/workspace/codes/datasets/FashionMNIST",
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# 训练和测试函数
def train(net, train_loader, optimizer, device):
    net.train()
    total_loss, total_acc, total_num = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (y_hat.argmax(dim=1) == y).sum().item()
        total_num += y.size(0)
    return total_loss / total_num, total_acc / total_num


def test(net, test_loader, device):
    net.eval()
    total_loss, total_acc, total_num = 0, 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = F.cross_entropy(y_hat, y)
            total_loss += loss.item()
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_num += y.size(0)
    return total_loss / total_num, total_acc / total_num


# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train(net, train_loader, optimizer, device)
    test_loss, test_acc = test(net, test_loader, device)
    print(
        f"epoch {epoch+1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, test loss {test_loss:.4f}, test acc {test_acc:.4f}"
    )
