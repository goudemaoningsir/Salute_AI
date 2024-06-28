import os
import torch
from torch import nn
import matplotlib.pyplot as plt

# 解决某些环境中的重复库问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 生成合成数据
n_train, n_test = 1000, 1000
num_inputs = 784
num_outputs = 10
num_hiddens1 = 256
num_hiddens2 = 256
drop_prob1, drop_prob2 = 0.2, 0.5

train_features = torch.randn((n_train, num_inputs))
train_labels = torch.randint(0, num_outputs, (n_train,))
test_features = torch.randn((n_test, num_inputs))
test_labels = torch.randint(0, num_outputs, (n_test,))


# 定义模型
class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2, num_outputs),
)

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# 定义损失函数和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


# 训练模型
def train_model(
    train_features,
    train_labels,
    test_features,
    test_labels,
    net,
    loss,
    optimizer,
    num_epochs=100,
    batch_size=10,
):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        net.train()  # 设置网络为训练模式
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
        print(
            f"epoch {epoch + 1}, train loss {train_ls[-1]:.4f}, test loss {test_ls[-1]:.4f}"
        )
    return train_ls, test_ls


train_ls, test_ls = train_model(
    train_features, train_labels, test_features, test_labels, net, loss, optimizer
)


# 绘制损失变化曲线
def plot_loss(train_ls, test_ls):
    plt.figure(figsize=(6, 4))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.semilogy(range(1, len(train_ls) + 1), train_ls, label="Train")
    plt.semilogy(range(1, len(test_ls) + 1), test_ls, label="Test", linestyle=":")
    plt.legend()
    plt.show()


plot_loss(train_ls, test_ls)
