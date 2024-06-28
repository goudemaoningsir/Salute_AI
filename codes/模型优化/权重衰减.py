import os
import torch
from torch import nn
import matplotlib.pyplot as plt

# 解决某些环境中的重复库问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

############ 1. 生成数据 ############
# 参数设置
n_train, n_test = 100, 100  # 训练集和测试集的样本数量
true_w, true_b = [1.2, -3.4, 5.6], 5  # 多项式回归的真实权重和偏置
num_epochs = 100  # 训练的轮数
lr = 0.01  # 学习率

# 生成合成数据
features = torch.randn((n_train + n_test, 1))  # 生成标准正态分布的特征
poly_features = torch.cat([features**i for i in range(1, 5)], 1)  # 构造多项式特征
labels = (
    sum(true_w[i] * poly_features[:, i] for i in range(len(true_w))) + true_b
)  # 生成标签
labels += torch.normal(0, 0.01, size=labels.size())  # 加入高斯噪声


############ 2. 定义模型 ############
def create_model(input_dim):
    """创建线性回归模型"""
    net = nn.Linear(input_dim, 1)
    return net


############ 3. 定义损失函数和优化器 ############
loss = nn.MSELoss()


def create_optimizer(net, lr=0.01):
    """创建优化器"""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return optimizer


############ 4. 训练模型 ############
def train_model(
    train_features,
    train_labels,
    net,
    loss,
    optimizer,
    num_epochs=100,
    batch_size=10,
    regularization=None,
):
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    train_ls = []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y.reshape(-1, 1))
            if regularization == "L1":
                l += l1_penalty(net.weight)
            elif regularization == "L2":
                l += l2_penalty(net.weight)
            elif regularization == "ElasticNet":
                l += elastic_net_penalty(net.weight)
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels.reshape(-1, 1)).item())
    return train_ls


############ 5. 绘制训练曲线 ############
def plot_loss(train_ls, test_ls):
    """绘制训练和测试损失曲线"""
    plt.figure(figsize=(6, 4))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.semilogy(range(1, num_epochs + 1), train_ls, label="Train")
    plt.semilogy(range(1, num_epochs + 1), test_ls, label="Test", linestyle=":")
    plt.legend()
    plt.show()


############ 6. 绘制拟合曲线 ############
def plot_fitted_curve(
    train_features, train_labels, test_features, test_labels, net, degree
):
    """绘制拟合曲线"""
    plt.figure(figsize=(10, 6))
    plt.scatter(
        train_features[:, 0].detach().numpy(),
        train_labels.detach().numpy(),
        color="blue",
        label="Training Data",
    )
    plt.scatter(
        test_features[:, 0].detach().numpy(),
        test_labels.detach().numpy(),
        color="green",
        label="Test Data",
    )

    sorted_train_features, sorted_train_indices = torch.sort(train_features[:, 0])
    plt.plot(
        sorted_train_features.detach().numpy(),
        net(train_features[sorted_train_indices]).detach().numpy(),
        color="red",
        label="Fitted Curve",
    )

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"Polynomial Regression (Degree={degree})")
    plt.show()


############ 7. L1、L2和弹性网正则化项 ############
def l1_penalty(w):
    """L1正则化"""
    return torch.sum(torch.abs(w))


def l2_penalty(w):
    """L2正则化"""
    return torch.sum(torch.pow(w, 2)) / 2


def elastic_net_penalty(w, alpha=0.5, l1_ratio=0.5):
    """弹性网正则化"""
    l1_term = torch.sum(torch.abs(w))
    l2_term = torch.sum(torch.pow(w, 2)) / 2
    return alpha * (l1_ratio * l1_term + (1 - l1_ratio) * l2_term)


############ 8. 综合运行 ############
def fit_and_plot(
    train_features,
    train_labels,
    test_features,
    test_labels,
    degree,
    regularization=None,
):
    net = create_model(train_features.shape[-1])
    optimizer = create_optimizer(net, lr)

    train_ls = train_model(
        train_features,
        train_labels,
        net,
        loss,
        optimizer,
        num_epochs,
        regularization=regularization,
    )
    test_ls = [
        loss(net(test_features), test_labels.reshape(-1, 1)).item()
        for _ in range(num_epochs)
    ]

    plot_loss(train_ls, test_ls)
    plot_fitted_curve(
        train_features, train_labels, test_features, test_labels, net, degree
    )


# 进行不同多项式次数的回归并绘图
fit_and_plot(
    poly_features[:n_train, :3],
    labels[:n_train],
    poly_features[n_train:, :3],
    labels[n_train:],
    degree=3,
)  # 三次多项式回归，无正则化
fit_and_plot(
    poly_features[:n_train, :3],
    labels[:n_train],
    poly_features[n_train:, :3],
    labels[n_train:],
    degree=3,
    regularization="L1",
)  # L1正则化
fit_and_plot(
    poly_features[:n_train, :3],
    labels[:n_train],
    poly_features[n_train:, :3],
    labels[n_train:],
    degree=3,
    regularization="L2",
)  # L2正则化
fit_and_plot(
    poly_features[:n_train, :3],
    labels[:n_train],
    poly_features[n_train:, :3],
    labels[n_train:],
    degree=3,
    regularization="ElasticNet",
)  # 弹性网正则化
