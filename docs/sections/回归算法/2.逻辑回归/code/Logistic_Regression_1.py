import numpy as np
import matplotlib.pyplot as plt

# 生成二分类数据集
np.random.seed(0)
mean1 = np.array([1, 2])
mean2 = np.array([6, 6])
cov = np.array([[1, 0.5], [0.5, 1]])
n_samples = 100

# 生成类别为0的数据
data1 = np.random.multivariate_normal(mean1, cov, n_samples)
labels1 = np.zeros(n_samples)

# 生成类别为1的数据
data2 = np.random.multivariate_normal(mean2, cov, n_samples)
labels2 = np.ones(n_samples)

# 合并数据
data = np.concatenate((data1, data2), axis=0)
labels = np.concatenate((labels1, labels2))

# 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data for Logistic Regression')
plt.grid(True)
plt.colorbar(label='Class', ticks=[0, 1])
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

# 转换数据为 PyTorch 张量
x_tensor = torch.tensor(data, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)


# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 输入维度为 input_dim，输出维度为1

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


# 初始化模型
model = LogisticRegression(input_dim=2)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')


def plot_decision_boundary(model, data, labels):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(mesh_tensor).numpy()
        Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Logistic Regression')
    plt.grid(True)
    plt.colorbar(label='Class', ticks=[0, 1])
    plt.show()


# 绘制决策边界和评估模型
plot_decision_boundary(model, data, labels)

# 计算在训练集上的准确率
model.eval()
with torch.no_grad():
    outputs = model(x_tensor)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_tensor).float().mean()
    print(f'Training Accuracy: {accuracy.item() * 100:.2f}%')
