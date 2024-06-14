import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
n_samples = 100
x = np.sort(np.random.rand(n_samples) * 10)  # 1D feature
true_b = [1, 2, 3]
true_W = [2, -1, 3]


def generate_y(x):
    y = np.piecewise(x, [x < 3, (x >= 3) & (x < 7), x >= 7],
                     [lambda x: true_W[0] * x + true_b[0],
                      lambda x: true_W[1] * x + true_b[1],
                      lambda x: true_W[2] * x + true_b[2]])
    return y + np.random.randn(n_samples) * 0.5


y = generate_y(x)

# 展示生成的数据
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='b', marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Data')
plt.grid(True)
plt.show()

import torch
import torch.nn as nn
import torch.optim as optim

# 转换数据为 PyTorch 张量
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


# 定义分段线性回归模型
class PiecewiseLinearModel(nn.Module):
    def __init__(self):
        super(PiecewiseLinearModel, self).__init__()
        self.b1 = nn.Parameter(torch.tensor(1.0))
        self.b2 = nn.Parameter(torch.tensor(2.0))
        self.b3 = nn.Parameter(torch.tensor(3.0))
        self.W1 = nn.Parameter(torch.tensor(2.0))
        self.W2 = nn.Parameter(torch.tensor(-1.0))
        self.W3 = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        y1 = self.W1 * x + self.b1
        y2 = self.W2 * x + self.b2
        y3 = self.W3 * x + self.b3
        return torch.where(x < 3, y1, torch.where(x < 7, y2, y3))


model = PiecewiseLinearModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
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

# 生成预测值
model.eval()
y_pred = model(x_tensor).detach().numpy()

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='b', marker='o', label='Original Data')
plt.plot(x, y_pred, 'r-', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Data and Fitted Line')
plt.legend()
plt.grid(True)
plt.show()

# 评估模型
train_mse = criterion(torch.tensor(y_pred, dtype=torch.float32), y_tensor).item()
print(f"Training MSE: {train_mse:.2f}")
