import torch
from torch import nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

############################################ 1、生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32
)

# 打印第一个样本的特征和标签
print(features[0], labels[0])

# 直接使用 plt 画特征 features[:, 1] 和标签 labels 的散点图
plt.scatter(features[:, 1].numpy(), labels.numpy(), s=1)
plt.xlabel("Feature 1")
plt.ylabel("Label")
plt.title("Feature 1 vs Label")
plt.show()


############################################ 2、读取数据集
# 定义加载数据的函数
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使用数据迭代器读取第一个小批量样本
first_batch = next(iter(data_iter))

# 打印第一个小批量样本
print(f"Batch of features:\n{first_batch[0]}")
print(f"Batch of labels:\n{first_batch[1]}")

############################################ 3、定义线性模型
# 使用 nn.Sequential 定义线性模型
net = nn.Sequential(nn.Linear(2, 1))

# 输出模型结构
print(net)

############################################ 4、初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

############################################ 5、定义损失函数
loss = nn.MSELoss()

############################################ 6、定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

############################################ 7、训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y.view(-1, 1))
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels.view(-1, 1))
    print(f"epoch {epoch + 1}, loss {l:f}")

############################################ 8、输出参数估计误差
# 获取训练后的模型参数
estimated_w = net[0].weight.data.numpy()
estimated_b = net[0].bias.data.numpy()

# 计算估计误差
w_error = np.abs(estimated_w - true_w)
b_error = np.abs(estimated_b - true_b)

print(f"估计的 w: {estimated_w.flatten()}")
print(f"真实的 w: {true_w}")
print(f"w 的绝对误差: {w_error.flatten()}")
print()
print(f"估计的 b: {estimated_b.flatten()}")
print(f"真实的 b: {true_b}")
print(f"b 的绝对误差: {b_error.flatten()}")
