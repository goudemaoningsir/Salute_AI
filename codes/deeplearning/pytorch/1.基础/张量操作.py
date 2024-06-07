import torch

# 创建一个 2x3 的张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("======================索引===============================")

# 访问第 1 行第 2 列的元素
print(tensor[0, 1])  # 输出: 2

# 访问第 2 行的所有元素
print(tensor[1])  # 输出: tensor([4, 5, 6])
print("=====================切片================================")

# 切片操作，获取第 1 行第 1 列到第 2 列的元素
print(tensor[0, 1:3])  # 输出: tensor([2, 3])

# 获取所有行的第 2 列
print(tensor[:, 1])  # 输出: tensor([2, 5])

print("====================连接=================================")

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 沿着第 0 维连接（行连接）
print(torch.cat((tensor1, tensor2), dim=0))
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# 沿着第 1 维连接（列连接）
print(torch.cat((tensor1, tensor2), dim=1))
# 输出:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# 使用 torch.stack 连接
print(torch.stack((tensor1, tensor2), dim=0))
# 输出:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])

print("=======================分割==============================")
# 按指定大小分割
tensor = torch.tensor([1, 2, 3, 4, 5, 6])
print(torch.split(tensor, 2))
# 输出: (tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))

# 按指定块数分割
print(torch.chunk(tensor, 3))
# 输出: (tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))

print("======================维度变换===============================")
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.view(3, 2))
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

print(tensor.reshape(3, 2))
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
print(tensor.shape)  # 输出: torch.Size([1, 2, 3])

print(tensor.squeeze().shape)  # 输出: torch.Size([2, 3])

tensor = torch.tensor([1, 2, 3])
print(tensor.unsqueeze(0).shape)  # 输出: torch.Size([1, 3])

print("======================算数操作===============================")

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print(x + y)  # 加法，输出: tensor([5, 7, 9])
print(x - y)  # 减法，输出: tensor([-3, -3, -3])
print(x * y)  # 乘法，输出: tensor([4, 10, 18])
print(x / y)  # 除法，输出: tensor([0.2500, 0.4000, 0.5000])

print("======================矩阵乘法===============================")
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

print(torch.mm(x, y))
# 输出:
# tensor([[19, 22],
#         [43, 50]])
print("======================转置===============================")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(x.t())
# 输出:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])

print("======================逆矩阵===============================")

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(torch.inverse(x))
# 输出:
# tensor([[-2.0000,  1.0000],
#         [ 1.5000, -0.5000]])

print("======================广播机制===============================")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([1, 2, 3])

print(x + y)
# 输出:
# tensor([[2, 4, 6],
#         [5, 7, 9]])

print("======================均值===============================")


tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 计算整个张量的均值
mean_all = torch.mean(tensor)
print(mean_all)  # 输出：tensor(2.5000)

# 计算沿第0维的均值
mean_dim0 = torch.mean(tensor, dim=0)
print(mean_dim0)  # 输出：tensor([2., 3.])

# 计算沿第1维的均值
mean_dim1 = torch.mean(tensor, dim=1)
print(mean_dim1)  # 输出：tensor([1.5000, 3.5000])

print("======================方差===============================")

# 计算整个张量的方差
var_all = torch.var(tensor)
print(var_all)  # 输出：tensor(1.6667)

# 计算沿第0维的方差
var_dim0 = torch.var(tensor, dim=0)
print(var_dim0)  # 输出：tensor([1., 1.])

# 计算沿第1维的方差
var_dim1 = torch.var(tensor, dim=1)
print(var_dim1)  # 输出：tensor([0.5000, 0.5000])

print("======================标准差===============================")

# 计算整个张量的标准差
std_all = torch.std(tensor)
print(std_all)  # 输出：tensor(1.290994)

# 计算沿第0维的标准差
std_dim0 = torch.std(tensor, dim=0)
print(std_dim0)  # 输出：tensor([1., 1.])

# 计算沿第1维的标准差
std_dim1 = torch.std(tensor, dim=1)
print(std_dim1)  # 输出：tensor([0.7071, 0.7071])

print("======================最小值===============================")

# 计算整个张量的最小值
min_all = torch.min(tensor)
print(min_all)  # 输出：tensor(1.)

# 计算沿第0维的最小值
min_dim0 = torch.min(tensor, dim=0)
print(min_dim0.values)  # 输出：tensor([1., 2.])
print(min_dim0.indices)  # 输出：tensor([0, 0])

# 计算沿第1维的最小值
min_dim1 = torch.min(tensor, dim=1)
print(min_dim1.values)  # 输出：tensor([1., 3.])
print(min_dim1.indices)  # 输出：tensor([0, 0])

print("======================最大值===============================")

# 计算整个张量的最大值
max_all = torch.max(tensor)
print(max_all)  # 输出：tensor(4.)

# 计算沿第0维的最大值
max_dim0 = torch.max(tensor, dim=0)
print(max_dim0.values)  # 输出：tensor([3., 4.])
print(max_dim0.indices)  # 输出：tensor([1, 1])

# 计算沿第1维的最大值
max_dim1 = torch.max(tensor, dim=1)
print(max_dim1.values)  # 输出：tensor([2., 4.])
print(max_dim1.indices)  # 输出：tensor([1, 1])

print("======================中位数===============================")

# 计算整个张量的中位数
median_all = torch.median(tensor)
print(median_all)  # 输出：tensor(2.5000)

# 计算沿第0维的中位数
median_dim0 = torch.median(tensor, dim=0)
print(median_dim0.values)  # 输出：tensor([2., 3.])
print(median_dim0.indices)  # 输出：tensor([0, 0])

# 计算沿第1维的中位数
median_dim1 = torch.median(tensor, dim=1)
print(median_dim1.values)  # 输出：tensor([1.5, 3.5])
print(median_dim1.indices)  # 输出：tensor([0, 0])