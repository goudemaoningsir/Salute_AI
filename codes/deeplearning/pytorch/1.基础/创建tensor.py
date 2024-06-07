import torch
import numpy as np

# 从列表创建
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)
print(tensor_from_list)

# 从 NumPy 数组创建
np_array = np.array(data)
tensor_from_np = torch.from_numpy(np_array)
print(tensor_from_np)

# 创建全零 Tensor
zero_tensor = torch.zeros((2, 2))
print(zero_tensor)

# 创建全一 Tensor
ones_tensor = torch.ones((2, 2))
print(ones_tensor)

# 创建未初始化的 Tensor
empty_tensor = torch.empty((2, 2))
print(empty_tensor)

# 创建随机初始化的 Tensor
random_tensor = torch.rand((2, 2))
print(random_tensor)

# 创建单位矩阵
eye_tensor = torch.eye(3)  # 3x3单位矩阵
print(eye_tensor)

# 创建包含均匀分布随机数的 Tensor
uniform_tensor = torch.rand((2, 2))
print(uniform_tensor)

# 创建包含正态分布随机数的 Tensor
normal_tensor = torch.randn((2, 2))
print(normal_tensor)

# 创建指定数据类型的 Tensor
float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
print(float_tensor)
int_tensor = torch.tensor([1, 2], dtype=torch.int32)
print(int_tensor)


# 创建 Tensor 在 GPU 上
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1.0, 2.0], device='cuda')
else:
    gpu_tensor = torch.tensor([1.0, 2.0], device='cpu')
print(gpu_tensor)

# 克隆一个 Tensor
original_tensor = torch.tensor([1, 2, 3])
cloned_tensor = original_tensor.clone()
print(cloned_tensor)

# 改变 Tensor 的形状
reshaped_tensor = original_tensor.view(1, 3)
print(reshaped_tensor)

# 创建包含范围序列的 Tensor
range_tensor = torch.arange(0, 10, step=2)
print(range_tensor)

# 创建线性序列的 Tensor
linspace_tensor = torch.linspace(0, 1, steps=5)
print(linspace_tensor)

# 通过条件创建 Tensor
mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
print(mask)
conditional_tensor = torch.ones(4).masked_fill(mask, 0)
print(conditional_tensor)