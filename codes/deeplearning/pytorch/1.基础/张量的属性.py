import torch

# 创建一个2x3的张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Shape of tensor:", tensor.shape) # Shape of tensor: torch.Size([2, 3])

# 改变张量形状为3x2
reshaped_tensor = tensor.view(3, 2)
print("Reshaped tensor (3x2):", reshaped_tensor)

# 使用reshape方法
reshaped_tensor = tensor.reshape(3, 2)
print("Reshaped tensor (3x2) with reshape:", reshaped_tensor)

# 查看张量的设备
print("Device tensor is stored on:", tensor.device)

# 检查是否有GPU可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor_on_gpu = tensor.to(device)
    print("Tensor is on GPU:", tensor_on_gpu.device)
else:
    print("CUDA is not available.")

# 检查可用设备，并赋值给device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device(device)
print(f"Using {device} device")

# 创建一个在默认设备（通常是CPU）上的Tensor
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print("Tensor on default device:", x)

# 将Tensor移动到指定设备
x = x.to(device)
print("Tensor on specified device:", x)
