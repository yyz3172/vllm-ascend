import torch

import numpy as np

print("===================创建张量===================")

# 1. 从 Python 列表或 NumPy 数组创建
tensor_from_list = torch.tensor([1, 2, 3])
print("tensor_from_list", tensor_from_list.shape)
np_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(np_array) # 与 numpy 数组共享内存
print("tensor_from_numpy", tensor_from_numpy.shape)
# 2. 创建特殊值的张量
zeros = torch.zeros(2, 3)       # 全0张量
print("zeros", zeros.shape)
ones = torch.ones(2, 3)         # 全1张量
print("ones", ones.shape)
empty = torch.empty(2, 3)       # 未初始化的张量（值为内存中的随机数）
print("empty", empty.shape)
# 3. 创建随机张量
rand = torch.rand(2, 3)         # [0, 1) 区间内的均匀分布
print("rand", rand.shape)
randn = torch.randn(2, 3)       # 标准正态分布（均值为0，方差为1）
print("randn", randn.shape)
# 4. 创建特定序列的张量
arange = torch.arange(0, 10, step=2) # [0, 2, 4, 6, 8]
print("arange", arange.shape)
linspace = torch.linspace(0, 1, steps=5) # [0.0, 0.25, 0.5, 0.75, 1.0]
print("linspace", linspace.shape)

print("===================张量属性===================")

x = torch.randn(3, 4, 5)

print(f"形状 (shape): {x.shape}")     # torch.Size([3, 4, 5])
print(f"维度 (dim): {x.dim()}")       # 3
print(f"元素: {x}") # 60
print(f"元素总数 (numel): {x.numel()}") # 60
print(f"数据类型 (dtype): {x.dtype}") # torch.float32
print(f"设备 (device): {x.device}")   # cpu

print("===================张量运算===================")

a = torch.tensor([[1., 2.], [3., 4.]])
print("a", a)
b = torch.tensor([[5., 6.], [7., 8.]])
print("b", b)

# 加法 (三种方式)
add1 = a + b
add2 = torch.add(a, b)
add3 = a.add(b)
print("add1", add1)

# 原地加法 (直接修改 a，节省内存)
a.add_(b) # 或者 a += b
print("a", a)

# 乘法
# 元素级乘法 (对应位置相乘)
mul = a * b
print("mul", mul)
mul = torch.mul(a, b)

# 矩阵乘法
matmul = a @ b
print("matmul", matmul)
matmul = torch.matmul(a, b)

# 其他运算
div = a / b       # 浮点除法
print("div", div)
div_int = a // b  # 整除
print("div_int", div_int)
sum_val = a.sum() # 求和
print("sum_val", sum_val)
mean_val = a.mean() # 求均值
print("mean_val", mean_val)
max_val = a.max()   # 求最大值
print("max_val", max_val)

print("===================张量形状操作===================")

x = torch.randn(4, 4)
print("x", x)
# 改变形状 (reshape 和 view 功能类似)
y = x.view(16)      # 展平成 16 个元素的一维张量
z = x.reshape(-1, 8) # -1 表示自动推断该维度，结果为 (2, 8)
print("y", y)
print("z", z)
# 增加/减少维度
tensor_2d = torch.randn(3, 4)
print("tensor_2d", tensor_2d)
tensor_3d = tensor_2d.unsqueeze(0) # 在第0维增加一个维度，形状变为 (1, 3, 4)
print("tensor_3d", tensor_3d)
tensor_2d_back = tensor_3d.squeeze(0) # 删除第0维，形状变回 (3, 4)
print("tensor_2d_back", tensor_2d_back)
# 转置
# 2D 张量转置
t = x.t()
print("t", t)
# 通用转置，可交换任意维度
p = torch.randn(2, 3, 4, 5)
print("p", p)   
print("p.shape", p.shape)
p_transposed = p.permute(0, 3, 2, 1) # 形状变为 (2, 5, 4, 3)
print("p_transposed", p_transposed)
print("p_transposed.shape", p_transposed.shape)


# 创建一个 (3, 1) 的张量
a = torch.tensor([[1], [2], [3]]) 
# 创建一个 (1, 4) 的张量
b = torch.tensor([[10, 20, 30, 40]])
print(f"A 的形状: {a.shape}") # torch.Size([3, 1])
print(f"B 的形状: {b.shape}") # torch.Size([1, 4])

# 直接相加
c = a + b
print(f"结果 C 的形状: {c.shape}") # torch.Size([3, 4])
print("结果内容:")
print(c)


print("===================张量索引与切片===================")

x = torch.arange(12).reshape(3, 4)
print("x", x)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 基本索引
print(x[0])        # 第一行: tensor([0, 1, 2, 3])
print(x[:, 1])     # 第二列: tensor([1, 5, 9])
print(x[1:3, 2])   # 第2、3行的第3列元素: tensor([6, 10])

# 高级索引
print(x[[0, 2], [1, 3]]) # 获取 (0,1) 和 (2,3) 位置的元素: tensor([1, 11])
print(x[x > 5])    # 获取所有大于5的元素: tensor([ 6,  7,  8,  9, 10, 11])


print("===================张量拼接与分割===================")

a = torch.ones(2, 3)
b = torch.ones(2, 3)
print("a", a)
print("b", b)
# 拼接
cat_0 = torch.cat([a, b], dim=0) # 沿第0维拼接，形状为 (4, 3)
cat_1 = torch.cat([a, b], dim=1) # 沿第1维拼接，形状为 (2, 6)
print("cat_0", cat_0)
print("cat_1", cat_1)
# 分割
chunks = torch.chunk(a, 2, dim=1) # 沿第1维分割成2块
print("chunks", chunks)

print("===================张量与numpy互转===================")

# Tensor 转 NumPy
tensor_x = torch.ones(5)
numpy_y = tensor_x.numpy() # 在 CPU 上，两者共享内存，修改一个会影响另一个

# NumPy 转 Tensor
numpy_z = np.array([1, 2, 3])
tensor_w = torch.from_numpy(numpy_z) # 同样共享内存


print("===================张量to操作===================")
#设备迁移：在 CPU 和 GPU/NPU 之间移动数据。
#类型转换：改变数据的精度（例如从 float32 转为 float64）。
# 1. 创建一个在 CPU 上的张量
x = torch.tensor([1.0, 2.0, 3.0])
print(x.device)  # 输出: cpu

# 2. 检查是否有可用的 npu
if torch.npu.is_available():
    # 定义设备对象 (推荐做法)
    device = torch.device("npu")
    # 将张量移动到 NPU
    x_npu = x.to(device) 
    # 或者简写为: x.to("npu")    
    print("npu", x_npu.device)  # 输出: npu:0 (表示在第0号显卡上)

# 默认是 float32
x = torch.ones(2, 3) 
print(x.dtype)  # 输出: torch.float32
# 转换为 float64 (双精度)
x_double = x.to(torch.float64)
print(x_double.dtype)  # 输出: torch.float64
# 转换为 int64 (长整型，常用于标签)
x_long = x.to(torch.long)
print(x_long.dtype)  

if torch.npu.is_available():
    # 同时移动到 GPU 并转换为 float16 (半精度，常用于加速训练)
    x = x.to("npu", torch.float16)


print("===================pythorch gather and index_select===================")
#torch.index_select：像是“整行/整列搬运工”。它比较“笨”，只能按固定的索引去搬运整行或整列，索引是共享的。
#torch.gather：像是“精准狙击手”。它非常灵活，可以为每一个位置指定不同的取值坐标，索引是独立的。

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 我想选第 0 行和第 2 行
indices = torch.tensor([0, 2]) 
print("indices", indices)
# 在 dim=0 (行) 方向上选
result = torch.index_select(x, dim=0, index=indices)
print(result)
# 输出:
# tensor([[1, 2, 3],  <- 原第 0 行
#         [7, 8, 9]]) <- 原第 2 行



x = torch.tensor([
    [1, 2, 3],  # 第0行
    [4, 5, 6],  # 第1行
    [7, 8, 9]   # 第2行
])

# 我想：
# 第0行取第2列的值 (3)
# 第1行取第0列的值 (4)
# 第2行取第1列的值 (8)

# index 的形状必须和输出一致 (这里是 3x1)
# index 里的值代表 "在 dim=1 (列) 方向上的坐标"
indices = torch.tensor([
    [2], 
    [0], 
    [1]
])

result = torch.gather(x, dim=1, index=indices)
print(result)
# 输出:
# tensor([[3],
#         [4],
#         [8]])