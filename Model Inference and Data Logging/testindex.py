import torch
from math import floor


tensor_3d = torch.randn(10, 16, 16)  # Example 3D tensor with shape (4, 3, 2)
indices = torch.tensor([0])
selected_slices = torch.index_select(tensor_3d, 0, indices)

S = 16
cellSize = 1 / S
x = 0.99
y = 0.99


n = floor(x/cellSize) + (floor(y/cellSize)) * S
n = 7 % 3
print(n)