import torch
import torch.nn as nn
import numpy as np

criterion = nn.MSELoss()
x = np.random.randn(1, 2, 2)
y = np.random.randn(1, 2, 2)
x = torch.from_numpy(x).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)
print(x)
print(y)
loss = criterion(x, y)
print(loss.item())

