import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.optim as optim


a = torch.zeros((4, 128, 512, 1024))
b = torch.zeros((4, 128, 512, 1024))
c = a*b


w = torch.zeros(512 * 128, 128, 2)
idx = torch.zeros(4, 512, 1024, dtype=torch.int64)
shp = idx.shape
idx = idx.reshape((idx.shape[0] * idx.shape[1] * idx.shape[2]))
b = torch.index_select(w, 0, idx)
b = b.reshape((shp[0], shp[1], shp[2], 128, 2))
#b = torch.zeros((4, 128, 2, 512, 1024))


a = a.permute([0, 2, 3, 1]).unsqueeze(3)

c = torch.matmul(a, b)
print(a.shape)
print(b.shape)
print(c.shape)




a = torch.tensor(1.0, dtype=torch.float32)
w = torch.tensor([0.0, 0.0], dtype=torch.float32)
w.requires_grad_(True)

optimizer = optim.SGD([w], lr=0.001, momentum=0.8)
ind = torch.tensor(1, dtype=torch.int64)

for i in range(1000):
    print("step")
    x = torch.tensor([1])
    y = torch.tensor([2])

    y_test = torch.gather(w, 0, ind) * x

    loss = (y - y_test) ** 2

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(w.detach().cpu().numpy())




