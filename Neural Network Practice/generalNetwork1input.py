import torch

X = torch.tensor([
    [2],
    [5]
]).float()

Y = torch.tensor([
    [5],
    [1]
]).float()

w = torch.tensor([
    [3]
]).float()

b = torch.tensor([
    [1]
]).float()


yHat = X@w+b
print(yHat)

r = yHat - Y
print(r)

SSE = r.T@r
print(SSE)

print(X.shape)
loss = SSE/X.shape[0]
print(loss.item())