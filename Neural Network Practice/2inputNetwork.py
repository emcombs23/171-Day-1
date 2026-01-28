import torch

X = torch.tensor([
    [2,3],
    [1,2]
]).float()

Y = torch.tensor([
    [30],
    [17]
]).float()

w = torch.tensor([
    [4],
    [5]
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