import torch
x = torch.tensor([
    [3.0]
])

y = torch.tensor([
    [10.0]
])

w = torch.tensor([
    [6.0]
], requires_grad = True)

b = torch.tensor([
    [1.0]
], requires_grad = True)

lr = .2

yHat = x@w+b

r = yHat - y

SSE = r.T@r
loss = SSE / x.shape[0]
print(loss.item())

loss.backward()

with torch.no_grad():
    w -= lr*w.grad
    b -= lr*b.grad

w.grad.zero_()
b.grad.zero_()

print(w,b)