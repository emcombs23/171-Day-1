import torch
x = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
])

y = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
])

w = torch.tensor([
    [0.0]
], requires_grad = True)

b = torch.tensor([
    [0.0]
], requires_grad = True)

lr = .01
count = 0
while True:
    oldW = w.item()
    oldB = b.item()

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


    print(w.item(),b.item())
    print(w.item()-oldW, b.item()-oldB)

    if abs(w.item()-oldW) < .0000000000001 and abs(b.item()-oldB) <.0000000000001:
        print(count)
        break
    count += 1
    print()

print('7', w.item()*7+b.item())