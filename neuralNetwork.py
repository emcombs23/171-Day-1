import torch
from matrepr import mprint
import numpy as np

X = torch.tensor([
    [4],
    [6]
    ])
w = torch.tensor(3)
b = torch.tensor(10)
y = torch.tensor([
    [5],
    [16]
    ])

yhat = X*w+b
print(yhat)

r = yhat - y

SSE  = r.T@r
print(SSE)

loss = SSE/2
print(loss)