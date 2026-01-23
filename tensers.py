import torch
from matrepr import mprint
import numpy as np


X = torch.tensor([
    [5],
    [1],
    [4]
    ])

print(X)
print(X.dim())
print(X.shape)
print("Hello")
mprint(X)

#Addition

A = torch.tensor([
    [1,3],
    [5,2]
    ])
B = torch.tensor([
    [2,7],
    [10,1]
    ])

mprint(A+B)

#Hadamard Multiplication
mprint(A*B)

#Scalar Multiplication
mprint(4*A)

#Multiplication

x = torch.tensor([
    [2,7],
    [3,4]
    ])
y = torch.tensor([
    [1,2],
    [5,3]
    ])

mprint(x@y)

