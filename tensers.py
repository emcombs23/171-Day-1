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
#mprint(X)
