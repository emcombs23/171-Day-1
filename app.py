import torch
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
print(df)

X = torch.tensor(df.drop('Y', axis = 1).to_numpy()).float()
print(X)

Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)
print(Y)


w = torch.tensor([
    [1.20],
    [-0.60],
    [1.60],
    [1.60]
]).float()

b = torch.tensor([
    [-0.60]
]).float()

print(X)
print(w)

yHat = X@w+b
r = yHat-Y
SSE = r.T@r
loss = SSE / X.shape[0]
print(loss)
