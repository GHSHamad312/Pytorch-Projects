import torch

x=torch.tensor(102,dtype=torch.float32, requires_grad=True)
y=5*x**4
print(x.shape)
y.backward()
print(x.grad)