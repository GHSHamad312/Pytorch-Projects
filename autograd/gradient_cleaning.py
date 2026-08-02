import torch

x=torch.tensor(5,dtype=torch.float32 ,requires_grad=True)
y=x**2
y.backward()
print(x.grad)
x.grad.zero_()
y=x**2
y.backward()
print(x.grad)

