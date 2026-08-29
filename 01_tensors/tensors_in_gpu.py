import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

device=torch.device("cuda")
print(device)

# making a deafult tensor and cheking device
x=torch.tensor([1,2,4,5])
print(x.device)
x=x.to(device)
print(x.device)

y=torch.tensor([2,3,4,5,6])
z=y.clone()
z[1]=100
# print(y)
# print(z)
