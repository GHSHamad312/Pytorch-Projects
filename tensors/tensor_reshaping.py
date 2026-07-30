from sympy import flatten
import torch
# generatin sequence of numbers
sequencial=torch.arange(start=1, end=25)
# print(sequencial)
# print(sequencial.shape)

# reshaping it
reshaped=sequencial.reshape(4,6)
# print(reshaped)
# print(reshaped.shape)

# again reshaping
reshape2=reshaped.reshape(2,3,4)
# print(reshape2)
# print(reshape2.shape)

# falttening
flattened_tensor=reshape2.flatten()
# print(flattened_tensor)

# adding dimension
dimension_tensor=flattened_tensor.unsqueeze(0)
# print(dimension_tensor.shape)

# removing dimension
prev_tensor=dimension_tensor.squeeze(0)
# print(prev_tensor.shape)

# creating 2 random tensors
ran1=torch.rand((2,3))
ran2=torch.rand((2,3))
print(ran1)
print(ran2)
stacked=torch.stack([ran1,ran2])
concat=torch.cat([ran1,ran2])
print(stacked)
print(concat)