import torch
import torch.nn as nn
criteration= nn.MSELoss()
prediction= torch.tensor([10], dtype=torch.float32)
target= torch.tensor([9], dtype=torch.float32)
loss=criteration(prediction, target)
print(f'MSELoss: {loss}')

classCriteration=nn.CrossEntropyLoss()
prediction=torch.tensor([[0.4, 3.5, 0.6]])
target=torch.tensor([0])
classLoss=classCriteration(prediction,target)
print(f'CrossEntropyLoss: {classLoss}')

