import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linier=nn.Linear(1,2)
    
    def forward(self, x):
        return self.linier

model=Model()
x=torch.tensor([10,20,12])
output=model(x)
print(output)
for name,parameter in model.named_parameters():
    print(name)
    print(parameter)
