import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        return x

data=torch.randn([5,4])
model=Model()
output=model(data)
print(output)