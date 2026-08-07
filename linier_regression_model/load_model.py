import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(1,16)
        self.layer2=nn.Linear(16,1)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        return x

model=Model()

model.load_state_dict(torch.load("./linier_regression_model/reg_model.pth"))

model.eval()

pred=model(torch.tensor([[20]], dtype=torch.float32))
print(pred)