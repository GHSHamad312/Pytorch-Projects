import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [1.0],  [2.0],  [3.0],  [4.0],  [5.0],
    [6.0],  [7.0],  [8.0],  [9.0],  [10.0],
    [11.0], [12.0], [13.0], [14.0], [15.0],
    [16.0], [17.0], [18.0], [19.0], [20.0]
])

Y = torch.tensor([
    [5.0],  [8.0],  [11.0], [14.0], [17.0],
    [20.0], [23.0], [26.0], [29.0], [32.0],
    [35.0], [38.0], [41.0], [44.0], [47.0],
    [50.0], [53.0], [56.0], [59.0], [62.0]
])

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

criteration=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    prediction=model(X)
    loss=criteration(prediction, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%20 == 0:
        print(f'epoch : {epoch:3d} : loss: {loss.item()}')

torch.save(
    model.state_dict(),
    "reg_model.pth"
)