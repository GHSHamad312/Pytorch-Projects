import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class StudentDataset(Dataset):
    def __init__(self,file):
        dataframe=pd.read_csv(file)
        self.x=torch.tensor(dataframe[['study_hours', 'attendance_pct', 'previous_score', 'sleep_hours']].values, dtype=torch.float32)
        self.y=torch.tensor(dataframe["passed"].values,  dtype=torch.long)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(4,16)
        self.linear2=nn.Linear(16,32)
        self.linear3=nn.Linear(32,2)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        return x

dataset=StudentDataset("./pass_fail_classifier/students.csv")
model=classifier()
optimizer=optim.Adam(
model.parameters(),lr=0.001)
criteration=nn.CrossEntropyLoss()
loader=DataLoader(dataset, batch_size=3)


for epoch in range (200):
    for x,y in loader:
        prediction=model(x)
        loss=criteration(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(f'epoch:{epoch:3d}: Loss: {loss}')

prediction=model(torch.tensor([2.5,65,95,6.0]))
print(prediction)