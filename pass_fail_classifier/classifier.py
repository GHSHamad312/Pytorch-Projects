from linier_regression_model.Regression_model import criteration
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader

class StudentDataset(Dataset):
    def __init__(self,file):
        dataframe=pd.read_csv(file)
        self.x=torch.tensor(dataframe[['study_hours', 'attendance_pct', 'previous_score', 'sleep_hours']].values)
        self.y=torch.tensor(dataframe["passed"].values)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Classifier(nn.Module):
    def __init__(self):
        self.linear1=nn.Linear(4,16)
        self.linear2=nn.Linear(16,32)
        self.linear3=nn.Linear(32,4)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        return x

dataset=StudentDataset("./pass_fail_classifier/students.csv")
model=Classifier()
optimizer=optim.Adam(
Classifier.parameters(),lr=0.001)
criteration=nn.CrossEntropyLoss()
loader=dataloader(dataset, batch_size=3)


for epoch in range (200):
    for x,y in loader:
        