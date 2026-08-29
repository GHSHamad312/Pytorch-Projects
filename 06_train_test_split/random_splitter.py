import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file):
        df=pd.read_csv(file)
        self.x=torch.tensor(df[["operating_hours","rotation_speed_rpm","torque_nm","power_kw","ambient_temp_c"]].values, dtype=torch.float32)
        self.y=torch.tensor(df["failure_mode_type"].values, dtype=torch.long)
    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        return self.x[index],self.y[index]

dataset=CustomDataset("./test_train_split/industrial_predictive_maintenance_data.csv")
test_split, train_split=random_split(dataset,[0.2,0.8])
print(f'test split length:{len(test_split)}')
print(f'trian split length:{len(train_split)}')

train_loader=DataLoader(train_split, batch_size=32, shuffle=True)
test_loader=DataLoader(test_split, batch_size=32, shuffle=False)

