from torch.utils.data import TensorDataset, DataLoader
import torch

x=torch.tensor([
    [1],[2],[3],[4]
],dtype=torch.float32)

y=torch.tensor([
    [5],[6],[7],[8]
],dtype=torch.float32)
set=TensorDataset(x,y)
# print(set[1])
# print(len(set))

# for x,y in set:
#     print(x,y)
loader=DataLoader(set,batch_size=2,shuffle=True)

for batch in loader:
    print(batch)