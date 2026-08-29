import torch

tensor1= torch.tensor([[1,2,4,2],[3,4,5,3],[4,3,5,2]])

# print(tensor1)
# print(tensor1.shape)

# building empty tensors

empty_tensor=torch.zeros(5)
ones=torch.ones(5)
randoms=torch.rand(5)

# print(empty_tensor)
# print(ones)
# print(randoms)
# print(randoms.dtype)
# print(ones.dtype)
# print(tensor1.dtype)

custom=torch.tensor([[10,2,4,3,5],[4,3,5,32,4]],dtype=torch.float32)
# print(custom)


# creating tensor of 10 zeros
zeros=torch.zeros(10)
# print(zeros)

# 4x4 random matrix
fourrand=torch.rand((4,4))
# print(fourrand.shape)
# print(fourrand)

# complex
comp=torch.rand((1,28,28))
# print(comp.shape)
# print(comp)
# print(comp.shape)
# print(comp)