import torch

# Generate 4x4 random matrix
random = torch.rand((4, 4))

print(random)

# Indexing and element slicing examples
# print(random[1])    # Print second row
# print(random[:, -1])  # Print last column

# In-place tensor arithmetic operations
# random[1, 1] = 0.999
# print(random)
# random += 0.001
# print(random)
# random *= 2
# print(random)

# Initialize tensors for multiplication
tensor1 = torch.ones((4, 4))
tensor2 = torch.rand((4, 4))

# Print input matrices
print(tensor1)
print(tensor2)

# Perform matrix multiplication using @
print(tensor1 @ tensor2)