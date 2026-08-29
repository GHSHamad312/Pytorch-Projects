# 01 - PyTorch Tensors Fundamentals

Tensors are the fundamental data structure in PyTorch. They are multi-dimensional arrays (n-dimensional matrices) optimized for mathematical operations, automatic differentiation, and hardware acceleration on GPUs (CUDA) and Apple Silicon (MPS).

---

## 📁 Module Structure

```text
01_tensors/
├── tensor_basics.py       # Tensor creation, dtypes, shapes, and dimensional initialization
├── tensor_operations.py   # Tensor slicing, in-place arithmetic, matrix multiplication
├── tensor_reshaping.py    # Reshaping, viewing, flattening, and dimension manipulations
├── tensors_in_gpu.py      # Device management, CUDA acceleration, and tensor migration
└── README.md              # Documentation & conceptual reference
```

---

## 🧠 Core Concepts & Mechanics

### 1. Tensor Initialization & Data Types (`tensor_basics.py`)
PyTorch provides several functions to initialize tensors from raw Python lists or with specific initial values:
- `torch.tensor(data, dtype=...)`: Creates a tensor from existing numeric arrays or lists.
- `torch.zeros(shape)` / `torch.ones(shape)`: Initializes tensors filled with 0s or 1s.
- `torch.rand(shape)`: Generates tensors with uniform random numbers between $[0, 1)$.
- `tensor.shape` & `tensor.dtype`: Inspects spatial dimensions and element precision (e.g. `torch.float32`, `torch.int64`).

```python
# Example: Creating a 2D float tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
```

### 2. Slicing & Operations (`tensor_operations.py`)
- **Indexing & Slicing**: Compatible with NumPy slicing conventions (`tensor[1]`, `tensor[:, -1]`).
- **In-Place Operations**: Operations ending with an underscore (`_`) mutate the tensor in-place without allocating new memory (`tensor.add_()`, `tensor += 1`).
- **Matrix Multiplication**: Supported via `torch.matmul(A, B)` or the `@` operator (`A @ B`), computing dot products across inner dimensions.

### 3. Dimension Manipulation (`tensor_reshaping.py`)
- **Reshaping & Viewing**: `tensor.view(shape)` and `tensor.reshape(shape)` transform tensor dimensions while preserving the total number of elements.
- **Squeeze & Unsqueeze**:
  - `tensor.squeeze(dim)`: Removes singleton dimensions of size 1.
  - `tensor.unsqueeze(dim)`: Inserts a new dimension of size 1 at the specified index (essential for adding batch dimensions).

### 4. GPU Acceleration & Device Agnosticism (`tensors_in_gpu.py`)
PyTorch seamlessly transfers computational workloads between host memory (CPU) and device memory (GPU):
- Check CUDA availability: `torch.cuda.is_available()`.
- Move tensors across devices: `tensor.to(device)` or `tensor.cuda()`.
- Verify current tensor device: `tensor.device`.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_on_gpu = tensor.to(device)
```

---

## 🚀 How to Run

Execute any script directly using Python:

```bash
python 01_tensors/tensor_basics.py
python 01_tensors/tensor_operations.py
python 01_tensors/tensor_reshaping.py
python 01_tensors/tensors_in_gpu.py
```
