# 02 - PyTorch Autograd Engine & Gradient Computation

`torch.autograd` is PyTorch's automatic differentiation engine that powers deep learning neural network optimization. It automatically records operations executed on tensors to build a dynamic computation graph (Directed Acyclic Graph or DAG), enabling exact gradient calculations via reverse-mode automatic differentiation (backpropagation).

---

## 📁 Module Structure

```text
02_autograd/
├── calculating_gradients.py   # Forward pass, DAG construction, and backward pass gradient calculation
├── gradient_cleaning.py       # Gradient accumulation mechanics and zeroing gradients (zero_())
└── README.md                  # Comprehensive autograd guide & best practices
```

---

## 🧠 Core Concepts & Mechanics

### 1. Dynamic Computation Graph (`calculating_gradients.py`)
When a tensor is declared with `requires_grad=True`, PyTorch begins tracking all mathematical operations applied to it.
- **Forward Pass**: Constructs an execution graph where input tensors are leaves and output nodes represent mathematical operations.
- **Backward Pass (`.backward()`)**: Traverses the DAG backward from the scalar output node, applying the chain rule of calculus to compute derivatives $\frac{\partial y}{\partial x}$.
- **Accessing Gradients (`.grad`)**: Stores the computed partial derivatives in the `.grad` attribute of leaf tensors.

#### Mathematical Example:
Given the scalar function:
$$y = 5x^4$$

The analytical derivative is:
$$\frac{dy}{dx} = 20x^3$$

For $x = 102$:
$$\frac{dy}{dx} = 20 \times (102)^3 = 20 \times 1,061,208 = 21,224,160$$

PyTorch computes this exact scalar derivative automatically upon calling `y.backward()`.

```python
x = torch.tensor(102.0, requires_grad=True)
y = 5 * x**4
y.backward()
print(x.grad)  # Outputs: tensor(21224160.)
```

---

### 2. Gradient Accumulation & Gradient Zeroing (`gradient_cleaning.py`)
In PyTorch, **gradients accumulate by default** whenever `.backward()` is called on subsequent operations.
- **Why?** Accumulating gradients allows effective batch sizes larger than GPU VRAM can fit (by calling `.backward()` over multiple micro-batches before stepping the optimizer).
- **The Catch in Standard Training**: In standard iterative optimization loops, old gradients from preceding epochs will corrupt new gradients if not cleared.
- **Gradient Zeroing**: `x.grad.zero_()` or `optimizer.zero_grad()` must be invoked before executing each new backward pass.

```python
x = torch.tensor(5.0, requires_grad=True)

# Step 1
y = x**2
y.backward()
print(x.grad)  # 2 * 5 = 10.0

# Without zeroing:
y = x**2
y.backward()
print(x.grad)  # 10.0 + 10.0 = 20.0 (Accumulated!)

# Correct practice: Clear gradients
x.grad.zero_()
y = x**2
y.backward()
print(x.grad)  # 10.0 (Fresh gradient)
```

---

## 🚀 How to Run

```bash
python 02_autograd/calculating_gradients.py
python 02_autograd/gradient_cleaning.py
```
