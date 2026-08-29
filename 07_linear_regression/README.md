# 07 - Linear Regression: From Scratch to `torch.nn`

This module provides a side-by-side progression of Linear Regression in PyTorch:
1. **From Scratch using Pure Autograd**: Deriving forward calculations, Mean Squared Error (MSE) loss, manual backpropagation, and gradient descent parameter updates in `torch.no_grad()`.
2. **Modular Implementation using `torch.nn` & `torch.optim`**: Implementing a 2-layer neural network with `nn.Linear`, `nn.ReLU`, Adam optimization, and model checkpoint persistence (`torch.save` / `torch.load`).

---

## 📁 Sub-Module Structure

```text
07_linear_regression/
├── 01_autograd_scratch/
│   ├── regression.py     # Pure autograd implementation (w, b, manual gradient update)
│   └── README.md         # Derivations, step-by-step autograd walkthrough
├── 02_nn_module/
│   ├── regression_model.py # nn.Module architecture, training loop, checkpoint saving
│   ├── load_model.py       # Loading saved weights (state_dict) and running inference
│   ├── reg_model.pth       # Saved model checkpoint (state_dict weights)
│   └── README.md           # Model serialization and production inference guide
└── README.md             # Comparative architecture & overview
```

---

## ⚖️ Comparison: Pure Autograd vs `torch.nn` Pipeline

| Component | Pure Autograd Approach | `torch.nn` Module Approach |
| :--- | :--- | :--- |
| **Parameters** | `w = torch.randn(1, requires_grad=True)` | Managed inside `nn.Linear(in, out)` |
| **Model** | Explicit function $\hat{y} = w \cdot x + b$ | Subclass of `nn.Module` with `forward()` |
| **Loss** | Manual formula `((y_pred - y)**2).mean()` | Standard `nn.MSELoss()` criterion |
| **Gradients** | `loss.backward()` then `w.grad` | `loss.backward()` |
| **Update** | Manual formula `w -= lr * w.grad` inside `no_grad()` | `optimizer.step()` (e.g. `optim.Adam` / `SGD`) |
| **Zero Grads**| `w.grad.zero_()` | `optimizer.zero_grad()` |
| **Checkpointing**| Manual tensor saving | `torch.save(model.state_dict(), path)` |

---

## 🚀 How to Run

```bash
# Run Autograd from scratch
python 07_linear_regression/01_autograd_scratch/regression.py

# Train and save the nn.Module model
python 07_linear_regression/02_nn_module/regression_model.py

# Load checkpoint and run inference
python 07_linear_regression/02_nn_module/load_model.py
```
