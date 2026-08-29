# Linear Regression from Scratch using PyTorch Autograd

An implementation of univariate Linear Regression built purely from fundamental mathematical equations and PyTorch's `autograd` engine without relying on high-level `torch.nn` modules or built-in optimizers.

---

## 📐 Mathematical Formulation

### 1. Hypothesis (Forward Pass)
$$\hat{y} = w \cdot x + b$$
Where:
- $w$ is the learnable weight (slope).
- $b$ is the learnable bias (intercept).

### 2. Objective Function (Mean Squared Error)
$$\mathcal{L}(w, b) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

### 3. Gradient Descent Updates
$$\begin{aligned}
w &\leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w} \\
b &\leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
\end{aligned}$$
Where $\eta$ is the learning rate.

---

## 🛠️ Step-by-Step Pipeline

1. **Parameter Initialization**: Initialize $w$ and $b$ as randomly sampled tensors with `requires_grad=True`.
2. **Prediction**: Compute linear prediction tensor $\hat{y}$.
3. **Loss Computation**: Calculate scalar MSE loss.
4. **Backpropagation**: Call `loss.backward()` to populate `w.grad` and `b.grad`.
5. **Parameter Update in `torch.no_grad()`**: Mutate weights and biases without tracking operations in the autograd history:
   ```python
   with torch.no_grad():
       w -= lr * w.grad
       b -= lr * b.grad
   ```
6. **Zero Gradients**: Explicitly invoke `w.grad.zero_()` and `b.grad.zero_()`.

---

## 🚀 Execution

```bash
python 07_linear_regression/01_autograd_scratch/regression.py
```
