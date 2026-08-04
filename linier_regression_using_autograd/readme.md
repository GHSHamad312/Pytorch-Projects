# Linear Regression using PyTorch Autograd

A simple implementation of Linear Regression from scratch using PyTorch's `autograd` without `torch.nn` or built-in optimizers.

## Model & Loss
- **Model**: $\hat{y} = w \cdot x + b$
- **Loss (MSE)**: $\text{Loss} = \text{mean}((\hat{y} - y)^2)$

## Core Steps
1. **Initialize**: Create $w$ and $b$ with `requires_grad=True`.
2. **Predict & Loss**: Compute predictions and Mean Squared Error.
3. **Backward Pass**: Call `loss.backward()` to calculate gradients.
4. **Update & Reset**: Update parameters within `torch.no_grad()` and clear gradients with `.grad.zero_()`.

## Run
```bash
python regression.py
```