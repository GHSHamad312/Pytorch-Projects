# Linear Regression with `torch.nn` & Model Checkpointing

A modular implementation of a 2-layer Neural Network for regression built using PyTorch's `torch.nn` framework and `torch.optim.Adam`. It demonstrates training, metric tracking, saving model weights (`state_dict`), and reloading weights for inference.

---

## 🏗️ Model Architecture

The regression model is constructed with a non-linear hidden layer:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

### Layer Specifications
- **Input Layer**: `nn.Linear(1, 16)` — Projects 1D input into 16 hidden dimensions.
- **Activation**: `nn.ReLU()` — Introduces non-linearity.
- **Output Layer**: `nn.Linear(16, 1)` — Maps 16 hidden features back to a scalar output prediction.

---

## 💾 Model Serialization (`state_dict`)

PyTorch models store learnable parameters in a dictionary called `state_dict`.

### Saving Weights:
```python
import os
import torch

model_path = os.path.join(os.path.dirname(__file__), "reg_model.pth")
torch.save(model.state_dict(), model_path)
```

### Loading Weights & Inference:
```python
model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode

# Run forward pass on new test tensor
with torch.no_grad():
    prediction = model(torch.tensor([[20.0]]))
    print(f"Prediction: {prediction.item()}")
```

---

## 🚀 Execution

```bash
# 1. Train and export weights to reg_model.pth
python 07_linear_regression/02_nn_module/regression_model.py

# 2. Load trained weights and run inference
python 07_linear_regression/02_nn_module/load_model.py
```
