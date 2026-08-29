# 03 - Neural Network Modules (`torch.nn`)

The `torch.nn` package provides modular building blocks—layers, activation functions, loss criteria, and containers—for constructing deep neural networks in PyTorch.

---

## 📁 Module Structure

```text
03_neural_networks/
├── simple_linear_layer.py   # Subclassing nn.Module, Linear transformations, parameter inspection
├── activations.py           # Multi-layer Perceptron (MLP) with non-linear activation functions (Sigmoid)
├── loss_functions.py        # Criterion metrics: Mean Squared Error (MSE) and Cross-Entropy Loss
└── README.md                # Detailed guide on layers, architectures, and loss criteria
```

---

## 🧠 Core Concepts & Mechanics

### 1. Subclassing `nn.Module` & Linear Layers (`simple_linear_layer.py`)
All custom neural network architectures in PyTorch subclass `nn.Module`.
- **`__init__()`**: Defines and initializes structural submodules, such as `nn.Linear(in_features, out_features)`.
- **`forward(x)`**: Implements the computation graph defining how input tensors flow through the layers.
- **`model.named_parameters()`**: Iterates through learnable weight matrices ($W$) and bias vectors ($b$).

#### Linear Layer Transformation:
$$y = x W^T + b$$

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=2)
    
    def forward(self, x):
        return self.linear(x)
```

---

### 2. Activation Functions (`activations.py`)
Linear layers alone can only model linear mappings. Non-linear activation functions allow neural networks to approximate complex, non-linear functions:
- **Sigmoid (`nn.Sigmoid`)**: Squeezes activations into range $(0, 1)$, historically used for probability mapping:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **ReLU (`nn.ReLU`)**: Rectified Linear Unit thresholding negative values to zero ($\max(0, z)$), providing efficient gradient propagation in deep networks.

---

### 3. Loss Functions & Optimization Criteria (`loss_functions.py`)
Loss functions quantify the discrepancy between model predictions ($\hat{y}$) and ground truth targets ($y$):

| Loss Criterion | PyTorch Class | Task Type | Formula |
| :--- | :--- | :--- | :--- |
| **Mean Squared Error** | `nn.MSELoss()` | Regression | $\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$ |
| **Cross-Entropy Loss** | `nn.CrossEntropyLoss()` | Multi-Class Classification | $\mathcal{L} = -\sum_{k} y_k \log(\text{softmax}(\hat{y}_k))$ |

> **Note on `nn.CrossEntropyLoss`**: In PyTorch, `nn.CrossEntropyLoss` combines `nn.LogSoftmax()` and `nn.NLLLoss()` into a single numerically stable function. Raw unbounded logits (not softmax outputs) should be passed directly into this loss function.

---

## 🚀 How to Run

```bash
python 03_neural_networks/simple_linear_layer.py
python 03_neural_networks/activations.py
python 03_neural_networks/loss_functions.py
```
