# Student Pass/Fail Classification using PyTorch MLP

An end-to-end PyTorch tabular classification pipeline that predicts whether a student will pass or fail based on academic habits and behavioral features.

---

## 📁 Project Structure

```text
pass_fail_classifier/
├── classifier.py     # Complete PyTorch training & inference script
├── students.csv      # Tabular dataset containing student features and binary targets
└── README.md         # Comprehensive project documentation
```

---

## 📊 Dataset Overview

The dataset (`students.csv`) contains continuous and integer academic telemetry features:
- **`study_hours`**: Number of hours spent studying per week.
- **`attendance_pct`**: Class attendance percentage.
- **`previous_score`**: Historical test score percentage.
- **`sleep_hours`**: Average hours of sleep per night.
- **`passed`** *(Target)*: Binary label (`0` = Fail, `1` = Pass).

---

## 🏗️ Neural Network Architecture

A 3-layer Multi-Layer Perceptron (MLP) built with PyTorch's `nn.Module`:

```python
class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
```

### Architecture Specifications:
| Layer | Type | Input Dim | Output Dim | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Input $\to$ Hidden 1** | `nn.Linear` | 4 | 16 | `nn.ReLU()` |
| **Hidden 1 $\to$ Hidden 2** | `nn.Linear` | 16 | 32 | `nn.ReLU()` |
| **Hidden 2 $\to$ Output** | `nn.Linear` | 32 | 2 (Class Logits) | Linear (Logits) |

---

## ⚙️ Training Setup & Hyperparameters

- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.Adam(model.parameters(), lr=0.001)`
- **Batch Size**: 3
- **Epochs**: 200
- **Evaluation**: Probability distribution generated via `nn.Softmax()` on raw output logits.

---

## 🚀 How to Run

```bash
python 08_classification_projects/pass_fail_classifier/classifier.py
```
