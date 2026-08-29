# 06 - Dataset Splitting & DataLoaders (`random_split`)

Splitting data into disjoint training and evaluation subsets prevents data leakage and ensures objective validation on unseen test data. This module demonstrates loading tabular CSV data directly into a custom PyTorch `Dataset` and partitioning it using PyTorch's native `random_split`.

---

## 📁 Module Structure

```text
06_train_test_split/
├── random_splitter.py                           # Custom Dataset class + random_split + Train/Test DataLoaders
├── industrial_predictive_maintenance_data.csv    # Clean industrial predictive maintenance sensor dataset
└── README.md                                    # Data splitting mechanics & best practices
```

---

## 🧠 Core Concepts & Mechanics

### 1. The `random_split` Function
PyTorch provides `torch.utils.data.random_split` to randomly partition a `Dataset` into non-overlapping subsets based on lengths or fractions:

```python
from torch.utils.data import random_split

# Split dataset into 80% train and 20% test
test_split, train_split = random_split(dataset, [0.2, 0.8])
```

### 2. Creating Separate Training and Testing DataLoaders
Once partitioned, independent `DataLoader` instances are instantiated for each split:
- **Training DataLoader**: `shuffle=True` to introduce stochastic mini-batch ordering and prevent cyclical gradient bias.
- **Testing DataLoader**: `shuffle=False` for deterministic evaluation metrics.

```python
train_loader = DataLoader(train_split, batch_size=32, shuffle=True)
test_loader = DataLoader(test_split, batch_size=32, shuffle=False)
```

---

## 📊 Dataset Schema

| Column | Type | Description |
| :--- | :--- | :--- |
| `operating_hours` | Float | Machine total operating hours |
| `rotation_speed_rpm` | Float | Operational rotational speed |
| `torque_nm` | Float | Torque output |
| `power_kw` | Float | Power consumption |
| `ambient_temp_c` | Float | Surrounding temperature |
| `failure_mode_type` | Int (Target) | Classification label for failure category |

---

## 🚀 How to Run

```bash
python 06_train_test_split/random_splitter.py
```
