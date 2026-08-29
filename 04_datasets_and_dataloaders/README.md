# 04 - Datasets & DataLoaders (`torch.utils.data`)

PyTorch decouples dataset storage from data batching and multiprocessing using two core primitives: `Dataset` and `DataLoader`.

---

## 📁 Module Structure

```text
04_datasets_and_dataloaders/
├── datasets.py               # TensorDataset pairing and DataLoader mini-batch iteration
├── custom_dataset_class.py   # Canonical template for subclassing torch.utils.data.Dataset
└── README.md                 # Complete guide to custom datasets and batch pipelines
```

---

## 🧠 Core Concepts & Mechanics

### 1. `Dataset` vs `DataLoader`
- **`Dataset`**: Represents a collection of samples and labels, responsible for loading individual items by index $(x_i, y_i)$.
- **`DataLoader`**: A Python iterable wrapping a `Dataset` to provide automated mini-batching, shuffling, multiprocessing (`num_workers`), and pinned memory transfer (`pin_memory=True`).

```mermaid
graph LR
    A["Raw Data (CSV, Images, Arrays)"] --> B["PyTorch Dataset (__getitem__)"]
    B --> C["DataLoader (Batching, Shuffling, Multi-workers)"]
    C --> D["Model Training Loop (Mini-batches)"]
```

---

### 2. Built-in `TensorDataset` (`datasets.py`)
When input features and targets are already loaded into memory as PyTorch tensors, `TensorDataset` pairs them into an indexed dataset:

```python
from torch.utils.data import TensorDataset, DataLoader

# Pair features and targets
dataset = TensorDataset(x_tensor, y_tensor)

# Wrap with DataLoader for batching and shuffling
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_x, batch_y in loader:
    # Process batch
    pass
```

---

### 3. Creating Custom Datasets (`custom_dataset_class.py`)
To load custom file formats, images, audio, or tabular files, subclass `torch.utils.data.Dataset` and implement three methods:

1. **`__init__(self, ...)`**: Initialize file paths, metadata, transforms, or load tabular data frames.
2. **`__len__(self)`**: Return the total number of samples in the dataset (`len(self.data)`).
3. **`__getitem__(self, index)`**: Given an integer index, retrieve, preprocess, and return the corresponding sample and label as PyTorch tensors `(x, y)`.

```python
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
```

---

## 🚀 How to Run

```bash
python 04_datasets_and_dataloaders/datasets.py
python 04_datasets_and_dataloaders/custom_dataset_class.py
```
