# CIFAR-10 Image Classification using PyTorch CNN

A deep learning computer vision project implementing a custom **Convolutional Neural Network (CNN)** in **PyTorch** to classify images from the **CIFAR-10** dataset into 10 distinct classes.

---

## 📌 Project Overview

This project demonstrates an end-to-end computer vision workflow using PyTorch:
- Automatic dataset downloading and preprocessing using `torchvision.datasets` and `torchvision.transforms`.
- Custom multi-layer **CNN Architecture** design for spatial feature extraction.
- GPU-accelerated (`cuda`) model training and evaluation loop over 100 epochs.
- Single-image inference pipeline on custom external images (e.g., custom PNG test images) with resizing, normalization, and Softmax class probability computation.

---

## 📁 Directory Structure

```text
CIFAR-10/
├── data/                       # Downloaded CIFAR-10 dataset (Git-ignored)
│   └── cifar-10-batches-py/
├── cat.png                     # Sample test image for custom inference
├── cat2.png                    # Sample test image for custom inference
├── truck1.png                  # Sample test image for custom inference
├── main.ipynb                  # Main Jupyter Notebook (Data loading, Model, Training, Inference)
└── README.md                   # Project documentation
```

---

## 📊 Dataset Details

The **CIFAR-10** dataset consists of 60,000 $32 \times 32$ color images across 10 categories, with 6,000 images per class:
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Target Classes**:
  1. `airplane`
  2. `automobile`
  3. `bird`
  4. `cat`
  5. `deer`
  6. `dog`
  7. `frog`
  8. `horse`
  9. `ship`
  10. `truck`

---

## 🏗️ Neural Network Architecture

The project builds a custom `CNNmodel` consisting of a feature extraction **backbone** and a dense **classifier**:

```python
class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Output: 16 x 16 x 16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Output: 32 x 8 x 8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)           # Output: 64 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

### Layer Summary
| Layer Type | Input Shape | Output Shape | Details |
| :--- | :--- | :--- | :--- |
| **Input Image** | `(3, 32, 32)` | - | RGB 3-channel image |
| **Conv Block 1** | `(3, 32, 32)` | `(16, 16, 16)` | Conv2D(3->16, k=3, p=1) + ReLU + MaxPool2D(2) |
| **Conv Block 2** | `(16, 16, 16)` | `(32, 8, 8)` | Conv2D(16->32, k=3, p=1) + ReLU + MaxPool2D(2) |
| **Conv Block 3** | `(32, 8, 8)` | `(64, 4, 4)` | Conv2D(32->64, k=3, p=1) + ReLU + MaxPool2D(2) |
| **Flatten** | `(64, 4, 4)` | `(1024)` | Reshape to 1D vector |
| **Dense 1** | `(1024)` | `(128)` | Linear(1024 -> 128) + ReLU |
| **Dense 2 (Output)**| `(128)` | `(10)` | Linear(128 -> 10) output logits |

---

## ⚙️ Hyperparameters & Training Config

- **Device**: CUDA GPU 
- **Batch Size**: 64
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.Adam(lr=0.001)`
- **Epochs**: 50

---

## 🚀 How to Run

### 1. Prerequisites & Environment Setup
Make sure Python 3.8+ and PyTorch are installed in your environment.

```bash
# Install required libraries
pip install torch torchvision matplotlib pillow
```

### 2. Launch the Notebook
Open and run `main.ipynb` in VS Code or Jupyter Lab:

```bash
jupyter notebook main.ipynb
```

### 3. Pipeline Steps in Notebook
1. **Load Data**: Automatically downloads CIFAR-10 data to `./data`.
2. **Visualize Batch**: Displays sample training images with target labels using `matplotlib`.
3. **Train Model**: Runs the optimization loop for 100 epochs and prints average epoch loss.
4. **Evaluate Model**: Computes loss on unseen test data in evaluation mode (`model.eval()`).
5. **Infer Custom Images**: Loads local PNG files (e.g., `cat2.png`), converts them to $32 \times 32$ tensor tensors, and prints predicted class labels.

---

## 🔮 Inference on Custom Images

To run inference on your own image:

```python
from torchvision.io import read_image
from torchvision import transforms

# Read image tensor
image = read_image("./cat2.png")

# Preprocessing transforms
transform_pipeline = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ConvertImageDtype(torch.float),
])

# Process image and add batch dimension
input_tensor = transform_pipeline(image).to(device).unsqueeze(0)

# Predict class
softmax = torch.nn.Softmax(dim=1)
predictions = softmax(model(input_tensor))
predicted_class_id = torch.argmax(predictions, dim=1).item()

print(f"Predicted Class: {classes[predicted_class_id]}")
```

---

