# CIFAR-10 Image Classification using PyTorch CNN

A deep learning computer vision project implementing an enhanced **Convolutional Neural Network (CNN)** in **PyTorch** with **Batch Normalization**, **Data Augmentation**, and **Dropout Regularization** to classify images from the **CIFAR-10** dataset into 10 distinct classes.

---

## 📌 Project Overview

This project demonstrates an end-to-end computer vision workflow using PyTorch:
- **Dataset Preprocessing & Data Augmentation**: Utilizes `torchvision.transforms` for Random Crop, Random Horizontal Flip, and dataset-specific channel normalization (`mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`).
- **Deep CNN Architecture**: 5 convolutional layers organized into 3 feature-extraction blocks with **Batch Normalization** (`BatchNorm2d`) after each convolution, Max Pooling, and a multi-layer classifier with **Dropout** (`p=0.5`) for regularization.
- **GPU-Accelerated Training**: Model training loop executed on CUDA GPU over 30 epochs with Adam optimizer and Cross-Entropy Loss, achieving steady loss reduction from ~1.58 to ~0.41.
- **Model Evaluation & Diagnostics**: Achieves **82.54% test accuracy** on the 10,000 unseen test images with a comprehensive **Confusion Matrix** visualization using `scikit-learn` and `matplotlib`.
- **Custom Image Inference Pipeline**: Predicts classes for external sample images (e.g., `plane.png`, `cat.png`, `truck1.png`) with proper tensor resizing, dtype conversion, and normalization.

---

## 📁 Directory Structure

```text
CIFAR-10/
├── data/                       # Downloaded CIFAR-10 dataset (Git-ignored)
│   └── cifar-10-batches-py/
├── cat.png                     # Sample test image for custom inference
├── cat2.png                    # Sample test image for custom inference
├── plane.png                   # Sample test image for custom inference
├── plane2.png                  # Sample test image for custom inference
├── truck1.png                  # Sample test image for custom inference
├── main.ipynb                  # Main Jupyter Notebook (Data loading, Model, Training, Evaluation, Inference)
└── README.md                   # Project documentation
```

---

## 📊 Dataset Details

The **CIFAR-10** dataset consists of 60,000 $32 \times 32$ color images across 10 categories (6,000 images per class):
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

### Transforms & Data Augmentation
- **Training Pipeline**:
  - `transforms.ToTensor()`
  - `transforms.RandomCrop(32, padding=4)`
  - `transforms.RandomHorizontalFlip()`
  - `transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`
- **Test Pipeline**:
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`

---

## 🏗️ Neural Network Architecture

The network consists of a deep convolutional **backbone** with Batch Normalization and an MLP **classifier** with Dropout:

```python
class CNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Output: 16 x 16 x 16

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # Output: 32 x 8 x 8

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)           # Output: 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
```

### Detailed Layer Summary
| Stage | Layer Type | Input Shape | Output Shape | Details |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | RGB Image | `(3, 32, 32)` | `(3, 32, 32)` | 3-channel normalized input |
| **Block 1** | Conv2d + BatchNorm + ReLU | `(3, 32, 32)` | `(16, 32, 32)` | Conv(3 $\to$ 16, k=3, p=1) + BN(16) + ReLU |
| | MaxPool2d | `(16, 32, 32)` | `(16, 16, 16)` | MaxPool2d(kernel_size=2, stride=2) |
| **Block 2** | Conv2d + BatchNorm + ReLU | `(16, 16, 16)` | `(32, 16, 16)` | Conv(16 $\to$ 32, k=3, p=1) + BN(32) + ReLU |
| | Conv2d + BatchNorm + ReLU | `(32, 16, 16)` | `(32, 16, 16)` | Conv(32 $\to$ 32, k=3, p=1) + BN(32) + ReLU |
| | MaxPool2d | `(32, 16, 16)` | `(32, 8, 8)` | MaxPool2d(kernel_size=2, stride=2) |
| **Block 3** | Conv2d + BatchNorm + ReLU | `(32, 8, 8)` | `(64, 8, 8)` | Conv(32 $\to$ 64, k=3, p=1) + BN(64) + ReLU |
| | Conv2d + BatchNorm + ReLU | `(64, 8, 8)` | `(128, 8, 8)` | Conv(64 $\to$ 128, k=3, p=1) + BN(128) + ReLU |
| | MaxPool2d | `(128, 8, 8)` | `(128, 4, 4)` | MaxPool2d(kernel_size=2, stride=2) |
| **Classifier**| Flatten | `(128, 4, 4)` | `(2048)` | Reshapes spatial feature maps to 1D |
| | Linear + ReLU | `(2048)` | `(512)` | Dense layer + ReLU activation |
| | Linear + ReLU | `(512)` | `(256)` | Dense layer + ReLU activation |
| | Linear + ReLU | `(256)` | `(128)` | Dense layer + ReLU activation |
| | Dropout | `(128)` | `(128)` | Dropout rate $p = 0.5$ |
| | Linear (Output) | `(128)` | `(10)` | Output logits for 10 classes |

---

## ⚙️ Hyperparameters & Training Setup

- **Device**: CUDA GPU (`torch.device('cuda')`)
- **Batch Size**: 64
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.Adam(model.parameters(), lr=0.001)`
- **Epochs**: 30
- **Training Progression**:
  - Epoch 1/30 Loss: `1.5841`
  - Epoch 10/30 Loss: `0.6826`
  - Epoch 20/30 Loss: `0.5008`
  - Epoch 30/30 Loss: `0.4138`

---

## 📈 Evaluation & Results

- **Test Accuracy**: **82.54%** across all 10,000 test set images.
- **Evaluation Loop**: Evaluates without gradient computation (`torch.no_grad()`) in evaluation mode (`model.eval()`).
- **Confusion Matrix**: Generates a 10-class confusion matrix using `sklearn.metrics.confusion_matrix` and `matplotlib.pyplot` to visualize model predictions across all target classes.

```python
model.eval()
total = 0
correct = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

---

## 🚀 How to Run

### 1. Prerequisites & Environment Setup
Ensure Python 3.8+ and PyTorch are installed in your environment:

```bash
# Install required libraries
pip install torch torchvision matplotlib pillow scikit-learn
```

### 2. Launch the Notebook
Open and run `main.ipynb` in VS Code or Jupyter Lab:

```bash
jupyter notebook main.ipynb
```

### 3. Pipeline Execution Flow
1. **Imports & Setup**: Loads PyTorch, Torchvision transforms, DataLoader, Matplotlib, and Scikit-learn.
2. **Data Augmentation & Loading**: Prepares train and test datasets with CIFAR-10 normalization and augmentation.
3. **Data Inspection**: Visualizes sample training batch images with their assigned class labels.
4. **Model Initialization**: Constructs `CNNmodel` with BatchNorm and Dropout layers and mounts onto GPU (`cuda`).
5. **Training Loop**: Optimizes weights over 30 epochs and logs batch loss per epoch.
6. **Testing & Confusion Matrix**: Calculates overall test accuracy (**82.54%**) and plots the confusion matrix heatmap.
7. **Custom Inference**: Loads a local image (e.g. `plane.png`), applies preprocessing and normalization transforms, and generates class predictions.

---

## 🔮 Inference on Custom Images

To run inference on an external image file (e.g., `plane.png` or `cat2.png`):

```python
from torchvision.io import read_image
from torchvision import transforms
import torch
import torch.nn as nn

# Load image tensor
image = read_image("./plane.png")

# Preprocessing & Normalization pipeline matching training distribution
transform_pipeline = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# Process image and add batch dimension
model.eval()
input_tensor = transform_pipeline(image).to(device).unsqueeze(0)

# Predict class
softmax = nn.Softmax(dim=1)
predictions = softmax(model(input_tensor))
predicted_class_id = torch.argmax(predictions, dim=1).item()

print(f"Predicted Class: {classes[predicted_class_id]}")
```

---
