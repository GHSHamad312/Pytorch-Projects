# 09 - Computer Vision with PyTorch & Torchvision

This module contains complete, end-to-end Computer Vision projects and architectures—ranging from image transformations and custom Convolutional Neural Networks (CNNs) to Transfer Learning with Pretrained ResNet backbones and Residual Network explorations.

---

## 📁 Sub-Projects & Modules

```text
09_computer_vision/
├── 01_image_transforms_basic.py     # Image loading (torchvision.io), tensor transforms, permute, visualization
├── cifar10_classification/           # Deep CNN on CIFAR-10 (82.54% accuracy, BatchNorm, Dropout, Inference)
├── intel_image_classification/      # Natural scene classification: Custom CNN vs ResNet18 Transfer Learning
├── residual_networks/               # Exploration of Residual Connections, Skip Connections & ResNet blocks
└── README.md                        # Overview of Computer Vision pipelines in PyTorch
```

---

## 🖼️ Computer Vision Concepts Covered

### 1. Image Tensor Conventions & Transforms (`01_image_transforms_basic.py`)
- **PyTorch Channel Convention**: `(Channels, Height, Width)` or `(C, H, W)` / `(B, C, H, W)`.
- **Matplotlib Channel Convention**: `(Height, Width, Channels)` or `(H, W, C)`.
- **Permutation**: `image.permute(1, 2, 0)` is used to reshape PyTorch tensors for plotting in Matplotlib.
- **Torchvision Transforms**: Composing spatial resizings (`transforms.Resize`), data augmentations (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`), and normalization (`transforms.Normalize`).

---

### 2. Project Spotlights

#### 🔹 [CIFAR-10 Image Classification](file:///09_computer_vision/cifar10_classification/README.md)
- **Dataset**: 60,000 $32 \times 32$ images across 10 classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Architecture**: 5-layer CNN backbone with Batch Normalization + 4-layer MLP classifier with Dropout (`p=0.5`).
- **Performance**: **82.54% Test Accuracy** with Confusion Matrix analysis and external image inference pipeline.

#### 🔹 [Intel Image Classification](file:///09_computer_vision/intel_image_classification/README.md)
- **Dataset**: ~25,000 natural scene images across 6 classes (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).
- **Comparative Study**:
  - **Custom CNN**: Deep multi-stage convolutional network with label smoothing (`nn.CrossEntropyLoss(label_smoothing=0.1)`).
  - **Transfer Learning (ResNet18)**: Pretrained ImageNet weights with fine-tuned layer 4 and custom classification head.

#### 🔹 [Residual Networks (ResNets)](file:///09_computer_vision/residual_networks/README.md)
- Deep residual learning formulation, identity shortcut connections ($F(x) + x$), and solving the vanishing gradient problem in deep neural architectures.

---

## 🚀 How to Run

```bash
# 1. Run basic image transforms
python 09_computer_vision/01_image_transforms_basic.py

# 2. Open CIFAR-10 notebook
jupyter notebook 09_computer_vision/cifar10_classification/main.ipynb

# 3. Open Intel Image Classification notebooks
jupyter notebook 09_computer_vision/intel_image_classification/main.ipynb
jupyter notebook 09_computer_vision/intel_image_classification/transfer_learning.ipynb
```
