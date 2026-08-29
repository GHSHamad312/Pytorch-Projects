# Intel Image Classification: Custom CNN vs Transfer Learning (ResNet18)

An end-to-end Computer Vision project comparing a **Custom Convolutional Neural Network (CNN)** built from scratch against **Transfer Learning with a Pretrained ResNet-18** architecture to classify natural scenes across 6 distinct categories.

---

## 📌 Project Highlights

- **Dataset**: Intel Image Classification dataset containing ~25,000 $150 \times 150$ images partitioned into `train`, `val`, and `test` splits.
- **Multi-Model Benchmark**:
  - **Custom CNN**: Deep 4-stage convolutional backbone with Batch Normalization, Max Pooling, Dropout (`0.5`), and Cross-Entropy Loss with **Label Smoothing** (`0.1`).
  - **Transfer Learning (ResNet-18)**: Pretrained ImageNet backbone with fine-tuned `layer4` residual blocks and a customized 6-class linear classification head.
- **Performance Analysis**: Per-class accuracy benchmarks and comprehensive **Confusion Matrix Visualizations** using Scikit-Learn and Matplotlib.
- **Inference Pipeline**: Single-image prediction with torchvision image transforms.

---

## 📁 Directory Layout

```text
intel_image_classification/
├── main.ipynb               # Custom CNN pipeline (Data loading, Architecture, Training, Evaluation, Confusion Matrix)
├── transfer_learning.ipynb  # Transfer Learning pipeline using Pretrained ResNet-18 with fine-tuning
├── Data/                    # Segregated dataset folders (Git-ignored)
│   ├── seg_train/           # Training dataset (~14,034 images)
│   ├── seg_test/            # Testing dataset (~3,000 images)
│   └── seg_val/             # Validation dataset
└── README.md                # Project documentation
```

---

## 📊 Dataset Details

The dataset consists of natural landscape and urban scenes categorized into **6 classes**:
1. `buildings`
2. `forest`
3. `glacier`
4. `mountain`
5. `sea`
6. `street`

### Preprocessing & Data Augmentation Pipelines

#### Custom CNN Pipeline ($150 \times 150$):
- **Train Transforms**:
  - `transforms.Resize((150, 150))`
  - `transforms.RandomHorizontalFlip(p=0.5)`
  - `transforms.ColorJitter(brightness=0.2, contrast=0.2)`
  - `transforms.ToTensor()`
- **Test / Val Transforms**:
  - `transforms.Resize((150, 150))`
  - `transforms.ToTensor()`

#### Transfer Learning Pipeline ($224 \times 224$):
- **Train Transforms**:
  - `transforms.Resize((224, 224))`
  - `transforms.RandomHorizontalFlip(p=0.5)`
  - `transforms.ColorJitter(brightness=0.2, contrast=0.2)`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Test / Val Transforms**:
  - `transforms.Resize((224, 224))`
  - `transforms.ToTensor()`
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

---

## 🏗️ Model Architectures

### 1. Custom CNN Architecture (`main.ipynb`)

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 150x150 -> 75x75

            # Stage 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 75x75 -> 37x37

            # Stage 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 37x37 -> 18x18

            # Stage 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)           # 18x18 -> 9x9
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
```

---

### 2. Transfer Learning Architecture (`transfer_learning.ipynb`)

Utilizes a **ResNet-18** backbone pretrained on ImageNet with fine-tuning on the deepest residual block (`layer4`):

```python
# Load pretrained ResNet-18
model = models.resnet18(weights="DEFAULT")

# Freeze early feature extraction layers
for params in model.parameters():
    params.requires_grad = False

# Replace final classification head for 6 target classes
model.fc = nn.Linear(model.fc.in_features, 6)

# Unfreeze layer4 for domain-specific fine-tuning
for params in model.layer4.parameters():
    params.requires_grad = True
```

---

## ⚙️ Hyperparameters & Training Configuration

| Parameter | Custom CNN (`main.ipynb`) | Transfer Learning (`transfer_learning.ipynb`) |
| :--- | :--- | :--- |
| **Backbone** | Custom 4-stage ConvNet | Pretrained ResNet-18 |
| **Input Resolution** | $150 \times 150 \times 3$ | $224 \times 224 \times 3$ |
| **Batch Size** | 32 | 32 |
| **Optimizer** | Adam ($\text{lr} = 0.001$) | Adam ($\text{lr} = 0.001$) |
| **Loss Function** | `CrossEntropyLoss(label_smoothing=0.1)` | `CrossEntropyLoss()` |
| **Epochs** | 15 | 10 |
| **Fine-Tuned Layers** | All layers trained from scratch | `layer4` + `fc` head |

---

## 📈 Evaluation & Diagnostics

Both pipelines compute overall test accuracy, evaluate test loss, generate confusion matrices with `ConfusionMatrixDisplay`, and calculate granular per-class accuracy across all 6 classes:

```python
# Per-Class Accuracy Evaluation
print("--- Per-Class Accuracy ---")
for i, class_name in enumerate(train_dataset.classes):
    total_class = cm[i].sum()
    correct_class = cm[i, i]
    class_accuracy = 100 * correct_class / total_class if total_class > 0 else 0
    print(f"{class_name:<12}: {class_accuracy:.2f}%")
```

---

## 🚀 How to Run

### 1. Launch Custom CNN Training:
```bash
jupyter notebook 09_computer_vision/intel_image_classification/main.ipynb
```

### 2. Launch ResNet-18 Transfer Learning:
```bash
jupyter notebook 09_computer_vision/intel_image_classification/transfer_learning.ipynb
```
