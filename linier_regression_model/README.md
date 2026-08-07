# Linear Regression Model (PyTorch)

A simple PyTorch project demonstrating linear regression using a 2-layer neural network with a ReLU activation function.

## 📁 Files

- `Regression_model.py`: Defines the PyTorch neural network model, trains it on sample data, and saves the trained weights to `reg_model.pth`.
- `load_model.py`: Loads the trained model weights from `reg_model.pth` and runs inference for predictions.
- `reg_model.pth`: Saved model weights file (`state_dict`).

## ⚡ Requirements

- Python 3.x
- PyTorch (`pip install torch`)

## 🚀 How to Run

1. **Train and save the model:**
   ```bash
   python Regression_model.py
   ```

2. **Load model and make predictions:**
   ```bash
   python load_model.py
   ```
