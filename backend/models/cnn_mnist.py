"""
CNN for MNIST (28x28 grayscale)
FLTalk-compatible model file.

Required functions:
- train_local_model(X_train, y_train, epochs, lr, batch_size)
- update_model(global_weights)
- evaluate_model(global_weights, X_test, y_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class CNNMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                            # 28x28 -> 14x14
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Globals
model = None
num_classes_global = None


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def _ensure_mnist_shape(X):
    """
    Accepts:
      - (N, 784) -> reshape to (N, 1, 28, 28)
      - (N, 28, 28) -> reshape to (N, 1, 28, 28)
      - (N, 1, 28, 28) -> ok
    Returns float32 torch tensor in (N,1,28,28).
    """
    X = np.array(X)

    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 1, 28, 28)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        X = X.reshape(-1, 1, 28, 28)
    elif X.ndim == 4 and X.shape[1:] == (1, 28, 28):
        pass
    else:
        raise ValueError(f"Unsupported MNIST input shape: {X.shape}. Expected (N,784) or (N,28,28) or (N,1,28,28).")

    # Normalize if not already [0,1]
    X = X.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0

    return torch.tensor(X, dtype=torch.float32)


def _state_dict_to_lists(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().cpu().numpy().tolist()
    return out


def _lists_to_state_dict(sd_lists):
    out = {}
    for k, v in sd_lists.items():
        out[k] = torch.tensor(v, dtype=torch.float32)
    return out


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
def train_local_model(X_train, y_train, epochs=1, lr=0.001, batch_size=32):
    global model, num_classes_global

    X_train_t = _ensure_mnist_shape(X_train)
    y_train = np.array(y_train).astype(np.int64)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    num_classes_global = int(len(np.unique(y_train)))

    if model is None:
        model = CNNMNIST(num_classes=num_classes_global)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return {
        "type": "pytorch",
        "state_dict": _state_dict_to_lists(model.state_dict()),
        "num_classes": num_classes_global
    }


# -------------------------------------------------------
# UPDATE (optional use in your client; safe to keep)
# -------------------------------------------------------
def update_model(global_weights):
    global model, num_classes_global

    num_classes_global = int(global_weights.get("num_classes", num_classes_global))

    if model is None:
        model = CNNMNIST(num_classes=num_classes_global)

    sd = _lists_to_state_dict(global_weights["state_dict"])
    model.load_state_dict(sd)


# --------------------------------------------------
# EVALUATE MODEL (Accuracy + Loss) — CNN MNIST
# --------------------------------------------------
def evaluate_model(global_weights, X_test, y_test):
    global model, num_classes_global

    try:
        # Load global into model for evaluation
        num_classes_global = int(global_weights.get("num_classes", num_classes_global))

        if model is None:
            model = CNNMNIST(num_classes=num_classes_global)

        sd = _lists_to_state_dict(global_weights["state_dict"])
        model.load_state_dict(sd)
        model.eval()

        # Prepare data
        X_test_t = _ensure_mnist_shape(X_test)
        y_test_np = np.array(y_test).astype(np.int64)
        y_test_t = torch.tensor(y_test_np, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = model(X_test_t)              # raw outputs
            loss = criterion(logits, y_test_t)    # compute loss
            preds = torch.argmax(logits, dim=1)   # predictions

        acc = accuracy_score(y_test_np, preds.cpu().numpy())

        # ✅ RETURN BOTH
        return float(acc), float(loss.item())

    except Exception as e:
        print(f"evaluate_model failed: {e}")
        return None
