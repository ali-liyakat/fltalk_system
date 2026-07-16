"""
MLP for MNIST (2-layer Fully Connected Network)

Accepts MNIST inputs in any of these shapes:
- (N, 28, 28)
- (N, 1, 28, 28)
- (N, 784)

Required functions:
- train_local_model(X_train, y_train, epochs=..., lr=..., batch_size=...)
- update_model(global_weights)
- evaluate_model(global_weights, X_test, y_test)   # LR-style signature
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score


# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class MLP2LayerMNIST(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Globals
model = None
_meta = {
    "input_dim": 784,
    "hidden_dim": 128,
    "num_classes": 10,
}


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def _ensure_mnist_flat(X):
    """
    Converts MNIST inputs to float32 features (N,784).
    Accepts:
      - (N, 28, 28)
      - (N, 1, 28, 28)
      - (N, 784)
    """
    X = np.array(X)

    if X.ndim == 4 and X.shape[1] == 1:         # (N,1,28,28)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):  # (N,28,28)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2 and X.shape[1] == 784:     # (N,784)
        pass
    else:
        raise ValueError(f"Unsupported MNIST input shape: {X.shape}")

    X = X.astype(np.float32)

    # If values look like 0..255, normalize to 0..1
    if X.max() > 1.0:
        X = X / 255.0

    return X


def _state_dict_to_lists(sd):
    return {k: v.detach().cpu().numpy().tolist() for k, v in sd.items()}


def _lists_to_state_dict(sd_lists):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in sd_lists.items()}


def _ensure_model():
    global model
    if model is None:
        model = MLP2LayerMNIST(
            input_dim=_meta["input_dim"],
            hidden_dim=_meta["hidden_dim"],
            num_classes=_meta["num_classes"],
        )


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
def train_local_model(X_train, y_train, epochs=1, lr=0.001, batch_size=32):
    """
    Trains local MLP on MNIST-style data.
    Returns JSON-safe weights dict for FLTalk.
    """
    global _meta

    X_train = _ensure_mnist_flat(X_train)
    y_train = np.array(y_train).astype(np.int64)

    # MNIST is fixed 10 classes, but keep this robust
    _meta["num_classes"] = int(max(10, len(np.unique(y_train))))
    _ensure_model()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return {
        "type": "pytorch",
        "state_dict": _state_dict_to_lists(model.state_dict()),
        "input_dim": _meta["input_dim"],
        "hidden_dim": _meta["hidden_dim"],
        "num_classes": _meta["num_classes"],
    }


# -------------------------------------------------------
# UPDATE
# -------------------------------------------------------
def update_model(global_weights):
    """
    Loads global weights into local model (lists -> tensors).
    """
    global _meta, model

    # Update metadata (keep defaults if missing)
    _meta["input_dim"] = int(global_weights.get("input_dim", 784))
    _meta["hidden_dim"] = int(global_weights.get("hidden_dim", 128))
    _meta["num_classes"] = int(global_weights.get("num_classes", 10))

    # Recreate model if needed (or if shape changed)
    model = MLP2LayerMNIST(
        input_dim=_meta["input_dim"],
        hidden_dim=_meta["hidden_dim"],
        num_classes=_meta["num_classes"],
    )

    sd = _lists_to_state_dict(global_weights["state_dict"])
    model.load_state_dict(sd)


# # -------------------------------------------------------
# # EVALUATE
# # -------------------------------------------------------
# def evaluate_model(global_weights, X_test, y_test):
#     """
#     Evaluates global MLP model on client local test set.
#     """
#     update_model(global_weights)

#     X_test = _ensure_mnist_flat(X_test)
#     y_test = np.array(y_test).astype(np.int64)

#     X_t = torch.tensor(X_test, dtype=torch.float32)

#     model.eval()
#     with torch.no_grad():
#         logits = model(X_t)
#         preds = torch.argmax(logits, dim=1).cpu().numpy()

#     return float(accuracy_score(y_test, preds))


# --------------------------------------------------
# EVALUATE MODEL (Accuracy + Loss) — MLP MNIST
# --------------------------------------------------
def evaluate_model(global_weights, X_test, y_test):
    """
    Evaluates global MLP model on client local test set (MNIST).
    Returns: (accuracy, loss)
    """
    try:
        # Load global weights into model
        update_model(global_weights)

        # Prepare data
        X_test = _ensure_mnist_flat(X_test)
        y_test = np.array(y_test).astype(np.int64)

        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()

        model.eval()
        with torch.no_grad():
            logits = model(X_t)              # 🔹 raw outputs
            loss = criterion(logits, y_t)    # 🔹 compute loss
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)

        # ✅ RETURN BOTH
        return float(acc), float(loss.item())

    except Exception as e:
        print(f"evaluate_model failed: {e}")
        return None
