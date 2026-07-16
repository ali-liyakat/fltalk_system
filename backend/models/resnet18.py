
"""
ResNet-18 (Head-only Federated Learning)
- Backbone frozen
- Only classifier head (fc.weight, fc.bias) is trained and shared
- JSON-friendly (small payload)
- Compatible with FLTalk: train_local_model(), update_model(), evaluate_model()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18

model = None
_meta = {
    "num_classes": 10,
    "head_only": True,
    "seed": 42,
}

def _ensure_nchw_float(X):
    X = np.array(X)
    if X.ndim != 4:
        raise ValueError(f"Expected 4D input, got shape: {X.shape}")

    # NHWC -> NCHW
    if X.shape[-1] == 3 and X.shape[1] != 3:
        X = np.transpose(X, (0, 3, 1, 2))

    if X.shape[1] != 3:
        raise ValueError(f"Expected channel dim=3, got shape: {X.shape}")

    X = X.astype(np.float32)
    if X.max() > 1.0:
        X = X / 255.0
    return X

def _ensure_model(num_classes: int):
    global model

    if model is None:
        torch.manual_seed(_meta["seed"])  # keep same init across clients
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        # Freeze backbone (everything except fc)
        for name, p in m.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

        model = m

def _get_head_state_dict(m):
    # Only fc params
    sd = m.state_dict()
    head = {
        "fc.weight": sd["fc.weight"].detach().cpu().numpy().tolist(),
        "fc.bias": sd["fc.bias"].detach().cpu().numpy().tolist(),
    }
    return head

def _load_head_state_dict(m, head_sd):
    with torch.no_grad():
        m.fc.weight.copy_(torch.tensor(head_sd["fc.weight"], dtype=torch.float32))
        m.fc.bias.copy_(torch.tensor(head_sd["fc.bias"], dtype=torch.float32))

def train_local_model(X_train, y_train, epochs=1, lr=0.001, batch_size=32):
    global _meta

    X_train = _ensure_nchw_float(X_train)
    y_train = np.array(y_train).astype(np.int64)

    num_classes = int(max(_meta.get("num_classes", 2), len(np.unique(y_train))))
    _meta["num_classes"] = num_classes

    _ensure_model(num_classes)
    device = torch.device("cpu")
    model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Optimizer only for fc params
    opt = optim.Adam(model.fc.parameters(), lr=lr)
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
        "arch": "resnet18",
        "num_classes": _meta["num_classes"],
        "head_only": True,
        "head_state_dict": _get_head_state_dict(model),
    }

def update_model(global_weights):
    global model, _meta

    num_classes = int(global_weights.get("num_classes", 10))
    _meta["num_classes"] = num_classes

    _ensure_model(num_classes)

    # Load only head
    head_sd = global_weights.get("head_state_dict", None)
    if head_sd is None:
        raise KeyError("global_weights missing 'head_state_dict'")

    _load_head_state_dict(model, head_sd)




# --------------------------------------------------
# EVALUATE MODEL (Accuracy + Loss) — ResNet18
# --------------------------------------------------
def evaluate_model(global_weights, X_test, y_test):
    try:
        # Load global weights into model
        update_model(global_weights)

        # Prepare data
        X_test = _ensure_nchw_float(X_test)
        y_test_np = np.array(y_test).astype(np.int64)

        device = torch.device("cpu")
        model.to(device)
        model.eval()

        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_test_np, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = model(X_t)               # raw outputs
            loss = criterion(logits, y_t)     # 🔹 compute loss
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test_np, preds)

       
        return float(acc), float(loss.item())

    except Exception as e:
        print(f"evaluate_model failed: {e}")
        return None
