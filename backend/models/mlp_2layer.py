"""
MLP (2-layer Fully Connected Network)

Works for tabular CSV data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


# -------------------------------------------------------
# MODEL DEFINITION
# -------------------------------------------------------
class MLP2Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Globals used across functions
model = None
input_dim_global = None
num_classes_global = None


# -------------------------------------------------------
# TRAIN LOCAL MODEL
# -------------------------------------------------------
def train_local_model(X_train, y_train, epochs=5, lr=0.001, batch_size=32):
    global model, input_dim_global, num_classes_global

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    input_dim_global = X_train.shape[1]
    num_classes_global = len(np.unique(y_train.numpy()))

    # Build model
    model = MLP2Layer(
        input_dim=input_dim_global,
        hidden_dim=128,
        num_classes=num_classes_global
    )

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Return weights in JSON-safe form (list-based)
    sd = {}
    for k, v in model.state_dict().items():
        sd[k] = v.detach().cpu().numpy().tolist()

    return {
        "type": "pytorch",
        "state_dict": sd,
        "input_dim": input_dim_global,
        "num_classes": num_classes_global
    }


# -------------------------------------------------------
# UPDATE MODEL (LOAD GLOBAL WEIGHTS)
# -------------------------------------------------------
def update_model(global_weights):
    global model, input_dim_global, num_classes_global

    input_dim_global = global_weights.get("input_dim")
    num_classes_global = global_weights.get("num_classes")

    # Initialize empty model if needed
    if model is None:
        model = MLP2Layer(
            input_dim=input_dim_global,
            hidden_dim=128,
            num_classes=num_classes_global
        )

    # Convert lists → tensors
    new_sd = {}
    for k, v in global_weights["state_dict"].items():
        new_sd[k] = torch.tensor(v, dtype=torch.float32)

    model.load_state_dict(new_sd)


# # -------------------------------------------------------
# # EVALUATE MODEL
# # -------------------------------------------------------
# def evaluate_model(global_weights, X_test, y_test):
#     global model

#     # Load global weights first (list -> tensor)
#     new_sd = {}
#     for k, v in global_weights["state_dict"].items():
#         new_sd[k] = torch.tensor(v, dtype=torch.float32)

#     input_dim = global_weights.get("input_dim")
#     num_classes = global_weights.get("num_classes")

#     # Rebuild model if needed
#     if model is None:
#         model = MLP2Layer(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)

#     model.load_state_dict(new_sd)

#     # Convert data
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_test = torch.tensor(y_test, dtype=torch.long)

#     with torch.no_grad():
#         preds = model(X_test)
#         preds = torch.argmax(preds, dim=1)

#     return float(accuracy_score(y_test.numpy(), preds.numpy()))


# --------------------------------------------------
# EVALUATE MODEL (Accuracy + Loss)
# --------------------------------------------------
def evaluate_model(global_weights, X_test, y_test):
    global model

    try:
        # Load global weights first (list -> tensor)
        new_sd = {}
        for k, v in global_weights["state_dict"].items():
            new_sd[k] = torch.tensor(v, dtype=torch.float32)

        input_dim = global_weights.get("input_dim")
        num_classes = global_weights.get("num_classes")

        # Rebuild model if needed
        if model is None:
            model = MLP2Layer(
                input_dim=input_dim,
                hidden_dim=128,
                num_classes=num_classes
            )

        model.load_state_dict(new_sd)
        model.eval()

        # Convert data
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            logits = model(X_test)                 # 🔹 raw outputs
            loss = criterion(logits, y_test)       # 🔹 compute loss
            preds = torch.argmax(logits, dim=1)    # 🔹 predictions

        acc = accuracy_score(y_test.numpy(), preds.numpy())

        # ✅ RETURN BOTH
        return float(acc), float(loss.item())

    except Exception as e:
        print(f"evaluate_model failed: {e}")
        return None
