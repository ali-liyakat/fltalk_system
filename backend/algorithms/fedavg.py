# ---------------------------------------
# FLTalk FedAvg Aggregation (Safe & Extensible)
# ---------------------------------------
# - Supports Sklearn weights: {"coef", "intercept", ...}
# - Supports Torch state_dict: {"layer.weight": [...], ...}
# - Ignores non-numeric fields (e.g., "classes", "n_features")
# - Automatically detects model type from weight dicts
# - Falls back to first client's weights if averaging is impossible
# ---------------------------------------

import numpy as np

def is_sklearn_weights(w):
    return isinstance(w, dict) and "coef" in w and "intercept" in w

def is_torch_weights(w):
    return isinstance(w, dict) and "state_dict" in w

def average_sklearn(weights_list):
    """FedAvg for sklearn-based models (e.g., Logistic/Linear Regression)."""
    try:
        coefs = np.array([w["coef"] for w in weights_list], dtype=object)
        intercepts = np.array([w["intercept"] for w in weights_list], dtype=object)

        # Average across clients (axis=0)
        avg_coef = np.mean(np.stack(coefs), axis=0).tolist()
        avg_intercept = np.mean(np.stack(intercepts), axis=0).tolist()

        # Preserve metadata from the first client's weights
        base = weights_list[0].copy()
        base.update({
            "coef": avg_coef,
            "intercept": avg_intercept
        })
        return base

    except Exception as e:
        print(f"⚠️ Sklearn FedAvg failed: {e} — returning first weights as fallback.")
        return weights_list[0]

def average_torch(weights_list):
    """FedAvg for PyTorch state_dict models (CNN, MLP, ResNet, etc.)."""
    try:
        # Collect the state_dicts
        sd_list = [w["state_dict"] for w in weights_list]

        # Initialize new global state_dict
        global_state = {}

        # Iterate over each key in the first model's state_dict
        for key in sd_list[0].keys():
            # Collect the weights across all clients for this key
            tensors = [np.array(sd[key]) for sd in sd_list]

            # Take element-wise mean
            avg_tensor = np.mean(tensors, axis=0)
            global_state[key] = avg_tensor.tolist()

        # Preserve metadata from the first model
        base = weights_list[0].copy()
        base["state_dict"] = global_state
        return base

    except Exception as e:
        print(f"⚠️ Torch FedAvg failed: {e} — returning first weights as fallback.")
        return weights_list[0]


def aggregate(weights_list):
    """Main dispatch function for FedAvg aggregation."""
    if not weights_list:
        print("⚠️ No client weights received — returning empty dict.")
        return {}

    sample = weights_list[0]

    # Detect type & route accordingly
    if is_sklearn_weights(sample):
        print("🧠 FedAvg applied to sklearn model.")
        return average_sklearn(weights_list)

    elif is_torch_weights(sample):
        print("🧠 FedAvg applied to torch model.")
        return average_torch(weights_list)

    else:
        print("⚠️ Unknown weight format — returning first weights.")
        return sample
