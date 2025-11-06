# ---------------------------------------
# FedAvg Algorithm (Flexible for Multiple Model Types)
# ---------------------------------------
import numpy as np

def aggregate(weights_list):
    """
    Federated Averaging (FedAvg) algorithm supporting multiple model formats.
    - Handles Linear / Logistic Regression (coef + intercept)
    - Ignores tree-based models (simulated combine)
    """
    if not weights_list:
        return {}

    # Detect model type automatically
    sample = weights_list[0]
    if "coef" in sample and "intercept" in sample:
        # For Linear / Logistic Regression
        avg_coef = np.mean([w["coef"] for w in weights_list], axis=0)
        avg_intercept = np.mean([w["intercept"] for w in weights_list], axis=0)
        print("FedAvg applied to regression-type models.")
        return {"coef": avg_coef.tolist(), "intercept": avg_intercept.tolist()}

    elif "params" in sample:
        # For Decision Tree / Random Forest (non-numeric params)
        print("FedAvg not applicable to tree models — returning first model.")
        return sample

    else:
        print("Unknown weight structure, returning first model as fallback.")
        return sample
