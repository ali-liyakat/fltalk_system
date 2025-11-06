# ---------------------------------------
# Logistic Regression Model Template (for FLTalk)
# ---------------------------------------
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_local_model(X_train, y_train):
    """
    Train Logistic Regression on provided local dataset (classification use case).
    User handles preprocessing & encoding.
    """
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    print("✅ Local Logistic Regression training complete.")
    return {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist()
    }

def update_model(global_model):
    """
    Update local logistic model using global weights.
    """
    print("🔁 Logistic Regression model updated with global parameters.")
    return global_model
