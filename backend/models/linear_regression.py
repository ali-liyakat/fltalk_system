from sklearn.linear_model import LinearRegression

def train_local_model(X_train, y_train):
    """Train Linear Regression using provided dataset (client-side)."""
    model = LinearRegression().fit(X_train, y_train)
    print("✅ Local Linear Regression trained.")
    return {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_)
    }

def update_model(global_model):
    """Apply received global weights."""
    print("🔁 Updated local model using global weights.")
    return global_model
