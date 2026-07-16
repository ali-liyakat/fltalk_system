# ---------------------------------------
# Logistic Regression Model Template 
# ---------------------------------------
# - Robust for binary & multiclass
# ---------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import numpy as np


STATE = {
    "classes": None,       
    "n_features": None,    
    "last_local_weights": None
}

def _as_list(x):
    """Ensure JSON-serializable Python lists."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _ensure_2d_coef(coef):
    """
    Ensure coef is 2D: (n_classes, n_features)
    sklearn gives (1, n_features) for binary; keep it 2D for uniformity.
    """
    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return coef

def _ensure_1d_intercept(intercept):
    """
    Ensure intercept is 1D: (n_classes,)
    """
    intercept = np.asarray(intercept)
    if intercept.ndim == 0:
        intercept = intercept.reshape(1,)
    return intercept

def _shape_sanity(weights):
    """
    Basic shape checks to avoid silent mismatches.
    Returns normalized (coef, intercept, classes)
    """
    coef = _ensure_2d_coef(weights.get("coef", []))
    intercept = _ensure_1d_intercept(weights.get("intercept", []))
    classes = weights.get("classes", None)


    if classes is None and STATE["classes"] is not None:
        classes = STATE["classes"]


    if intercept.shape[0] not in (1, coef.shape[0]):
        
        if intercept.shape[0] == 1:
            intercept = np.repeat(intercept, coef.shape[0], axis=0)
        else:
            
            k = coef.shape[0]
            if intercept.shape[0] > k:
                intercept = intercept[:k]
            else:
                pad = np.zeros((k - intercept.shape[0],), dtype=coef.dtype)
                intercept = np.concatenate([intercept, pad], axis=0)

    
    if classes is not None:
        if len(classes) != coef.shape[0]:
            
            if coef.shape[0] == 1 and len(classes) in (1, 2):
                pass  
            else:
                
                classes = list(classes)[:coef.shape[0]]

    return coef, intercept, classes

def train_local_model(
    X_train,
    y_train,
    *,
    penalty="l2",
    C=1.0,
    max_iter=5000,
    solver="lbfgs",
    multi_class="auto",
    n_jobs=None,
    class_weight=None,
    fit_intercept=True,
    tol=1e-4,
):
    """
    Train Logistic Regression on provided local dataset (classification use case).
    Returns JSON-serializable weights including classes & feature count.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    
    STATE["n_features"] = int(X_train.shape[1])

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        solver=solver,
        multi_class=multi_class,
        n_jobs=n_jobs,
        class_weight=class_weight,
        fit_intercept=fit_intercept,
        tol=tol,
    )
    model.fit(X_train, y_train)

    coef = _ensure_2d_coef(model.coef_)
    intercept = _ensure_1d_intercept(model.intercept_)

    
    STATE["classes"] = _as_list(model.classes_)

    weights = {
        "type": "sklearn",
        "model": "logistic_regression",
        "coef": _as_list(coef),
        "intercept": _as_list(intercept),
        "classes": STATE["classes"],
        "n_features": STATE["n_features"],
    }

    STATE["last_local_weights"] = weights
    print("Local Logistic Regression training complete.")
    return weights

def update_model(global_model):
    """
    Update local logistic model using received global parameters.
    This is a lightweight function because sklearn LogisticRegression
    cannot be 'partially' updated with coef_/intercept_ without re-instantiation.
    Our convention: we simply keep the synced weights in STATE so the *next*
    local training round starts consistent w.r.t. class order & shapes.

    - Accepts global_model that may or may not include 'classes'.
    - If 'classes' missing (e.g., FedAvg stripped it), we reuse STATE['classes'].
    - Performs shape sanity & reconciliation.
    """
    if not isinstance(global_model, dict):
        print("update_model: global_model not a dict; ignoring.")
        return STATE.get("last_local_weights", global_model)

    coef, intercept, classes = _shape_sanity(global_model)

    
    if STATE["n_features"] is not None and coef.shape[1] != STATE["n_features"]:
        print(f"update_model: feature mismatch (global {coef.shape[1]} != local {STATE['n_features']}). "
              "Ensure consistent preprocessing/columns across clients.")
        

    updated = {
        "type": "sklearn",
        "model": "logistic_regression",
        "coef": _as_list(coef),
        "intercept": _as_list(intercept),
        "classes": _as_list(classes) if classes is not None else STATE["classes"],
        "n_features": STATE["n_features"],
    }

    
    if updated["classes"] is None and STATE["classes"] is not None:
        updated["classes"] = STATE["classes"]

    
    STATE["last_local_weights"] = updated
    print("Logistic Regression model updated with global parameters.")
    return updated

# def evaluate_model(global_model, X_test, y_test):
#     """Evaluate global Logistic Regression model on client local test set"""
#     try:
#         coef = np.array(global_model["coef"])
#         intercept = np.array(global_model["intercept"]).ravel()
#         classes = global_model.get("classes", [0, 1])

#         model = LogisticRegression()
#         model.classes_ = np.array(classes)
#         model.coef_ = coef
#         model.intercept_ = intercept

#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         return acc

#     except Exception as e:
#         print(f"evaluate_model failed: {e}")
#         return None

def evaluate_model(global_model, X_test, y_test):
    """Evaluate global Logistic Regression model on client local test set (accuracy + logloss)"""
    try:
        coef = np.array(global_model["coef"])
        intercept = np.array(global_model["intercept"]).ravel()
        classes = global_model.get("classes", [0, 1])

        model = LogisticRegression()
        model.classes_ = np.array(classes)
        model.coef_ = coef
        model.intercept_ = intercept

        # Accuracy
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Loss (Log Loss) needs probabilities
        proba = model.predict_proba(X_test)

        # Make sure labels match the class order used in proba columns
        loss = log_loss(y_test, proba, labels=model.classes_)

        # ✅ return BOTH
        return float(acc), float(loss)

    except Exception as e:
        print(f"evaluate_model failed: {e}")
        return None

