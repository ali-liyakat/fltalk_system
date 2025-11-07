# ---------------------------------------
# Logistic Regression Model Template (FLTalk-safe)
# ---------------------------------------
# - Compatible with your current client loop:
#     exec(model_code, globals()); train_local_model(X, y); update_model(global)
# - Robust for binary & multiclass
# - Keeps "classes" across rounds even if aggregator drops them
# ---------------------------------------

from sklearn.linear_model import LogisticRegression
import numpy as np

# Module-level state persists across rounds within the same client process
STATE = {
    "classes": None,       # list of class labels
    "n_features": None,    # int
    "last_local_weights": None  # for debugging / fallback
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

    # If classes are missing but we have STATE, reuse them.
    if classes is None and STATE["classes"] is not None:
        classes = STATE["classes"]

    # Align lengths: intercept size should match number of rows in coef
    if intercept.shape[0] not in (1, coef.shape[0]):
        # Fallback: if single intercept is given for multiclass, tile it
        if intercept.shape[0] == 1:
            intercept = np.repeat(intercept, coef.shape[0], axis=0)
        else:
            # As a last resort, trim/pad to match
            k = coef.shape[0]
            if intercept.shape[0] > k:
                intercept = intercept[:k]
            else:
                pad = np.zeros((k - intercept.shape[0],), dtype=coef.dtype)
                intercept = np.concatenate([intercept, pad], axis=0)

    # If classes length mismatches but exists, try to reconcile
    if classes is not None:
        if len(classes) != coef.shape[0]:
            # Try simple fixes: for binary with (1, n_features) coef, keep classes as-is.
            if coef.shape[0] == 1 and len(classes) in (1, 2):
                pass  # acceptable
            else:
                # Trim/pad classes for safety
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

    # Remember feature count for future checks
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

    # Persist classes for future rounds
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
    print("✅ Local Logistic Regression training complete.")
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
        print("⚠️ update_model: global_model not a dict; ignoring.")
        return STATE.get("last_local_weights", global_model)

    coef, intercept, classes = _shape_sanity(global_model)

    # If global n_features mismatches local data, warn (client preprocessing issue)
    if STATE["n_features"] is not None and coef.shape[1] != STATE["n_features"]:
        print(f"⚠️ update_model: feature mismatch (global {coef.shape[1]} != local {STATE['n_features']}). "
              "Ensure consistent preprocessing/columns across clients.")
        # We still proceed, but this likely indicates a user data issue.

    updated = {
        "type": "sklearn",
        "model": "logistic_regression",
        "coef": _as_list(coef),
        "intercept": _as_list(intercept),
        "classes": _as_list(classes) if classes is not None else STATE["classes"],
        "n_features": STATE["n_features"],
    }

    # Keep classes sticky across rounds even if aggregator drops them
    if updated["classes"] is None and STATE["classes"] is not None:
        updated["classes"] = STATE["classes"]

    # Persist as the last known synchronized weights
    STATE["last_local_weights"] = updated
    print("🔁 Logistic Regression model updated with global parameters.")
    return updated
