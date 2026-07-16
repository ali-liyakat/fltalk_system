"""
FedAvgM (Federated Averaging with Server Momentum)

Keeps velocity per experiment on server process memory.
Works for:
- sklearn coef/intercept
- pytorch state_dict
- pytorch head_state_dict (head-only)
"""

import numpy as np

# Server-side state (in federated_server process)
_STATE = {
    "velocity": None,   # same structure as weights
}

def _to_np(x):
    return np.array(x, dtype=np.float32)

def _avg(arrs):
    return np.mean(np.stack(arrs, axis=0), axis=0)

def _apply_momentum(v_prev, delta, beta):
    if v_prev is None:
        return delta
    return beta * v_prev + (1.0 - beta) * delta

def aggregate(weights_list, beta=0.9):
    if not weights_list:
        raise ValueError("No client weights received.")

    first = weights_list[0]

    # ---------- sklearn logistic regression ----------
    if "coef" in first and "intercept" in first:
        coefs = [_to_np(w["coef"]) for w in weights_list]
        ints  = [_to_np(w["intercept"]) for w in weights_list]

        avg_coef = _avg(coefs)
        avg_int  = _avg(ints)

        delta = {"coef": avg_coef, "intercept": avg_int}

        v_prev = _STATE["velocity"]
        if v_prev is None:
            v_new = {"coef": delta["coef"], "intercept": delta["intercept"]}
        else:
            v_new = {
                "coef": _apply_momentum(v_prev["coef"], delta["coef"], beta),
                "intercept": _apply_momentum(v_prev["intercept"], delta["intercept"], beta),
            }
        _STATE["velocity"] = v_new

        return {
            "type": "sklearn",
            "coef": v_new["coef"].tolist(),
            "intercept": v_new["intercept"].tolist(),
            "classes": first.get("classes", None),
        }

    # ---------- pytorch head-only ----------
    if "head_state_dict" in first:
        keys = first["head_state_dict"].keys()
        avg_head = {}
        for k in keys:
            vals = [_to_np(w["head_state_dict"][k]) for w in weights_list]
            avg_head[k] = _avg(vals)

        delta = {"head_state_dict": avg_head}

        v_prev = _STATE["velocity"]
        if v_prev is None:
            v_new = {"head_state_dict": avg_head}
        else:
            v_new = {"head_state_dict": {}}
            for k in keys:
                v_new["head_state_dict"][k] = _apply_momentum(
                    v_prev["head_state_dict"][k], avg_head[k], beta
                )

        _STATE["velocity"] = v_new

        out = {
            "type": "pytorch",
            "head_only": True,
            "head_state_dict": {k: v_new["head_state_dict"][k].tolist() for k in keys},
        }
        for mk in ["num_classes", "arch"]:
            if mk in first:
                out[mk] = first[mk]
        return out

    # ---------- pytorch full state_dict ----------
    if "state_dict" in first:
        keys = first["state_dict"].keys()
        avg_sd = {}
        for k in keys:
            vals = [_to_np(w["state_dict"][k]) for w in weights_list]
            avg_sd[k] = _avg(vals)

        v_prev = _STATE["velocity"]
        if v_prev is None:
            v_new = {"state_dict": avg_sd}
        else:
            v_new = {"state_dict": {}}
            for k in keys:
                v_new["state_dict"][k] = _apply_momentum(
                    v_prev["state_dict"][k], avg_sd[k], beta
                )
        _STATE["velocity"] = v_new

        out = {
            "type": "pytorch",
            "state_dict": {k: v_new["state_dict"][k].tolist() for k in keys},
        }
        for mk in ["input_dim", "hidden_dim", "num_classes", "arch"]:
            if mk in first:
                out[mk] = first[mk]
        return out

    raise ValueError("Unsupported payload format.")
