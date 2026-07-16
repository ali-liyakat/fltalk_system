"""
FedAdam (Server optimizer aggregation)

Idea:
- Treat average client model as "gradient step" direction vs previous global.
- Maintain m, v like Adam on server.

We maintain server-side state in-process.
"""

import numpy as np

_STATE = {
    "t": 0,
    "m": None,
    "v": None,
    "w_prev": None,   # previous global weights
}

def _to_np(x):
    return np.array(x, dtype=np.float32)

def _avg(arrs):
    return np.mean(np.stack(arrs, axis=0), axis=0)

def _adam_update(w_prev, w_avg, m_prev, v_prev, lr, b1, b2, eps):
    # pseudo-gradient: g = w_prev - w_avg  (move towards avg)
    g = w_prev - w_avg
    m = b1 * m_prev + (1.0 - b1) * g
    v = b2 * v_prev + (1.0 - b2) * (g * g)
    return m, v

def _bias_correct(m, v, t, b1, b2, eps):
    mhat = m / (1.0 - (b1 ** t))
    vhat = v / (1.0 - (b2 ** t))
    step = mhat / (np.sqrt(vhat) + eps)
    return step

def aggregate(weights_list, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    if not weights_list:
        raise ValueError("No client weights received.")
    first = weights_list[0]

    # ---------- sklearn ----------
    if "coef" in first and "intercept" in first:
        coefs = [_to_np(w["coef"]) for w in weights_list]
        ints  = [_to_np(w["intercept"]) for w in weights_list]

        w_avg = {"coef": _avg(coefs), "intercept": _avg(ints)}

        if _STATE["w_prev"] is None:
            _STATE["w_prev"] = w_avg
            _STATE["m"] = {"coef": np.zeros_like(w_avg["coef"]), "intercept": np.zeros_like(w_avg["intercept"])}
            _STATE["v"] = {"coef": np.zeros_like(w_avg["coef"]), "intercept": np.zeros_like(w_avg["intercept"])}

        _STATE["t"] += 1
        t = _STATE["t"]

        w_prev = _STATE["w_prev"]
        m_prev, v_prev = _STATE["m"], _STATE["v"]

        m_prev["coef"], v_prev["coef"] = _adam_update(w_prev["coef"], w_avg["coef"], m_prev["coef"], v_prev["coef"], lr, beta1, beta2, eps)
        m_prev["intercept"], v_prev["intercept"] = _adam_update(w_prev["intercept"], w_avg["intercept"], m_prev["intercept"], v_prev["intercept"], lr, beta1, beta2, eps)

        step_coef = _bias_correct(m_prev["coef"], v_prev["coef"], t, beta1, beta2, eps)
        step_int  = _bias_correct(m_prev["intercept"], v_prev["intercept"], t, beta1, beta2, eps)

        w_new = {
            "coef": w_prev["coef"] - lr * step_coef,
            "intercept": w_prev["intercept"] - lr * step_int,
        }

        _STATE["w_prev"] = w_new
        _STATE["m"], _STATE["v"] = m_prev, v_prev

        return {
            "type": "sklearn",
            "coef": w_new["coef"].tolist(),
            "intercept": w_new["intercept"].tolist(),
            "classes": first.get("classes", None),
        }

    # ---------- pytorch head-only ----------
    if "head_state_dict" in first:
        keys = first["head_state_dict"].keys()
        w_avg = {k: _avg([_to_np(w["head_state_dict"][k]) for w in weights_list]) for k in keys}

        if _STATE["w_prev"] is None:
            _STATE["w_prev"] = {k: w_avg[k] for k in keys}
            _STATE["m"] = {k: np.zeros_like(w_avg[k]) for k in keys}
            _STATE["v"] = {k: np.zeros_like(w_avg[k]) for k in keys}

        _STATE["t"] += 1
        t = _STATE["t"]

        w_prev = _STATE["w_prev"]
        m_prev, v_prev = _STATE["m"], _STATE["v"]

        for k in keys:
            g = w_prev[k] - w_avg[k]
            m_prev[k] = beta1 * m_prev[k] + (1.0 - beta1) * g
            v_prev[k] = beta2 * v_prev[k] + (1.0 - beta2) * (g * g)

            mhat = m_prev[k] / (1.0 - (beta1 ** t))
            vhat = v_prev[k] / (1.0 - (beta2 ** t))
            step = mhat / (np.sqrt(vhat) + eps)

            w_prev[k] = w_prev[k] - lr * step

        _STATE["w_prev"] = w_prev
        _STATE["m"], _STATE["v"] = m_prev, v_prev

        out = {
            "type": "pytorch",
            "head_only": True,
            "head_state_dict": {k: w_prev[k].tolist() for k in keys},
        }
        for mk in ["num_classes", "arch"]:
            if mk in first:
                out[mk] = first[mk]
        return out

    # ---------- pytorch full state_dict ----------
    if "state_dict" in first:
        keys = first["state_dict"].keys()
        w_avg = {k: _avg([_to_np(w["state_dict"][k]) for w in weights_list]) for k in keys}

        if _STATE["w_prev"] is None:
            _STATE["w_prev"] = {k: w_avg[k] for k in keys}
            _STATE["m"] = {k: np.zeros_like(w_avg[k]) for k in keys}
            _STATE["v"] = {k: np.zeros_like(w_avg[k]) for k in keys}

        _STATE["t"] += 1
        t = _STATE["t"]

        w_prev = _STATE["w_prev"]
        m_prev, v_prev = _STATE["m"], _STATE["v"]

        for k in keys:
            g = w_prev[k] - w_avg[k]
            m_prev[k] = beta1 * m_prev[k] + (1.0 - beta1) * g
            v_prev[k] = beta2 * v_prev[k] + (1.0 - beta2) * (g * g)

            mhat = m_prev[k] / (1.0 - (beta1 ** t))
            vhat = v_prev[k] / (1.0 - (beta2 ** t))
            step = mhat / (np.sqrt(vhat) + eps)

            w_prev[k] = w_prev[k] - lr * step

        out = {
            "type": "pytorch",
            "state_dict": {k: w_prev[k].tolist() for k in keys},
        }
        for mk in ["input_dim", "hidden_dim", "num_classes", "arch"]:
            if mk in first:
                out[mk] = first[mk]
        return out

    raise ValueError("Unsupported payload format.")
