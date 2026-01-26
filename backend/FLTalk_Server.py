from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time

app = FastAPI(title="FLTalk Main Server")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


EXPERIMENTS = {}   



#================================================================
#------UI HEARTBEAT CODE-----------------------------------------
#================================================================
def _ensure_ui_keys(exp_id: str):
    """Ensure experiment dict has UI monitoring buffers."""
    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {
            "client_weights": [],
            "global_weights": None,
            "global_round": 0,
        }

    exp = EXPERIMENTS[exp_id]
    exp.setdefault("ui_nodes", {})       
    exp.setdefault("ui_logs", [])           
    exp.setdefault("ui_log_seq", 0)



def _ensure_ui_metrics_keys(exp_id: str):
    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {
            "client_weights": [],
            "global_weights": None,
            "global_round": 0,
        }
    exp = EXPERIMENTS[exp_id]
    exp.setdefault("ui_metrics", [])
    exp.setdefault("ui_metric_seq", 0)



@app.post("/ui/metrics")
async def ui_metrics_post(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    node_id = data.get("node_id", "unknown")
    round_id = int(data.get("round", 0))
    metrics = data.get("metrics", {})

    _ensure_ui_metrics_keys(exp_id)
    exp = EXPERIMENTS[exp_id]

    exp["ui_metric_seq"] += 1
    item = {
        "id": exp["ui_metric_seq"],
        "ts": time.time(),
        "node_id": node_id,
        "round": round_id,
        "metrics": metrics,
    }
    exp["ui_metrics"].append(item)

    
    if len(exp["ui_metrics"]) > 5000:
        exp["ui_metrics"] = exp["ui_metrics"][-5000:]

    return {"status": "ok", "id": item["id"]}


@app.get("/ui/metrics")
async def ui_metrics_get(experiment_id: str = "default", since: int = 0, limit: int = 500):
    _ensure_ui_metrics_keys(experiment_id)
    exp = EXPERIMENTS[experiment_id]

    since = int(since)
    limit = max(1, min(int(limit), 1000))

    items = [x for x in exp["ui_metrics"] if int(x["id"]) > since]
    items = items[:limit]

    next_since = since
    if items:
        next_since = int(items[-1]["id"])

    return {"status": "ok", "next": next_since, "items": items}




@app.post("/ui/heartbeat")
async def ui_heartbeat(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    node_type = data.get("node_type", "client")  
    node_id = data.get("node_id", "unknown")    
    status = data.get("status", "running")       
    round_id = int(data.get("round", 0))

    _ensure_ui_keys(exp_id)

    EXPERIMENTS[exp_id]["ui_nodes"][node_id] = {
        "node_type": node_type,
        "status": status,
        "round": round_id,
        "last_seen_ts": time.time(),
    }
    return {"status": "ok"}



@app.post("/ui/log")
async def ui_log(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    node_type = data.get("node_type", "client")
    node_id = data.get("node_id", "unknown")
    message = data.get("message", "")

    _ensure_ui_keys(exp_id)

    exp = EXPERIMENTS[exp_id]
    exp["ui_log_seq"] += 1

    item = {
        "id": exp["ui_log_seq"],
        "ts": time.time(),
        "node_type": node_type,
        "node_id": node_id,
        "msg": str(message)[:2000], 
    }
    exp["ui_logs"].append(item)

    
    if len(exp["ui_logs"]) > 2000:
        exp["ui_logs"] = exp["ui_logs"][-2000:]

    return {"status": "ok", "id": item["id"]}



@app.get("/ui/status")
async def ui_status(experiment_id: str = "default", offline_after_sec: int = 6):
    _ensure_ui_keys(experiment_id)
    exp = EXPERIMENTS[experiment_id]

    now = time.time()
    nodes_out = {}

    for node_id, info in exp["ui_nodes"].items():
        last_seen = float(info.get("last_seen_ts", 0))
        age = now - last_seen if last_seen else 1e9
        online = age <= offline_after_sec

        nodes_out[node_id] = {
            "node_type": info.get("node_type"),
            "status": info.get("status") if online else "offline",
            "round": info.get("round", 0),
            "last_seen_sec": round(age, 2),
            "online": online,
        }

    return {"status": "ok", "experiment_id": experiment_id, "nodes": nodes_out}



@app.get("/ui/logs")
async def ui_logs(experiment_id: str = "default", since: int = 0, limit: int = 200):
    _ensure_ui_keys(experiment_id)
    exp = EXPERIMENTS[experiment_id]

    since = int(since)
    limit = max(1, min(int(limit), 500))

    logs = [x for x in exp["ui_logs"] if int(x["id"]) > since]
    logs = logs[:limit]

    next_since = since
    if logs:
        next_since = int(logs[-1]["id"])

    return {"status": "ok", "next": next_since, "logs": logs}



@app.post("/ui/reset")
async def ui_reset(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")

    _ensure_ui_keys(exp_id)          
    exp = EXPERIMENTS[exp_id]

    
    exp["ui_nodes"] = {}
    exp["ui_logs"] = []
    exp["ui_log_seq"] = 0

    
    exp.setdefault("ui_metrics", [])
    exp.setdefault("ui_metric_seq", 0)
    exp["ui_metrics"].clear()
    exp["ui_metric_seq"] = 0

    return {"status": "ok", "experiment_id": exp_id}


#================================================================
#------UI HEARTBEAT CODE-----------------------------------------
#================================================================




# -------------------------------
# Receive client weights
# -------------------------------
@app.post("/send_weights")
async def receive_weights(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    weights = data.get("weights")

    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {"client_weights": [], "global_weights": None}

    if weights:
        EXPERIMENTS[exp_id]["client_weights"].append(weights)
        print(f"[{exp_id}] Received client weights ({len(EXPERIMENTS[exp_id]['client_weights'])})")
        return {"status": "received", "count": len(EXPERIMENTS[exp_id]["client_weights"])}

    return {"status": "error", "msg": "No weights provided"}



# -------------------------------
# Send client weights
# -------------------------------
@app.get("/fetch_weights")
async def fetch_weights(experiment_id: str = "default"):
    exp = EXPERIMENTS.get(experiment_id)

    if exp and exp["client_weights"]:
        print(f"[{experiment_id}] Sending {len(exp['client_weights'])} weights to aggregator")

        weights_to_send = exp["client_weights"].copy()
        EXPERIMENTS[experiment_id]["client_weights"] = []   

        return {"status": "ready", "weights": weights_to_send}

    return {"status": "waiting"}





# -------------------------------
# Receive global weights from federated server 
# -------------------------------
@app.post("/send_global")
async def receive_global(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    global_weights = data.get("global_weights")
    round_id = int(data.get("round", 0))  

    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {
            "client_weights": [],
            "global_weights": None,
            "global_round": 0,            
        }

    
    EXPERIMENTS[exp_id]["global_weights"] = global_weights
    EXPERIMENTS[exp_id]["global_round"] = round_id
    EXPERIMENTS[exp_id]["client_weights"] = []

    print(f"[{exp_id}] Received global weights for round {round_id}")
    return {"status": "stored", "round": round_id}






# -------------------------------
# Send global weights back to clients 
# -------------------------------
@app.get("/fetch_global")
async def fetch_global(experiment_id: str = "default", expected_round: int = 1):
    exp = EXPERIMENTS.get(experiment_id, None)
    if not exp or not exp.get("global_weights"):
        return {"status": "waiting"}

    current_round = int(exp.get("global_round", 0))

    if current_round < int(expected_round):
        return {"status": "waiting", "current_round": current_round}

    print(f"[{experiment_id}] Sending global weights (round {current_round}) to clients")
    return {
        "status": "ready",
        "round": current_round,
        "global_weights": exp["global_weights"]
    }



# -------------------------------
# Serve Model Code 
# -------------------------------

@app.get("/get_model_code")
async def get_model_code(model_name: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
    model_path = os.path.join(BASE_DIR, "models", f"{model_name}.py")

    print(f" Looking for model file at: {model_path}")
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            code = f.read()
        print(f" Found model: {model_name}")
        return {"status": "ok", "model_name": model_name, "code": code}
    except Exception as e:
        print(f" Error loading model: {e}")
        return {"status": "error", "msg": str(e)}



# -------------------------------
# Serve Algorithm Code 
# -------------------------------
@app.get("/get_algorithm_code")
async def get_algorithm_code(algo_name: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    algo_path = os.path.join(BASE_DIR, "algorithms", f"{algo_name}.py")

    print(f" Looking for algo file at: {algo_path}")
    try:
        with open(algo_path, "r", encoding="utf-8") as f:
            code = f.read()
        print(f" Found algorithm: {algo_name}")
        return {"status": "ok", "algorithm": algo_name, "code": code}
    except Exception as e:
        print(f" Error loading algorithm: {e}")
        return {"status": "error", "msg": str(e)}






# =========================================================
# FedDeepSMOTE: Parallel endpoints (Skeleton)
# =========================================================

def _ensure_feddeepsmote_keys(exp_id: str):
    """Make sure experiment dict has FedDeepSMOTE buffers."""
    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {
            "client_weights": [],
            "global_weights": None,
            "global_round": 0,
        }

    exp = EXPERIMENTS[exp_id]

    
    exp.setdefault("feddeepsmote_client_updates", [])   
    exp.setdefault("feddeepsmote_global", None)        
    exp.setdefault("feddeepsmote_global_round", 0)      


# -------------------------------
# Receive client FedDeepSMOTE updates
# -------------------------------
@app.post("/send_feddeepsmote_weights")
async def send_feddeepsmote_weights(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    client_id = data.get("client_id", "unknown")
    round_id = int(data.get("round", 0))

    
    enc = data.get("enc") 
    dec = data.get("dec")
    n_samples = int(data.get("n_samples", 0))

    _ensure_feddeepsmote_keys(exp_id)

    if enc is None or dec is None:
        return {"status": "error", "msg": "Missing enc/dec in payload"}

    EXPERIMENTS[exp_id]["feddeepsmote_client_updates"].append({
        "client_id": client_id,
        "round": round_id,
        "n_samples": n_samples,
        "enc": enc,
        "dec": dec,
    })

    count = len(EXPERIMENTS[exp_id]["feddeepsmote_client_updates"])
    print(f"[{exp_id}] Received FedDeepSMOTE update from {client_id} (count={count}, round={round_id})")
    return {"status": "received", "count": count, "round": round_id}


# -------------------------------
# Fetch client FedDeepSMOTE updates for aggregator
# -------------------------------
@app.get("/fetch_feddeepsmote_weights")
async def fetch_feddeepsmote_weights(experiment_id: str = "default"):
    _ensure_feddeepsmote_keys(experiment_id)
    buf = EXPERIMENTS[experiment_id]["feddeepsmote_client_updates"]

    if buf:
        updates_to_send = buf.copy()
        EXPERIMENTS[experiment_id]["feddeepsmote_client_updates"] = []  
        print(f"[{experiment_id}] Sending {len(updates_to_send)} FedDeepSMOTE updates to aggregator")
        return {"status": "ready", "updates": updates_to_send}

    return {"status": "waiting"}


# -------------------------------
# Receive global FedDeepSMOTE from aggregator (round-aware)
# -------------------------------
@app.post("/send_feddeepsmote_global")
async def send_feddeepsmote_global(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    round_id = int(data.get("round", 0))

    global_payload = data.get("global")  
    if not global_payload or ("enc" not in global_payload) or ("dec" not in global_payload):
        return {"status": "error", "msg": "global must be {'enc':..., 'dec':...}"}

    _ensure_feddeepsmote_keys(exp_id)

    EXPERIMENTS[exp_id]["feddeepsmote_global"] = global_payload
    EXPERIMENTS[exp_id]["feddeepsmote_global_round"] = round_id

    print(f"[{exp_id}] Stored FedDeepSMOTE global for round {round_id}")
    return {"status": "stored", "round": round_id}


# -------------------------------
# Fetch global FedDeepSMOTE for clients (round-aware)
# -------------------------------
@app.get("/fetch_feddeepsmote_global")
async def fetch_feddeepsmote_global(experiment_id: str = "default", expected_round: int = 1):
    _ensure_feddeepsmote_keys(experiment_id)

    exp = EXPERIMENTS.get(experiment_id)
    if not exp or not exp.get("feddeepsmote_global"):
        return {"status": "waiting"}

    current_round = int(exp.get("feddeepsmote_global_round", 0))
    if current_round < int(expected_round):
        return {"status": "waiting", "current_round": current_round}

    print(f"[{experiment_id}] Sending FedDeepSMOTE global (round {current_round}) to clients")
    return {
        "status": "ready",
        "round": current_round,
        "global": exp["feddeepsmote_global"]
    }


@app.get("/")
def root():
    return {"msg": "FLTalk Main Server Running"}
