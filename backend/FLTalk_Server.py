from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI(title="FLTalk Main Server")

# -------------------------------
# CORS Setup
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Global Storage (temporary)
# -------------------------------
EXPERIMENTS = {}   # { experiment_id: { 'client_weights': [], 'global_weights': None } }

# -------------------------------
# 1️⃣ Receive client weights
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
# 2️⃣ Send client weights to federated server
# -------------------------------
@app.get("/fetch_weights")
async def fetch_weights(experiment_id: str = "default"):
    exp = EXPERIMENTS.get(experiment_id, None)
    if exp and exp["client_weights"]:
        print(f"[{experiment_id}] Sending {len(exp['client_weights'])} weights to aggregator")
        return {"status": "ready", "weights": exp["client_weights"]}
    return {"status": "waiting"}

# -------------------------------
# 3️⃣ Receive global weights from federated server
# -------------------------------
@app.post("/send_global")
async def receive_global(req: Request):
    data = await req.json()
    exp_id = data.get("experiment_id", "default")
    global_weights = data.get("global_weights")

    if exp_id not in EXPERIMENTS:
        EXPERIMENTS[exp_id] = {"client_weights": [], "global_weights": None}

    EXPERIMENTS[exp_id]["global_weights"] = global_weights
    EXPERIMENTS[exp_id]["client_weights"] = []  # reset for next round
    print(f"[{exp_id}] Received global weights")
    return {"status": "stored"}

# -------------------------------
# 4️⃣ Send global weights back to clients
# -------------------------------
@app.get("/fetch_global")
async def fetch_global(experiment_id: str = "default"):
    exp = EXPERIMENTS.get(experiment_id, None)
    if exp and exp["global_weights"]:
        print(f"[{experiment_id}] Sending global weights to clients")
        return {"status": "ready", "global_weights": exp["global_weights"]}
    return {"status": "waiting"}

# -------------------------------
# 5️⃣ Serve Model Code (for client)
# -------------------------------

@app.get("/get_model_code")
async def get_model_code(model_name: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend folder
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
# 6️⃣ Serve Algorithm Code (for server)
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



@app.get("/")
def root():
    return {"msg": "FLTalk Main Server Running"}
