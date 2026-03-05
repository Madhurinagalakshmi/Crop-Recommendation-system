# app.py — Member 3: FastAPI Backend
# Place this file at the ROOT of the project (same level as src/, artifacts/)
# Run with: uvicorn app:app --reload --port 8000

import os
import sys
import yaml
import torch
import joblib
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make sure src/ is importable ──────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.inference_utils import preprocess_single_input, decode_prediction
from src.models.gcn import GCN

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CropSense API",
    description="GNN-powered crop recommendation system — Member 3 Backend",
    version="1.0.0"
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ─────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"

def load_model(path: str) -> GCN:
    """Load GCN model using dims from graph_config.yaml."""
    with open("configs/graph_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = GCN(
        input_dim  = config.get("input_dim",  7),
        hidden_dim = config.get("hidden_dim", 64),
        output_dim = config.get("output_dim", 22),
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model

print("Loading models...")
normal_model = load_model(f"{ARTIFACTS_DIR}/gcn_model.pth")
adv_model    = load_model(f"{ARTIFACTS_DIR}/gcn_adv_model.pth")
le           = joblib.load(f"{ARTIFACTS_DIR}/label_encoder.pkl")
CROP_CLASSES = list(le.classes_)
print(f"✅ Models loaded · {len(CROP_CLASSES)} crops: {CROP_CLASSES}")


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class SoilInput(BaseModel):
    N:           float = Field(..., ge=0,   le=140,  description="Nitrogen (kg/ha)")
    P:           float = Field(..., ge=5,   le=145,  description="Phosphorus (kg/ha)")
    K:           float = Field(..., ge=5,   le=205,  description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=8,   le=44,   description="Temperature (°C)")
    humidity:    float = Field(..., ge=14,  le=100,  description="Humidity (%)")
    ph:          float = Field(..., ge=3.5, le=9.9,  description="pH level")
    rainfall:    float = Field(..., ge=20,  le=300,  description="Rainfall (mm)")

class RobustnessRequest(BaseModel):
    input:   SoilInput
    epsilon: float = Field(0.1, ge=0.01, le=0.5, description="FGSM noise strength")


# ── Helper: single-node graph for inference ───────────────────────────────────
def build_single_node_graph(x_tensor):
    from torch_geometric.data import Data
    return Data(
        x          = x_tensor,
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    )


# ══════════════════════════════════════════════════════════════
#  GET /health
# ══════════════════════════════════════════════════════════════
@app.get("/health", tags=["System"])
def health():
    """Check that the API and models are running."""
    return {
        "status":      "running",
        "num_crops":   len(CROP_CLASSES),
        "crops":       CROP_CLASSES
    }


# ══════════════════════════════════════════════════════════════
#  POST /predict
# ══════════════════════════════════════════════════════════════
@app.post("/predict", tags=["Prediction"])
def predict(data: SoilInput):
    """
    Takes 7 soil/climate inputs and returns:
    - Recommended crop name
    - Confidence score
    - Top 3 crop candidates with probabilities
    """
    try:
        # Step 1: Preprocess — Member 1's function
        x = preprocess_single_input(data.dict(), artifacts_dir=ARTIFACTS_DIR)

        # Step 2: GNN inference — Member 2's model
        graph = build_single_node_graph(x)
        with torch.no_grad():
            logits = normal_model(graph)
            probs  = F.softmax(logits, dim=1)[0]

        # Step 3: Decode prediction
        pred_idx   = probs.argmax().item()
        crop_name  = decode_prediction(pred_idx, artifacts_dir=ARTIFACTS_DIR)
        confidence = round(probs[pred_idx].item(), 4)

        # Step 4: Top 3 crops
        top3_idx = probs.topk(3).indices.tolist()
        top3 = [
            {
                "crop": decode_prediction(i, artifacts_dir=ARTIFACTS_DIR),
                "prob": round(probs[i].item(), 4)
            }
            for i in top3_idx
        ]

        return {
            "crop":       crop_name,
            "confidence": confidence,
            "top3":       top3,
            "status":     "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
#  POST /robustness-test
# ══════════════════════════════════════════════════════════════
@app.post("/robustness-test", tags=["Robustness"])
def robustness_test(body: RobustnessRequest):
    """
    Compares normal model vs adversarial model (FGSM noise).
    Returns both predictions and confidence drop.
    """
    try:
        data    = body.input.dict()
        epsilon = body.epsilon

        x     = preprocess_single_input(data, artifacts_dir=ARTIFACTS_DIR)
        graph = build_single_node_graph(x)

        # Normal prediction
        with torch.no_grad():
            n_probs = F.softmax(normal_model(graph), dim=1)[0]
            n_idx   = n_probs.argmax().item()
            n_conf  = round(n_probs[n_idx].item(), 4)
            n_crop  = decode_prediction(n_idx, artifacts_dir=ARTIFACTS_DIR)

        # Adversarial prediction — FGSM simulated noise
        x_adv     = x + epsilon * torch.sign(torch.randn_like(x))
        graph_adv = build_single_node_graph(x_adv)

        with torch.no_grad():
            a_probs = F.softmax(adv_model(graph_adv), dim=1)[0]
            a_idx   = a_probs.argmax().item()
            a_conf  = round(a_probs[a_idx].item(), 4)
            a_crop  = decode_prediction(a_idx, artifacts_dir=ARTIFACTS_DIR)

        return {
            "normal":          {"crop": n_crop, "confidence": n_conf},
            "adversarial":     {"crop": a_crop, "confidence": a_conf},
            "same_prediction": n_crop == a_crop,
            "confidence_drop": round(n_conf - a_conf, 4),
            "epsilon":         epsilon,
            "status":          "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))