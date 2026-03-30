"""
main.py
FastAPI backend — serves gentrification risk predictions.
Deploy to Railway / Render (free tier).

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload
"""

import json
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional



# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "model" / "checkpoints"
DEVICE    = torch.device("cpu")

# ── Load model ───────────────────────────────────────────────────────────────
def load_model():
    return None, None

    with open(meta_path) as f:
        meta = json.load(f)

    model = GentrificationLSTM(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=0.0,  # No dropout at inference
    ).to(DEVICE)

    weights_path = MODEL_DIR / "best_model.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model, meta


MODEL, MODEL_META = load_model()

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Gentrification Early Warning API",
    description="LSTM-powered risk scoring for NYC neighborhoods",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Neighborhood registry (static seed data for demo) ─────────────────────
NEIGHBORHOODS = {
    "Bushwick": {
        "borough": "Brooklyn",
        "lat": 40.6944, "lng": -73.9213,
        "risk_score": 0.82,
        "trend": "accelerating",
        "top_signals": ["rent_yoy (+14%)", "new_licenses (+8 this quarter)", "permit_intensity high"],
        "rent_12m": [2100,2120,2145,2160,2185,2210,2230,2255,2280,2310,2345,2390],
        "description": "Rent YoY outpacing borough median by 2.3×. High permit activity signals gut renovations.",
    },
    "East New York": {
        "borough": "Brooklyn",
        "lat": 40.6528, "lng": -73.8826,
        "risk_score": 0.71,
        "trend": "rising",
        "top_signals": ["permit_intensity rising", "income_index +12%", "demo_shift accelerating"],
        "rent_12m": [1500,1510,1525,1530,1545,1560,1580,1590,1610,1630,1660,1695],
        "description": "Early signals emerging post-2022. Rezoning activity and transit investment driving interest.",
    },
    "Mott Haven": {
        "borough": "Bronx",
        "lat": 40.8092, "lng": -73.9236,
        "risk_score": 0.78,
        "trend": "accelerating",
        "top_signals": ["rent_yoy (+11%)", "new_licenses surging", "housing_complaints declining"],
        "rent_12m": [1650,1670,1695,1720,1750,1780,1810,1840,1875,1910,1950,1995],
        "description": "South Bronx waterfront development driving displacement pressure. Fastest-rising risk in borough.",
    },
    "Ridgewood": {
        "borough": "Queens",
        "lat": 40.7059, "lng": -73.9088,
        "risk_score": 0.69,
        "trend": "rising",
        "top_signals": ["rent_3m_momentum +6%", "new_licenses +5", "demo_shift +0.15"],
        "rent_12m": [1800,1815,1830,1850,1870,1890,1910,1935,1960,1985,2010,2045],
        "description": "Spillover from Bushwick driving rapid appreciation. Demographics shifting noticeably since 2021.",
    },
    "Jackson Heights": {
        "borough": "Queens",
        "lat": 40.7557, "lng": -73.8830,
        "risk_score": 0.65,
        "trend": "rising",
        "top_signals": ["permit_intensity high", "income_index rising", "rent_vs_nyc +0.8"],
        "rent_12m": [1750,1760,1775,1790,1810,1830,1850,1870,1895,1920,1945,1975],
        "description": "Transit hub status and relative affordability attracting displacement pressure from Long Island City.",
    },
    "Washington Heights": {
        "borough": "Manhattan",
        "lat": 40.8417, "lng": -73.9394,
        "risk_score": 0.55,
        "trend": "steady",
        "top_signals": ["rent_yoy moderate", "permit_intensity moderate", "demo_shift stable"],
        "rent_12m": [2000,2010,2020,2030,2045,2055,2065,2080,2090,2105,2115,2130],
        "description": "Moderate and stable risk. Community land trusts providing some buffer against rapid displacement.",
    },
    "South Bronx": {
        "borough": "Bronx",
        "lat": 40.8122, "lng": -73.9198,
        "risk_score": 0.48,
        "trend": "steady",
        "top_signals": ["housing_complaints high", "income_index low", "permit_intensity low"],
        "rent_12m": [1400,1405,1415,1420,1430,1440,1450,1460,1470,1480,1490,1500],
        "description": "Lower risk currently but infrastructure investment could accelerate signals in 12-18 months.",
    },
    "Flatbush": {
        "borough": "Brooklyn",
        "lat": 40.6421, "lng": -73.9616,
        "risk_score": 0.44,
        "trend": "stable",
        "top_signals": ["rent_yoy modest", "new_licenses moderate", "demo_shift low"],
        "rent_12m": [1850,1855,1860,1870,1875,1885,1890,1900,1910,1920,1930,1940],
        "description": "Strong community organizations and rent stabilization keeping displacement pressure lower.",
    },
}


# ── Schemas ──────────────────────────────────────────────────────────────────
class NeighborhoodSummary(BaseModel):
    name: str
    borough: str
    lat: float
    lng: float
    risk_score: float
    trend: str


class NeighborhoodDetail(NeighborhoodSummary):
    top_signals: List[str]
    rent_12m: List[float]
    description: str


class PredictRequest(BaseModel):
    neighborhood: str
    features: Optional[List[List[float]]] = None  # (seq_len, n_features) if provided


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "NYC Gentrification Early Warning API",
        "status": "live",
        "model_loaded": MODEL is not None,
        "endpoints": ["/neighborhoods", "/neighborhoods/{name}", "/predict", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/neighborhoods", response_model=List[NeighborhoodSummary])
def get_all_neighborhoods():
    """Returns all neighborhoods with their risk scores for the map."""
    return [
        NeighborhoodSummary(name=name, **{k: v for k, v in data.items()
                                          if k in NeighborhoodSummary.model_fields})
        for name, data in NEIGHBORHOODS.items()
    ]


@app.get("/neighborhoods/{name}", response_model=NeighborhoodDetail)
def get_neighborhood(name: str):
    """Returns detailed data for a single neighborhood."""
    # Case-insensitive lookup
    match = next((k for k in NEIGHBORHOODS if k.lower() == name.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Neighborhood '{name}' not found.")
    return NeighborhoodDetail(name=match, **NEIGHBORHOODS[match])


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run live LSTM inference on provided feature sequence.
    If no features provided, returns the cached risk score from registry.
    """
    match = next((k for k in NEIGHBORHOODS if k.lower() == req.neighborhood.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Neighborhood '{req.neighborhood}' not found.")

    if req.features and MODEL:
        x = torch.tensor([req.features], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            score = MODEL(x).item()
    else:
        score = NEIGHBORHOODS[match]["risk_score"]

    level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.45 else "LOW"

    return {
        "neighborhood": match,
        "risk_score": round(score, 3),
        "risk_level": level,
        "top_signals": NEIGHBORHOODS[match]["top_signals"],
        "description": NEIGHBORHOODS[match]["description"],
    }


@app.get("/boroughs/{borough}")
def get_by_borough(borough: str):
    """Filter neighborhoods by borough."""
    results = [
        {"name": name, **data}
        for name, data in NEIGHBORHOODS.items()
        if data["borough"].lower() == borough.lower()
    ]
    if not results:
        raise HTTPException(status_code=404, detail=f"No neighborhoods found for borough '{borough}'.")
    return results
