"""
Production FastAPI Application for Indo-Gangetic PM2.5 Spatiotemporal Air Quality Forecasting.
Provides multi-horizon predictions (24h, 48h, 72h), calibrated uncertainty intervals,
and satellite-driven agricultural stubble burning source attribution.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.air_quality_dataset import IndoGangeticAirQualityDataset, STATIONS
from src.spatiotemporal_model import SpatiotemporalPM25Forecaster
from src.quantile_loss import MultiQuantileLoss
from src.forecaster import SpatiotemporalForecasterManager
from src.attribution_engine import StubbleAttributionEngine

# Global context
context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes dataset, trains Spatiotemporal Network with Multi-Quantile Loss, and sets up attribution."""
    print("[API Startup] Loading Indo-Gangetic monitoring network & training Spatiotemporal Forecaster...")
    dataset_gen = IndoGangeticAirQualityDataset(num_days=60, seq_len=48, random_state=42)
    data = dataset_gen.generate()

    model = SpatiotemporalPM25Forecaster(in_features=6, hidden_dim=32, num_stations=len(data['stations']))
    criterion = MultiQuantileLoss(quantiles=(0.10, 0.50, 0.90))
    manager = SpatiotemporalForecasterManager(model=model, criterion=criterion, lr=0.008, weight_decay=1e-4)

    results = manager.fit(data, epochs=12, verbose=False)
    engine = StubbleAttributionEngine(model, data["dist_matrix"], data["stations"])

    context["model"] = model
    context["data"] = data
    context["engine"] = engine
    context["test_metrics"] = results["test_metrics"]

    print(f"[API Startup Ready] Overall Test MAE: {results['test_metrics']['overall_avg_mae']:.2f} ug/m3")
    yield
    context.clear()


app = FastAPI(
    title="Indo-Gangetic PM2.5 Air Quality Spatiotemporal Forecaster API",
    description="Multi-Horizon Spatiotemporal Attention Network with Quantile Uncertainty and Satellite Stubble-Burning Attribution",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class ForecastRequest(BaseModel):
    station_id: int = Field(0, ge=0, le=14, description="Station ID index (0 to 14, default 0 for Delhi - Anand Vihar)")


class AttributionRequest(BaseModel):
    station_id: int = Field(0, ge=0, le=14, description="Station ID index (0 to 14)")


# --- Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "service": "Indo-Gangetic PM2.5 Spatiotemporal Smog Forecasting Engine",
        "spatial_coverage": "Indo-Gangetic Plain (Delhi-NCR, Punjab, Haryana, Uttar Pradesh)",
        "num_monitored_stations": len(STATIONS),
        "horizons_hours": [24, 48, 72],
        "uncertainty_intervals": "80% Confidence Interval (10th to 90th percentile)",
        "cpcb_aqi_scale": {
            "0-30": "Good",
            "31-60": "Satisfactory",
            "61-90": "Moderate",
            "91-120": "Poor",
            "121-250": "Very Poor",
            ">250": "Severe / Hazardous"
        },
        "documentation": "/docs"
    }


@app.get("/api/v1/stations", tags=["Ground Network"])
def get_stations():
    """Returns metadata for all 15 ground monitoring stations across the Indo-Gangetic basin."""
    return {"stations": STATIONS}


@app.get("/api/v1/metrics", tags=["Model Evaluation"])
def get_metrics():
    """Returns test evaluation metrics across 24h, 48h, and 72h forecast horizons (MAE, RMSE, PICP Coverage)."""
    return {
        "test_metrics": context["test_metrics"],
        "loss_formulation": "Multi-Quantile Pinball Loss (q=0.10, 0.50, 0.90)"
    }


@app.post("/api/v1/forecast", tags=["Forecasting"])
def get_station_forecast(req: ForecastRequest):
    """Generates 24h, 48h, and 72h PM2.5 predictions with 80% confidence interval for a given station."""
    model: SpatiotemporalPM25Forecaster = context["model"]
    data = context["data"]

    test_sample = data["x_test"][0:1]
    model.eval()
    with torch.no_grad():
        preds = model(test_sample, data["dist_matrix"])[0, req.station_id].cpu().numpy()

    station = STATIONS[req.station_id]
    horizons = [24, 48, 72]
    forecast_results = []

    for h_idx, h in enumerate(horizons):
        q10 = float(preds[h_idx, 0])
        q50 = float(preds[h_idx, 1])
        q90 = float(preds[h_idx, 2])

        if q50 <= 30:
            cat = "Good"
        elif q50 <= 60:
            cat = "Satisfactory"
        elif q50 <= 90:
            cat = "Moderate"
        elif q50 <= 120:
            cat = "Poor"
        elif q50 <= 250:
            cat = "Very Poor"
        else:
            cat = "Severe / Hazardous"

        forecast_results.append({
            "horizon_hours": h,
            "pm25_median_ug_m3": round(q50, 1),
            "confidence_interval_80pct": [round(q10, 1), round(q90, 1)],
            "cpcb_aqi_category": cat
        })

    return {
        "station_id": req.station_id,
        "station_name": station["name"],
        "state": station["state"],
        "coordinates": {"lat": station["lat"], "lon": station["lon"]},
        "forecasts": forecast_results
    }


@app.post("/api/v1/stubble-attribution", tags=["Source Attribution"])
def get_stubble_attribution(req: AttributionRequest):
    """Quantifies the percentage contribution of upstream agricultural fires vs local urban baseline."""
    engine: StubbleAttributionEngine = context["engine"]
    data = context["data"]

    peak_idx = int(torch.argmax(data["x_val"][..., 5].mean(dim=(1, 2))).item())
    sample = data["x_val"][peak_idx:peak_idx + 1]
    report = engine.attribute_pollution(sample, station_id=req.station_id)
    return report


@app.get("/api/v1/regional-summary", tags=["Regional Early Warning"])
def get_regional_summary():
    """Returns regional risk aggregation across Delhi-NCR, Punjab, Haryana, and Uttar Pradesh."""
    model: SpatiotemporalPM25Forecaster = context["model"]
    data = context["data"]

    test_sample = data["x_test"][0:1]
    model.eval()
    with torch.no_grad():
        preds = model(test_sample, data["dist_matrix"])[0, :, 0, 1].cpu().numpy()  # 24h q50

    state_groups = {}
    for s_idx, station in enumerate(STATIONS):
        state = station["state"]
        pm25 = float(preds[s_idx])
        if state not in state_groups:
            state_groups[state] = []
        state_groups[state].append(pm25)

    summary = {}
    for state, vals in state_groups.items():
        avg_pm25 = float(np.mean(vals))
        risk = "CRITICAL / SEVERE" if avg_pm25 > 250 else ("VERY POOR" if avg_pm25 > 120 else "MODERATE")
        summary[state] = {
            "mean_predicted_24h_pm25": round(avg_pm25, 1),
            "regional_risk_level": risk,
            "monitored_stations": len(vals)
        }

    return {"regional_air_quality_summary": summary}
