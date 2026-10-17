"""
Integration tests for Indo-Gangetic Air Quality Forecasting FastAPI service.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Indo-Gangetic" in data["service"]
    assert data["num_monitored_stations"] == 15


def test_stations_endpoint(client):
    response = client.get("/api/v1/stations")
    assert response.status_code == 200
    data = response.json()
    assert len(data["stations"]) == 15
    assert data["stations"][0]["name"] == "Delhi - Anand Vihar"


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert "mae_24h" in data["test_metrics"]
    assert "overall_avg_mae" in data["test_metrics"]


def test_forecast_endpoint(client):
    payload = {"station_id": 0}  # Delhi - Anand Vihar
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["station_name"] == "Delhi - Anand Vihar"
    assert len(data["forecasts"]) == 3
    for fc in data["forecasts"]:
        assert fc["horizon_hours"] in [24, 48, 72]
        assert fc["pm25_median_ug_m3"] > 0
        ci = fc["confidence_interval_80pct"]
        assert ci[0] <= fc["pm25_median_ug_m3"] <= ci[1]


def test_attribution_endpoint(client):
    payload = {"station_id": 0}
    response = client.post("/api/v1/stubble-attribution", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "average_stubble_share_pct" in data
    assert 0.0 <= data["average_stubble_share_pct"] <= 100.0
    assert "horizon_breakdown" in data


def test_regional_summary_endpoint(client):
    response = client.get("/api/v1/regional-summary")
    assert response.status_code == 200
    data = response.json()
    summary = data["regional_air_quality_summary"]
    assert "Delhi" in summary
    assert "Punjab" in summary
    assert "Haryana" in summary
