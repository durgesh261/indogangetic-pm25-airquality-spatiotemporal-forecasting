"""
Unit tests for Indo-Gangetic Air Quality dataset generator.
"""

import pytest
import torch
from src.air_quality_dataset import IndoGangeticAirQualityDataset, STATIONS


def test_dataset_generation_shapes():
    gen = IndoGangeticAirQualityDataset(num_days=40, seq_len=48, random_state=42)
    data = gen.generate()

    assert data["x_train"].shape[1] == 15  # 15 stations
    assert data["x_train"].shape[2] == 48  # seq_len
    assert data["x_train"].shape[3] == 6   # in_features
    assert data["y_train"].shape[1] == 15  # 15 stations
    assert data["y_train"].shape[2] == 3   # 3 horizons (24h, 48h, 72h)
    assert data["dist_matrix"].shape == (15, 15)


def test_station_metadata():
    assert len(STATIONS) == 15
    for station in STATIONS:
        assert "name" in station
        assert "state" in station
        assert 25.0 <= station["lat"] <= 33.0  # North India latitude range
        assert 73.0 <= station["lon"] <= 83.0  # Indo-Gangetic longitude range
