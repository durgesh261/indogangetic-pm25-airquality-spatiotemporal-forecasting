"""
Indo-Gangetic PM2.5 Spatiotemporal Air Quality Forecasting Package.
"""

from .air_quality_dataset import IndoGangeticAirQualityDataset, STATIONS
from .spatiotemporal_model import SpatiotemporalPM25Forecaster, SpatialStationAttention
from .quantile_loss import MultiQuantileLoss
from .attribution_engine import StubbleAttributionEngine
from .forecaster import SpatiotemporalForecasterManager

__all__ = [
    "IndoGangeticAirQualityDataset",
    "STATIONS",
    "SpatiotemporalPM25Forecaster",
    "SpatialStationAttention",
    "MultiQuantileLoss",
    "StubbleAttributionEngine",
    "SpatiotemporalForecasterManager"
]
