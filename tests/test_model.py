"""
Unit tests for Spatiotemporal model architecture and multi-quantile loss.
"""

import pytest
import torch
from src.spatiotemporal_model import SpatiotemporalPM25Forecaster
from src.quantile_loss import MultiQuantileLoss


def test_spatiotemporal_forward_and_quantiles():
    B = 4
    N = 15
    T = 24
    F_in = 6

    x = torch.randn(B, N, T, F_in)
    dist_matrix = torch.rand(N, N) * 200.0  # Pairwise distances

    model = SpatiotemporalPM25Forecaster(in_features=F_in, hidden_dim=32, num_stations=N)
    quantiles = model(x, dist_matrix)

    # Output shape: [B, N, 3 horizons, 3 quantiles]
    assert quantiles.shape == (B, N, 3, 3)

    # Crucial property: Monotonic non-crossing quantiles q10 <= q50 <= q90
    q10 = quantiles[..., 0]
    q50 = quantiles[..., 1]
    q90 = quantiles[..., 2]

    assert (q50 >= q10).all(), "q50 must be >= q10 everywhere"
    assert (q90 >= q50).all(), "q90 must be >= q50 everywhere"
    assert (q10 >= 0.0).all(), "PM2.5 predictions must be strictly positive"


def test_multi_quantile_loss_backward():
    B = 4
    N = 15
    preds = torch.randn(B, N, 3, 3, requires_grad=True)
    targets = torch.rand(B, N, 3) * 150.0

    criterion = MultiQuantileLoss(quantiles=(0.10, 0.50, 0.90))
    loss = criterion(preds, targets)

    assert loss.item() > 0.0
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()
