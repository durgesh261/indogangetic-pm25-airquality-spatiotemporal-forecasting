"""
Spatiotemporal Graph-Temporal Attention Network for Multi-Horizon Air Quality Forecasting.
Outputs calibrated prediction intervals (q10, q50, q90) across 24h, 48h, and 72h forecast horizons.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialStationAttention(nn.Module):
    """
    Spatial Attention Layer that learns transport dependencies between upwind and downwind stations.
    Combines geographic distance priors with learned meteorological affinity.
    """

    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(in_features, hidden_dim)
        self.key_proj = nn.Linear(in_features, hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_stations, hidden_dim]
            dist_matrix: [num_stations, num_stations] pairwise distances in km
        Returns:
            out: [B, num_stations, hidden_dim] spatially aggregated features
        """
        Q = self.query_proj(x)  # [B, N, H]
        K = self.key_proj(x)    # [B, N, H]

        # Learned attention: [B, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Distance decay prior: closer stations interact more strongly
        # dist_decay = exp(-D / 150 km)
        dist_prior = torch.exp(-dist_matrix / 150.0).unsqueeze(0)  # [1, N, N]
        attn_scores = attn_scores + torch.log(dist_prior + 1e-6)

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, x)
        return out


class SpatiotemporalPM25Forecaster(nn.Module):
    """
    Deep Spatiotemporal Network for Indo-Gangetic PM2.5 forecasting.
    Features:
    - GRU Temporal Sequence Encoder over 168-hour history
    - Spatial Cross-Station Attention mechanism
    - Multi-Horizon (24h, 48h, 72h) simultaneous regression
    - Non-crossing Quantile outputs (q10, q50, q90) via cumulative positive increments
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 32,
        num_stations: int = 15,
        num_horizons: int = 3,  # 24h, 48h, 72h
        dropout: float = 0.20
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_stations = num_stations
        self.num_horizons = num_horizons

        # 1. Temporal Sequence Encoder: Lightweight 1-layer GRU
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 2. Spatial Station Attention Layer
        self.spatial_attn = SpatialStationAttention(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Multi-Horizon Quantile Heads (Outputs q10, delta_50, delta_90 for each horizon)
        # 3 horizons * 3 outputs = 9 outputs per station
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_horizons * 3)
        )

    def forward(
        self,
        x: torch.Tensor,
        dist_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, num_stations, seq_len, in_features]
            dist_matrix: [num_stations, num_stations]
        Returns:
            quantiles: [B, num_stations, num_horizons, 3] where last dim is [q10, q50, q90]
        """
        B, N, T, F_in = x.shape

        # Reshape to run GRU over all (B * N) station sequences in parallel
        x_flat = x.view(B * N, T, F_in)
        _, h_n = self.gru(x_flat)  # h_n: [num_layers, B*N, hidden_dim]
        temporal_emb = h_n[-1].view(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        # Spatial interaction across stations
        spatial_emb = self.spatial_attn(temporal_emb, dist_matrix)
        fused = self.norm(temporal_emb + spatial_emb)

        # Output raw predictions: [B, N, num_horizons * 3]
        raw_preds = self.regressor(fused).view(B, N, self.num_horizons, 3)

        # Monotonic non-crossing quantiles:
        # q10 = base (positive PM2.5 concentration)
        # q50 = q10 + Softplus(delta_1)
        # q90 = q50 + Softplus(delta_2)
        q10 = F.relu(raw_preds[..., 0]) + 15.0  # PM2.5 baseline > 15 ug/m3
        delta_50 = F.softplus(raw_preds[..., 1])
        delta_90 = F.softplus(raw_preds[..., 2])

        q50 = q10 + delta_50
        q90 = q50 + delta_90

        quantiles = torch.stack([q10, q50, q90], dim=-1)  # [B, N, num_horizons, 3]
        return quantiles
