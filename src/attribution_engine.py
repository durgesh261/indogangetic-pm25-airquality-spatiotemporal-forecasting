"""
Stubble-Burning Attribution & Counterfactual Analysis Engine.
Quantifies the percentage contribution of upstream agricultural biomass fires
versus local urban baseline emissions during severe Indo-Gangetic smog episodes.
"""

import numpy as np
import torch
from typing import Dict, List, Any


class StubbleAttributionEngine:
    """
    Evaluates causal contribution of upstream satellite fire detections on ground station PM2.5.
    Uses model counterfactual intervention (setting satellite fire intensity feature to zero).
    """

    def __init__(self, model: torch.nn.Module, dist_matrix: torch.Tensor, stations: List[Dict]):
        self.model = model
        self.dist_matrix = dist_matrix
        self.stations = stations

    @torch.no_grad()
    def attribute_pollution(
        self,
        x_sample: torch.Tensor,
        station_id: int,
        fire_feature_idx: int = 5
    ) -> Dict[str, Any]:
        """
        Computes the stubble-burning attribution for a station across 24h, 48h, 72h horizons.
        
        Args:
            x_sample: [1, num_stations, seq_len, num_features]
            station_id: station index (0 to 14)
            fire_feature_idx: index of the satellite fire feature (default 5)
        Returns:
            Dictionary containing predicted PM2.5, counterfactual baseline, and attribution percentage.
        """
        self.model.eval()

        # 1. Actual prediction (with real satellite fire signals)
        preds_actual = self.model(x_sample, self.dist_matrix)  # [1, N, 3 horizons, 3 quantiles]
        # Median predictions (q50) for the target station
        med_actual = preds_actual[0, station_id, :, 1].cpu().numpy()  # [3 horizons]
        q10_actual = preds_actual[0, station_id, :, 0].cpu().numpy()
        q90_actual = preds_actual[0, station_id, :, 2].cpu().numpy()

        # 2. Counterfactual prediction: set upstream satellite fires to 0 (baseline scenario)
        x_counterfactual = x_sample.clone()
        # Set fire feature (index 5) to minimum normalized value (-1.5)
        x_counterfactual[..., fire_feature_idx] = -1.5

        preds_counterfactual = self.model(x_counterfactual, self.dist_matrix)
        med_counterfactual = preds_counterfactual[0, station_id, :, 1].cpu().numpy()

        # 3. Calculate Stubble Burning Contribution
        station_name = self.stations[station_id]["name"]
        horizons_hours = [24, 48, 72]
        horizon_reports = []

        for h_idx, h in enumerate(horizons_hours):
            tot_pm25 = float(med_actual[h_idx])
            base_pm25 = float(np.clip(med_counterfactual[h_idx], 20.0, tot_pm25))
            stubble_pm25 = float(max(0.0, tot_pm25 - base_pm25))
            stubble_pct = float(round((stubble_pm25 / max(tot_pm25, 1e-4)) * 100.0, 1))

            # AQI Category for PM2.5 according to Central Pollution Control Board (CPCB)
            if tot_pm25 <= 30:
                category = "Good"
            elif tot_pm25 <= 60:
                category = "Satisfactory"
            elif tot_pm25 <= 90:
                category = "Moderate"
            elif tot_pm25 <= 120:
                category = "Poor"
            elif tot_pm25 <= 250:
                category = "Very Poor"
            else:
                category = "Severe / Hazardous"

            horizon_reports.append({
                "horizon_hours": h,
                "predicted_pm25_median": round(tot_pm25, 1),
                "confidence_interval_80pct": [round(float(q10_actual[h_idx]), 1), round(float(q90_actual[h_idx]), 1)],
                "local_urban_baseline_pm25": round(base_pm25, 1),
                "stubble_burning_pm25": round(stubble_pm25, 1),
                "stubble_burning_share_pct": stubble_pct,
                "cpcb_aqi_category": category
            })

        avg_stubble_share = float(np.mean([r["stubble_burning_share_pct"] for r in horizon_reports]))
        summary_text = (
            f"At {station_name}, the 24-72h predicted PM2.5 levels indicate {horizon_reports[0]['cpcb_aqi_category']} air quality. "
            f"Upstream agricultural biomass burning accounts for approximately {avg_stubble_share:.1f}% "
            f"of total particulate loading under current northwesterly wind transport conditions."
        )

        return {
            "station_id": station_id,
            "station_name": station_name,
            "state": self.stations[station_id]["state"],
            "horizon_breakdown": horizon_reports,
            "average_stubble_share_pct": round(avg_stubble_share, 1),
            "executive_summary": summary_text
        }
