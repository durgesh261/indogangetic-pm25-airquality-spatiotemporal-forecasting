"""
End-to-End Demonstration: Indo-Gangetic PM2.5 Spatiotemporal Smog Forecasting & Stubble Attribution.
Trains spatial-temporal attention network with Multi-Quantile Loss, benchmarks on test set,
and quantifies upstream agricultural fire contributions across Delhi-NCR ground monitoring stations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.air_quality_dataset import IndoGangeticAirQualityDataset, STATIONS
from src.spatiotemporal_model import SpatiotemporalPM25Forecaster
from src.quantile_loss import MultiQuantileLoss
from src.forecaster import SpatiotemporalForecasterManager
from src.attribution_engine import StubbleAttributionEngine


def run_full_demo():
    print("=" * 80)
    print("  INDO-GANGETIC PM2.5 SPATIOTEMPORAL FORECASTING & STUBBLE BURNING ATTRIBUTION  ")
    print("=" * 80)

    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # 1. Generate Dataset
    print("\n[Step 1] Loading Indo-Gangetic CPCB stations & satellite active fire data...")
    dataset_gen = IndoGangeticAirQualityDataset(num_days=60, seq_len=48, random_state=42)
    data = dataset_gen.generate()

    print(f" -> Monitored Ground Stations: {len(data['stations'])} (Delhi-NCR, Punjab, Haryana, UP)")
    print(f" -> Train Samples: {len(data['x_train'])} | Val: {len(data['x_val'])} | Test: {len(data['x_test'])}")
    print(f" -> Forecast Horizons: 24h, 48h, 72h ahead")
    print(f" -> Multi-Quantile Prediction Bands: 10th percentile (lower), 50th (median), 90th (upper)")

    # 2. Build Model & Loss
    print("\n[Step 2] Initializing Spatiotemporal Attention Network & Multi-Quantile Pinball Loss...")
    model = SpatiotemporalPM25Forecaster(in_features=6, hidden_dim=32, num_stations=len(data['stations'])).to(device)
    criterion = MultiQuantileLoss(quantiles=(0.10, 0.50, 0.90))
    manager = SpatiotemporalForecasterManager(model=model, criterion=criterion, lr=0.008, weight_decay=1e-4)

    # Move tensors to device
    for k in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test", "dist_matrix"]:
        data[k] = data[k].to(device)

    # 3. Train
    print("\n[Step 3] Training Spatiotemporal Network on 48-hour lag sequences...")
    results = manager.fit(data, epochs=15, verbose=True)

    # 4. Benchmark Evaluation
    test_metrics = results["test_metrics"]
    print("\n[Step 4] Held-Out Test Evaluation Results across Forecast Horizons:")
    print("-" * 65)
    print(f" -> 24h Horizon | MAE: {test_metrics['mae_24h']} ug/m3 | RMSE: {test_metrics['rmse_24h']} | 80% Coverage (PICP): {test_metrics['picp_coverage_pct_24h']}%")
    print(f" -> 48h Horizon | MAE: {test_metrics['mae_48h']} ug/m3 | RMSE: {test_metrics['rmse_48h']} | 80% Coverage (PICP): {test_metrics['picp_coverage_pct_48h']}%")
    print(f" -> 72h Horizon | MAE: {test_metrics['mae_72h']} ug/m3 | RMSE: {test_metrics['rmse_72h']} | 80% Coverage (PICP): {test_metrics['picp_coverage_pct_72h']}%")
    print(f" -> Overall Test MAE: {test_metrics['overall_avg_mae']} ug/m3")
    print("-" * 65)

    # 5. Stubble Burning Attribution
    print("\n[Step 5] Performing Upstream Stubble-Burning Counterfactual Attribution...")
    engine = StubbleAttributionEngine(model, data["dist_matrix"], data["stations"])

    # Analyze Delhi - Anand Vihar (Station 0) during peak winter smog stubble-burning episode
    peak_idx = int(torch.argmax(data["x_val"][..., 5].mean(dim=(1, 2))).item())
    peak_sample = data["x_val"][peak_idx:peak_idx + 1]
    anand_vihar_report = engine.attribute_pollution(peak_sample, station_id=0)

    print(f"\nAttribution Summary for {anand_vihar_report['station_name']}:")
    print(f" -> {anand_vihar_report['executive_summary']}")
    for r in anand_vihar_report["horizon_breakdown"]:
        print(f"    * {r['horizon_hours']}h Forecast: Median {r['predicted_pm25_median']} ug/m3 [{r['confidence_interval_80pct'][0]} - {r['confidence_interval_80pct'][1]}] | Stubble Share: {r['stubble_burning_share_pct']}% ({r['cpcb_aqi_category']})")

    # 6. Generate Publication-Quality Visualizations
    print("\n[Step 6] Saving publication figures to outputs/...")
    model.eval()
    with torch.no_grad():
        test_preds = model(data["x_test"], data["dist_matrix"]).cpu().numpy()
        test_true = data["y_test"].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Subplot 1: Delhi Anand Vihar 24h rolling forecast with 80% confidence interval
    station_idx = 0  # Anand Vihar
    y_true_series = test_true[:, station_idx, 0]  # 24h true
    q10_series = test_preds[:, station_idx, 0, 0]
    q50_series = test_preds[:, station_idx, 0, 1]
    q90_series = test_preds[:, station_idx, 0, 2]

    time_steps = np.arange(len(y_true_series))
    axes[0].plot(time_steps, y_true_series, color="black", lw=1.8, label="Ground Truth PM2.5 (CPCB Sensor)")
    axes[0].plot(time_steps, q50_series, color="#d9534f", lw=2.0, label="Spatiotemporal Forecast (Median q50)")
    axes[0].fill_between(time_steps, q10_series, q90_series, color="#d9534f", alpha=0.25, label="80% Calibrated Prediction Interval [q10, q90]")

    # CPCB Warning lines
    axes[0].axhline(y=250, color="purple", linestyle="--", alpha=0.7, label="CPCB 'Severe' Threshold (250 ug/m3)")
    axes[0].axhline(y=120, color="orange", linestyle=":", alpha=0.7, label="CPCB 'Poor' Threshold (120 ug/m3)")

    axes[0].set_title(f"24h Ahead Rolling PM2.5 Forecast: Delhi - Anand Vihar", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Test Timeline (6-hour intervals)", fontsize=10)
    axes[0].set_ylabel("PM2.5 Concentration (ug/m3)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Stubble Burning Attribution across Delhi-NCR stations
    ncr_station_indices = [0, 1, 2, 3, 4, 5]
    ncr_names = [data["stations"][i]["name"].replace("Delhi - ", "") for i in ncr_station_indices]
    stubble_shares = []
    urban_shares = []

    for s_id in ncr_station_indices:
        rep = engine.attribute_pollution(peak_sample, station_id=s_id)
        share = rep["average_stubble_share_pct"]
        stubble_shares.append(share)
        urban_shares.append(100.0 - share)

    x_pos = np.arange(len(ncr_names))
    axes[1].bar(x_pos, urban_shares, color="#5bc0de", edgecolor="black", label="Local Urban Baseline (Vehicular/Industry)")
    axes[1].bar(x_pos, stubble_shares, bottom=urban_shares, color="#d9534f", edgecolor="black", label="Upstream Stubble-Burning Transport")

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(ncr_names, rotation=25, ha="right", fontsize=9)
    axes[1].set_title("Source Attribution: Upstream Stubble vs Urban Baseline", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Attribution Share (%)", fontsize=10)
    axes[1].set_ylim(0, 115)
    axes[1].legend(loc="upper right", fontsize=9, frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join("outputs", "pm25_spatiotemporal_forecast_and_attribution.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f" -> Output chart saved to: {output_path}")
    print("\n[SUCCESS] Indo-Gangetic PM2.5 Spatiotemporal Demonstration Completed Successfully!\n")


if __name__ == "__main__":
    run_full_demo()
