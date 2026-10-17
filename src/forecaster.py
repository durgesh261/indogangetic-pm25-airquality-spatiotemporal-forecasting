"""
Training, Validation, and Multi-Horizon Forecaster Manager for Air Quality Modeling.
Computes MAE, RMSE, and Prediction Interval Coverage Probability (PICP) for uncertainty evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


class SpatiotemporalForecasterManager:
    """
    Manages model training, validation, early stopping, and metric evaluation across horizons.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        lr: float = 0.005,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dist_matrix: torch.Tensor,
        batch_size: int = 16
    ) -> float:
        self.model.train()
        n_samples = len(x)
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            b_x = x[batch_idx]
            b_y = y[batch_idx]

            self.optimizer.zero_grad()
            preds = self.model(b_x, dist_matrix)
            loss = self.criterion(preds, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dist_matrix: torch.Tensor
    ) -> Dict[str, Any]:
        self.model.eval()
        preds = self.model(x, dist_matrix)  # [B, N, num_horizons, 3]

        # Extract median point forecast (q50) and interval bounds (q10, q90)
        q10 = preds[..., 0].cpu().numpy()
        q50 = preds[..., 1].cpu().numpy()
        q90 = preds[..., 2].cpu().numpy()
        targets = y.cpu().numpy()  # [B, N, num_horizons]

        # Compute metrics per horizon (24h, 48h, 72h)
        horizon_names = ["24h", "48h", "72h"]
        metrics = {}

        for h_idx, h_name in enumerate(horizon_names):
            y_true = targets[:, :, h_idx]
            y_pred = q50[:, :, h_idx]
            lower = q10[:, :, h_idx]
            upper = q90[:, :, h_idx]

            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            # Prediction Interval Coverage Probability (PICP, target ~ 80%)
            covered = (y_true >= lower) & (y_true <= upper)
            picp = float(np.mean(covered) * 100.0)

            # Mean Prediction Interval Width (MPIW)
            mpiw = float(np.mean(upper - lower))

            metrics[f"mae_{h_name}"] = round(mae, 2)
            metrics[f"rmse_{h_name}"] = round(rmse, 2)
            metrics[f"picp_coverage_pct_{h_name}"] = round(picp, 1)
            metrics[f"interval_width_{h_name}"] = round(mpiw, 1)

        # Average MAE across all horizons
        avg_mae = np.mean([metrics[f"mae_{h}"] for h in horizon_names])
        metrics["overall_avg_mae"] = round(float(avg_mae), 2)

        return metrics

    def fit(
        self,
        data: Dict[str, Any],
        epochs: int = 25,
        verbose: bool = True
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "val_mae": []}
        best_mae = float("inf")
        best_state = None

        x_train, y_train = data["x_train"], data["y_train"]
        x_val, y_val = data["x_val"], data["y_val"]
        x_test, y_test = data["x_test"], data["y_test"]
        dist_matrix = data["dist_matrix"]

        for ep in range(1, epochs + 1):
            train_loss = self.train_epoch(x_train, y_train, dist_matrix)
            val_metrics = self.evaluate(x_val, y_val, dist_matrix)
            val_mae = val_metrics["overall_avg_mae"]

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_mae)

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (ep % 5 == 0 or ep == epochs):
                print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} ug/m3 | 24h Coverage: {val_metrics['picp_coverage_pct_24h']:.1f}%")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_metrics = self.evaluate(x_test, y_test, dist_matrix)
        return {
            "history": history,
            "best_val_mae": best_mae,
            "test_metrics": test_metrics
        }
