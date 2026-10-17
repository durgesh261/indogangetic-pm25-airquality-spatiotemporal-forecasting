"""
Multi-Quantile Pinball Loss Function for Uncertainty Quantification.
Trains the network to output calibrated 10th, 50th (median), and 90th percentile prediction bands.
"""

import torch
import torch.nn as nn


class MultiQuantileLoss(nn.Module):
    """
    Pinball / Quantile Loss for simultaneous multi-horizon prediction intervals.
    
    L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """

    def __init__(self, quantiles=(0.10, 0.50, 0.90)):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: [B, N, num_horizons, 3] predicted quantiles (q10, q50, q90)
            targets: [B, N, num_horizons] ground truth PM2.5 concentrations
        Returns:
            Scalar average pinball loss
        """
        total_loss = 0.0
        # Expand targets to align with quantiles: [B, N, num_horizons, 1]
        targets_exp = targets.unsqueeze(-1)

        for q_idx, q in enumerate(self.quantiles):
            pred_q = preds[..., q_idx:q_idx + 1]  # [B, N, H, 1]
            errors = targets_exp - pred_q
            # Pinball loss: max(q * e, (q - 1) * e)
            loss_q = torch.max(q * errors, (q - 1.0) * errors)
            total_loss = total_loss + loss_q.mean()

        return total_loss / len(self.quantiles)
