"""Decision heads: alpha, risk, intensity over frozen transformer latents."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionModule(nn.Module):
    """Three linear probes over transformer hidden states.

    Heads:
        alpha  — expected forward return (signed, no activation)
        risk   — realized volatility (positive, softplus)
        intensity — normalized trade intensity (positive, softplus)

    Input:  (B, T, d_model) hidden states
    Output: dict of (B, T) predictions per head
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.alpha_head = nn.Linear(d_model, 1)
        self.risk_head = nn.Linear(d_model, 1)
        self.intensity_head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for head in [self.alpha_head, self.risk_head, self.intensity_head]:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        # Bias risk and intensity heads toward reasonable positive values
        # softplus(0.5) ≈ 1.0
        self.risk_head.bias.data.fill_(0.5)
        self.intensity_head.bias.data.fill_(0.5)

    def forward(
        self, hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden: (B, T, d_model) transformer hidden states

        Returns:
            dict with keys "alpha", "risk", "intensity", each (B, T)
        """
        return {
            "alpha": self.alpha_head(hidden).squeeze(-1),
            "risk": F.softplus(self.risk_head(hidden).squeeze(-1)),
            "intensity": F.softplus(self.intensity_head(hidden).squeeze(-1)),
        }

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: torch.Tensor,
        huber_delta: float = 1e-4,
        w_alpha: float = 1.0,
        w_risk: float = 1.0,
        w_intensity: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute per-head losses with NaN masking.

        Args:
            preds:   dict of (B, T) predictions
            targets: dict of (B, T) target values
            mask:    (B, T) bool — True where targets are valid
            huber_delta: delta for Huber loss (alpha head)
            w_alpha, w_risk, w_intensity: loss weights

        Returns:
            dict with per-head losses and "total"
        """
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=mask.device, requires_grad=True)
            return {"alpha": zero, "risk": zero, "intensity": zero, "total": zero}

        # Alpha: Huber loss
        loss_alpha = F.huber_loss(
            preds["alpha"][mask],
            targets["alpha"][mask],
            delta=huber_delta,
            reduction="mean",
        )

        # Risk: MSE in log space
        eps = 1e-8
        loss_risk = F.mse_loss(
            torch.log(preds["risk"][mask] + eps),
            torch.log(targets["risk"][mask] + eps),
        )

        # Intensity: MSE in log space
        loss_intensity = F.mse_loss(
            torch.log(preds["intensity"][mask] + eps),
            torch.log(targets["intensity"][mask] + eps),
        )

        total = w_alpha * loss_alpha + w_risk * loss_risk + w_intensity * loss_intensity

        return {
            "alpha": loss_alpha,
            "risk": loss_risk,
            "intensity": loss_intensity,
            "total": total,
        }
