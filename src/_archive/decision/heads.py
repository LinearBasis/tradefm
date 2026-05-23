"""Decision heads: alpha, risk, intensity over frozen transformer latents."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(d_in: int, d_hidden: int, dropout: float = 0.1) -> nn.Sequential:
    """Two-layer MLP: d_in → d_hidden → 1, with GELU + dropout in the middle."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, 1),
    )


class DecisionModule(nn.Module):
    """Three MLP heads over transformer hidden states.

    Each head is a 2-layer MLP (d_model → d_model//2 → 1) with GELU activation.
    Compared with the previous linear-probe version (~3·d_model params), this
    gives each head ~d_model² / 2 params, enabling non-linear combinations of
    latent features.

    Heads:
        alpha     — expected forward return (signed, no activation)
        risk      — realized volatility (positive, softplus)
        intensity — normalized trade intensity (positive, softplus)

    Input:  (B, T, d_model) hidden states
    Output: dict of (B, T) predictions per head
    """

    def __init__(self, d_model: int, hidden_mult: float = 0.5, dropout: float = 0.1):
        super().__init__()
        d_hidden = max(int(d_model * hidden_mult), 16)
        self.alpha_head = _build_mlp(d_model, d_hidden, dropout)
        self.risk_head = _build_mlp(d_model, d_hidden, dropout)
        self.intensity_head = _build_mlp(d_model, d_hidden, dropout)
        self._init_weights()

    def _init_weights(self):
        """GPT-2 style init for linears, plus positive bias on the final
        layer of risk/intensity heads so softplus output starts near 1.0."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Final-layer bias on risk and intensity → softplus(0.5) ≈ 0.97
        self.risk_head[-1].bias.data.fill_(0.5)
        self.intensity_head[-1].bias.data.fill_(0.5)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
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
            huber_delta: delta for Huber loss (alpha head). Targets are scaled
                         ~1e-4 (per-second relative return), so 1e-4 is sensible.
            w_alpha, w_risk, w_intensity: loss weights. Note that raw scales
                differ by orders of magnitude — α-Huber ~1e-7 vs σ/κ log-MSE ~10.
                Use w_alpha >> 1 to balance gradients across heads.

        Returns:
            dict with per-head losses and "total"
        """
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=mask.device, requires_grad=True)
            return {"alpha": zero, "risk": zero, "intensity": zero, "total": zero}

        loss_alpha = F.huber_loss(
            preds["alpha"][mask],
            targets["alpha"][mask],
            delta=huber_delta,
            reduction="mean",
        )

        eps = 1e-8
        loss_risk = F.mse_loss(
            torch.log(preds["risk"][mask] + eps),
            torch.log(targets["risk"][mask] + eps),
        )
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
