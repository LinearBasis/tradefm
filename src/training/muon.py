"""Muon optimizer (DeepSeek-V4 Algorithm 1) with hybrid Newton-Schulz orthogonalization.

References:
- Jordan et al. 2024 (NanoGPT speedrun)
- Liu et al. 2025 (Moonshot Kimi K2)
- DeepSeek-V4 paper, Section 2.4 + Algorithm 1

For 2D matrix parameters only. Use AdamW for embeddings, biases, and norm scales.
"""

import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz_hybrid(
    G: torch.Tensor,
    phase1_steps: int = 8,
    phase2_steps: int = 2,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Hybrid two-phase Newton-Schulz orthogonalization (DeepSeek-V4).

    Approximately maps G = U·Σ·V^T to U·V^T (i.e. singular values → 1).

    Phase 1 (default 8 iters): (a, b, c) = (3.4445, -4.7750, 2.0315) — rapid convergence.
    Phase 2 (default 2 iters): (a, b, c) = (2, -1.5, 0.5) — stabilizes singular values at 1.

    Input must be 2D. Runs in bf16 for speed; output cast back to input dtype.
    """
    assert G.ndim == 2, f"newton_schulz_hybrid expects a 2D matrix, got shape {tuple(G.shape)}"

    orig_dtype = G.dtype
    X = G.to(torch.bfloat16) if orig_dtype != torch.bfloat16 else G.clone()
    X = X / (X.norm() + eps)

    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    # Phase 1: rapid convergence
    a1, b1, c1 = 3.4445, -4.7750, 2.0315
    for _ in range(phase1_steps):
        A = X @ X.T
        B = b1 * A + c1 * (A @ A)
        X = a1 * X + B @ X

    # Phase 2: stabilize singular values at 1
    a2, b2, c2 = 2.0, -1.5, 0.5
    for _ in range(phase2_steps):
        A = X @ X.T
        B = b2 * A + c2 * (A @ A)
        X = a2 * X + B @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


class Muon(Optimizer):
    """Muon optimizer for 2D matrix parameters (DeepSeek-V4 Algorithm 1).

    Update rule per step:
        G_t  = grad
        M_t  = μ·M_{t-1} + G_t
        O'_t = HybridNewtonSchulz(μ·M_t + G_t)         # Nesterov-style input
        O_t  = O'_t · sqrt(max(n, m)) · γ              # rescale to AdamW-equivalent magnitude
        W_t  = W_{t-1}·(1 − η·λ) − η·O_t                # multiplicative weight decay + update

    Args:
        params: iterable of 2D parameter tensors (not embeddings/biases/norms).
        lr: learning rate η. Typical ~0.02 for transformer linear weights.
        momentum: μ. Typical 0.95.
        weight_decay: λ. Applied multiplicatively (`W *= 1 - lr*λ`).
        update_rescale: γ. Maps Muon step magnitude to the AdamW-equivalent range so
                        the AdamW LR scale roughly carries over. Default 0.2.
        ns_phase1_steps / ns_phase2_steps: Newton-Schulz iteration counts.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        update_rescale: float = 0.2,
        ns_phase1_steps: int = 8,
        ns_phase2_steps: int = 2,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_rescale=update_rescale,
            ns_phase1_steps=ns_phase1_steps,
            ns_phase2_steps=ns_phase2_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            gamma = group["update_rescale"]
            p1 = group["ns_phase1_steps"]
            p2 = group["ns_phase2_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon got a non-2D parameter (shape {tuple(p.shape)}). "
                        "Route embeddings, biases, and norm scales through AdamW instead."
                    )

                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                # Standard momentum: M_t = μ·M_{t-1} + G_t
                buf.mul_(mu).add_(g)
                # Nesterov-style input to Newton-Schulz: μ·M_t + G_t
                ns_input = mu * buf + g
                o_prime = newton_schulz_hybrid(ns_input, p1, p2)

                # Rescale to AdamW-equivalent magnitude
                n, m = p.shape
                scale = (max(n, m) ** 0.5) * gamma

                # Multiplicative weight decay + update
                p.data.mul_(1.0 - lr * wd)
                p.data.add_(o_prime, alpha=-lr * scale)

        return loss


def split_params_for_muon(model: torch.nn.Module) -> tuple[list, list, list, list]:
    """Split model parameters into Muon-eligible vs AdamW-only groups.

    Following DeepSeek-V4 Section 2.4:
      AdamW handles: embeddings, output head (if untied), biases, RMSNorm/LayerNorm scales.
      Muon handles: all other 2D matrix parameters (linear-layer weights).

    Returns:
        (muon_params, adamw_params, muon_names, adamw_names) — names returned for logging.
    """
    muon_params: list = []
    adamw_params: list = []
    muon_names: list = []
    adamw_names: list = []
    seen_ids: set[int] = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip duplicates from weight tying (e.g. head.weight === token_emb.weight).
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))

        lower = name.lower()
        is_2d = p.ndim == 2
        is_embedding = "emb" in lower or "embedding" in lower
        is_norm = "norm" in lower or ".ln" in lower or "layernorm" in lower
        is_bias = lower.endswith(".bias") or lower.endswith("_bias")
        # Output projection head: tied with token_emb, but if untied a separate `head.weight`
        # of shape [vocab, d_model] would land here. Treat it as embedding-like.
        is_head = name.split(".")[-2:][0] == "head" if "." in name else False

        if is_2d and not is_embedding and not is_norm and not is_bias and not is_head:
            muon_params.append(p)
            muon_names.append(name)
        else:
            adamw_params.append(p)
            adamw_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


class HybridMuonAdamW:
    """Thin wrapper that drives Muon and AdamW as a single 'optimizer'.

    Exposes the standard subset of `torch.optim.Optimizer` interface used by the training
    loop: `step()`, `zero_grad()`, `state_dict()`, `load_state_dict()`, and `param_groups`.

    State dict layout:
        { "muon": Muon.state_dict(), "adamw": AdamW.state_dict() }
    """

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float,
        muon_momentum: float,
        muon_weight_decay: float,
        muon_update_rescale: float,
        adamw_lr: float,
        adamw_weight_decay: float,
        adamw_betas: tuple[float, float],
    ):
        muon_params, adamw_params, muon_names, adamw_names = split_params_for_muon(model)
        self.muon_param_names = muon_names
        self.adamw_param_names = adamw_names

        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            update_rescale=muon_update_rescale,
        )
        # fused=True selects the CUDA multi-tensor-apply kernel on cuda devices,
        # silently falls back to the non-fused path on cpu/mps. ~5-10% faster
        # per optimizer step at no cost.
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            weight_decay=adamw_weight_decay,
            betas=adamw_betas,
            fused=True,
        )

    @property
    def param_groups(self):
        """Combined view of both sub-optimizers' groups (used by warmup/LR schedulers)."""
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        if "muon" in state_dict and "adamw" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])
        else:
            # Backward-compat: prior runs used a plain AdamW state. Try to load into AdamW only.
            self.adamw.load_state_dict(state_dict)
