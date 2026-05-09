import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def load_from_json(cls: type[T], path: str | Path) -> T:
    """Load a dataclass config from JSON. Unspecified fields use dataclass defaults.
    Lists in JSON are coerced to tuples for fields whose default is a tuple."""
    with open(path) as f:
        data = json.load(f)
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields in {path}: {sorted(unknown)}")
    instance = cls()
    for k, v in data.items():
        default = getattr(instance, k)
        if isinstance(default, tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(instance, k, v)
    return instance


@dataclass
class ModelConfig:
    # --- Architecture ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    d_context: int = 32
    context_length: int = 512
    dropout: float = 0.1

    # --- Vocab (derived from PipelineConfig, but stored for standalone use) ---
    vocab_size: int = 16_384
    n_price_level_bins: int = 32
    n_liquidity_bins: int = 3
    n_instruments: int = 20

    # --- Training ---
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_epochs: int = 20
    warmup_fraction: float = 0.05
    grad_clip_norm: float = 1.0
    val_days: int = 5  # last N days → validation (mirrors PipelineConfig)

    # --- Dataset ---
    stride: int = 256
    sequences_dir: str = "data/processed/sequences"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    patience: int = 5  # early stopping patience (epochs)


@dataclass
class HeadConfig:
    # --- Targets ---
    tau: int = 512  # forward horizon in events
    session_length_sec: float = 30840.0  # 18:39 - 10:05

    # --- Architecture ---
    d_model: int = 128  # must match ModelConfig.d_model
    latent_layer: int = -1  # which transformer layer to extract from

    # --- Training ---
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 50
    patience: int = 10
    warmup_fraction: float = 0.05

    # --- Loss weights ---
    w_alpha: float = 1.0
    w_risk: float = 1.0
    w_intensity: float = 1.0
    huber_delta: float = 1e-4  # scale of returns is ~1e-4

    # --- Paths ---
    sequences_dir: str = "data/processed/sequences"
    transformer_checkpoint: str = "checkpoints/best.pt"
    checkpoint_dir: str = "checkpoints/heads"


@dataclass
class PipelineConfig:
    # --- Data ---
    raw_dir: str = "data/raw"  # directory with per-day CSVs
    raw_path: str | None = None  # single file fallback (backward compat)
    output_dir: str = "data/processed"

    # --- Session filter (continuous trading only) ---
    continuous_start_sec: float = 36300.0  # 10:05:00 — skip opening auction
    continuous_end_sec: float = 67140.0  # 18:39:00 — skip closing auction

    # --- Instrument filter ---
    top_n_instruments: int | None = 20  # None = use all
    min_events_per_day: int = 5000

    # --- EW-VWAP ---
    ewvwap_halflife_sec: float = 10.0

    # --- Binning ---
    n_bins_price_depth: int = 16
    n_bins_volume: int = 16
    n_bins_interarrival: int = 16
    n_bins_price_level: int = 32
    n_liquidity_bins: int = 3
    outlier_lower_pct: float = 1.0
    outlier_upper_pct: float = 99.0

    # --- Train/val split ---
    val_days: int = 5  # last N days → validation
    calibration_days: int | None = None  # None = all training days

    # --- Parallelism ---
    n_jobs: int = 1  # number of worker processes for per-day stages (1 = serial)
    polars_threads_per_worker: int = 2  # POLARS_MAX_THREADS in each worker

    # --- Features to include (extensible) ---
    predicted_features: list[str] = field(
        default_factory=lambda: ["interarrival", "price_depth", "volume", "action", "side"]
    )
    context_features: list[str] = field(
        default_factory=lambda: ["price_level", "liquidity"]
    )

    @property
    def vocab_size(self) -> int:
        return (
            2  # action: add/cancel
            * 2  # side: buy/sell
            * self.n_bins_price_depth
            * self.n_bins_volume
            * self.n_bins_interarrival
        )
