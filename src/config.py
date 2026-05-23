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
    n_heads: int = 4         # Query heads
    n_kv_heads: int = 1      # Key/Value heads for GQA. Must divide n_heads.
    n_layers: int = 4
    d_ff: int = 512          # SwiGLU intermediate size (typically 4*d_model in Llama family)
    d_context: int = 32
    context_length: int = 512
    dropout: float = 0.1
    # Apply RMSNorm to Q and K before SDPA (Gemma 2 / DeepSeek-V4 style).
    # Stabilizes training, plays well with Muon. Default off → old checkpoints compatible.
    qk_norm: bool = False

    # --- Vocab (derived from PipelineConfig, but stored for standalone use) ---
    vocab_size: int = 16_384
    n_price_level_bins: int = 32
    n_liquidity_bins: int = 3
    n_instruments: int = 20
    # Paper-faithful default: no asset-specific embedding (scale-invariant features
    # are the whole point of TradeFM). Enable for ablation on small-N datasets.
    use_instrument_emb: bool = False

    # --- Training ---
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_epochs: int = 20
    warmup_fraction: float = 0.05
    grad_clip_norm: float = 1.0
    val_days: int = 5  # last N days → validation (mirrors PipelineConfig)
    # Optimizer: "adamw" (default) or "muon_hybrid" (Muon for 2D matrices + AdamW for the rest).
    # Old configs without this field stay on AdamW.
    optimizer: str = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_update_rescale: float = 0.2

    # --- Dataset ---
    stride: int = 256
    sequences_dir: str = "data/processed/sequences"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    patience: int = 5  # early stopping patience (epochs)

    # --- Stylized-fact rollout (paper §9.1-9.2) ---
    # 0 = disabled; otherwise run a reduced closed-loop rollout every N epochs +
    # at the end of training, logging stylized-fact metrics to TensorBoard.
    rollout_every_n_epochs: int = 0
    rollout_n_events: int = 512
    rollout_n_rollouts: int = 3
    rollout_init_mid: float = 100.0
    tokenizer_path: str = "data/processed/tokenizer.json"


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
