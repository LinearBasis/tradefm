from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- Data ---
    raw_path: str = "data.csv"
    output_dir: str = "data/processed"

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

    # --- Tokenizer calibration ---
    calibration_date: str | None = None  # None = use all available data

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
