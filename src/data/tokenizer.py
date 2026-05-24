"""Tokenizer for MOEX order flow (TradeFM paper §6).

Stateful class with fit/transform/save/load. Calibrated edges include
explicit outlier bins per paper §6.1 / Algorithm 1, and liquidity uses
Average Daily Volume per instrument (paper §6.2) instead of per-day volume.

Bin layout per feature (vocab still 16,384):
  - Double-sided (price_depth, price_level): bin 0 = low outlier, bin n-1 =
    high outlier, bins 1..n-2 = quantile interior.
  - Single-sided (volume, interarrival): bins 0..n-2 = log-uniform interior,
    bin n-1 = high outlier.
  - Liquidity (ADV): plain quantile bins; no outliers (ADV is bounded by data).

`edges` for each feature is a 1D array of length (n_bins - 1) such that
`np.digitize(value, edges, right=False)` returns the bin index in [0, n_bins).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from src.config import PipelineConfig


# --------------------------------------------------------------------------- #
# Edges container                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class BinEdges:
    interarrival: np.ndarray = field(default_factory=lambda: np.array([]))
    price_depth: np.ndarray = field(default_factory=lambda: np.array([]))
    volume: np.ndarray = field(default_factory=lambda: np.array([]))
    price_level: np.ndarray = field(default_factory=lambda: np.array([]))
    liquidity: np.ndarray = field(default_factory=lambda: np.array([]))


# --------------------------------------------------------------------------- #
# Tokenizer                                                                    #
# --------------------------------------------------------------------------- #

class Tokenizer:
    """Calibrate-once, transform-many tokenizer.

    Persisted as JSON: edges + adv map + cfg snapshot. Re-usable across runs
    (pipeline supports `--reuse-tokenizer`).
    """

    def __init__(
        self,
        edges: BinEdges,
        adv: dict[str, float],
        cfg: PipelineConfig,
        extents: dict[str, tuple[float, float]] | None = None,
    ):
        self.edges = edges
        self.adv = adv
        self.cfg = cfg
        # extents[feature] = (min_observed, max_observed) in the same scale as
        # edges (log-space for interarrival/volume, linear elsewhere). Used by
        # the centroid LUT to give honest representatives for outlier bins.
        self.extents: dict[str, tuple[float, float]] = extents or {}
        self._centroid_lut: dict[str, np.ndarray] = self._build_centroid_luts()

    def _build_centroid_luts(self) -> dict[str, np.ndarray]:
        """Precompute centroid value per bin per feature.

        For each bin i, centroid = midpoint(lower_boundary, upper_boundary),
        where boundaries are [extent_min, *edges, extent_max]. Returns values
        in the natural (raw) scale: log-space features go through expm1.
        """
        spec = {
            "interarrival": (self.cfg.n_bins_interarrival, True),   # log space
            "price_depth":  (self.cfg.n_bins_price_depth, False),
            "volume":       (self.cfg.n_bins_volume, True),         # log space
            "price_level":  (self.cfg.n_bins_price_level, False),
            "liquidity":    (self.cfg.n_liquidity_bins, False),
        }
        luts: dict[str, np.ndarray] = {}
        for feat, (n_bins, log_space) in spec.items():
            edges = np.asarray(getattr(self.edges, feat), dtype=np.float64)
            if len(edges) == 0:
                luts[feat] = np.zeros(n_bins)
                continue
            # Fall back to edges[0]/edges[-1] if extents missing (old tokenizer.json
            # files lack them — produces the previous, slightly biased centroids).
            lo, hi = self.extents.get(feat, (float(edges[0]), float(edges[-1])))
            bounds = np.concatenate([[lo], edges, [hi]])
            cent = 0.5 * (bounds[:-1] + bounds[1:])
            luts[feat] = np.expm1(cent) if log_space else cent
        return luts

    @property
    def instruments(self) -> list[str]:
        """Instruments this tokenizer was calibrated on, sorted alphabetically
        (matches OrderFlowDataset.instrument_map convention).

        Derived from ADV keys, which is the source of truth.
        """
        return sorted(self.adv.keys())

    # ----------------------------------------------------------------- fit ---

    @classmethod
    def fit_from_features(
        cls,
        features: pl.DataFrame,
        adv_per_day: dict[str, list[float]],
        cfg: PipelineConfig,
    ) -> "Tokenizer":
        """Pure-compute fit from already-aggregated feature columns and per-day ADV map.

        `features` must contain: interarrival, log_volume, price_depth, price_level.
        `adv_per_day[seccode]` is the list of per-day daily_volume across training days.
        """
        edges = BinEdges()
        extents: dict[str, tuple[float, float]] = {}

        def _extent(arr: np.ndarray) -> tuple[float, float]:
            arr = arr[np.isfinite(arr)]
            return (float(arr.min()), float(arr.max())) if len(arr) else (0.0, 0.0)

        # Interarrival: log-uniform single-sided. Extent stored in log space.
        iat = features["interarrival"].drop_nulls().to_numpy()
        iat = iat[iat > 0]
        iat_log = np.log1p(iat)
        edges.interarrival = _calibrate_one_sided(
            iat_log, cfg.n_bins_interarrival, cfg.outlier_upper_pct,
        )
        extents["interarrival"] = _extent(iat_log)

        # log_volume: already in log-space.
        vol = features["log_volume"].drop_nulls().to_numpy()
        edges.volume = _calibrate_one_sided(
            vol, cfg.n_bins_volume, cfg.outlier_upper_pct,
        )
        extents["volume"] = _extent(vol)

        # Price depth / price level: quantile double-sided.
        pd_vals = features["price_depth"].drop_nulls().to_numpy()
        edges.price_depth = _calibrate_two_sided(
            pd_vals, cfg.n_bins_price_depth, cfg.outlier_lower_pct, cfg.outlier_upper_pct,
        )
        extents["price_depth"] = _extent(pd_vals)

        pl_vals = features["price_level"].drop_nulls().to_numpy()
        edges.price_level = _calibrate_two_sided(
            pl_vals, cfg.n_bins_price_level, cfg.outlier_lower_pct, cfg.outlier_upper_pct,
        )
        extents["price_level"] = _extent(pl_vals)

        # ADV: per-asset mean of per-day daily volume (paper §6.2).
        adv = {sec: float(np.mean(vals)) for sec, vals in adv_per_day.items()}

        # Liquidity bins: plain quantile over ADV distribution; no outliers.
        adv_values = np.array(list(adv.values()), dtype=np.float64)
        if len(adv_values) >= cfg.n_liquidity_bins:
            n_cuts = cfg.n_liquidity_bins - 1
            qs = np.linspace(1.0 / cfg.n_liquidity_bins, 1.0 - 1.0 / cfg.n_liquidity_bins, n_cuts)
            edges.liquidity = np.quantile(adv_values, qs)
        else:
            edges.liquidity = np.array([])
        extents["liquidity"] = _extent(adv_values)

        return cls(edges=edges, adv=adv, cfg=cfg, extents=extents)

    # ----------------------------------------------------------- transform ---

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Digitize, look up bin_liquidity per ADV, compose trade_token.

        Input columns required: interarrival, log_volume, price_depth,
        price_level, order_action, side, SECCODE.
        Output adds: bin_interarrival, bin_price_depth, bin_volume,
        bin_price_level, bin_liquidity, trade_token.
        """
        cfg = self.cfg

        iat_log = np.log1p(np.maximum(df["interarrival"].to_numpy(), 0.0))
        i_iat = _digitize(iat_log, self.edges.interarrival)
        i_vol = _digitize(df["log_volume"].to_numpy(), self.edges.volume)
        i_depth = _digitize(df["price_depth"].to_numpy(), self.edges.price_depth)
        i_plevel = _digitize(df["price_level"].to_numpy(), self.edges.price_level)

        # Liquidity: per-instrument constant from ADV lookup.
        seccodes = df["SECCODE"].to_numpy()
        adv_lookup = np.array(
            [self.adv.get(str(s), np.nan) for s in seccodes], dtype=np.float64,
        )
        i_liq = _digitize(adv_lookup, self.edges.liquidity)

        df = df.with_columns(
            bin_interarrival=pl.Series(i_iat, dtype=pl.Int16),
            bin_price_depth=pl.Series(i_depth, dtype=pl.Int16),
            bin_volume=pl.Series(i_vol, dtype=pl.Int16),
            bin_price_level=pl.Series(i_plevel, dtype=pl.Int16),
            bin_liquidity=pl.Series(i_liq, dtype=pl.Int16),
        )

        # Composite trade token, mixed-base (paper §6.2 Eq. 1).
        nd, nv, nt = cfg.n_bins_price_depth, cfg.n_bins_volume, cfg.n_bins_interarrival
        df = df.with_columns(
            trade_token=(
                pl.col("order_action").cast(pl.Int32) * (2 * nd * nv * nt)
                + pl.col("side").cast(pl.Int32) * (nd * nv * nt)
                + pl.col("bin_price_depth").cast(pl.Int32) * (nv * nt)
                + pl.col("bin_volume").cast(pl.Int32) * nt
                + pl.col("bin_interarrival").cast(pl.Int32)
            )
        )
        return df

    # ---------------------------------------------------------- detokenize ---

    def bin_centroid(self, feature: str, bin_idx: int) -> float:
        """Representative continuous value for a bin via precomputed LUT.

        Uses observed (min, max) extents from training data to give honest
        midpoints for outlier bins (bin 0 and bin n-1). For a one-sided log
        feature, bin 0 = midpoint(min_log, edges[0]), inverted via expm1;
        previously returned edges[0] itself (upper bound), which was biased.
        """
        lut = self._centroid_lut.get(feature)
        if lut is None or len(lut) == 0:
            return 0.0
        n_bins = lut.shape[0]
        bin_idx = int(max(0, min(bin_idx, n_bins - 1)))
        return float(lut[bin_idx])

    # ------------------------------------------------------------- persist ---

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "edges": {k: getattr(self.edges, k).tolist() for k in
                      ("interarrival", "price_depth", "volume", "price_level", "liquidity")},
            "adv": self.adv,
            # Observed (min, max) per feature in the edges' scale (log for
            # interarrival/volume). Used by bin_centroid for outlier bins.
            "extents": {k: list(v) for k, v in self.extents.items()},
            # Explicit instruments list (also derivable from adv.keys); written
            # for human-readability and forward-compat.
            "instruments": self.instruments,
            "cfg_snapshot": {
                "n_bins_interarrival": self.cfg.n_bins_interarrival,
                "n_bins_price_depth": self.cfg.n_bins_price_depth,
                "n_bins_volume": self.cfg.n_bins_volume,
                "n_bins_price_level": self.cfg.n_bins_price_level,
                "n_liquidity_bins": self.cfg.n_liquidity_bins,
                "outlier_lower_pct": self.cfg.outlier_lower_pct,
                "outlier_upper_pct": self.cfg.outlier_upper_pct,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path, cfg: PipelineConfig | None = None) -> "Tokenizer":
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)

        edges = BinEdges(
            interarrival=np.array(payload["edges"]["interarrival"], dtype=np.float64),
            price_depth=np.array(payload["edges"]["price_depth"], dtype=np.float64),
            volume=np.array(payload["edges"]["volume"], dtype=np.float64),
            price_level=np.array(payload["edges"]["price_level"], dtype=np.float64),
            liquidity=np.array(payload["edges"]["liquidity"], dtype=np.float64),
        )
        adv = {str(k): float(v) for k, v in payload["adv"].items()}
        # Old tokenizer.json files lack "extents" → bin_centroid silently falls
        # back to edges[0]/edges[-1] for outlier bins (the previous behaviour).
        extents_raw = payload.get("extents", {})
        extents = {k: (float(v[0]), float(v[1])) for k, v in extents_raw.items()}
        if cfg is None:
            cfg = PipelineConfig()
        for k, v in payload["cfg_snapshot"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cls(edges=edges, adv=adv, cfg=cfg, extents=extents)


# --------------------------------------------------------------------------- #
# Calibration helpers                                                          #
# --------------------------------------------------------------------------- #

def _calibrate_two_sided(
    values: np.ndarray, n_bins: int, lo_pct: float, hi_pct: float,
) -> np.ndarray:
    """Edges for two-sided binning with low/high outlier bins.

    Returns array of length (n_bins - 1): [lo_thr, *interior_cuts, hi_thr].
    Bin 0 captures values < lo_thr; bin n-1 captures values > hi_thr;
    bins 1..n-2 are equal-frequency quantile partitions of the [lo_thr, hi_thr] range.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros(n_bins - 1, dtype=np.float64)
    lo_thr = float(np.percentile(finite, lo_pct))
    hi_thr = float(np.percentile(finite, hi_pct))
    interior = finite[(finite >= lo_thr) & (finite <= hi_thr)]
    n_main = n_bins - 2  # 2 outlier bins reserved
    if n_main <= 1 or len(interior) == 0:
        cuts = np.array([])
    else:
        qs = np.linspace(1.0 / n_main, 1.0 - 1.0 / n_main, n_main - 1)
        cuts = np.quantile(interior, qs)
    return np.concatenate([[lo_thr], cuts, [hi_thr]])


def _calibrate_one_sided(
    log_values: np.ndarray, n_bins: int, hi_pct: float,
) -> np.ndarray:
    """Edges for one-sided log-uniform binning with high outlier bin.

    `log_values` is already in log-space. Returns array of length (n_bins - 1):
    [*interior_cuts, hi_thr]. Bins 0..n-2 are equal-width in log-space within
    [min, hi_thr]; bin n-1 captures values > hi_thr.
    """
    finite = log_values[np.isfinite(log_values)]
    if len(finite) == 0:
        return np.zeros(n_bins - 1, dtype=np.float64)
    hi_thr = float(np.percentile(finite, hi_pct))
    interior = finite[finite <= hi_thr]
    n_main = n_bins - 1  # 1 outlier bin reserved
    if n_main <= 1 or len(interior) == 0:
        cuts = np.array([])
    else:
        cuts = np.linspace(float(interior.min()), hi_thr, n_main + 1)[1:-1]
    return np.concatenate([cuts, [hi_thr]])


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """np.digitize with NaN→0 safety. Returns int32 bin indices in [0, len(edges)]."""
    if len(edges) == 0:
        return np.zeros(len(values), dtype=np.int32)
    safe = np.where(np.isfinite(values), values, edges[0])
    return np.digitize(safe, edges, right=False).astype(np.int32)


# --------------------------------------------------------------------------- #
# Composite token decode (for diagnostics / per-subtoken accuracy)             #
# --------------------------------------------------------------------------- #

def decompose_token_torch(token, factors: tuple[int, int, int, int, int]):
    """Vectorized torch-tensor inverse of compose_trade_token.

    Mirrors decode_trade_token but returns a 5-tuple of tensors instead of a
    dict, and uses torch.div(rounding_mode='floor') for correct integer
    division across torch dtypes. Same arithmetic as Tokenizer.transform's
    composition step (paper §6.2 Eq. 1).
    """
    import torch  # local import — tokenizer module stays torch-free otherwise
    _, n_s, n_d, n_v, n_t = factors
    block_a = n_s * n_d * n_v * n_t
    block_s = n_d * n_v * n_t
    block_d = n_v * n_t
    i_action = torch.div(token, block_a, rounding_mode="floor")
    rem = token % block_a
    i_side = torch.div(rem, block_s, rounding_mode="floor")
    rem = rem % block_s
    i_depth = torch.div(rem, block_d, rounding_mode="floor")
    rem = rem % block_d
    i_vol = torch.div(rem, n_t, rounding_mode="floor")
    i_iat = rem % n_t
    return i_action, i_side, i_depth, i_vol, i_iat


def subtoken_factors(vocab_size: int) -> tuple[int, int, int, int, int]:
    """Infer (n_action, n_side, n_depth, n_volume, n_interarrival) from vocab.

    Assumes paper layout: n_action = n_side = 2 and the remaining three
    feature bin counts are equal (n_depth == n_volume == n_interarrival).
    For default vocab = 16,384 this yields (2, 2, 16, 16, 16).
    """
    n_a, n_s = 2, 2
    cube = vocab_size // (n_a * n_s)
    n_per = round(cube ** (1.0 / 3.0))
    if n_a * n_s * n_per ** 3 != vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} does not factor as n_a*n_s*n_per^3 "
            f"with n_per integer; got cube={cube}, n_per={n_per}"
        )
    return n_a, n_s, n_per, n_per, n_per


def decode_trade_token(token, cfg: PipelineConfig) -> dict:
    """Inverse of the mixed-base encoding in Tokenizer.transform.

    Accepts an int or np.ndarray. Returns dict with int / ndarray per component.
    """
    nd = cfg.n_bins_price_depth
    nv = cfg.n_bins_volume
    nt = cfg.n_bins_interarrival

    block_action = 2 * nd * nv * nt
    block_side = nd * nv * nt
    block_depth = nv * nt

    i_action = token // block_action
    rem = token % block_action
    i_side = rem // block_side
    rem = rem % block_side
    i_depth = rem // block_depth
    rem = rem % block_depth
    i_vol = rem // nt
    i_iat = rem % nt

    return {
        "action": i_action,
        "side": i_side,
        "price_depth": i_depth,
        "volume": i_vol,
        "interarrival": i_iat,
    }
