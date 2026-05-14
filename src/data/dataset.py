"""PyTorch Dataset for tokenized order flow sequences."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.config import ModelConfig


class OrderFlowDataset(Dataset):
    """Sliding-window dataset over tokenized order flow.

    Loads per-instrument-per-day parquet sequences using manifest.json
    for date-based train/val split. Creates overlapping windows of
    length `context_length + 1` with configurable stride.

    If include_targets=True, also loads target_alpha/risk/intensity columns
    (requires pipeline to have been run with target computation).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        split: str = "train",
        sequences_dir: str | None = None,
        include_targets: bool = False,
    ):
        assert split in ("train", "val")
        self.cfg = cfg
        self.context_length = cfg.context_length
        self.include_targets = include_targets

        seq_dir = Path(sequences_dir or cfg.sequences_dir)
        manifest_path = seq_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {seq_dir}. Run the pipeline first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Stable instrument → id mapping from manifest (alphabetical)
        self.instrument_names = manifest["instruments"]
        self.instrument_map = {name: i for i, name in enumerate(self.instrument_names)}
        self.manifest_tau_sec = manifest.get("tau_sec")  # None for pre-fix parquets

        self.windows: list[tuple[np.ndarray, np.ndarray | None, int]] = []

        target_cols = ["target_alpha", "target_risk", "target_intensity"]

        for seq_name, meta in manifest["sequences"].items():
            if meta["split"] != split:
                continue

            # Extract instrument name (everything before last _YYYY-MM-DD)
            seccode = seq_name.rsplit("_", 1)[0]
            inst_id = self.instrument_map[seccode]

            pf = seq_dir / f"{seq_name}.parquet"
            if not pf.exists():
                continue

            df = pl.read_parquet(pf)
            data = np.column_stack([
                df["trade_token"].to_numpy().astype(np.int32),
                df["bin_price_level"].to_numpy().astype(np.int16),
                df["bin_liquidity"].to_numpy().astype(np.int16),
            ])

            targets = None
            if include_targets and all(c in df.columns for c in target_cols):
                targets = np.column_stack([
                    df[c].to_numpy().astype(np.float32) for c in target_cols
                ])
                # Floor risk/intensity to prevent log(0) outliers in MSE loss (NaN preserved).
                # risk: ~3 bp (3 ticks moves), intensity: 1% of daily rate
                targets[:, 1] = np.maximum(targets[:, 1], 3e-4)
                targets[:, 2] = np.maximum(targets[:, 2], 1e-2)

            window_len = cfg.context_length + 1  # input + 1 target token

            for start in range(0, len(data) - window_len + 1, cfg.stride):
                end = start + window_len
                t = targets[start:end] if targets is not None else None
                self.windows.append((data[start:end], t, inst_id))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window, targets, inst_id = self.windows[idx]
        result = {
            "trade_tokens": torch.from_numpy(window[:, 0].astype(np.int64)),
            "price_levels": torch.from_numpy(window[:, 1].astype(np.int64)),
            "liquidities": torch.from_numpy(window[:, 2].astype(np.int64)),
            "instrument_id": torch.tensor(inst_id, dtype=torch.long),
        }

        if targets is not None:
            # NaN → 0.0 in values; separate mask tracks validity
            t = torch.from_numpy(targets)
            valid = ~torch.isnan(t).any(dim=1)  # any-NaN-across-heads → invalid
            t = torch.nan_to_num(t, nan=0.0)
            result["target_alpha"] = t[:, 0]
            result["target_risk"] = t[:, 1]
            result["target_intensity"] = t[:, 2]
            result["target_mask"] = valid

        return result
