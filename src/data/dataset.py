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
    """

    def __init__(
        self,
        cfg: ModelConfig,
        split: str = "train",
        sequences_dir: str | None = None,
    ):
        assert split in ("train", "val")
        self.cfg = cfg
        self.context_length = cfg.context_length

        seq_dir = Path(sequences_dir or cfg.sequences_dir)
        manifest_path = seq_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {seq_dir}. Run the pipeline first."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.instrument_names = manifest["instruments"]
        self.instrument_map = {name: i for i, name in enumerate(self.instrument_names)}

        self.windows: list[tuple[np.ndarray, int]] = []

        for seq_name, meta in manifest["sequences"].items():
            if meta["split"] != split:
                continue

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

            window_len = cfg.context_length + 1  # input + 1 target token

            for start in range(0, len(data) - window_len + 1, cfg.stride):
                end = start + window_len
                self.windows.append((data[start:end], inst_id))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window, inst_id = self.windows[idx]
        return {
            "trade_tokens": torch.from_numpy(window[:, 0].astype(np.int64)),
            "price_levels": torch.from_numpy(window[:, 1].astype(np.int64)),
            "liquidities": torch.from_numpy(window[:, 2].astype(np.int64)),
            "instrument_id": torch.tensor(inst_id, dtype=torch.long),
        }
