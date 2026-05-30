"""MMEnv + латент TradeFM."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.mm_env import MMEnv, EnvConfig


class MMEnvTradeFM(MMEnv):
    def __init__(self, md, emb_path: str | Path, cfg: EnvConfig | None = None,
                 exec_latency_ns: int = 10_000_000,
                 md_latency_ns: int = 10_000_000,
                 maker_fee: float = -0.0004 / 100):
        d = np.load(emb_path, allow_pickle=True)
        self._emb_ts = np.ascontiguousarray(d["ts_ns"].astype(np.int64))
        self._emb = np.ascontiguousarray(d["emb"].astype(np.float32))
        self._emb_dim = int(self._emb.shape[1])
        self._emb_zero = np.zeros(self._emb_dim, dtype=np.float32)
        super().__init__(md, cfg=cfg,
                         exec_latency_ns=exec_latency_ns,
                         md_latency_ns=md_latency_ns, maker_fee=maker_fee)

    def _current_emb(self) -> np.ndarray:
        ts = getattr(self, "current_ts", None)
        if ts is None:
            return self._emb_zero
        idx = int(np.searchsorted(self._emb_ts, ts, side="right")) - 1
        if idx < 0:
            return self._emb_zero
        return self._emb[idx]

    def _get_state(self) -> np.ndarray:
        base = super()._get_state()
        return np.concatenate([base, self._current_emb()], axis=0)
