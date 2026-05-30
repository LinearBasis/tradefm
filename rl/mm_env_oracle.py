"""MMEnv + future-mid признаки (proxy-конфигурация)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.mm_env import MMEnv, EnvConfig


@dataclass
class OracleEnvConfig(EnvConfig):
    oracle_horizons_s: tuple = (1, 5, 10, 30)


class MMEnvOracle(MMEnv):
    def __init__(self, md, cfg: OracleEnvConfig | None = None,
                 exec_latency_ns: int = 10_000_000,
                 md_latency_ns: int = 10_000_000,
                 maker_fee: float = -0.0004 / 100):
        cfg = cfg or OracleEnvConfig()
        ts_list, mid_list = [], []
        for u in md:
            ob = getattr(u, "orderbook", None)
            if ob is None or not ob.asks or not ob.bids:
                continue
            ts_list.append(u.exchange_ts)
            mid_list.append(0.5 * (ob.asks[0][0] + ob.bids[0][0]))
        self._oracle_ts = np.asarray(ts_list, dtype=np.int64)
        self._oracle_mid = np.asarray(mid_list, dtype=np.float64)
        self._oracle_horizons_ns = tuple(int(h * 1_000_000_000) for h in cfg.oracle_horizons_s)
        super().__init__(md, cfg=cfg,
                         exec_latency_ns=exec_latency_ns,
                         md_latency_ns=md_latency_ns,
                         maker_fee=maker_fee)

    def _oracle_features(self) -> np.ndarray:
        cur_mid = self.fc.last_mid
        n = len(self._oracle_horizons_ns)
        if cur_mid <= 0:
            return np.zeros(n, dtype=np.float32)
        cur_ts = self.fc.last_ts
        out = np.zeros(n, dtype=np.float32)
        for i, dt_ns in enumerate(self._oracle_horizons_ns):
            target_ts = cur_ts + dt_ns
            idx = int(np.searchsorted(self._oracle_ts, target_ts, side="left"))
            idx = min(idx, len(self._oracle_mid) - 1)
            future_mid = float(self._oracle_mid[idx]) if idx >= 0 else cur_mid
            rel = (future_mid - cur_mid) / cur_mid
            out[i] = float(np.clip(rel * 1000.0, -100.0, 100.0))
        return out

    def _get_state(self) -> np.ndarray:
        base = super()._get_state()
        return np.concatenate([base, self._oracle_features()], dtype=np.float32)
