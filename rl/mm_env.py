"""Среда маркет-мейкинга над событийным симулятором."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from simulator.simulator import MdUpdate, OwnTrade, Sim


ACTIONS: list[tuple[int, int]] = [
    (0, 0),
    (1, 1), (1, 3), (1, 5),
    (3, 1), (3, 3), (3, 5),
    (5, 1), (5, 3), (5, 5),
]
N_ACTIONS = len(ACTIONS)


@dataclass
class EnvConfig:
    inventory_max: float = 30.0
    trade_size: float = 1.0
    step_delay_ns: int = int(1e9)
    rsi_window_ns: int = 60 * int(1e9)
    feature_depth: int = 10
    mid_window_size: int = 5
    risk_alpha: float = 0.05
    initial_inventory: float = 0.0


class FeatureCache:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._rsi_buf: deque = deque()
        self._rsi_gains: float = 0.0
        self._rsi_losses: float = 0.0
        self._mid_window: deque = deque(maxlen=cfg.mid_window_size)
        self._prev_bid_px: float = -1.0
        self._prev_bid_q: float = 0.0
        self._prev_ask_px: float = -1.0
        self._prev_ask_q: float = 0.0
        self._prev_mid: float = -1.0
        self.last_asks: list[tuple[float, float]] = []
        self.last_bids: list[tuple[float, float]] = []
        self.last_mid: float = -1.0
        self.last_bid: float = -1.0
        self.last_ask: float = -1.0
        self.last_spread: float = 0.0
        self.last_ofi: float = 0.0
        self.last_rsi: float = 0.0
        self.last_ts: int = 0

    def update(self, asks, bids, ts):
        if not asks or not bids:
            return
        self.last_asks = list(asks)
        self.last_bids = list(bids)
        bid_px, bid_q = bids[0]
        ask_px, ask_q = asks[0]
        mid = 0.5 * (bid_px + ask_px)
        self.last_mid = mid
        self.last_bid = bid_px
        self.last_ask = ask_px
        self.last_spread = ask_px - bid_px

        ofi = 0.0
        if self._prev_bid_px > 0:
            if bid_px >= self._prev_bid_px:
                ofi += bid_q
            if bid_px <= self._prev_bid_px:
                ofi -= self._prev_bid_q
            if ask_px <= self._prev_ask_px:
                ofi -= ask_q
            if ask_px >= self._prev_ask_px:
                ofi += self._prev_ask_q
        self.last_ofi = ofi
        self._prev_bid_px, self._prev_bid_q = bid_px, bid_q
        self._prev_ask_px, self._prev_ask_q = ask_px, ask_q

        if self._prev_mid > 0:
            ret = (mid / self._prev_mid) - 1.0
            self._rsi_buf.append((ts, ret))
            if ret > 0:
                self._rsi_gains += ret
            elif ret < 0:
                self._rsi_losses += -ret
            cutoff = ts - self.cfg.rsi_window_ns
            while self._rsi_buf and self._rsi_buf[0][0] < cutoff:
                _, r_old = self._rsi_buf.popleft()
                if r_old > 0:
                    self._rsi_gains -= r_old
                elif r_old < 0:
                    self._rsi_losses -= -r_old
            denom = self._rsi_gains + self._rsi_losses
            self.last_rsi = (self._rsi_gains - self._rsi_losses) / denom if denom > 1e-12 else 0.0
        self._prev_mid = mid
        self._mid_window.append(mid)
        self.last_ts = ts

    def state_full(self, inventory_ratio: float) -> np.ndarray:
        D = self.cfg.feature_depth
        W = self.cfg.mid_window_size
        out = np.zeros(1 + 4 * D + 3 + W, dtype=np.float32)
        if self.last_mid <= 0:
            return out
        out[0] = self.last_spread / self.last_mid
        for i in range(D):
            if i < len(self.last_bids):
                p, q = self.last_bids[i]
                out[1 + i] = p / self.last_mid - 1.0
                out[1 + 2 * D + i] = float(np.log1p(max(p * q, 0.0)))
            if i < len(self.last_asks):
                p, q = self.last_asks[i]
                out[1 + D + i] = p / self.last_mid - 1.0
                out[1 + 3 * D + i] = float(np.log1p(max(p * q, 0.0)))
        out[1 + 4 * D] = self.last_ofi / 100.0
        out[1 + 4 * D + 1] = self.last_rsi
        out[1 + 4 * D + 2] = inventory_ratio
        base = 1 + 4 * D + 3
        for i, m in enumerate(self._mid_window):
            out[base + i] = m / self.last_mid - 1.0
        return out


@dataclass
class EpisodeStats:
    rewards: list[float] = field(default_factory=list)
    pnl_history: list[float] = field(default_factory=list)
    inv_history: list[float] = field(default_factory=list)
    n_trades: int = 0
    total_volume: float = 0.0
    final_cash: float = 0.0
    final_inv: float = 0.0
    final_mid: float = 0.0
    final_pnl: float = 0.0


class MMEnv:
    def __init__(self, md: list[MdUpdate], cfg: EnvConfig | None = None,
                 exec_latency_ns: int = 10_000_000,
                 md_latency_ns: int = 10_000_000,
                 maker_fee: float = -0.0004 / 100):
        self.md = md
        self.cfg = cfg or EnvConfig()
        self.exec_latency_ns = exec_latency_ns
        self.md_latency_ns = md_latency_ns
        self.maker_fee = maker_fee
        self.reset()

    def reset(self) -> np.ndarray:
        self.sim = Sim(list(self.md), self.exec_latency_ns, self.md_latency_ns)
        self.fc = FeatureCache(self.cfg)
        self.inv = float(self.cfg.initial_inventory)
        self.cash = 0.0
        self.initial_inv = float(self.cfg.initial_inventory)
        self.realized_pnl = 0.0
        self.prev_mid = -1.0
        self.last_decision_ts = -10**18
        self.ongoing: dict[int, object] = {}
        self._cur_ask_px = None
        self._cur_bid_px = None
        self._initial_mid = None
        self.stats = EpisodeStats()
        self._advance_to_observation()
        return self._get_state()

    @property
    def state_dim(self) -> int:
        return self._get_state().shape[0]

    def _process_updates(self, updates):
        for u in updates:
            if isinstance(u, MdUpdate):
                if u.orderbook is not None:
                    self.fc.update(u.orderbook.asks, u.orderbook.bids, u.exchange_ts)
            elif isinstance(u, OwnTrade):
                if u.order_id in self.ongoing:
                    self.ongoing.pop(u.order_id)
                if u.side == "BID":
                    self.inv += u.size
                    self.cash -= u.size * u.price * (1 + self.maker_fee)
                else:
                    self.inv -= u.size
                    self.cash += u.size * u.price * (1 - self.maker_fee)
                self.stats.n_trades += 1
                self.stats.total_volume += u.size

    def _advance_to_observation(self):
        while True:
            ts, updates = self.sim.tick()
            if updates is None:
                self.done = True
                return None
            self._process_updates(updates)
            if (self.fc.last_mid > 0
                and len(self.fc._mid_window) == self.cfg.mid_window_size
                and ts - self.last_decision_ts >= self.cfg.step_delay_ns):
                if self._initial_mid is None:
                    self._initial_mid = float(self.fc.last_mid)
                self.last_decision_ts = ts
                self.current_ts = ts
                self.done = False
                return ts

    def _get_state(self) -> np.ndarray:
        inv_ratio = float(np.clip(self.inv / self.cfg.inventory_max, -1.0, 1.0))
        return self.fc.state_full(inv_ratio)

    def _post_quotes(self, action_idx: int, ts: int):
        ask_lvl, bid_lvl = ACTIONS[action_idx]
        target_ask_px = None
        target_bid_px = None
        if ask_lvl > 0 and self.inv > -self.cfg.inventory_max and ask_lvl <= len(self.fc.last_asks):
            target_ask_px = self.fc.last_asks[ask_lvl - 1][0]
        if bid_lvl > 0 and self.inv < self.cfg.inventory_max and bid_lvl <= len(self.fc.last_bids):
            target_bid_px = self.fc.last_bids[bid_lvl - 1][0]
        unchanged = (target_ask_px == self._cur_ask_px
                     and target_bid_px == self._cur_bid_px)
        n_target = int(target_ask_px is not None) + int(target_bid_px is not None)
        if unchanged and len(self.ongoing) >= n_target:
            return
        for oid in list(self.ongoing.keys()):
            self.sim.cancel_order(ts, oid)
            self.ongoing.pop(oid, None)
        if target_ask_px is not None:
            o = self.sim.place_order(ts, self.cfg.trade_size, "ASK", target_ask_px)
            self.ongoing[o.order_id] = o
        if target_bid_px is not None:
            o = self.sim.place_order(ts, self.cfg.trade_size, "BID", target_bid_px)
            self.ongoing[o.order_id] = o
        self._cur_ask_px = target_ask_px
        self._cur_bid_px = target_bid_px

    def current_mtm(self) -> float:
        mid = self.fc.last_mid if self.fc.last_mid > 0 else 0.0
        current_value = self.cash + self.inv * mid
        initial_mid = self._initial_mid if self._initial_mid is not None else mid
        return current_value - self.initial_inv * initial_mid

    def snapshot_stats(self) -> dict:
        return {
            "n_steps": len(self.stats.rewards),
            "n_trades": self.stats.n_trades,
            "total_volume": self.stats.total_volume,
            "final_pnl": self.current_mtm(),
            "final_inv": self.inv,
            "final_mid": self.fc.last_mid,
            "initial_inv": self.initial_inv,
            "initial_mid": self._initial_mid,
        }

    def step(self, action_idx: int):
        prev_mid = self.fc.last_mid
        prev_mtm = self.cash + self.inv * prev_mid
        self._post_quotes(action_idx, self.current_ts)
        self._advance_to_observation()
        if self.done:
            self.stats.final_cash = self.cash
            self.stats.final_inv = self.inv
            self.stats.final_mid = self.fc.last_mid
            self.stats.final_pnl = self.cash + self.inv * self.fc.last_mid
            return self._get_state(), 0.0, True, {}
        new_mid = self.fc.last_mid
        d_m = (new_mid / prev_mid) - 1.0 if prev_mid > 0 else 0.0
        inv_ratio = self.inv / self.cfg.inventory_max
        new_mtm = self.cash + self.inv * new_mid
        delta_pnl = new_mtm - prev_mtm
        norm = max(self.cfg.inventory_max * (new_mid if new_mid > 0 else 1.0), 1.0)
        reward = (
            d_m * inv_ratio
            + delta_pnl / norm
            - self.cfg.risk_alpha * abs(inv_ratio)
        )
        self.stats.rewards.append(reward)
        self.stats.pnl_history.append(new_mtm)
        self.stats.inv_history.append(self.inv)
        return self._get_state(), reward, False, {"mtm": new_mtm, "inv": self.inv}
