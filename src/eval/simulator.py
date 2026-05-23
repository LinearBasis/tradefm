"""Minimal limit order book simulator for closed-loop TradeFM evaluation.

Adapted from paper Algorithm 2-3 with one pragmatic simplification: cancels target
the closest matching price level on the named side (the model emits anonymous
events, not order IDs, so paper's ID-based matching isn't applicable to
generated rollouts).

The simulator maintains:
  - bid and ask sides as price -> FIFO queue of (qty, order_id, ts)
  - current mid-price estimate (avg of best bid and best ask)
  - fill log + per-step history (mid, spread, OBI, bid/ask volume)

Convention: side=0 means buy/bid, side=1 means sell/ask (matches tokenizer).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass
class Fill:
    ts: float
    aggressor_side: int   # 0=buy aggressor (lifted ask), 1=sell aggressor (hit bid)
    price: float
    qty: int


class MinimalLOB:
    def __init__(self, init_mid: float, init_ts: float = 0.0):
        self.opening_price = float(init_mid)
        self.mid = float(init_mid)
        self.last_ts = float(init_ts)

        # price -> list[(qty, order_id, ts)] in time order
        self.bids: dict[float, list[tuple[int, int, float]]] = {}
        self.asks: dict[float, list[tuple[int, int, float]]] = {}

        # Sorted price lists for fast best-lookup
        self._bid_prices: list[float] = []  # ascending; best = last
        self._ask_prices: list[float] = []  # ascending; best = first

        self.next_order_id = 1
        self.fills: list[Fill] = []
        self.history: list[dict] = []

    # ------------------------------------------------------------ accessors --

    def best_bid(self) -> float | None:
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> float | None:
        return self._ask_prices[0] if self._ask_prices else None

    def spread(self) -> float:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return float("nan")
        return a - b

    def obi(self) -> float:
        """Order book imbalance at top of book: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        b, a = self.best_bid(), self.best_ask()
        bv = sum(q for q, _, _ in self.bids[b]) if b is not None else 0
        av = sum(q for q, _, _ in self.asks[a]) if a is not None else 0
        if bv + av == 0:
            return 0.0
        return (bv - av) / (bv + av)

    def update_mid(self) -> None:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            self.mid = 0.5 * (a + b)
        elif b is not None:
            self.mid = b
        elif a is not None:
            self.mid = a
        # If both empty, keep last mid.

    # -------------------------------------------------------------- helpers --

    def _insert(self, side: int, price: float, qty: int, ts: float, oid: int | None = None) -> None:
        """Insert resting order at price (no matching). FIFO queue per level."""
        levels = self.bids if side == 0 else self.asks
        prices = self._bid_prices if side == 0 else self._ask_prices
        if oid is None:
            oid = self.next_order_id
            self.next_order_id += 1
        if price not in levels:
            bisect.insort(prices, price)
            levels[price] = []
        levels[price].append((qty, oid, ts))

    def _pop_oldest_opposite(self, aggressor_side: int) -> tuple[float, int, int, float] | None:
        """Pop and return the front-of-queue order at the best price on the opposite side.

        Returns (price, qty, order_id, ts) or None if that side is empty.
        """
        if aggressor_side == 0:  # buyer aggresses asks
            if not self._ask_prices:
                return None
            best = self._ask_prices[0]
            qty, oid, ts = self.asks[best].pop(0)
            if not self.asks[best]:
                del self.asks[best]
                self._ask_prices.pop(0)
        else:                    # seller aggresses bids
            if not self._bid_prices:
                return None
            best = self._bid_prices[-1]
            qty, oid, ts = self.bids[best].pop(0)
            if not self.bids[best]:
                del self.bids[best]
                self._bid_prices.pop()
        return best, qty, oid, ts

    def _put_back_front(self, side: int, price: float, qty: int, oid: int, ts: float) -> None:
        """Re-insert a partially-filled order at the front of its level's queue."""
        levels = self.bids if side == 0 else self.asks
        prices = self._bid_prices if side == 0 else self._ask_prices
        if price not in levels:
            bisect.insort(prices, price)
            levels[price] = []
        levels[price].insert(0, (qty, oid, ts))

    # -------------------------------------------------------------- actions --

    def add_order(self, side: int, price: float, qty: int, ts: float) -> int:
        """Add an order. Matches across opposite side via price-time priority.

        Returns volume actually filled. Unfilled remainder rests on the book.
        """
        opp_side = 1 - side
        filled = 0
        remaining = qty

        while remaining > 0:
            opp_best = self.best_ask() if side == 0 else self.best_bid()
            if opp_best is None:
                break
            crosses = (side == 0 and price >= opp_best) or (side == 1 and price <= opp_best)
            if not crosses:
                break
            top = self._pop_oldest_opposite(side)
            if top is None:
                break
            opp_price, opp_qty, opp_oid, opp_ts = top
            trade_qty = min(remaining, opp_qty)
            self.fills.append(Fill(ts=ts, aggressor_side=side, price=opp_price, qty=trade_qty))
            remaining -= trade_qty
            filled += trade_qty
            if opp_qty > trade_qty:
                self._put_back_front(opp_side, opp_price, opp_qty - trade_qty, opp_oid, opp_ts)

        if remaining > 0:
            self._insert(side, price, remaining, ts)
        return filled

    def cancel_order(self, side: int, price: float, qty: int, ts: float) -> int:
        """Cancel up to `qty` shares at the price level CLOSEST to `price` on `side`.

        Removes from the FIFO front of that level (oldest orders first).
        Returns volume actually cancelled.
        """
        levels = self.bids if side == 0 else self.asks
        prices = self._bid_prices if side == 0 else self._ask_prices
        if not prices:
            return 0
        idx = bisect.bisect_left(prices, price)
        candidates = []
        if idx < len(prices):
            candidates.append(prices[idx])
        if idx > 0:
            candidates.append(prices[idx - 1])
        target = min(candidates, key=lambda p: abs(p - price))

        remaining = qty
        orders = levels[target]
        while orders and remaining > 0:
            q, oid, t = orders[0]
            if q <= remaining:
                remaining -= q
                orders.pop(0)
            else:
                orders[0] = (q - remaining, oid, t)
                remaining = 0
        if not orders:
            del levels[target]
            prices.pop(bisect.bisect_left(prices, target))
        return qty - remaining

    # ----------------------------------------------------------------- step --

    def step(
        self,
        action: int,
        side: int,
        price_depth: float,
        volume: int,
        iat_sec: float,
    ) -> dict:
        """One event step. Returns post-event snapshot.

        `price_depth` is the signed relative depth (e.g. +0.001 = 10 bps above mid).
        Absolute price = mid * (1 + price_depth).
        """
        ts = self.last_ts + max(float(iat_sec), 0.0)
        price = self.mid * (1.0 + price_depth)
        qty = max(int(volume), 1)
        if action == 0:
            self.add_order(side, price, qty, ts)
        else:
            self.cancel_order(side, price, qty, ts)
        self.update_mid()

        b, a = self.best_bid(), self.best_ask()
        bv = sum(q for q, _, _ in self.bids[b]) if b is not None else 0
        av = sum(q for q, _, _ in self.asks[a]) if a is not None else 0
        snap = {
            "ts": ts,
            "mid": self.mid,
            "spread": self.spread(),
            "obi": self.obi(),
            "bid_vol": bv,
            "ask_vol": av,
            "n_bid_levels": len(self._bid_prices),
            "n_ask_levels": len(self._ask_prices),
        }
        self.history.append(snap)
        self.last_ts = ts
        return snap

    # ----------------------------------------------------------------- seed --

    def seed_book(self, n_levels: int = 10, qty_per_level: int = 100, tick: float = 0.01) -> None:
        """Populate a symmetric synthetic book around current mid for rollout warmup."""
        for i in range(1, n_levels + 1):
            self._insert(0, self.mid - i * tick, qty_per_level, self.last_ts)
            self._insert(1, self.mid + i * tick, qty_per_level, self.last_ts)
        self.update_mid()
