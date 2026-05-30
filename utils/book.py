"""Инкрементальная реконструкция L2-стакана из MOEX OrderLog L3-событий."""
from __future__ import annotations

from dataclasses import dataclass, field

from sortedcontainers import SortedDict


CANCEL = 0
PLACE = 1
TRADE = 2

BUY = "B"
SELL = "S"


@dataclass
class BookStats:
    place_limit: int = 0
    place_market: int = 0
    trade_applied: int = 0
    trade_skipped_unknown: int = 0
    cancel_applied: int = 0
    cancel_skipped_unknown: int = 0
    negative_volume_clamped: int = 0


@dataclass
class L2Book:
    bids: SortedDict = field(default_factory=SortedDict)
    asks: SortedDict = field(default_factory=SortedDict)
    orders: dict[int, tuple[str, float, int]] = field(default_factory=dict)
    stats: BookStats = field(default_factory=BookStats)

    def _book(self, side: str) -> SortedDict:
        return self.bids if side == BUY else self.asks

    def _add_volume(self, side: str, price: float, delta: int) -> None:
        if price == 0.0 or delta == 0:
            return
        book = self._book(side)
        new = book.get(price, 0) + delta
        if new < 0:
            self.stats.negative_volume_clamped += 1
            new = 0
        if new == 0:
            book.pop(price, None)
        else:
            book[price] = new

    def apply(self, action: int, side: str, orderno: int,
              price: float, volume: int) -> None:
        if action == PLACE:
            if price == 0.0:
                self.orders[orderno] = (side, 0.0, volume)
                self.stats.place_market += 1
            else:
                self.orders[orderno] = (side, price, volume)
                self._add_volume(side, price, volume)
                self.stats.place_limit += 1

        elif action == TRADE:
            ord_state = self.orders.get(orderno)
            if ord_state is None:
                self.stats.trade_skipped_unknown += 1
                return
            s, p, remaining = ord_state
            self._add_volume(s, p, -volume)
            remaining -= volume
            if remaining <= 0:
                self.orders.pop(orderno, None)
            else:
                self.orders[orderno] = (s, p, remaining)
            self.stats.trade_applied += 1

        elif action == CANCEL:
            ord_state = self.orders.pop(orderno, None)
            if ord_state is None:
                self.stats.cancel_skipped_unknown += 1
                return
            s, p, _ = ord_state
            self._add_volume(s, p, -volume)
            self.stats.cancel_applied += 1

    def top(self, height: int) -> tuple[list[float], list[int], list[float], list[int]] | None:
        if len(self.asks) < height or len(self.bids) < height:
            return None
        ask_keys = self.asks.keys()
        bid_keys = self.bids.keys()
        ask_prices = [ask_keys[i] for i in range(height)]
        ask_sizes = [self.asks[p] for p in ask_prices]
        n = len(bid_keys)
        bid_prices = [bid_keys[n - 1 - i] for i in range(height)]
        bid_sizes = [self.bids[p] for p in bid_prices]
        return ask_prices, ask_sizes, bid_prices, bid_sizes

    def bbo(self) -> tuple[float, int, float, int] | None:
        if not self.bids or not self.asks:
            return None
        bb_px = self.bids.keys()[-1]
        ba_px = self.asks.keys()[0]
        return bb_px, self.bids[bb_px], ba_px, self.asks[ba_px]
