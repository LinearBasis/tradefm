"""Событийный симулятор биржевого стакана с задержками исполнения и market-data."""
from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
from math import inf as _INF
from typing import List, Optional, Tuple, Union, Deque, Dict


@dataclass
class Order:
    place_ts: float
    exchange_ts: float
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete: int


@dataclass
class AnonTrade:
    exchange_ts: float
    receive_ts: float
    side: str
    size: float
    price: float


@dataclass
class OwnTrade:
    place_ts: float
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute: str

    def __post_init__(self):
        assert isinstance(self.side, str)


@dataclass
class OrderbookSnapshotUpdate:
    exchange_ts: float
    receive_ts: float
    asks: List[Tuple[float, float]]
    bids: List[Tuple[float, float]]


@dataclass
class MdUpdate:
    exchange_ts: float
    receive_ts: float
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


def update_best_positions(best_bid: float, best_ask: float, md: MdUpdate,
                          levels: bool = False) -> Tuple[float, float]:
    if md.orderbook is not None:
        best_bid = md.orderbook.bids[0][0]
        best_ask = md.orderbook.asks[0][0]
        if levels:
            asks = [level[0] for level in md.orderbook.asks]
            bids = [level[0] for level in md.orderbook.bids]
            return best_bid, best_ask, asks, bids
        return best_bid, best_ask
    if md.trade.side == 'BID':
        best_ask = max(md.trade.price, best_ask)
    elif md.trade.side == 'ASK':
        best_bid = min(best_bid, md.trade.price)
    return best_bid, best_ask


class Sim:
    def __init__(self, market_data: List[MdUpdate],
                 execution_latency: float, md_latency: float) -> None:
        self.md_queue: Deque[MdUpdate] = deque(market_data)
        self.actions_queue: Deque[Union[Order, CancelOrder]] = deque()
        self.strategy_updates_queue: "OrderedDict[float, list]" = OrderedDict()
        self.ready_to_execute_orders: Dict[int, Order] = {}

        self.md: Optional[MdUpdate] = None
        self.order_id = 0
        self.trade_id = 0
        self.latency = execution_latency
        self.md_latency = md_latency
        self.best_bid = -_INF
        self.best_ask = _INF
        self.trade_price = {'BID': -_INF, 'ASK': _INF}
        self.last_order: Optional[Order] = None

    def get_md_queue_event_time(self) -> float:
        q = self.md_queue
        return _INF if not q else q[0].exchange_ts

    def get_actions_queue_event_time(self) -> float:
        q = self.actions_queue
        return _INF if not q else q[0].exchange_ts

    def get_strategy_updates_queue_event_time(self) -> float:
        sq = self.strategy_updates_queue
        return _INF if not sq else next(iter(sq))

    def get_order_id(self) -> int:
        res = self.order_id
        self.order_id += 1
        return res

    def get_trade_id(self) -> int:
        res = self.trade_id
        self.trade_id += 1
        return res

    def update_best_pos(self) -> None:
        md = self.md
        if md is not None and md.orderbook is not None:
            self.best_bid = md.orderbook.bids[0][0]
            self.best_ask = md.orderbook.asks[0][0]

    def update_last_trade(self) -> None:
        md = self.md
        if md is not None and md.trade is not None:
            self.trade_price[md.trade.side] = md.trade.price

    def delete_last_trade(self) -> None:
        self.trade_price['BID'] = -_INF
        self.trade_price['ASK'] = _INF

    def update_md(self, md: MdUpdate) -> None:
        self.md = md
        ob = md.orderbook
        if ob is not None:
            self.best_bid = ob.bids[0][0]
            self.best_ask = ob.asks[0][0]
        tr = md.trade
        if tr is not None:
            self.trade_price[tr.side] = tr.price

        sq = self.strategy_updates_queue
        rts = md.receive_ts
        bucket = sq.get(rts)
        if bucket is None:
            sq[rts] = [md]
        else:
            bucket.append(md)

    def update_action(self, action: Union[Order, CancelOrder]) -> None:
        if isinstance(action, Order):
            self.last_order = action
        elif isinstance(action, CancelOrder):
            self.ready_to_execute_orders.pop(action.id_to_delete, None)
        else:
            assert False, f"Unknown action type: {type(action).__name__}"

    def tick(self) -> Tuple[float, Optional[list]]:
        md_queue = self.md_queue
        actions_queue = self.actions_queue
        sq = self.strategy_updates_queue

        while True:
            mq_et = md_queue[0].exchange_ts if md_queue else _INF
            aq_et = actions_queue[0].exchange_ts if actions_queue else _INF
            sq_et = next(iter(sq)) if sq else _INF

            if mq_et == _INF and aq_et == _INF:
                break
            if sq_et < mq_et and sq_et < aq_et:
                break

            if mq_et <= aq_et:
                self.update_md(md_queue.popleft())
            if aq_et <= mq_et:
                self.update_action(actions_queue.popleft())

            self.execute_last_order()
            self.execute_orders()
            self.delete_last_trade()

        if not sq:
            return _INF, None
        key, res = sq.popitem(last=False)
        return key, res

    def execute_last_order(self) -> None:
        lo = self.last_order
        if lo is None:
            return
        executed_price = None
        execute = None
        side = lo.side
        price = lo.price
        if side == 'BID' and price >= self.best_ask:
            executed_price = self.best_ask
            execute = 'BOOK'
        elif side == 'ASK' and price <= self.best_bid:
            executed_price = self.best_bid
            execute = 'BOOK'

        if executed_price is not None:
            md = self.md
            rts = md.exchange_ts + self.md_latency
            ot = OwnTrade(
                lo.place_ts, md.exchange_ts, rts,
                self.get_trade_id(), lo.order_id,
                side, lo.size, executed_price, execute,
            )
            sq = self.strategy_updates_queue
            bucket = sq.get(rts)
            if bucket is None:
                sq[rts] = [ot]
            else:
                bucket.append(ot)
        else:
            self.ready_to_execute_orders[lo.order_id] = lo

        self.last_order = None

    def execute_orders(self) -> None:
        ready = self.ready_to_execute_orders
        if not ready:
            return
        best_ask = self.best_ask
        best_bid = self.best_bid
        trade_ask = self.trade_price['ASK']
        trade_bid = self.trade_price['BID']
        md = self.md
        ex_ts = md.exchange_ts
        rts = ex_ts + self.md_latency
        sq = self.strategy_updates_queue

        executed_orders_id = []
        for order_id, order in ready.items():
            side = order.side
            price = order.price
            executed_price = None
            execute = None

            if side == 'BID':
                if price >= best_ask:
                    executed_price = price
                    execute = 'BOOK'
                elif price >= trade_ask:
                    executed_price = price
                    execute = 'TRADE'
            else:
                if price <= best_bid:
                    executed_price = price
                    execute = 'BOOK'
                elif price <= trade_bid:
                    executed_price = price
                    execute = 'TRADE'

            if executed_price is not None:
                ot = OwnTrade(
                    order.place_ts, ex_ts, rts,
                    self.get_trade_id(), order_id,
                    side, order.size, executed_price, execute,
                )
                executed_orders_id.append(order_id)
                bucket = sq.get(rts)
                if bucket is None:
                    sq[rts] = [ot]
                else:
                    bucket.append(ot)

        for k in executed_orders_id:
            del ready[k]

    def place_order(self, ts: float, size: float, side: str, price: float) -> Order:
        order = Order(ts, ts + self.latency, self.get_order_id(), side, size, price)
        self.actions_queue.append(order)
        return order

    def cancel_order(self, ts: float, id_to_delete: int) -> CancelOrder:
        ts += self.latency
        delete_order = CancelOrder(ts, id_to_delete)
        self.actions_queue.append(delete_order)
        return delete_order
