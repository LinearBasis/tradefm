"""RLMarketMakingEnv — gym-style wrapper around hftbacktest L3.

One rollout = one full trading session of one instrument on one date.

Decision cadence is `quote_refresh_ms` (default 1000 ms). At every step:
    1. Cancel previous quotes (if any).
    2. Compute A-S baseline from frozen transformer + heads.
    3. Apply action via `compute_quotes()` to get final bid/ask.
    4. Place new limit orders (subject to inventory cap).
    5. Elapse `quote_refresh_ns`.
    6. Read new market state, detect fills, compute reward.
    7. Build next observation; check termination.

The transformer + heads are frozen and loaded once in __init__. hftbacktest is
torn down and re-built in `reset()` (one rollout per reset).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import hftbacktest as hb
import numpy as np
import polars as pl
import torch

from src._archive.decision.heads import DecisionModule
from src._archive.decision.rl.action import (
    QuoteContext,
    action_space_size,
    compute_quotes,
    is_continuous,
)
from src._archive.decision.rl.config import EnvConfig
from src._archive.decision.rl.reward import PnLSnapshot, compute_step_reward
from src._archive.decision.rl.state import (
    build_fast_tier,
    build_state_both,
    build_state_heads,
    build_state_hidden,
    state_dim,
)
from src.models.transformer import OrderFlowTransformer


SECONDS_FROM_MIDNIGHT_AT_T_FIRST = 36000.0  # 10:00:00


# Order-ID base offsets so cancels are unambiguous
_BID_OID_BASE = 10**8
_ASK_OID_BASE = 2 * 10**8
_EOD_OID_BASE = 9 * 10**8


@dataclass
class StepInfo:
    """Diagnostic info returned alongside reward / obs."""
    delta_pnl: float
    inv_penalty: float
    q_norm: float
    sigma_rel: float
    position: float
    balance: float
    mid: float
    our_bid: float
    our_ask: float
    fills: int
    is_terminal: bool


class RLMarketMakingEnv:
    """Gym-style environment. No external gym dependency."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # --- Locate tokens via manifest (mirrors src/data/dataset.py) ---
        # Manifest gives a stable inst_id (alphabetical position in
        # manifest["instruments"]) — same convention the transformer was
        # trained with, so the instrument embedding is consistent.
        seq_dir = Path(cfg.sequences_dir)
        manifest_path = seq_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}. "
                f"Run src/data/pipeline.py to generate sequences."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        instrument_names = manifest["instruments"]
        if cfg.instrument not in instrument_names:
            raise ValueError(
                f"instrument {cfg.instrument} not in manifest "
                f"({len(instrument_names)} instruments)"
            )
        self._inst_id = instrument_names.index(cfg.instrument)
        self._manifest_tau_sec = manifest.get("tau_sec")

        # --- Pick token file: override, or canonical sequences/<INST>_<DATE>.parquet ---
        if cfg.tokens_parquet_override is not None:
            tokens_path = Path(cfg.tokens_parquet_override)
        else:
            tokens_path = seq_dir / f"{cfg.instrument}_{cfg.date}.parquet"
        if not tokens_path.exists():
            raise FileNotFoundError(f"tokens parquet not found: {tokens_path}")

        tokens_df = pl.read_parquet(tokens_path)
        if "time_sec" not in tokens_df.columns:
            raise ValueError(
                f"{tokens_path} has no `time_sec` column. Regenerate sequences "
                f"via src/data/pipeline.py (pipeline.py:87 writes time_sec)."
            )
        # If SECCODE is present (e.g. flat tokens.parquet via override) — filter.
        if "SECCODE" in tokens_df.columns:
            tokens_df = tokens_df.filter(pl.col("SECCODE") == cfg.instrument)
        tokens_df = tokens_df.sort("time_sec")
        if len(tokens_df) == 0:
            raise RuntimeError(f"no tokens for {cfg.instrument} in {tokens_path}")
        self._tok_time = tokens_df["time_sec"].to_numpy()
        self._tok_trade = tokens_df["trade_token"].to_numpy().astype(np.int64)
        self._tok_pl = tokens_df["bin_price_level"].to_numpy().astype(np.int64)
        self._tok_liq = tokens_df["bin_liquidity"].to_numpy().astype(np.int64)

        # --- Load frozen transformer + heads (mirrors train_heads.load_transformer) ---
        ck_xfmr = torch.load(cfg.transformer_checkpoint, map_location=self.device, weights_only=False)
        self._model_cfg = ck_xfmr["config"]
        self._transformer = OrderFlowTransformer(self._model_cfg).to(self.device)
        self._transformer.load_state_dict(ck_xfmr["model_state_dict"])
        self._transformer.eval()
        for p in self._transformer.parameters():
            p.requires_grad = False

        self._heads = DecisionModule(self._model_cfg.d_model).to(self.device)
        ck_heads = torch.load(cfg.heads_checkpoint, map_location=self.device, weights_only=False)
        self._heads.load_state_dict(ck_heads["heads_state_dict"])
        self._heads.eval()
        for p in self._heads.parameters():
            p.requires_grad = False

        self._inst_id_t = torch.tensor([self._inst_id], dtype=torch.long, device=self.device)
        self._ctx_len = self._model_cfg.context_length

        # --- Action/state space ---
        self.action_size = action_space_size(cfg.action_mode)
        self.is_continuous = is_continuous(cfg.action_mode)
        self.observation_dim = state_dim(cfg.state_mode, self._model_cfg.d_model)

        # --- Per-rollout state (set in reset) ---
        self._hbt: hb.HashMapMarketDepthBacktest | None = None
        self._t_first = 0
        self._t_last = 0
        self._t_flatten = 0
        self._t_end = 0
        self._session_length_sec = 1.0
        self._quote_refresh_ns = int(cfg.quote_refresh_ms * 1e6)
        self._latency_ns = int(cfg.latency_ms * 1e6)
        self._dt_sec = cfg.quote_refresh_ms / 1000.0

        self._bid_oid: int | None = None
        self._ask_oid: int | None = None
        self._next_oid_counter = 0

        self._prev_position = 0.0
        self._prev_balance = 0.0
        self._prev_mid: float | None = None
        self._prev_mtm = 0.0
        self._our_bid: float | None = None
        self._our_ask: float | None = None
        self._terminated = False
        self._step_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Build a fresh hftbacktest instance and return the first observation."""
        if self._hbt is not None:
            self._hbt.close()
            self._hbt = None

        cfg = self.cfg
        npz_path = Path(cfg.data_dir) / cfg.instrument / f"{cfg.date}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        raw = np.load(str(npz_path), allow_pickle=False)
        arr = raw[raw.files[0]]
        self._t_first = int(arr["exch_ts"][0])
        self._t_last = int(arr["exch_ts"][-1])
        self._t_end = self._t_last
        self._t_flatten = self._t_end - int(cfg.flatten_min_before_end * 60 * 1e9)
        self._session_length_sec = (self._t_end - self._t_first) / 1e9

        asset = (
            hb.BacktestAsset()
            .data([str(npz_path)])
            .linear_asset(1.0)
            .constant_order_latency(self._latency_ns, self._latency_ns)
            .l3_fifo_queue_model()
            .tick_size(cfg.tick_size)
            .lot_size(cfg.lot_size)
            .flat_per_trade_fee_model(cfg.maker_fee, cfg.taker_fee)
        )
        self._hbt = hb.HashMapMarketDepthBacktest([asset])

        self._bid_oid = self._ask_oid = None
        self._next_oid_counter = 0
        self._prev_position = 0.0
        self._prev_balance = 0.0
        self._prev_mid = None
        self._our_bid = None
        self._our_ask = None
        self._terminated = False
        self._step_count = 0

        # Initial elapse so we have a non-trivial book and α/σ/κ from heads.
        # One quote_refresh tick. If session ends here something is very wrong.
        if self._hbt.elapse(self._quote_refresh_ns) != 0:
            raise RuntimeError("session ended immediately on reset()")

        obs, _book = self._build_observation()
        self._prev_mtm = self._read_mtm()
        return obs

    def close(self):
        if self._hbt is not None:
            self._hbt.close()
            self._hbt = None

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action) -> tuple[np.ndarray, float, bool, StepInfo]:
        """Apply action, advance by one quote_refresh, return (obs, r, done, info)."""
        if self._terminated:
            raise RuntimeError("step() called after termination; call reset()")

        cfg = self.cfg
        hbt = self._hbt
        assert hbt is not None

        # --- Cancel previous quotes ---
        if self._bid_oid is not None:
            hbt.cancel(0, self._bid_oid, False)
            self._bid_oid = None
        if self._ask_oid is not None:
            hbt.cancel(0, self._ask_oid, False)
            self._ask_oid = None

        # --- Read current book / model outputs ---
        ctx, sigma_rel, alpha_rel, kappa, h_t, has_valid_book = self._read_context()

        # --- EOD: cancel and submit crossing orders to flatten; terminate ---
        now = hbt.current_timestamp
        if now >= self._t_flatten:
            self._do_eod_flatten(ctx)
            # Elapse the remaining time so unwind fills land in balance.
            remaining = max(self._t_end - now, self._quote_refresh_ns)
            hbt.elapse(remaining)
            self._our_bid = self._our_ask = None
            return self._finalize_step(ctx, alpha_rel, sigma_rel, kappa, h_t, terminal=True)

        # --- Apply action ---
        if not has_valid_book:
            # Bad book snapshot — skip quoting this step, just elapse.
            self._our_bid = self._our_ask = None
        else:
            quotes = compute_quotes(
                cfg.action_mode, action, ctx,
                c_spread=cfg.c_spread, c_skew=cfg.c_skew,
                c_gamma=cfg.c_gamma, c_skew_b=cfg.c_skew_b,
            )
            if quotes is None:  # only c_disc no-op
                self._our_bid = self._our_ask = None
            else:
                bid_px, ask_px = quotes
                self._place_quotes(bid_px, ask_px, ctx)

        # --- Elapse one decision interval ---
        ended = hbt.elapse(self._quote_refresh_ns) != 0
        terminal = ended or hbt.current_timestamp > self._t_end
        if cfg.max_steps_per_rollout is not None and self._step_count + 1 >= cfg.max_steps_per_rollout:
            terminal = True

        # Re-read context (post-elapse) for state and reward; reuse heads forward.
        # We re-run the forward pass at the new time to get fresh α/σ/κ for the
        # next observation. NB: reward σ is from the *post*-step state.
        new_ctx, new_sigma_rel, new_alpha_rel, new_kappa, new_h_t, _ = self._read_context()
        return self._finalize_step(new_ctx, new_alpha_rel, new_sigma_rel, new_kappa, new_h_t, terminal=terminal)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_mtm(self) -> float:
        hbt = self._hbt
        assert hbt is not None
        depth = hbt.depth(0)
        bbt, bat = depth.best_bid_tick, depth.best_ask_tick
        if bbt <= 0 or bat == 2**31 - 1 or bat <= bbt:
            return self._prev_mtm  # bad book; reuse previous to avoid spurious reward
        mid = 0.5 * (bbt + bat) * self.cfg.tick_size
        position = float(hbt.position(0))
        sv = hbt.state_values(0)
        return float(sv.balance) + position * mid

    def _read_context(self) -> tuple[QuoteContext, float, float, float, np.ndarray | None, bool]:
        """Read book + run frozen model → (ctx, σ_rel, α_rel, κ, h_t, has_valid_book)."""
        cfg = self.cfg
        hbt = self._hbt
        assert hbt is not None

        depth = hbt.depth(0)
        bbt, bat = depth.best_bid_tick, depth.best_ask_tick
        valid_book = not (bbt <= 0 or bat == 2**31 - 1 or bat <= bbt)
        if valid_book:
            best_bid = bbt * cfg.tick_size
            best_ask = bat * cfg.tick_size
            best_bid_qty = float(depth.best_bid_qty)
            best_ask_qty = float(depth.best_ask_qty)
        else:
            # Use last known mid if available; otherwise return zero spread placeholder
            mid = self._prev_mid if self._prev_mid is not None else 0.0
            best_bid = best_ask = mid
            best_bid_qty = best_ask_qty = 0.0

        mid = 0.5 * (best_bid + best_ask)
        position = float(hbt.position(0))

        # Run heads at this time
        now = hbt.current_timestamp
        cur_time_sec = SECONDS_FROM_MIDNIGHT_AT_T_FIRST + (now - self._t_first) / 1e9
        i_end = int(np.searchsorted(self._tok_time, cur_time_sec, side="right"))
        i_start = max(0, i_end - self._ctx_len)
        if i_end - i_start < 2:
            # Not enough context yet — fall back to sigma_floor, neutral signals
            alpha_rel = 0.0
            sigma_rel = cfg.sigma_floor
            kappa = 1.0
            h_t = None
        else:
            tok_t = torch.from_numpy(self._tok_trade[i_start:i_end]).unsqueeze(0).to(self.device)
            plv_t = torch.from_numpy(self._tok_pl[i_start:i_end]).unsqueeze(0).to(self.device)
            liq_t = torch.from_numpy(self._tok_liq[i_start:i_end]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                hidden = self._transformer.extract_hidden_states(tok_t, plv_t, liq_t, self._inst_id_t)
                preds = self._heads(hidden)
            alpha_rel = float(preds["alpha"][0, -1].item())
            sigma_rel = max(float(preds["risk"][0, -1].item()), cfg.sigma_floor)
            kappa = float(preds["intensity"][0, -1].item())
            h_t = hidden[0, -1, :].detach().cpu().numpy().astype(np.float32)

        ctx = QuoteContext(
            mid=mid,
            best_bid=best_bid,
            best_ask=best_ask,
            book_spread=best_ask - best_bid,
            best_bid_qty=best_bid_qty,
            best_ask_qty=best_ask_qty,
            inventory=position,
            sigma_price=sigma_rel * mid,
            kappa=kappa,
            dt_sec=self._dt_sec,
            tick_size=cfg.tick_size,
            max_half_spread_ticks=cfg.max_half_spread_ticks,
            gamma_as=cfg.gamma_as,
        )
        return ctx, sigma_rel, alpha_rel, kappa, h_t, valid_book

    def _place_quotes(self, bid_px: float, ask_px: float, ctx: QuoteContext) -> None:
        """Submit limit orders for both sides, subject to inventory cap and book."""
        cfg = self.cfg
        hbt = self._hbt
        assert hbt is not None

        position = ctx.inventory
        place_bid = position < cfg.max_inventory and np.isfinite(bid_px)
        place_ask = position > -cfg.max_inventory and np.isfinite(ask_px)

        if place_bid:
            # Round down so we stay on the maker side
            bid_tick_raw = int(np.floor(bid_px / cfg.tick_size))
            # Cap so we don't cross the spread
            bid_tick = min(bid_tick_raw, int(round(ctx.best_bid / cfg.tick_size)))
            if bid_tick > 0:
                self._next_oid_counter += 1
                self._bid_oid = _BID_OID_BASE + self._next_oid_counter
                px = bid_tick * cfg.tick_size
                hbt.submit_buy_order(0, self._bid_oid, px, cfg.order_qty,
                                     hb.GTC, hb.LIMIT, False)
                self._our_bid = px
            else:
                self._our_bid = None
        else:
            self._our_bid = None

        if place_ask:
            ask_tick_raw = int(np.ceil(ask_px / cfg.tick_size))
            ask_tick = max(ask_tick_raw, int(round(ctx.best_ask / cfg.tick_size)))
            self._next_oid_counter += 1
            self._ask_oid = _ASK_OID_BASE + self._next_oid_counter
            px = ask_tick * cfg.tick_size
            hbt.submit_sell_order(0, self._ask_oid, px, cfg.order_qty,
                                  hb.GTC, hb.LIMIT, False)
            self._our_ask = px
        else:
            self._our_ask = None

    def _do_eod_flatten(self, ctx: QuoteContext) -> None:
        cfg = self.cfg
        hbt = self._hbt
        assert hbt is not None

        if self._bid_oid is not None:
            hbt.cancel(0, self._bid_oid, False); self._bid_oid = None
        if self._ask_oid is not None:
            hbt.cancel(0, self._ask_oid, False); self._ask_oid = None

        position = float(hbt.position(0))
        if abs(position) > 0:
            self._next_oid_counter += 1
            oid = _EOD_OID_BASE + self._next_oid_counter
            if position > 0:
                # Sell at best bid (cross the spread)
                hbt.submit_sell_order(0, oid, ctx.best_bid, abs(position),
                                      hb.GTC, hb.LIMIT, False)
            else:
                hbt.submit_buy_order(0, oid, ctx.best_ask, abs(position),
                                     hb.GTC, hb.LIMIT, False)

    def _finalize_step(
        self,
        ctx: QuoteContext,
        alpha_rel: float,
        sigma_rel: float,
        kappa: float,
        h_t: np.ndarray | None,
        *,
        terminal: bool,
    ) -> tuple[np.ndarray, float, bool, StepInfo]:
        """Compute reward + next obs after a step or terminal flatten."""
        cfg = self.cfg
        hbt = self._hbt
        assert hbt is not None

        # Read current PnL state
        position = float(hbt.position(0))
        sv = hbt.state_values(0)
        balance = float(sv.balance)
        cur = PnLSnapshot(balance=balance, position=position, mid=ctx.mid)
        prev = PnLSnapshot(
            balance=self._prev_balance,
            position=self._prev_position,
            mid=self._prev_mid if self._prev_mid is not None else ctx.mid,
        )

        reward, br = compute_step_reward(
            prev, cur, sigma_rel,
            max_inventory=cfg.max_inventory,
            gamma_R=cfg.gamma_R,
            dt_sec=self._dt_sec,
            scale=cfg.reward_scale,
        )

        # Detect fills via position delta (for logging only)
        fills = int(round(abs(position - self._prev_position) / cfg.order_qty))

        # Build next observation
        fast_tier = build_fast_tier(
            position=position,
            max_inventory=cfg.max_inventory,
            now_sec=(hbt.current_timestamp - self._t_first) / 1e9,
            t_flatten_sec=(self._t_flatten - self._t_first) / 1e9,
            session_length_sec=self._session_length_sec,
            mid=ctx.mid,
            last_mid=self._prev_mid,
            tick_size=cfg.tick_size,
            book_spread=ctx.book_spread,
            best_bid_size=ctx.best_bid_qty,
            best_ask_size=ctx.best_ask_qty,
            our_bid_price=self._our_bid,
            our_ask_price=self._our_ask,
        )
        if h_t is None:
            h_t = np.zeros(self._model_cfg.d_model, dtype=np.float32)
        if cfg.state_mode == "heads":
            obs = build_state_heads(
                alpha_rel=alpha_rel, sigma_rel=sigma_rel, kappa=kappa,
                fast_tier=fast_tier,
            )
        elif cfg.state_mode == "hidden":
            obs = build_state_hidden(h_t=h_t, fast_tier=fast_tier)
        else:  # "both"
            obs = build_state_both(
                alpha_rel=alpha_rel, sigma_rel=sigma_rel, kappa=kappa,
                h_t=h_t, fast_tier=fast_tier,
            )

        # Persist for next step
        self._prev_position = position
        self._prev_balance = balance
        self._prev_mid = ctx.mid
        self._prev_mtm = cur.mtm()
        self._step_count += 1
        if terminal:
            self._terminated = True

        info = StepInfo(
            delta_pnl=br["delta_pnl"],
            inv_penalty=br["inv_penalty"],
            q_norm=br["q_norm"],
            sigma_rel=br["sigma_rel"],
            position=position,
            balance=balance,
            mid=ctx.mid,
            our_bid=self._our_bid if self._our_bid is not None else float("nan"),
            our_ask=self._our_ask if self._our_ask is not None else float("nan"),
            fills=fills,
            is_terminal=terminal,
        )
        return obs, reward, terminal, info

    def _build_observation(self) -> tuple[np.ndarray, QuoteContext]:
        """Used by reset() to produce the first observation without a step()."""
        ctx, sigma_rel, alpha_rel, kappa, h_t, _ = self._read_context()
        fast_tier = build_fast_tier(
            position=ctx.inventory,
            max_inventory=self.cfg.max_inventory,
            now_sec=(self._hbt.current_timestamp - self._t_first) / 1e9,
            t_flatten_sec=(self._t_flatten - self._t_first) / 1e9,
            session_length_sec=self._session_length_sec,
            mid=ctx.mid,
            last_mid=None,
            tick_size=self.cfg.tick_size,
            book_spread=ctx.book_spread,
            best_bid_size=ctx.best_bid_qty,
            best_ask_size=ctx.best_ask_qty,
            our_bid_price=None,
            our_ask_price=None,
        )
        if h_t is None:
            h_t = np.zeros(self._model_cfg.d_model, dtype=np.float32)
        if self.cfg.state_mode == "heads":
            obs = build_state_heads(alpha_rel=alpha_rel, sigma_rel=sigma_rel, kappa=kappa, fast_tier=fast_tier)
        elif self.cfg.state_mode == "hidden":
            obs = build_state_hidden(h_t=h_t, fast_tier=fast_tier)
        else:  # "both"
            obs = build_state_both(
                alpha_rel=alpha_rel, sigma_rel=sigma_rel, kappa=kappa,
                h_t=h_t, fast_tier=fast_tier,
            )

        self._prev_mid = ctx.mid
        self._prev_position = ctx.inventory
        self._prev_balance = float(self._hbt.state_values(0).balance)
        return obs, ctx
