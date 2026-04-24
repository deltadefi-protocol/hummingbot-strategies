import asyncio
import json
import logging
import logging.handlers
import math
import os
import random
import time
from collections import deque
from decimal import Decimal
from typing import Dict, List, NamedTuple, Optional

import numpy as np
from pydantic import Field, model_validator

from hummingbot.client.settings import DEFAULT_LOG_FILE_PATH
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.event.events import MarketOrderFailureEvent, OrderCancelledEvent
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

D = Decimal
ZERO = D("0")

PAIR_PRESETS = {
    "ADA-USDM": {"initial_price": D("0.27")},
    "IAG-USDM": {"initial_price": D("0.26")},
    "NIGHT-USDM": {"initial_price": D("0.0585")},
}


class OrderProposal(NamedTuple):
    side: TradeType
    price: Decimal
    size: Decimal


class AvgCostBook:
    """O(1) weighted-average-cost position tracker.

    Tracks one open position at a time. Same-side fills blend into avg_cost;
    opposite-side fills realize P&L at the current avg_cost. A fill larger
    than the open position closes it and flips the side.

    Used for both spot trading P&L and futures hedge P&L — same math applies.
    """

    __slots__ = ("size", "avg_cost", "side", "realized_pnl", "fees")

    def __init__(self):
        self.size: Decimal = ZERO
        self.avg_cost: Decimal = ZERO
        self.side: Optional[str] = None  # "long" | "short" | None
        self.realized_pnl: Decimal = ZERO
        self.fees: Decimal = ZERO

    def apply_fill(self, side: str, size: Decimal, price: Decimal, fee: Decimal):
        """Apply a fill. side is 'long' (buy) or 'short' (sell).
        Returns the realized P&L delta from this fill (0 if same side or open)."""
        self.fees += fee
        if size <= ZERO or price <= ZERO:
            return ZERO

        delta = ZERO
        if self.side is None or self.size <= ZERO:
            # Open new
            self.side = side
            self.size = size
            self.avg_cost = price
        elif self.side == side:
            # Blend into weighted avg
            new_size = self.size + size
            self.avg_cost = (self.size * self.avg_cost + size * price) / new_size
            self.size = new_size
        elif size <= self.size:
            # Partial / full close
            if self.side == "long":
                delta = size * (price - self.avg_cost)
            else:
                delta = size * (self.avg_cost - price)
            self.realized_pnl += delta
            self.size -= size
            if self.size <= ZERO:
                self.side = None
                self.size = ZERO
                self.avg_cost = ZERO
        else:
            # Flip
            if self.side == "long":
                delta = self.size * (price - self.avg_cost)
            else:
                delta = self.size * (self.avg_cost - price)
            self.realized_pnl += delta
            leftover = size - self.size
            self.side = side
            self.size = leftover
            self.avg_cost = price
        return delta

    def signed_position(self) -> Decimal:
        """+ for long, - for short, 0 for flat."""
        if self.size <= ZERO or self.side is None:
            return ZERO
        return self.size if self.side == "long" else -self.size

    def unrealized(self, mid: Optional[Decimal]) -> Optional[Decimal]:
        """Mark-to-market on the open position. None if mid unavailable."""
        signed = self.signed_position()
        if signed == ZERO:
            return ZERO
        if mid is None or mid <= ZERO:
            return None
        return signed * (mid - self.avg_cost)

    def to_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}size": str(self.size),
            f"{prefix}avg_cost": str(self.avg_cost),
            f"{prefix}side": self.side or "",
            f"{prefix}realized_pnl": str(self.realized_pnl),
            f"{prefix}fees": str(self.fees),
        }

    def load_from(self, state: dict, prefix: str = ""):
        """Restore from a dict produced by to_dict()."""
        try:
            sz = D(str(state.get(f"{prefix}size", "0")))
            av = D(str(state.get(f"{prefix}avg_cost", "0")))
            sd = state.get(f"{prefix}side") or ""
            if sz > ZERO and av > ZERO and sd in ("long", "short"):
                self.size = sz
                self.avg_cost = av
                self.side = sd
            self.realized_pnl = D(str(state.get(f"{prefix}realized_pnl", "0")))
            self.fees = D(str(state.get(f"{prefix}fees", "0")))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DeltaDefiCLAMMConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    exchange: str = Field("deltadefi")
    trading_pair: str = Field(default="ADA-USDM", json_schema_extra={
        "prompt": lambda mi: "Trading pair (e.g. ADA-USDM, NIGHT-USDM)",
        "prompt_on_new": True})
    initial_price: Optional[Decimal] = Field(default=None)
    base_spread_bps: Decimal = Field(D("40"))
    max_cumulative_loss: Decimal = Field(D("500"))
    min_base_balance: Decimal = Field(D("1000"))
    min_quote_balance: Decimal = Field(D("500"))
    max_base_budget: Optional[Decimal] = Field(default=None, json_schema_extra={
        "prompt": lambda mi: "Max base asset budget for this strategy (e.g. 370370). Leave empty for no limit",
        "prompt_on_new": True})
    max_quote_budget: Optional[Decimal] = Field(default=None, json_schema_extra={
        "prompt": lambda mi: "Max quote asset budget for this strategy (e.g. 100000). Leave empty for no limit",
        "prompt_on_new": True})
    price_decimals: Optional[int] = Field(default=None)
    amount_decimals: Optional[int] = Field(default=None)

    # Concentrated liquidity
    base_concentration_pct: Decimal = Field(D("5"))
    min_concentration_pct: Decimal = Field(D("5"))
    max_concentration_pct: Decimal = Field(D("20"))

    # NATR
    natr_period: int = Field(14)
    natr_baseline: Decimal = Field(D("0.005"))
    natr_range_scale: Decimal = Field(D("1.0"))

    # ADX
    adx_period: int = Field(14)

    # Hurst
    hurst_min_candles: int = Field(100)
    hurst_update_interval_sec: int = Field(60)

    # HMM
    hmm_n_states: int = Field(3)
    hmm_min_candles: int = Field(200)
    hmm_refit_interval_sec: int = Field(1800)
    hmm_window: int = Field(500)
    hmm_confidence_threshold: Decimal = Field(D("0.80"))

    # Range control
    trend_sensitivity: Decimal = Field(D("0.5"))
    range_ema_alpha: Decimal = Field(D("0.1"))
    range_update_dead_band_pct: Decimal = Field(D("0.5"))

    # Trend protection
    trend_order_scale_factor: Decimal = Field(D("0.0"))
    trend_halt_threshold: Decimal = Field(D("0.0"))

    # Outer range architecture (inner/outer zone split)
    outer_capital_fraction: Decimal = Field(D("0.30"))
    outer_spread_mult: Decimal = Field(D("2.5"))
    outer_range_mult: Decimal = Field(D("2.5"))
    outer_recenter_trigger_pct: Decimal = Field(D("0.50"))

    # Soft recenter: re-anchor range when anchor drifts from range center
    soft_recenter_drift_pct: Decimal = Field(D("2.0"))

    # Conditional rebalance after recenter
    enable_rebalance: bool = Field(True)
    rebalance_max_slippage_bps: Decimal = Field(D("30"))
    rebalance_partial_fraction: Decimal = Field(D("0.5"))
    rebalance_lvr_threshold_pct: Decimal = Field(D("0.30"))

    # Futures hedge overlay (Binance perpetual)
    enable_hedge: bool = Field(False)
    hedge_exchange: str = Field("binance_perpetual_testnet")
    hedge_trading_pair: str = Field("ADA-USDT")
    hedge_size_cap_pct: Decimal = Field(D("0.30"))   # max hedge as % of spot inventory
    hedge_min_notional_quote: Decimal = Field(D("10"))  # skip if too small
    hedge_leverage: int = Field(2)                    # set on connector at start
    hedge_min_state_change_interval_sec: int = Field(60)  # debounce open/close cycling
    hedge_in_flight_timeout_sec: int = Field(30)      # release latch if no fill/fail event
    # Concentration-scaled toxicity gate.
    # Tight range (min_conc) → HIGH bar: fills are frequent so buy_ratio is
    # noisy; require strong consensus to avoid false positives.
    # Loose range (max_conc) → LOW bar: fills are sparse and each one carries
    # more inventory impact; modest imbalance already worth hedging.
    hedge_toxicity_threshold_at_min_conc: Decimal = Field(D("0.75"))
    hedge_toxicity_threshold_at_max_conc: Decimal = Field(D("0.55"))
    # Concentration-scaled inventory skew gate (replaces the discrete
    # inv_state >= soft check). Threshold = activation_fraction × δ/c, where
    # δ = soft_recenter_drift_pct/100 and c = current concentration_pct/100.
    # See docs §1: max_skew(c, δ) ≈ δ/c — the threshold must be reachable
    # within one recenter window.
    hedge_inventory_skew_activation_fraction: Decimal = Field(D("0.60"))

    # Order generation
    num_levels: int = Field(1)
    size_decay: Decimal = Field(D("0.85"))
    spread_multiplier: Decimal = Field(D("1.5"))
    order_safe_ratio: Decimal = Field(D("0.5"))
    order_refresh_time: int = Field(5)
    refresh_on_fill_only: bool = Field(True)
    balance_buffer_pct: Decimal = Field(D("0.90"))
    pool_price_weight: Decimal = Field(D("0.70"))
    anchor_ema_alpha: Decimal = Field(D("0.05"))

    # Enhancement flags
    enable_fill_velocity_detector: bool = Field(True)
    fill_velocity_window_sec: int = Field(10)
    fill_velocity_max_same_side: int = Field(3)
    enable_asymmetric_spread: bool = Field(True)
    skew_sensitivity: Decimal = Field(D("0.5"))
    min_spread_bps: Decimal = Field(D("20"))
    enable_order_randomization: bool = Field(False)
    randomization_pct: Decimal = Field(D("0.15"))
    enable_momentum_spread: bool = Field(True)
    momentum_window_sec: int = Field(300)
    momentum_spread_bps: Decimal = Field(D("10"))

    # Fill-toxicity guard (widen + shrink on toxic side)
    toxicity_window_sec: int = Field(300)
    toxicity_window_fills: int = Field(20)  # fill-count window for regime signal
    toxicity_activation_count: int = Field(8)
    toxicity_buy_ratio_soft: Decimal = Field(D("0.65"))
    toxicity_buy_ratio_hard: Decimal = Field(D("0.80"))
    toxicity_sell_ratio_soft: Decimal = Field(D("0.65"))
    toxicity_sell_ratio_hard: Decimal = Field(D("0.80"))
    toxicity_soft_spread_mult: Decimal = Field(D("1.30"))
    toxicity_hard_spread_mult: Decimal = Field(D("1.80"))
    toxicity_soft_size_mult: Decimal = Field(D("0.70"))
    toxicity_hard_size_mult: Decimal = Field(D("0.40"))

    # Inventory risk controls
    inventory_skew_soft_limit: Decimal = Field(D("0.60"))
    inventory_skew_hard_limit: Decimal = Field(D("0.80"))
    inventory_soft_size_mult: Decimal = Field(D("0.60"))
    inventory_hard_size_mult: Decimal = Field(D("0.20"))
    inventory_soft_spread_mult: Decimal = Field(D("1.30"))
    inventory_hard_spread_mult: Decimal = Field(D("1.80"))
    inventory_hard_disable_accumulation_side: bool = Field(True)

    # HMM v2 options
    hmm_use_fill_asymmetry: bool = Field(False)
    hmm_refit_interval_sec: int = Field(1800)

    # FLAIR / LVR monitoring
    enable_flair_monitor: bool = Field(True)
    flair_markout_sec: int = Field(30)
    flair_window_sec: int = Field(1800)

    @model_validator(mode="after")
    def apply_pair_presets(self):
        preset = PAIR_PRESETS.get(self.trading_pair, {})
        if self.initial_price is None:
            self.initial_price = preset.get("initial_price", D("0.01"))
        if not (ZERO <= self.pool_price_weight <= D(1)):
            raise ValueError("pool_price_weight must be in [0, 1]")
        if not (ZERO < self.anchor_ema_alpha <= D(1)):
            raise ValueError("anchor_ema_alpha must be in (0, 1]")
        if not (ZERO < self.order_safe_ratio <= D(1)):
            raise ValueError("order_safe_ratio must be in (0, 1]")
        if self.base_concentration_pct <= ZERO:
            raise ValueError("base_concentration_pct must be > 0")
        if self.min_concentration_pct >= self.max_concentration_pct:
            raise ValueError("min_concentration_pct must be < max_concentration_pct")
        if self.toxicity_window_sec <= 0:
            raise ValueError("toxicity_window_sec must be > 0")
        if self.toxicity_window_fills < 5:
            raise ValueError("toxicity_window_fills must be >= 5")
        if self.toxicity_activation_count <= 0:
            raise ValueError("toxicity_activation_count must be > 0")
        if self.hmm_refit_interval_sec < 60:
            raise ValueError("hmm_refit_interval_sec must be >= 60")
        if self.flair_markout_sec <= 0:
            raise ValueError("flair_markout_sec must be > 0")
        if self.flair_window_sec <= 0:
            raise ValueError("flair_window_sec must be > 0")
        for name in (
            "toxicity_buy_ratio_soft",
            "toxicity_buy_ratio_hard",
            "toxicity_sell_ratio_soft",
            "toxicity_sell_ratio_hard",
        ):
            value = getattr(self, name)
            if not (ZERO < value < D(1)):
                raise ValueError(f"{name} must be in (0, 1)")
        for name in (
            "toxicity_soft_spread_mult",
            "toxicity_hard_spread_mult",
            "inventory_soft_spread_mult",
            "inventory_hard_spread_mult",
        ):
            if getattr(self, name) < D(1):
                raise ValueError(f"{name} must be >= 1")
        for name in (
            "toxicity_soft_size_mult",
            "toxicity_hard_size_mult",
            "inventory_soft_size_mult",
            "inventory_hard_size_mult",
            "inventory_skew_soft_limit",
            "inventory_skew_hard_limit",
        ):
            value = getattr(self, name)
            if not (ZERO < value <= D(1)):
                raise ValueError(f"{name} must be in (0, 1]")
        if self.inventory_skew_soft_limit >= self.inventory_skew_hard_limit:
            raise ValueError("inventory_skew_soft_limit must be < inventory_skew_hard_limit")
        return self


# ---------------------------------------------------------------------------
# ConcentratedPool — Uni V3 math
# ---------------------------------------------------------------------------


def _sqrt(val: Decimal) -> Decimal:
    return D(str(math.sqrt(float(val))))


class ConcentratedPool:
    """Virtual AMM with Uni V3 concentrated liquidity math.

    Reserves are confined to [p_lower, p_upper]. Liquidity L is the invariant
    (constant between fills). Only base changes on fills; quote is re-derived.
    """

    def __init__(self, initial_price: Decimal, pool_depth: Decimal, concentration_pct: Decimal):
        self.anchor_price = D(str(initial_price))
        self.concentration_pct = D(str(concentration_pct))
        pct = self.concentration_pct / D(100)
        self.p_lower = self.anchor_price * (D(1) - pct)
        self.p_upper = self.anchor_price * (D(1) + pct)

        sqrt_p = _sqrt(self.anchor_price)
        sqrt_pa = _sqrt(self.p_lower)
        sqrt_pb = _sqrt(self.p_upper)

        denom = sqrt_p - sqrt_pa
        if denom <= ZERO:
            denom = D("0.0001")
        self.L = D(str(pool_depth)) / denom

        self.base = self.L * (D(1) / sqrt_p - D(1) / sqrt_pb)
        self.quote = self.L * (sqrt_p - sqrt_pa)
        self.initial_base = self.base
        self.initial_quote = self.quote

    @classmethod
    def from_state(cls, state: dict) -> "ConcentratedPool":
        pool = cls.__new__(cls)
        pool.anchor_price = D(str(state["anchor_price"]))
        pool.p_lower = D(str(state["p_lower"]))
        pool.p_upper = D(str(state["p_upper"]))
        pool.L = D(str(state["L"]))
        pool.base = D(str(state["base"]))
        pool.quote = D(str(state["quote"]))
        pool.initial_base = D(str(state["initial_base"]))
        pool.initial_quote = D(str(state["initial_quote"]))
        pool.concentration_pct = D(str(state["concentration_pct"]))
        return pool

    @classmethod
    def from_legacy_state(cls, state: dict) -> "ConcentratedPool":
        """Migrate from old VirtualPool (amplification-based) state."""
        amp = D(str(state["amplification"]))
        concentration_pct = D(100) / amp
        initial_price = D(str(state.get("anchor_price", "0")))
        if initial_price <= ZERO:
            ib = D(str(state["initial_base"]))
            iq = D(str(state["initial_quote"]))
            initial_price = iq / ib if ib > ZERO else D("0.01")
        pool_depth = D(str(state["initial_quote"]))
        return cls(initial_price, pool_depth, concentration_pct)

    def to_state(self) -> dict:
        return {
            "anchor_price": str(self.anchor_price),
            "p_lower": str(self.p_lower),
            "p_upper": str(self.p_upper),
            "L": str(self.L),
            "base": str(self.base),
            "quote": str(self.quote),
            "initial_base": str(self.initial_base),
            "initial_quote": str(self.initial_quote),
            "concentration_pct": str(self.concentration_pct),
        }

    def get_mid_price(self) -> Decimal:
        if self.base <= ZERO:
            return self.p_upper
        sqrt_pb = _sqrt(self.p_upper)
        sqrt_p = self.L / (self.base + self.L / sqrt_pb)
        return sqrt_p * sqrt_p

    def get_pool_price(self) -> Optional[Decimal]:
        if self.base <= ZERO:
            return None
        return self.get_mid_price()

    def update_on_fill(self, side: TradeType, amount: Decimal):
        filled = D(str(amount))
        if side == TradeType.SELL:
            self.base -= filled
        elif side == TradeType.BUY:
            self.base += filled
        self.base = max(ZERO, self.base)
        # Re-derive quote from L invariant
        sqrt_pa = _sqrt(self.p_lower)
        sqrt_p = _sqrt(self.get_mid_price())
        self.quote = self.L * (sqrt_p - sqrt_pa)
        self.quote = max(ZERO, self.quote)

    def recenter(self, new_anchor: Decimal):
        pct = self.concentration_pct / D(100)
        self.anchor_price = new_anchor
        self.p_lower = new_anchor * (D(1) - pct)
        self.p_upper = new_anchor * (D(1) + pct)
        sqrt_p = _sqrt(new_anchor)
        sqrt_pa = _sqrt(self.p_lower)
        sqrt_pb = _sqrt(self.p_upper)
        self.base = self.L * (D(1) / sqrt_p - D(1) / sqrt_pb)
        self.quote = self.L * (sqrt_p - sqrt_pa)
        self.initial_base = self.base
        self.initial_quote = self.quote

    def set_concentration(self, new_pct: Decimal):
        self.concentration_pct = new_pct
        pct = new_pct / D(100)
        self.p_lower = self.anchor_price * (D(1) - pct)
        self.p_upper = self.anchor_price * (D(1) + pct)
        pool_price = self.get_mid_price()
        if not self.is_in_range(pool_price):
            self.recenter(self.anchor_price)

    def is_in_range(self, price: Decimal) -> bool:
        return self.p_lower <= price <= self.p_upper

    def get_inventory_skew(self) -> float:
        if self.initial_base <= ZERO or self.initial_quote <= ZERO:
            return 0.0
        base_ratio = float(self.base / self.initial_base)
        quote_ratio = float(self.quote / self.initial_quote)
        denom = base_ratio + quote_ratio
        if denom == 0:
            return 0.0
        return (base_ratio - quote_ratio) / denom


# ---------------------------------------------------------------------------
# Market Regime Indicators
# ---------------------------------------------------------------------------


class NATRIndicator:
    """Normalized Average True Range — volatility dimension."""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, candles) -> Optional[Decimal]:
        needed = self.period + 1
        if len(candles) < needed:
            return None
        recent = candles[-needed:]
        trs = []
        for i in range(1, len(recent)):
            h = float(recent[i].high)
            l = float(recent[i].low)
            pc = float(recent[i - 1].close)
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        atr = sum(trs[-self.period:]) / self.period
        close = float(recent[-1].close)
        if close <= 0:
            return None
        return D(str(atr / close))


class ADXIndicator:
    """Average Directional Index — directionality dimension."""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, candles) -> Optional[Decimal]:
        needed = self.period * 2 + 1
        if len(candles) < needed:
            return None
        recent = candles[-needed:]

        plus_dm_list = []
        minus_dm_list = []
        tr_list = []
        for i in range(1, len(recent)):
            h = float(recent[i].high)
            l = float(recent[i].low)
            ph = float(recent[i - 1].high)
            pl = float(recent[i - 1].low)
            pc = float(recent[i - 1].close)

            up_move = h - ph
            down_move = pl - l
            plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
            minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

        # Wilder's smoothing for first period
        p = self.period
        smoothed_plus_dm = sum(plus_dm_list[:p])
        smoothed_minus_dm = sum(minus_dm_list[:p])
        smoothed_tr = sum(tr_list[:p])

        dx_values = []
        for i in range(p, len(plus_dm_list)):
            smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / p + plus_dm_list[i]
            smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / p + minus_dm_list[i]
            smoothed_tr = smoothed_tr - smoothed_tr / p + tr_list[i]

            if smoothed_tr == 0:
                continue
            plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
            di_sum = plus_di + minus_di
            if di_sum == 0:
                continue
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
            dx_values.append(dx)

        if len(dx_values) < p:
            return None

        # ADX = Wilder's smoothing of DX
        adx = sum(dx_values[:p]) / p
        for i in range(p, len(dx_values)):
            adx = (adx * (p - 1) + dx_values[i]) / p

        return D(str(adx))


class HurstExponent:
    """Hurst exponent via R/S analysis — persistence dimension."""

    def __init__(self, min_candles: int = 100, update_interval_sec: int = 60):
        self.min_candles = min_candles
        self.update_interval_sec = update_interval_sec
        self._last_value: Optional[Decimal] = None
        self._last_compute_time: float = 0

    def compute(self, candles) -> Optional[Decimal]:
        now = time.time()
        # Rate-limit computation
        if self._last_value is not None and (now - self._last_compute_time) < self.update_interval_sec:
            return self._last_value

        if len(candles) < self.min_candles:
            return D("0.5")  # neutral fallback

        closes = [float(c.close) for c in candles[-self.min_candles:]]
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i] > 0 and closes[i - 1] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))
            else:
                log_returns.append(0.0)

        if len(log_returns) < 16:
            return D("0.5")

        sub_lengths = [n for n in [8, 16, 32, 64] if n <= len(log_returns) // 2]
        if len(sub_lengths) < 2:
            return D("0.5")

        log_ns = []
        log_rs_values = []
        for n in sub_lengths:
            num_chunks = len(log_returns) // n
            if num_chunks == 0:
                continue
            rs_vals = []
            for c in range(num_chunks):
                chunk = log_returns[c * n:(c + 1) * n]
                mean_r = sum(chunk) / len(chunk)
                adjusted = [x - mean_r for x in chunk]
                cumsum = []
                s = 0.0
                for v in adjusted:
                    s += v
                    cumsum.append(s)
                R = max(cumsum) - min(cumsum)
                S = (sum((x - mean_r) ** 2 for x in chunk) / len(chunk)) ** 0.5
                if S > 1e-15:
                    rs_vals.append(R / S)
            if rs_vals:
                avg_rs = sum(rs_vals) / len(rs_vals)
                if avg_rs > 0:
                    log_ns.append(math.log(n))
                    log_rs_values.append(math.log(avg_rs))

        if len(log_ns) < 2:
            return D("0.5")

        # Linear regression: log(R/S) = H * log(n) + c
        n_pts = len(log_ns)
        sum_x = sum(log_ns)
        sum_y = sum(log_rs_values)
        sum_xy = sum(x * y for x, y in zip(log_ns, log_rs_values))
        sum_xx = sum(x * x for x in log_ns)
        denom = n_pts * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-15:
            return D("0.5")
        H = (n_pts * sum_xy - sum_x * sum_y) / denom
        H = max(0.0, min(1.0, H))

        self._last_value = D(str(round(H, 4)))
        self._last_compute_time = now
        return self._last_value


class HMMRegimeDetector:
    """Hidden Markov Model regime detector — probabilistic regime synthesis."""

    REGIME_LABELS = ["ranging", "trending", "volatile"]

    def __init__(self, n_states: int = 3, min_candles: int = 200,
                 refit_interval_sec: int = 1800, window: int = 500):
        self.n_states = n_states
        self.min_candles = min_candles
        self.refit_interval_sec = refit_interval_sec
        self.window = window
        self._model = None
        self._label_map: Optional[Dict[int, str]] = None
        self._last_fit_time: float = 0
        self._available = HMM_AVAILABLE

    def _build_observations(self, candles,
                            fill_asymmetry_ratio: Optional[float] = None) -> Optional[np.ndarray]:
        if len(candles) < 11:
            return None
        closes = [float(c.close) for c in candles]
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                returns.append(closes[i] / closes[i - 1] - 1.0)
            else:
                returns.append(0.0)

        if len(returns) < 10:
            return None

        obs = []
        for i in range(10, len(returns)):
            ret = returns[i]
            abs_ret = abs(ret)
            roll_vol = float(np.std(returns[i - 10:i])) if i >= 10 else 0.0
            row = [ret, abs_ret, roll_vol]
            if fill_asymmetry_ratio is not None:
                row.append(fill_asymmetry_ratio)
            obs.append(row)
        return np.array(obs) if obs else None

    def _fit(self, candles, fill_asymmetry_ratio: Optional[float] = None):
        if not self._available:
            return
        obs = self._build_observations(candles[-self.window:], fill_asymmetry_ratio)
        if obs is None or len(obs) < 50:
            return
        try:
            model = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                                n_iter=50, random_state=42)
            model.fit(obs)
            # Label states by variance (sum of diagonal covariances)
            variances = [model.covars_[i].sum() for i in range(self.n_states)]
            sorted_indices = sorted(range(self.n_states), key=lambda i: variances[i])
            self._label_map = {}
            for rank, idx in enumerate(sorted_indices):
                self._label_map[idx] = self.REGIME_LABELS[min(rank, len(self.REGIME_LABELS) - 1)]
            self._model = model
            self._last_fit_time = time.time()
        except Exception:
            pass  # graceful degradation

    def predict(self, candles,
                fill_asymmetry_ratio: Optional[float] = None) -> Optional[Dict[str, float]]:
        if not self._available:
            return None
        if len(candles) < self.min_candles:
            return None

        now = time.time()
        if self._model is None or (now - self._last_fit_time) >= self.refit_interval_sec:
            self._fit(candles, fill_asymmetry_ratio)

        if self._model is None or self._label_map is None:
            return None

        obs = self._build_observations(candles[-20:], fill_asymmetry_ratio)
        if obs is None or len(obs) == 0:
            return None
        try:
            probs = self._model.predict_proba(obs[-1:])
            result = {label: 0.0 for label in self.REGIME_LABELS}
            for state_idx, prob in enumerate(probs[0]):
                label = self._label_map.get(state_idx, "volatile")
                result[label] += prob
            return result
        except Exception:
            return None


# ---------------------------------------------------------------------------
# DynamicRangeController
# ---------------------------------------------------------------------------


class RangeDecision(NamedTuple):
    """Full breakdown of DynamicRangeController's decision pipeline."""
    natr_mult: float
    natr_adjusted_pct: float
    adx_normalized: float
    persistence: float
    effective_trend: float
    hmm_override: str          # "none", "ranging_dampen", "trending_floor"
    raw_pct: float
    smoothed_pct: Decimal


class DynamicRangeController:
    """Combines NATR + ADX + Hurst + HMM into concentration_pct adjustments."""

    def __init__(self, config: DeltaDefiCLAMMConfig):
        self.config = config
        self._previous_smoothed_pct: Optional[Decimal] = None
        self.last_decision: Optional[RangeDecision] = None

    def compute_concentration_pct(
        self,
        natr: Optional[Decimal],
        adx: Optional[Decimal],
        hurst: Optional[Decimal],
        hmm_probs: Optional[Dict[str, float]],
    ) -> Decimal:
        cfg = self.config
        base_pct = cfg.base_concentration_pct

        # Step 1: NATR -> baseline range width
        if natr is not None:
            natr_f = float(natr)
            baseline_f = float(cfg.natr_baseline)
            scale_f = float(cfg.natr_range_scale)
            deviation = (natr_f - baseline_f) / baseline_f if baseline_f > 0 else 0.0
            natr_mult = max(0.5, min(3.0, 1.0 + deviation * scale_f))
        else:
            natr_mult = 1.0
        natr_adjusted_pct = float(base_pct) * natr_mult

        # Step 2: ADX -> trend strength (0-1)
        if adx is not None:
            adx_normalized = min(float(adx) / 50.0, 1.0)
        else:
            adx_normalized = 0.0

        # Step 3: Hurst -> persistence modifier
        if hurst is not None:
            persistence = (float(hurst) - 0.5) * 2.0
        else:
            persistence = 0.0  # neutral
        effective_trend = adx_normalized * max(0.0, 1.0 + persistence)

        # Step 4: HMM regime override
        hmm_override = "none"
        if hmm_probs is not None:
            threshold = float(cfg.hmm_confidence_threshold)
            if hmm_probs.get("ranging", 0) > threshold:
                effective_trend *= 0.3
                hmm_override = f"ranging_dampen({hmm_probs['ranging']:.0%})"
            elif hmm_probs.get("trending", 0) > threshold:
                effective_trend = max(effective_trend, 0.5)
                hmm_override = f"trending_floor({hmm_probs['trending']:.0%})"

        # Step 5: Final range with anti-oscillation
        trend_sens = float(cfg.trend_sensitivity)
        raw_pct = natr_adjusted_pct * (1.0 + trend_sens * effective_trend)
        raw_pct = max(float(cfg.min_concentration_pct), min(float(cfg.max_concentration_pct), raw_pct))

        # EMA smoothing
        alpha = float(cfg.range_ema_alpha)
        if self._previous_smoothed_pct is not None:
            prev = float(self._previous_smoothed_pct)
            smoothed = alpha * raw_pct + (1.0 - alpha) * prev
        else:
            smoothed = raw_pct
        self._previous_smoothed_pct = D(str(round(smoothed, 4)))

        self.last_decision = RangeDecision(
            natr_mult=round(natr_mult, 3),
            natr_adjusted_pct=round(natr_adjusted_pct, 2),
            adx_normalized=round(adx_normalized, 3),
            persistence=round(persistence, 3),
            effective_trend=round(effective_trend, 3),
            hmm_override=hmm_override,
            raw_pct=round(raw_pct, 2),
            smoothed_pct=self._previous_smoothed_pct,
        )

        return self._previous_smoothed_pct


# ---------------------------------------------------------------------------
# BalanceGate
# ---------------------------------------------------------------------------


class BalanceGate:
    """Reads real balance from exchange connector each cycle. Scales or
    removes orders that exceed what the account can support."""

    def __init__(self, connector: ConnectorBase, config: DeltaDefiCLAMMConfig):
        self.connector = connector
        self.config = config

    def get_real_balances(self):
        base_token, quote_token = self.config.trading_pair.split("-")
        base_bal = self.connector.get_available_balance(base_token)
        quote_bal = self.connector.get_available_balance(quote_token)
        if self.config.max_base_budget is not None:
            base_bal = min(base_bal, self.config.max_base_budget)
        if self.config.max_quote_budget is not None:
            quote_bal = min(quote_bal, self.config.max_quote_budget)
        return base_bal, quote_bal

    def scale_orders(self, orders: List[OrderProposal]) -> List[OrderProposal]:
        real_base, real_quote = self.get_real_balances()
        usable_base = real_base * self.config.balance_buffer_pct
        usable_quote = real_quote * self.config.balance_buffer_pct

        if real_base < self.config.min_base_balance:
            orders = [o for o in orders if o.side != TradeType.SELL]
        if real_quote < self.config.min_quote_balance:
            orders = [o for o in orders if o.side != TradeType.BUY]

        if not orders:
            return orders

        total_ask = sum(o.size for o in orders if o.side == TradeType.SELL)
        if total_ask > ZERO and total_ask > usable_base:
            scale = usable_base / total_ask
            orders = [
                OrderProposal(o.side, o.price, o.size * scale) if o.side == TradeType.SELL else o
                for o in orders
            ]

        total_bid_quote = sum(o.size * o.price for o in orders if o.side == TradeType.BUY)
        if total_bid_quote > ZERO and total_bid_quote > usable_quote:
            scale = usable_quote / total_bid_quote
            orders = [
                OrderProposal(o.side, o.price, o.size * scale) if o.side == TradeType.BUY else o
                for o in orders
            ]

        return orders


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class DeltaDefiCLAMM(StrategyV2Base):
    _default_config = DeltaDefiCLAMMConfig()
    markets = {_default_config.exchange: {_default_config.trading_pair}}
    _refresh_timestamp: float = 0
    _stopped: bool = False
    _refreshing: bool = False
    _pool_scaled: bool = False

    @classmethod
    def init_markets(cls, config: DeltaDefiCLAMMConfig):
        markets = {config.exchange: {config.trading_pair}}
        if config.enable_hedge:
            markets.setdefault(config.hedge_exchange, set()).add(
                config.hedge_trading_pair)
        cls.markets = markets

    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[DeltaDefiCLAMMConfig] = None):
        super().__init__(connectors, config)
        if self.config is None:
            self.config = DeltaDefiCLAMMConfig()

        # Per-pair logger
        pair_tag = self.config.trading_pair
        self._pair_logger = logging.getLogger(f"{__name__}.{pair_tag}")
        self._pair_logger.propagate = False

        logs_dir = str(DEFAULT_LOG_FILE_PATH)
        os.makedirs(logs_dir, exist_ok=True)
        pair_log_file = os.path.join(logs_dir, f"logs_deltadefi_cl_amm_mm_{pair_tag}.log")
        if not self._pair_logger.handlers:
            fh = logging.handlers.TimedRotatingFileHandler(
                pair_log_file, when="D", interval=1, backupCount=7, encoding="utf8"
            )
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._pair_logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._pair_logger.addHandler(ch)
            self._pair_logger.setLevel(logging.DEBUG)

        connector = connectors[self.config.exchange]
        network = getattr(connector, "deltadefi_network", "mainnet")
        self._state_dir = os.path.join(os.path.dirname(logs_dir), "state", network)
        os.makedirs(self._state_dir, exist_ok=True)
        self._state_file = os.path.join(self._state_dir, f"{self.config.trading_pair}_cl_pool_state.json")

        # Initialize indicators
        self._natr_indicator = NATRIndicator(self.config.natr_period)
        self._adx_indicator = ADXIndicator(self.config.adx_period)
        self._hurst_indicator = HurstExponent(self.config.hurst_min_candles, self.config.hurst_update_interval_sec)
        self._hmm_detector = HMMRegimeDetector(
            self.config.hmm_n_states, self.config.hmm_min_candles,
            self.config.hmm_refit_interval_sec, self.config.hmm_window,
        )
        self._range_controller = DynamicRangeController(self.config)

        # Indicator cache for status display
        self._last_natr: Optional[Decimal] = None
        self._last_adx: Optional[Decimal] = None
        self._last_hurst: Optional[Decimal] = None
        self._last_hmm: Optional[Dict[str, float]] = None

        # Load state
        state = self._load_state()
        if state and "L" in state:
            self.pool = ConcentratedPool.from_state(state)
            self._base_flow = D(str(state.get("base_flow", "0")))
            self._quote_flow = D(str(state.get("quote_flow", "0")))
            self._pair_logger.info(f"Restored CL pool state from {self._state_file}")
        elif state and "amplification" in state:
            self.pool = ConcentratedPool.from_legacy_state(state)
            self._base_flow = D(str(state.get("base_flow", "0")))
            self._quote_flow = D(str(state.get("quote_flow", "0")))
            self._pair_logger.info(f"Migrated legacy pool state to concentrated liquidity")
        else:
            self.pool = None
            self._base_flow = ZERO
            self._quote_flow = ZERO

        self.balance_gate = BalanceGate(connectors[self.config.exchange], self.config)
        self._recent_fills: deque = deque(maxlen=1000)
        self._toxicity_recent_fills: deque = deque(maxlen=2000)
        self._flair_pending: deque = deque(maxlen=4000)
        self._flair_events: deque = deque(maxlen=8000)
        self._flair_roll_fee: Decimal = ZERO
        self._flair_roll_lvr: Decimal = ZERO
        self._flair_roll_net: Decimal = ZERO
        self._flair_lifetime_fee: Decimal = ZERO
        self._flair_lifetime_lvr: Decimal = ZERO
        self._flair_lifetime_net: Decimal = ZERO

        # Post-fill adverse selection tracking (+1m, +5m, +15m markouts)
        self._adverse_pending: deque = deque(maxlen=12000)
        self._adverse_stats = {
            "1m":  {"count": 0, "total_move": 0.0, "adverse_count": 0},
            "5m":  {"count": 0, "total_move": 0.0, "adverse_count": 0},
            "15m": {"count": 0, "total_move": 0.0, "adverse_count": 0},
        }

        # LVR accumulator for rebalance decision
        self._lvr_since_reposition: Decimal = ZERO
        self._fills_since_reposition: int = 0

        # Per-tick caches (invalidated by self._tick_seq)
        self._tick_seq: int = 0
        self._cached_toxicity: tuple = (-1, None)  # (tick_seq, profile)
        self._cached_inventory: tuple = (-1, None)

        # Trading P&L: shared avg-cost book (spot + hedge)
        self._spot_book = AvgCostBook()
        self._hedge_book = AvgCostBook()
        self._hedge_fills: int = 0
        # Pending hedge order ids so we can route fills correctly even if the
        # trading_pair check is ambiguous.
        self._hedge_order_ids: set = set()
        self._hedge_in_flight: bool = False  # avoid double-firing
        self._hedge_in_flight_ts: float = 0.0
        self._hedge_leverage_set: bool = False
        self._hedge_last_state_change_ts: float = 0.0
        # Throttle "connector unavailable" warnings to once per 60s
        self._hedge_unavailable_warned_ts: float = 0.0
        if state:
            # Spot — try new keys, then legacy "position_*" / FIFO lots
            if "spot_size" in state:
                self._spot_book.load_from(state, prefix="spot_")
            elif "position_size" in state:
                self._spot_book.load_from({
                    "size": state.get("position_size", "0"),
                    "avg_cost": state.get("position_avg_cost", "0"),
                    "side": state.get("position_side", ""),
                    "realized_pnl": state.get("realized_pnl", "0"),
                    "fees": state.get("cumulative_fees", "0"),
                })
            elif "position_lots" in state:
                # Collapse old FIFO lots into a single avg
                try:
                    lots = state.get("position_lots") or []
                    total_sz = sum(D(str(e[0])) for e in lots if D(str(e[0])) > ZERO)
                    total_n = sum(D(str(e[0])) * D(str(e[1])) for e in lots
                                  if D(str(e[0])) > ZERO and D(str(e[1])) > ZERO)
                    sd = state.get("position_side") or ""
                    if total_sz > ZERO and sd in ("long", "short"):
                        self._spot_book.size = total_sz
                        self._spot_book.avg_cost = total_n / total_sz
                        self._spot_book.side = sd
                    self._spot_book.realized_pnl = D(str(state.get("realized_pnl", "0")))
                    self._spot_book.fees = D(str(state.get("cumulative_fees", "0")))
                except Exception:
                    pass
            # Hedge
            self._hedge_book.load_from(state, prefix="hedge_")
            try:
                self._hedge_fills = int(state.get("hedge_fills", 0) or 0)
            except Exception:
                pass
        self._last_max_safe: Decimal = ZERO
        self._rebalancing: bool = False
        self._last_rebalance_action: str = ""
        self._indicator_log_counter: int = 0
        self._last_fill_ts: float = 0.0
        self._last_reposition_price: Optional[Decimal] = None
        self._gap_check_ts: float = 0.0
        self._last_autocorr: Optional[float] = None
        self._last_vol_ratio: Optional[float] = None

    # ---- Main loop --------------------------------------------------------

    def on_tick(self):
        if self._stopped:
            return

        # Invalidate per-tick caches at the start of each tick
        self._tick_seq += 1

        connector = self.connectors[self.config.exchange]
        if self.config.trading_pair not in connector.trading_rules:
            return

        # Initialize or auto-scale pool
        if self.pool is None:
            if not self._init_pool_from_balance():
                return
        elif not self._pool_scaled:
            self._auto_scale_pool()

        if self._check_circuit_breakers():
            return

        # Keep anchor fresh
        book_mid = self._get_book_mid()
        if book_mid:
            self._update_anchor_ema(book_mid)
            self._update_flair_monitor(book_mid)
            self._resolve_adverse_markouts(book_mid)

        # Outer wing trigger: fire when price drifts past outer zone from pool center
        _recentered = False
        if book_mid:
            pool_center = (self.pool.p_lower + self.pool.p_upper) / D(2)
            if pool_center > ZERO:
                price_drift = float(abs(book_mid - pool_center) / pool_center)
                outer_trig = self._outer_trigger_pct()
                if price_drift >= outer_trig:
                    self._pair_logger.info(
                        f"Outer wing trigger: drift {price_drift*100:.2f}% >= "
                        f"{outer_trig*100:.2f}% from pool center {pool_center:.6f}")
                    self.pool.recenter(book_mid)
                    self._last_reposition_price = book_mid
                    self._save_state()
                    safe_ensure_future(self._conditional_rebalance(book_mid))
                    _recentered = True

        # Hard recenter: book mid exits bounds (fallback if outer trigger didn't catch it)
        if not _recentered and book_mid and not self.pool.is_in_range(book_mid):
            self._pair_logger.info(
                f"Hard recenter: book_mid {book_mid:.6f} outside "
                f"[{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}]"
            )
            self.pool.recenter(book_mid)
            self._last_reposition_price = book_mid
            self._save_state()
            safe_ensure_future(self._conditional_rebalance(book_mid))
            _recentered = True

        # Soft recenter: anchor drifted from range center (price still in range)
        if not _recentered and book_mid and self.config.soft_recenter_drift_pct > ZERO:
            range_center = (self.pool.p_lower + self.pool.p_upper) / D(2)
            if range_center > ZERO:
                drift_pct = abs(self.pool.anchor_price - range_center) / range_center * D(100)
                if drift_pct >= self.config.soft_recenter_drift_pct:
                    self._pair_logger.info(
                        f"Soft recenter: anchor {self.pool.anchor_price:.6f} drifted "
                        f"{drift_pct:.2f}% from range center {range_center:.6f}"
                    )
                    self.pool.recenter(self.pool.anchor_price)
                    self._last_reposition_price = self.pool.anchor_price
                    self._save_state()
                    safe_ensure_future(self._conditional_rebalance(self.pool.anchor_price))

        # Gap detection heartbeat: catches flash crashes that skip the outer wing trigger.
        # Skip if we've already recentered this tick — outer/hard/soft already handled it.
        now_ts = time.time()
        if (not _recentered and book_mid and self._last_reposition_price
                and self._last_reposition_price > ZERO
                and now_ts - self._gap_check_ts >= 30.0):
            self._gap_check_ts = now_ts
            gap_pct = float(abs(book_mid - self._last_reposition_price) / self._last_reposition_price)
            half_width = float(self.pool.concentration_pct) / 100.0
            if gap_pct > half_width:
                self._pair_logger.warning(
                    f"Gap detected: oracle moved {gap_pct * 100:.2f}% since last reposition "
                    f"(threshold: {half_width * 100:.2f}%). Emergency reposition."
                )
                self.pool.recenter(book_mid)
                self._last_reposition_price = book_mid
                self._save_state()
                safe_ensure_future(self._conditional_rebalance(book_mid))

        # Dynamic range update from indicators
        self._update_dynamic_range(connector)

        # Futures hedge check
        if self.config.enable_hedge:
            self._check_and_update_hedge()

        # Don't place orders while refreshing
        if self._refreshing:
            return

        has_active = bool(self.get_active_orders(connector_name=self.config.exchange))

        if self.config.refresh_on_fill_only:
            if has_active:
                return
        else:
            if self.current_timestamp < self._refresh_timestamp:
                return
            if has_active:
                self._cancel_all_orders()
                return

        # Compute effective mid: (1-w)*anchor + w*pool_mid
        w = self.config.pool_price_weight
        pool_mid = self.pool.get_mid_price()
        mid = (D(1) - w) * self.pool.anchor_price + w * pool_mid
        RateOracle.get_instance().set_price(self.config.trading_pair, mid)

        orders = self._generate_orders(mid)
        orders = self.balance_gate.scale_orders(orders)
        self._place_orders(orders)
        self._refresh_timestamp = self.current_timestamp + self.config.order_refresh_time

    # ---- Dynamic range update ---------------------------------------------

    def _update_dynamic_range(self, connector):
        candle_builder = getattr(connector, "candle_builder", None)
        if candle_builder is None:
            return

        max_needed = max(
            self.config.natr_period + 1,
            self.config.adx_period * 2 + 1,
            self.config.hurst_min_candles,
            self.config.hmm_window,
        )
        candles = candle_builder.get_candles(max_needed)
        if not candles:
            return

        self._last_natr = self._natr_indicator.compute(candles)
        self._last_adx = self._adx_indicator.compute(candles)
        self._last_hurst = self._hurst_indicator.compute(candles)

        # Compute autocorr and vol_ratio for regime composite signal
        self._last_autocorr = self._compute_autocorr_lag1(candles)
        self._last_vol_ratio = self._compute_vol_ratio(candles)

        # HMM: pass fill asymmetry as 4th feature if configured
        fill_asym = None
        if self.config.hmm_use_fill_asymmetry:
            tox = self._get_toxicity_profile()
            n = int(tox["sample_count"])
            if n >= self.config.toxicity_activation_count:
                fill_asym = float(tox["buy_ratio"])
        self._last_hmm = self._hmm_detector.predict(candles, fill_asym)

        new_pct = self._range_controller.compute_concentration_pct(
            self._last_natr, self._last_adx, self._last_hurst, self._last_hmm,
        )

        # Periodic indicator log (every 300 ticks ≈ 5 min)
        self._indicator_log_counter += 1
        decision = self._range_controller.last_decision
        range_changed = abs(new_pct - self.pool.concentration_pct) >= self.config.range_update_dead_band_pct

        if self._indicator_log_counter >= 300 or range_changed:
            self._indicator_log_counter = 0
            natr_str = f"{float(self._last_natr)*100:.3f}%" if self._last_natr is not None else "n/a"
            adx_str = f"{float(self._last_adx):.1f}" if self._last_adx is not None else "n/a"
            hurst_str = f"{float(self._last_hurst):.3f}" if self._last_hurst is not None else "n/a"
            if self._last_hmm is not None:
                top = max(self._last_hmm, key=self._last_hmm.get)
                hmm_str = f"{top}({self._last_hmm[top]:.0%})"
            else:
                hmm_str = "n/a"
            self._pair_logger.info(
                f"Trend analysis [{len(candles)} candles]: "
                f"NATR={natr_str} ADX={adx_str} Hurst={hurst_str} HMM={hmm_str}"
            )
            if decision is not None:
                self._pair_logger.info(
                    f"  Pipeline: natr_mult={decision.natr_mult:.2f} "
                    f"natr_adj={decision.natr_adjusted_pct:.2f}% "
                    f"adx_norm={decision.adx_normalized:.2f} "
                    f"persist={decision.persistence:+.2f} "
                    f"eff_trend={decision.effective_trend:.3f} "
                    f"hmm_override={decision.hmm_override}"
                )
                self._pair_logger.info(
                    f"  Result: raw={decision.raw_pct:.2f}% "
                    f"smoothed={decision.smoothed_pct:.2f}% "
                    f"current={self.pool.concentration_pct:.2f}% "
                    f"{'-> UPDATING' if range_changed else '(within dead-band)'}"
                )

        if range_changed:
            old_pct = self.pool.concentration_pct
            self.pool.set_concentration(new_pct)
            self._save_state()
            self._pair_logger.info(
                f"Dynamic range: {old_pct:.2f}% -> {new_pct:.2f}% "
                f"[{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}]"
            )

    # ---- Order generation -------------------------------------------------

    def _max_safe_order_base(self) -> Decimal:
        """Max order size (in base) for concentrated liquidity no-ping-pong.

        dp/p = dB * sqrt(p) / L
        w * dp/p < spread => dB < spread * L / (w * sqrt(p))
        """
        w = self.config.pool_price_weight
        spread = self.config.base_spread_bps / D("10000")
        if w <= ZERO or self.pool.L <= ZERO:
            return D("999999999")
        sqrt_p = _sqrt(self.pool.anchor_price)
        if sqrt_p <= ZERO:
            return D("999999999")
        max_base = spread * self.pool.L / (w * sqrt_p)
        self._last_max_safe = max_base
        return max_base

    def _generate_orders(self, mid_price: Decimal) -> List[OrderProposal]:
        orders: List[OrderProposal] = []

        # Trend protection
        decision = self._range_controller.last_decision
        eff_trend = decision.effective_trend if decision is not None else 0.0

        halt_thresh = float(self.config.trend_halt_threshold)
        if halt_thresh > 0 and eff_trend >= halt_thresh:
            return orders

        scale_factor = float(self.config.trend_order_scale_factor)
        trend_scale = max(0.0, 1.0 - scale_factor * eff_trend)
        if trend_scale <= 0.0:
            return orders

        max_safe_base = self._max_safe_order_base()
        order_base = max_safe_base * self.config.order_safe_ratio * D(str(round(trend_scale, 6)))
        order_value = order_base * mid_price

        connector = self.connectors[self.config.exchange]
        pair = self.config.trading_pair
        toxicity = self._get_toxicity_profile()
        inventory = self._inventory_adjustments()
        momentum_bid_extra, momentum_ask_extra = self._momentum_spread_adjustment()

        inner_spread = self.config.base_spread_bps / D("10000")
        inner_value = order_value * (D(1) - self.config.outer_capital_fraction)
        orders += self._make_order_pair(
            mid_price, inner_spread, inner_value, toxicity, inventory,
            momentum_bid_extra, momentum_ask_extra, connector, pair, outer_zone=False)

        outer_spread = inner_spread * self.config.outer_spread_mult
        outer_value = order_value * self.config.outer_capital_fraction
        orders += self._make_order_pair(
            mid_price, outer_spread, outer_value, toxicity, inventory,
            momentum_bid_extra, momentum_ask_extra, connector, pair, outer_zone=True)

        return orders

    def _make_order_pair(self, mid_price: Decimal, base_spread: Decimal,
                         total_value: Decimal, toxicity: dict, inventory: dict,
                         momentum_bid_extra: Decimal, momentum_ask_extra: Decimal,
                         connector, pair: str, outer_zone: bool = False) -> List[OrderProposal]:
        if self.config.enable_asymmetric_spread:
            bid_spread, ask_spread = self._asymmetric_spreads(base_spread)
        else:
            bid_spread, ask_spread = base_spread, base_spread

        if toxicity["buy_state"] >= D(2):
            bid_spread *= self.config.toxicity_hard_spread_mult
        elif toxicity["buy_state"] >= D(1):
            bid_spread *= self.config.toxicity_soft_spread_mult
        if toxicity["sell_state"] >= D(2):
            ask_spread *= self.config.toxicity_hard_spread_mult
        elif toxicity["sell_state"] >= D(1):
            ask_spread *= self.config.toxicity_soft_spread_mult

        if inventory["state"] >= D(2):
            if inventory["accumulation_side"] == "buy":
                bid_spread *= self.config.inventory_hard_spread_mult
            elif inventory["accumulation_side"] == "sell":
                ask_spread *= self.config.inventory_hard_spread_mult
        elif inventory["state"] >= D(1):
            if inventory["accumulation_side"] == "buy":
                bid_spread *= self.config.inventory_soft_spread_mult
            elif inventory["accumulation_side"] == "sell":
                ask_spread *= self.config.inventory_soft_spread_mult

        bid_spread += momentum_bid_extra
        ask_spread += momentum_ask_extra

        ask_price = mid_price * (D(1) + ask_spread)
        bid_price = mid_price * (D(1) - bid_spread)

        ask_size = total_value / ask_price if ask_price > ZERO else ZERO
        bid_size = total_value / bid_price if bid_price > ZERO else ZERO

        if toxicity["buy_state"] >= D(2):
            bid_size *= self.config.toxicity_hard_size_mult
        elif toxicity["buy_state"] >= D(1):
            bid_size *= self.config.toxicity_soft_size_mult
        if toxicity["sell_state"] >= D(2):
            ask_size *= self.config.toxicity_hard_size_mult
        elif toxicity["sell_state"] >= D(1):
            ask_size *= self.config.toxicity_soft_size_mult

        if inventory["state"] >= D(2):
            if inventory["accumulation_side"] == "buy":
                if self.config.inventory_hard_disable_accumulation_side:
                    bid_size = ZERO
                else:
                    bid_size *= self.config.inventory_hard_size_mult
            elif inventory["accumulation_side"] == "sell":
                if self.config.inventory_hard_disable_accumulation_side:
                    ask_size = ZERO
                else:
                    ask_size *= self.config.inventory_hard_size_mult
        elif inventory["state"] >= D(1):
            if inventory["accumulation_side"] == "buy":
                bid_size *= self.config.inventory_soft_size_mult
            elif inventory["accumulation_side"] == "sell":
                ask_size *= self.config.inventory_soft_size_mult

        if self.config.enable_order_randomization:
            ask_size = self._randomize(ask_size)
            bid_size = self._randomize(bid_size)

        ask_price = self._quantize_price(connector, pair, ask_price)
        bid_price = self._quantize_price(connector, pair, bid_price)
        ask_size = self._quantize_amount(connector, pair, ask_size)
        bid_size = self._quantize_amount(connector, pair, bid_size)

        result = []
        if ask_size > ZERO:
            result.append(OrderProposal(TradeType.SELL, ask_price, ask_size))
        if bid_size > ZERO:
            result.append(OrderProposal(TradeType.BUY, bid_price, bid_size))
        return result

    def _asymmetric_spreads(self, base_spread: Decimal):
        skew = D(str(self.pool.get_inventory_skew()))
        sens = self.config.skew_sensitivity
        floor = self.config.min_spread_bps / D("10000")
        bid_spread = max(base_spread * (D(1) + skew * sens), floor)
        ask_spread = max(base_spread * (D(1) - skew * sens), floor)
        return bid_spread, ask_spread

    def _outer_trigger_pct(self) -> float:
        inner_half = float(self.pool.concentration_pct) / 100
        outer_half = inner_half * float(self.config.outer_range_mult)
        wing_half = outer_half - inner_half
        return inner_half + wing_half * float(self.config.outer_recenter_trigger_pct)

    def _get_toxicity_profile(self) -> Dict[str, Decimal]:
        cached_seq, cached_val = self._cached_toxicity
        if cached_seq == self._tick_seq and cached_val is not None:
            return cached_val
        val = self._compute_toxicity_profile()
        self._cached_toxicity = (self._tick_seq, val)
        return val

    def _compute_toxicity_profile(self) -> Dict[str, Decimal]:
        # Fill-count window: use last N fills regardless of elapsed time
        window_n = self.config.toxicity_window_fills
        recent = list(self._toxicity_recent_fills)[-window_n:]

        buy_count = sum(1 for _, t in recent if t == TradeType.BUY)
        sell_count = sum(1 for _, t in recent if t == TradeType.SELL)
        total = buy_count + sell_count
        if total <= 0:
            return {
                "buy_ratio": ZERO,
                "sell_ratio": ZERO,
                "sample_count": ZERO,
                "buy_toxicity": ZERO,
                "sell_toxicity": ZERO,
                "buy_state": D(0),
                "sell_state": D(0),
            }

        buy_ratio = D(str(buy_count / total))
        sell_ratio = D(str(sell_count / total))
        sample_count = D(str(total))
        min_count = self.config.toxicity_activation_count

        buy_state = D(0)
        if total >= min_count:
            if buy_ratio >= self.config.toxicity_buy_ratio_hard:
                buy_state = D(2)
            elif buy_ratio >= self.config.toxicity_buy_ratio_soft:
                buy_state = D(1)

        sell_state = D(0)
        if total >= min_count:
            if sell_ratio >= self.config.toxicity_sell_ratio_hard:
                sell_state = D(2)
            elif sell_ratio >= self.config.toxicity_sell_ratio_soft:
                sell_state = D(1)

        return {
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "sample_count": sample_count,
            "buy_toxicity": buy_state,
            "sell_toxicity": sell_state,
            "buy_state": buy_state,
            "sell_state": sell_state,
        }

    @staticmethod
    def _compute_autocorr_lag1(candles) -> Optional[float]:
        """Lag-1 return autocorrelation over rolling 30-candle window."""
        if len(candles) < 32:
            return None
        closes = [float(c.close) for c in candles[-31:]]
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            rets.append((closes[i] / prev - 1.0) if prev > 0 else 0.0)
        if len(rets) < 4:
            return None
        x, y = rets[:-1], rets[1:]
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        sx = math.sqrt(sum((v - mx) ** 2 for v in x))
        sy = math.sqrt(sum((v - my) ** 2 for v in y))
        return cov / (sx * sy) if sx > 0 and sy > 0 else None

    @staticmethod
    def _compute_vol_ratio(candles) -> Optional[float]:
        """Ratio of short-term realized vol to long-term: EWMA_60 / EWMA_240."""
        if len(candles) < 62:
            return None
        closes = [float(c.close) for c in candles]
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            rets.append((closes[i] / prev - 1.0) if prev > 0 else 0.0)
        if len(rets) < 61:
            return None
        short_rets = rets[-60:]
        long_rets = rets[-min(240, len(rets)):]
        short_var = sum(r * r for r in short_rets) / len(short_rets)
        long_var = sum(r * r for r in long_rets) / len(long_rets)
        if long_var <= 0:
            return None
        return math.sqrt(short_var / long_var)

    def _inventory_adjustments(self) -> Dict[str, object]:
        cached_seq, cached_val = self._cached_inventory
        if cached_seq == self._tick_seq and cached_val is not None:
            return cached_val
        val = self._compute_inventory_adjustments()
        self._cached_inventory = (self._tick_seq, val)
        return val

    def _compute_inventory_adjustments(self) -> Dict[str, object]:
        skew = D(str(self.pool.get_inventory_skew()))
        abs_skew = abs(skew)
        soft = self.config.inventory_skew_soft_limit
        hard = self.config.inventory_skew_hard_limit

        state = D(0)
        if abs_skew >= hard:
            state = D(2)
        elif abs_skew >= soft:
            state = D(1)

        side = None
        if skew > ZERO:
            side = "sell"
        elif skew < ZERO:
            side = "buy"

        return {
            "skew": skew,
            "state": state,
            "accumulation_side": side,
        }

    @staticmethod
    def _to_decimal(value) -> Decimal:
        try:
            return D(str(value))
        except Exception:
            return ZERO

    def _fee_quote_equivalent(self, event: OrderFilledEvent) -> Decimal:
        fee = getattr(event, "trade_fee", None)
        if fee is None:
            return ZERO

        base_token, quote_token = self.config.trading_pair.split("-")
        fill_price = self._to_decimal(getattr(event, "price", ZERO))
        fill_amount = self._to_decimal(getattr(event, "amount", ZERO))
        notional_quote = fill_price * fill_amount
        total_fee_quote = ZERO

        percent = self._to_decimal(
            fee.get("percent", ZERO) if isinstance(fee, dict) else getattr(fee, "percent", ZERO)
        )
        if percent > ZERO:
            total_fee_quote += notional_quote * percent

        flat_fees = fee.get("flat_fees", []) if isinstance(fee, dict) else getattr(fee, "flat_fees", [])
        for flat_fee in flat_fees or []:
            if isinstance(flat_fee, dict):
                token = flat_fee.get("token")
                amount = self._to_decimal(flat_fee.get("amount", ZERO))
            else:
                token = getattr(flat_fee, "token", None)
                amount = self._to_decimal(getattr(flat_fee, "amount", ZERO))
            if amount <= ZERO:
                continue
            if token == quote_token:
                total_fee_quote += amount
            elif token == base_token:
                total_fee_quote += amount * fill_price

        return total_fee_quote

    def _queue_flair_markout(self, event: OrderFilledEvent):
        if not self.config.enable_flair_monitor:
            return
        now_ts = time.time()
        self._flair_pending.append((
            now_ts + self.config.flair_markout_sec,
            event.trade_type,
            self._to_decimal(event.price),
            self._to_decimal(event.amount),
            self._fee_quote_equivalent(event),
        ))

    def _prune_flair_window(self, now_ts: Optional[float] = None):
        if now_ts is None:
            now_ts = time.time()
        while self._flair_events and now_ts - self._flair_events[0][0] > self.config.flair_window_sec:
            _, fee_quote, lvr_quote, flair_net = self._flair_events.popleft()
            self._flair_roll_fee -= fee_quote
            self._flair_roll_lvr -= lvr_quote
            self._flair_roll_net -= flair_net

    def _update_flair_monitor(self, book_mid: Decimal):
        if not self.config.enable_flair_monitor or book_mid is None or book_mid <= ZERO:
            return
        now_ts = time.time()
        while self._flair_pending and self._flair_pending[0][0] <= now_ts:
            _, side, fill_price, amount, fee_quote = self._flair_pending.popleft()
            if side == TradeType.BUY:
                lvr_quote = max(ZERO, amount * (fill_price - book_mid))
            else:
                lvr_quote = max(ZERO, amount * (book_mid - fill_price))
            flair_net = fee_quote - lvr_quote

            self._flair_lifetime_fee += fee_quote
            self._flair_lifetime_lvr += lvr_quote
            self._flair_lifetime_net += flair_net

            # Accumulate LVR since last reposition for LVR-based rebalance decision
            self._lvr_since_reposition += lvr_quote

            self._flair_events.append((now_ts, fee_quote, lvr_quote, flair_net))
            self._flair_roll_fee += fee_quote
            self._flair_roll_lvr += lvr_quote
            self._flair_roll_net += flair_net

        self._prune_flair_window(now_ts)

    def _flair_summary(self) -> Dict[str, Decimal]:
        self._prune_flair_window()
        eps = D("0.00000001")
        roll_ratio = self._flair_roll_fee / max(self._flair_roll_lvr, eps) if self._flair_roll_fee > ZERO else ZERO
        life_ratio = self._flair_lifetime_fee / max(self._flair_lifetime_lvr, eps) if self._flair_lifetime_fee > ZERO else ZERO
        return {
            "roll_fee": self._flair_roll_fee,
            "roll_lvr": self._flair_roll_lvr,
            "roll_net": self._flair_roll_net,
            "roll_ratio": roll_ratio,
            "lifetime_fee": self._flair_lifetime_fee,
            "lifetime_lvr": self._flair_lifetime_lvr,
            "lifetime_net": self._flair_lifetime_net,
            "lifetime_ratio": life_ratio,
            "pending": D(str(len(self._flair_pending))),
            "samples": D(str(len(self._flair_events))),
        }

    # ---- Post-fill adverse selection (1m / 5m / 15m markouts) ----

    _ADVERSE_HORIZONS = (("1m", 60), ("5m", 300), ("15m", 900))

    def _queue_adverse_markout(self, event: OrderFilledEvent):
        now_ts = time.time()
        fill_price = self._to_decimal(event.price)
        amount = self._to_decimal(event.amount)
        for label, sec in self._ADVERSE_HORIZONS:
            self._adverse_pending.append((
                now_ts + sec, label, event.trade_type, fill_price, amount))

    def _resolve_adverse_markouts(self, book_mid: Optional[Decimal]):
        if book_mid is None or book_mid <= ZERO:
            return
        now_ts = time.time()
        # _adverse_pending is not ordered by deadline (3 horizons interleave),
        # so iterate-and-filter rather than popleft.
        kept = deque(maxlen=self._adverse_pending.maxlen)
        for entry in self._adverse_pending:
            deadline, label, side, fill_price, amount = entry
            if deadline > now_ts:
                kept.append(entry)
                continue
            if side == TradeType.BUY:
                adverse_pct = float((fill_price - book_mid) / fill_price) if fill_price > ZERO else 0.0
            else:
                adverse_pct = float((book_mid - fill_price) / fill_price) if fill_price > ZERO else 0.0
            stats = self._adverse_stats[label]
            stats["count"] += 1
            stats["total_move"] += adverse_pct
            if adverse_pct > 0:
                stats["adverse_count"] += 1
        self._adverse_pending = kept

    def _adverse_summary_str(self) -> str:
        parts = []
        for label, _ in self._ADVERSE_HORIZONS:
            s = self._adverse_stats[label]
            if s["count"] == 0:
                parts.append(f"{label}=n/a")
            else:
                avg = s["total_move"] / s["count"] * 100
                parts.append(f"{label}={avg:+.3f}%")
        return " ".join(parts)

    def _adverse_status_lines(self) -> str:
        parts = []
        for label, _ in self._ADVERSE_HORIZONS:
            s = self._adverse_stats[label]
            if s["count"] == 0:
                parts.append(f"{label}=n/a")
            else:
                avg = s["total_move"] / s["count"] * 100
                parts.append(
                    f"{label}={avg:+.3f}% "
                    f"({s['adverse_count']}/{s['count']} adverse)")
        return " | ".join(parts)

    def _momentum_spread_adjustment(self):
        if not self.config.enable_momentum_spread:
            return ZERO, ZERO
        now = time.time()
        window = self.config.momentum_window_sec
        buys = sum(1 for ts, t in self._recent_fills if t == TradeType.BUY and now - ts <= window)
        sells = sum(1 for ts, t in self._recent_fills if t == TradeType.SELL and now - ts <= window)
        net = sells - buys
        extra_bps = self.config.momentum_spread_bps / D("10000")
        if net > 0:
            return ZERO, extra_bps * net
        elif net < 0:
            return extra_bps * abs(net), ZERO
        return ZERO, ZERO

    def _randomize(self, size: Decimal) -> Decimal:
        pct = float(self.config.randomization_pct)
        jitter = D(str(1.0 + random.uniform(-pct, pct)))
        return max(ZERO, size * jitter)

    def _quantize_price(self, connector: ConnectorBase, pair: str, price: Decimal) -> Decimal:
        if self.config.price_decimals is not None:
            return round(price, self.config.price_decimals)
        return connector.quantize_order_price(pair, price)

    def _quantize_amount(self, connector: ConnectorBase, pair: str, amount: Decimal) -> Decimal:
        if self.config.amount_decimals is not None:
            return round(amount, self.config.amount_decimals)
        return connector.quantize_order_amount(pair, amount)

    # ---- Order execution --------------------------------------------------

    def _place_orders(self, orders: List[OrderProposal]) -> List[str]:
        order_ids = []
        for o in orders:
            if o.side == TradeType.SELL:
                oid = self.sell(self.config.exchange, self.config.trading_pair, o.size, OrderType.LIMIT, o.price)
            else:
                oid = self.buy(self.config.exchange, self.config.trading_pair, o.size, OrderType.LIMIT, o.price)
            order_ids.append(oid)
        return order_ids

    def _cancel_all_orders(self):
        safe_ensure_future(self._cancel_all_orders_async())

    async def _cancel_all_orders_async(self):
        connector = self.connectors[self.config.exchange]
        await connector.cancel_all(timeout_seconds=10.0)

    async def _await_order_confirmations(self, order_ids: List[str], timeout: float = 10.0):
        connector = self.connectors[self.config.exchange]
        for oid in order_ids:
            tracked = connector.in_flight_orders.get(oid)
            if tracked is None:
                continue
            try:
                await asyncio.wait_for(tracked.get_exchange_order_id(), timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                pass

    # ---- Circuit breakers -------------------------------------------------

    def _check_circuit_breakers(self) -> bool:
        pnl = self._get_pnl()
        if pnl is not None and pnl < -self.config.max_cumulative_loss:
            self._pair_logger.error(f"Loss limit exceeded: P&L {pnl:.2f}. Shutting down.")
            self._cancel_all_orders()
            self._stopped = True
            return True

        now = time.time()
        max_window = 0
        if self.config.enable_fill_velocity_detector:
            max_window = max(max_window, self.config.fill_velocity_window_sec)
        if self.config.enable_momentum_spread:
            max_window = max(max_window, self.config.momentum_window_sec)
        max_window = max(max_window, self.config.toxicity_window_sec)
        if max_window > 0:
            while self._recent_fills and now - self._recent_fills[0][0] > max_window:
                self._recent_fills.popleft()

        if self.config.enable_fill_velocity_detector:
            vel_window = self.config.fill_velocity_window_sec
            buy_count = sum(1 for ts, t in self._recent_fills if t == TradeType.BUY and now - ts <= vel_window)
            sell_count = sum(1 for ts, t in self._recent_fills if t == TradeType.SELL and now - ts <= vel_window)
            if max(buy_count, sell_count) >= self.config.fill_velocity_max_same_side:
                self._pair_logger.warning(f"Fill velocity: {buy_count}B/{sell_count}S in {vel_window}s. Pausing.")
                self._cancel_all_orders()
                return True

        return False

    # ---- Anchor EMA -------------------------------------------------------

    def _update_anchor_ema(self, book_mid: Decimal):
        if self.pool.anchor_price and self.pool.anchor_price > ZERO:
            alpha = self.config.anchor_ema_alpha
            self.pool.anchor_price = alpha * book_mid + (D(1) - alpha) * self.pool.anchor_price
        else:
            self.pool.anchor_price = book_mid

    # ---- Conditional rebalance after recenter ------------------------------

    def _should_rebalance(self) -> tuple:
        """Decide rebalance fraction by comparing accumulated adverse selection
        cost (FLAIR LVR since last reposition) against the rebalance slippage
        budget. Only rebalance when leakage exceeds the cost of rebalancing.

        Returns (fraction, reason):
          fraction=0   → skip rebalance (cheaper to leave inventory alone)
          fraction=1.0 → full rebalance (LVR exceeds slippage budget)
        """
        if not self.config.enable_rebalance:
            return 0.0, "disabled"

        book_mid = self._get_book_mid()
        if book_mid is None or book_mid <= ZERO:
            return 0.0, "no book mid"

        real_base, real_quote = self.balance_gate.get_real_balances()
        total_capital = real_base * book_mid + real_quote
        if total_capital <= ZERO:
            return 0.0, "no capital"

        lvr_pct = float(self._lvr_since_reposition / total_capital * D(100))
        threshold = float(self.config.rebalance_lvr_threshold_pct)

        if lvr_pct >= threshold:
            return 1.0, (
                f"LVR {lvr_pct:.3f}% >= {threshold}% "
                f"({self._fills_since_reposition} fills)")
        return 0.0, (
            f"LVR {lvr_pct:.3f}% < {threshold}% "
            f"({self._fills_since_reposition} fills)")

    async def _conditional_rebalance(self, book_mid: Decimal):
        """After recenter, optionally place a market order to realign real
        balance with the virtual pool's balanced reserves."""
        fraction, reason = self._should_rebalance()
        # Reset LVR-since-reposition counter regardless — the decision is made
        self._lvr_since_reposition = ZERO
        self._fills_since_reposition = 0
        if fraction <= 0:
            self._last_rebalance_action = f"skip ({reason})"
            self._pair_logger.info(f"Rebalance skipped: {reason}")
            return

        real_base, real_quote = self.balance_gate.get_real_balances()
        # Target: pool's balanced reserves, scaled to real total capital
        pool_base_value = self.pool.initial_base * book_mid
        pool_quote_value = self.pool.initial_quote
        pool_total = pool_base_value + pool_quote_value
        if pool_total <= ZERO:
            return

        real_total = real_base * book_mid + real_quote
        scale = real_total / pool_total
        target_base = self.pool.initial_base * scale
        target_quote = self.pool.initial_quote * scale

        base_deficit = (target_base - real_base) * D(str(fraction))
        connector = self.connectors[self.config.exchange]
        pair = self.config.trading_pair

        max_slippage = self.config.rebalance_max_slippage_bps / D("10000")

        if base_deficit > ZERO:
            # Need to buy base
            buy_amount = self._quantize_amount(connector, pair, base_deficit)
            if buy_amount <= ZERO:
                return
            limit_price = self._quantize_price(connector, pair, book_mid * (D(1) + max_slippage))
            self._pair_logger.info(
                f"Rebalance BUY {buy_amount:.2f} base @ limit {limit_price:.6f} "
                f"({fraction:.0%}, {reason})"
            )
            self._last_rebalance_action = f"BUY {buy_amount:.0f} ({fraction:.0%}, {reason})"
            self.buy(self.config.exchange, pair, buy_amount, OrderType.LIMIT, limit_price)
        elif base_deficit < ZERO:
            # Need to sell base
            sell_amount = self._quantize_amount(connector, pair, abs(base_deficit))
            if sell_amount <= ZERO:
                return
            limit_price = self._quantize_price(connector, pair, book_mid * (D(1) - max_slippage))
            self._pair_logger.info(
                f"Rebalance SELL {sell_amount:.2f} base @ limit {limit_price:.6f} "
                f"({fraction:.0%}, {reason})"
            )
            self._last_rebalance_action = f"SELL {sell_amount:.0f} ({fraction:.0%}, {reason})"
            self.sell(self.config.exchange, pair, sell_amount, OrderType.LIMIT, limit_price)
        else:
            self._last_rebalance_action = f"balanced ({reason})"

    # ---- Futures hedge ----------------------------------------------------

    def _hedge_connector(self):
        if not self.config.enable_hedge:
            return None
        return self.connectors.get(self.config.hedge_exchange)

    def _hedge_book_mid(self) -> Optional[Decimal]:
        c = self._hedge_connector()
        if c is None:
            return None
        try:
            mid = c.get_mid_price(self.config.hedge_trading_pair)
            return mid if mid is not None and mid > ZERO else None
        except Exception:
            return None

    def _hedge_status(self) -> tuple:
        """Returns (ready, status_string). ready=False means hedge can't fire.
        Common reasons: connector not registered (config or workflow issue),
        connector still warming up, no order book data yet."""
        if not self.config.enable_hedge:
            return False, "disabled"
        c = self._hedge_connector()
        if c is None:
            return False, (
                f"connector '{self.config.hedge_exchange}' not loaded — "
                f"run `connect {self.config.hedge_exchange}` in Hummingbot "
                f"client BEFORE starting strategy")
        # Hummingbot connectors expose `ready` (dict) and `status_dict`
        is_ready = getattr(c, "ready", True)
        if isinstance(is_ready, dict):
            is_ready = all(is_ready.values())
        if not is_ready:
            return False, f"connector '{self.config.hedge_exchange}' warming up"
        if self._hedge_book_mid() is None:
            return False, f"no order book data for '{self.config.hedge_trading_pair}' yet"
        return True, "ready"

    def _warn_hedge_unavailable(self, reason: str):
        """Emit a warning at most once per 60 seconds about why hedge can't fire."""
        now = time.time()
        if now - self._hedge_unavailable_warned_ts < 60.0:
            return
        self._hedge_unavailable_warned_ts = now
        self._pair_logger.warning(f"Hedge unavailable: {reason}")

    def _hedge_inventory_skew_threshold(self) -> Decimal:
        """Inventory skew threshold for hedge trigger, derived analytically
        from concentration. Using max_skew(c, δ) ≈ δ/c (see docs §1), the
        threshold = activation_fraction × δ/c. Tight ranges → high threshold
        (more skew naturally reachable), loose ranges → low threshold.
        Clamped to [0.03, 0.50] for safety.
        """
        cfg = self.config
        if self.pool is None:
            return D("0.20")
        c = float(self.pool.concentration_pct) / 100.0
        delta = float(cfg.soft_recenter_drift_pct) / 100.0
        if c <= 0:
            return D("0.20")
        natural_max = delta / c
        activation = float(cfg.hedge_inventory_skew_activation_fraction)
        threshold = natural_max * activation
        return D(str(max(0.03, min(0.50, threshold))))

    def _hedge_toxicity_threshold(self) -> Decimal:
        """Toxicity ratio threshold for hedge trigger, scaled linearly by
        current concentration. Threshold typically DECREASES as concentration
        widens — calm/tight regimes need strong consensus to filter noise,
        volatile/loose regimes need a lower bar because each fill carries
        more inventory impact.
        """
        cfg = self.config
        cur = self.pool.concentration_pct if self.pool is not None else cfg.min_concentration_pct
        min_c = cfg.min_concentration_pct
        max_c = cfg.max_concentration_pct
        thresh_at_min = cfg.hedge_toxicity_threshold_at_min_conc
        thresh_at_max = cfg.hedge_toxicity_threshold_at_max_conc
        if max_c <= min_c:
            return thresh_at_min
        span = (cur - min_c) / (max_c - min_c)
        if span < ZERO:
            span = ZERO
        elif span > D(1):
            span = D(1)
        return thresh_at_min + span * (thresh_at_max - thresh_at_min)

    def _check_and_update_hedge(self):
        """Open/close a futures hedge based on toxicity + inventory accumulation.
        Mirrors the backtest mock but places real market orders on the hedge
        connector. Idempotent — safe to call every tick."""
        if not self.config.enable_hedge:
            return

        # Visible readiness check — without this, the gate silently does
        # nothing when the connector is missing/warming, and operators have
        # no idea the hedge subsystem isn't running.
        ready, status = self._hedge_status()
        if not ready:
            self._warn_hedge_unavailable(status)
            return

        c = self._hedge_connector()  # already verified non-None by _hedge_status

        # Auto-release a stale in-flight latch (e.g., if a fail event was
        # missed). Without this, one rejected order disables hedging forever.
        now_ts = time.time()
        if (self._hedge_in_flight and self._hedge_in_flight_ts > 0
                and now_ts - self._hedge_in_flight_ts
                    > self.config.hedge_in_flight_timeout_sec):
            self._pair_logger.warning(
                f"Hedge in-flight latch timed out after "
                f"{self.config.hedge_in_flight_timeout_sec}s — releasing")
            self._hedge_in_flight = False
            self._hedge_in_flight_ts = 0.0
            self._hedge_order_ids.clear()
        if self._hedge_in_flight:
            return

        # Set leverage once after connector is ready
        if not self._hedge_leverage_set:
            try:
                c.set_leverage(self.config.hedge_trading_pair,
                               self.config.hedge_leverage)
                self._hedge_leverage_set = True
                self._pair_logger.info(
                    f"Hedge leverage set to {self.config.hedge_leverage}x on "
                    f"{self.config.hedge_trading_pair}")
            except Exception as e:
                self._pair_logger.warning(f"set_leverage failed (will retry): {e}")
                # don't block hedging if exchange already has the right leverage
                self._hedge_leverage_set = True

        # Debounce: don't toggle open/close too rapidly
        if (self._hedge_last_state_change_ts > 0
                and now_ts - self._hedge_last_state_change_ts
                    < self.config.hedge_min_state_change_interval_sec):
            return

        hedge_mid = self._hedge_book_mid()
        if hedge_mid is None:
            return

        toxicity = self._get_toxicity_profile()
        inventory = self._inventory_adjustments()
        inv_side = inventory["accumulation_side"]

        # Concentration-scaled gates (see docs §1, §8): both must pass.
        skew = inventory["skew"]
        abs_skew = abs(skew)
        skew_thresh = self._hedge_inventory_skew_threshold()
        tox_thresh = self._hedge_toxicity_threshold()

        # No open hedge: maybe open one
        if self._hedge_book.size <= ZERO:
            spot_mid = self._get_book_mid()
            if spot_mid is None or spot_mid <= ZERO:
                return  # cannot size a hedge against an unknown spot
            real_base, real_quote = self.balance_gate.get_real_balances()

            # Spot SHORT (inv_side="buy" = skew<0) → LONG hedge to protect
            # against price rising while we still owe base.
            # Toxicity gate: SELL ratio high = our asks getting hit = market
            # is buying aggressively from us, which is exactly the flow that
            # drove us short. High sell_ratio → expect price to keep rising.
            if (inv_side == "buy" and abs_skew >= skew_thresh
                    and toxicity["sell_ratio"] >= tox_thresh):
                # Size against the value of remaining base inventory
                size_quote = real_base * spot_mid * self.config.hedge_size_cap_pct
                if size_quote >= self.config.hedge_min_notional_quote:
                    size = self._quantize_amount(c, self.config.hedge_trading_pair,
                                                 size_quote / hedge_mid)
                    if size > ZERO:
                        self._pair_logger.info(
                            f"Hedge gate met: skew={skew:+.3f} "
                            f"(thresh ±{float(skew_thresh):.3f}), "
                            f"sell_ratio={float(toxicity['sell_ratio']):.2f} "
                            f"(thresh ≥{float(tox_thresh):.2f})")
                        self._open_hedge("long", size)
                return

            # Spot LONG (inv_side="sell" = skew>0) → SHORT hedge to protect
            # against price dropping while we hold extra base.
            # Toxicity gate: BUY ratio high = our bids getting hit = market
            # is selling aggressively to us, which drove us long. High
            # buy_ratio → expect price to keep dropping.
            if (inv_side == "sell" and abs_skew >= skew_thresh
                    and toxicity["buy_ratio"] >= tox_thresh):
                size_quote = real_quote * self.config.hedge_size_cap_pct
                if size_quote >= self.config.hedge_min_notional_quote:
                    size = self._quantize_amount(c, self.config.hedge_trading_pair,
                                                 size_quote / hedge_mid)
                    if size > ZERO:
                        self._pair_logger.info(
                            f"Hedge gate met: skew={skew:+.3f} "
                            f"(thresh ±{float(skew_thresh):.3f}), "
                            f"buy_ratio={float(toxicity['buy_ratio']):.2f} "
                            f"(thresh ≥{float(tox_thresh):.2f})")
                        self._open_hedge("short", size)
                return
            return

        # Open hedge exists: maybe close it.
        # Close when: skew falls below half the open threshold (hysteresis to
        # avoid oscillation), OR the spot side flips from what we hedged.
        close_skew_thresh = skew_thresh / D(2)
        should_close = (
            abs_skew < close_skew_thresh
            or (self._hedge_book.side == "long" and inv_side != "buy")
            or (self._hedge_book.side == "short" and inv_side != "sell")
        )
        if should_close:
            self._close_hedge()

    def _place_hedge_market(self, direction: str, size: Decimal,
                            position_action: PositionAction, label: str):
        """Common entry point for opening/closing a hedge market order.
        Sets the in-flight latch + timestamp; routes to buy/sell."""
        pair = self.config.hedge_trading_pair
        try:
            self._hedge_in_flight = True
            self._hedge_in_flight_ts = time.time()
            if direction == "short":
                oid = self.sell(self.config.hedge_exchange, pair, size,
                                OrderType.MARKET, Decimal("0"),
                                position_action=position_action)
            else:
                oid = self.buy(self.config.hedge_exchange, pair, size,
                               OrderType.MARKET, Decimal("0"),
                               position_action=position_action)
            if oid:
                self._hedge_order_ids.add(oid)
                self._hedge_last_state_change_ts = time.time()
            else:
                # No order id returned — release latch immediately
                self._hedge_in_flight = False
                self._hedge_in_flight_ts = 0.0
            self._pair_logger.info(
                f"Hedge {label} {direction.upper()} {size} {pair} (market, "
                f"{position_action.value}) order_id={oid}")
        except Exception as e:
            self._pair_logger.error(
                f"Failed to {label.lower()} hedge: {e}", exc_info=True)
            self._hedge_in_flight = False
            self._hedge_in_flight_ts = 0.0

    def _open_hedge(self, direction: str, size: Decimal):
        self._place_hedge_market(direction, size, PositionAction.OPEN, "OPEN")

    def _close_hedge(self):
        if self._hedge_book.size <= ZERO or self._hedge_book.side is None:
            return
        c = self._hedge_connector()
        pair = self.config.hedge_trading_pair
        size = self._quantize_amount(c, pair, self._hedge_book.size)
        if size <= ZERO:
            return  # rounded to zero — leave residual for next tick
        # To close: opposite side market order
        close_dir = "long" if self._hedge_book.side == "short" else "short"
        self._place_hedge_market(
            close_dir, size, PositionAction.CLOSE,
            f"CLOSE-{self._hedge_book.side.upper()}")

    def _process_hedge_fill(self, event: OrderFilledEvent):
        """Apply a hedge fill to the hedge avg-cost book. Latch released last
        so concurrent ticks can't fire a duplicate before state is committed."""
        amount = self._to_decimal(event.amount)
        price = self._to_decimal(event.price)
        fee = self._fee_quote_equivalent(event)
        side = "long" if event.trade_type == TradeType.BUY else "short"
        self._hedge_book.apply_fill(side, amount, price, fee)
        self._hedge_fills += 1
        # Release latch only after all state mutations are committed
        self._hedge_in_flight = False
        self._hedge_in_flight_ts = 0.0
        self._hedge_order_ids.discard(getattr(event, "order_id", None))

    # ---- Fill handling ----------------------------------------------------

    def did_fill_order(self, event: OrderFilledEvent):
        # Route hedge fills to the hedge tracker; they don't touch spot P&L.
        is_hedge = (
            self.config.enable_hedge
            and (
                getattr(event, "trading_pair", None) == self.config.hedge_trading_pair
                or getattr(event, "order_id", None) in self._hedge_order_ids
            )
        )
        if is_hedge:
            self._hedge_order_ids.discard(getattr(event, "order_id", None))
            self._process_hedge_fill(event)
            self._save_state()
            self._pair_logger.info(
                f"Hedge FILL {event.trade_type.name} {event.amount} @ {event.price} | "
                f"hedge pos: {self._hedge_book.side or 'flat'} {self._hedge_book.size} @ "
                f"{self._hedge_book.avg_cost:.6f} | hedge realized: {self._hedge_book.realized_pnl:+.4f}"
            )
            return

        amount = self._to_decimal(event.amount)
        price = self._to_decimal(event.price)
        fee_quote = self._fee_quote_equivalent(event)

        if event.trade_type == TradeType.SELL:
            self._base_flow -= amount
            self._quote_flow += price * amount
        else:
            self._base_flow += amount
            self._quote_flow -= price * amount

        # Realized/unrealized P&L tracking via FIFO lots
        self._process_fill_for_pnl(event.trade_type, amount, price, fee_quote)

        self.pool.update_on_fill(event.trade_type, amount)

        self._queue_flair_markout(event)
        self._queue_adverse_markout(event)
        self._fills_since_reposition += 1
        now_fill = time.time()
        self._last_fill_ts = now_fill
        self._toxicity_recent_fills.append((now_fill, event.trade_type))
        if self.config.enable_fill_velocity_detector or self.config.enable_momentum_spread:
            self._recent_fills.append((time.time(), event.trade_type))

        self._save_state()

        pnl = self._get_pnl_breakdown()
        spread_str = f"{pnl['spread_capture']:+.4f}"
        fees_str = f"{pnl['fees']:.4f}"
        lvr_str = f"{pnl['lvr_estimate']:.4f}"
        hedge_str = f"{pnl['hedge_pnl']:+.4f}"
        net_str = f"{pnl['total_net']:+.4f}"
        pos_str = (f"{pnl['spot_position']:+.2f}@{pnl['spot_avg_cost']:.6f}"
                   if pnl['spot_position'] != ZERO else "flat")

        pool_price = self.pool.get_pool_price()
        pp_str = f"{pool_price:.6f}" if pool_price else "n/a"

        # Zone label: compare fill distance from mid to inner spread threshold
        w = self.config.pool_price_weight
        pool_mid = self.pool.get_mid_price()
        blended_mid = (D(1) - w) * self.pool.anchor_price + w * pool_mid
        inner_spread = self.config.base_spread_bps / D("10000")
        fill_dist = abs(price - blended_mid) / blended_mid if blended_mid > ZERO else ZERO
        zone = "outer" if fill_dist > inner_spread * D("1.5") else "inner"

        toxicity = self._get_toxicity_profile()
        inventory = self._inventory_adjustments()
        flair = self._flair_summary()

        def _st(state):
            if state >= D(2): return "hard"
            if state >= D(1): return "soft"
            return "off"

        inv_side = inventory["accumulation_side"] or "none"
        msg = (
            f"CL-AMM {event.trade_type.name} {amount:.4f} @ {price:.6f} [{zone}] | "
            f"P&L: spread={spread_str} - fees={fees_str} + hedge={hedge_str} "
            f"= net={net_str} | "
            f"pos: {pos_str} | pool: {pp_str} | "
            f"range: [{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}] ({self.pool.concentration_pct:.1f}%) | "
            f"inv: {inventory['skew']:+.2f} {_st(inventory['state'])} {inv_side} | "
            f"tox: buy={_st(toxicity['buy_state'])}({toxicity['buy_ratio']:.2f}) "
            f"sell={_st(toxicity['sell_state'])}({toxicity['sell_ratio']:.2f}) | "
            f"FLAIR(ref): lvr~{lvr_str} roll_net={flair['roll_net']:+.4f} | "
            f"adv: {self._adverse_summary_str()}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        safe_ensure_future(self._refresh_after_fill())

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """Release the hedge in-flight latch if the failed order was a hedge order."""
        oid = getattr(event, "order_id", None)
        if oid in self._hedge_order_ids:
            self._hedge_order_ids.discard(oid)
            self._hedge_in_flight = False
            self._hedge_in_flight_ts = 0.0
            self._pair_logger.warning(
                f"Hedge order failed (order_id={oid}) — latch released")

    def did_cancel_order(self, event: OrderCancelledEvent):
        """Release the hedge in-flight latch if a hedge order was cancelled."""
        oid = getattr(event, "order_id", None)
        if oid in self._hedge_order_ids:
            self._hedge_order_ids.discard(oid)
            self._hedge_in_flight = False
            self._hedge_in_flight_ts = 0.0
            self._pair_logger.warning(
                f"Hedge order cancelled (order_id={oid}) — latch released")

    async def _refresh_after_fill(self):
        if self._refreshing:
            return
        self._refreshing = True
        try:
            connector = self.connectors[self.config.exchange]

            book_mid = self._get_book_mid()
            if book_mid:
                self._update_anchor_ema(book_mid)
                self._update_flair_monitor(book_mid)

            # Range check after fill
            if book_mid and not self.pool.is_in_range(book_mid):
                self._pair_logger.info(f"Recentering after fill: {book_mid:.6f}")
                self.pool.recenter(book_mid)
                self._last_reposition_price = book_mid
                self._save_state()
                await self._conditional_rebalance(book_mid)

            results = await connector.cancel_all(timeout_seconds=10.0)
            if results and any(not r.success for r in results):
                self._pair_logger.warning("cancel_all had failures — skipping requote.")
                return

            w = self.config.pool_price_weight
            pool_mid = self.pool.get_mid_price()
            mid = (D(1) - w) * self.pool.anchor_price + w * pool_mid
            RateOracle.get_instance().set_price(self.config.trading_pair, mid)
            orders = self._generate_orders(mid)
            orders = self.balance_gate.scale_orders(orders)
            order_ids = self._place_orders(orders)
            self._refresh_timestamp = self.current_timestamp + self.config.order_refresh_time

            await self._await_order_confirmations(order_ids)
        except Exception:
            self._pair_logger.error("Error in refresh after fill.", exc_info=True)
        finally:
            self._refreshing = False

    # ---- P&L --------------------------------------------------------------

    def _process_fill_for_pnl(self, side: TradeType, size: Decimal,
                              price: Decimal, fee_quote: Decimal):
        """Apply a spot fill to the spot avg-cost book."""
        side_str = "long" if side == TradeType.BUY else "short"
        self._spot_book.apply_fill(side_str, size, price, fee_quote)

    def _get_pnl_breakdown(self) -> Dict[str, Optional[Decimal]]:
        """Cash decomposition of total P&L (inventory mark excluded).

            total_net = spread_capture - fees + hedge_pnl

        spread_capture : realized P&L from closed spot round-trips
        fees           : actual spot exchange fees paid (cumulative)
        hedge_pnl      : realized + unrealized of futures hedge book,
                         net of hedge fees

        inventory_pnl is also returned for visibility, but it is NOT in
        total_net — open spot inventory mark-to-market is treated as
        unrealized exposure to ignore until the position is closed.

        lvr_estimate is an INDEPENDENT microstructure metric from FLAIR
        (oracle-markout adverse selection). Not a component of total_net.
        """
        sb, hb = self._spot_book, self._hedge_book

        book_mid = self._get_book_mid()
        inventory_pnl = sb.unrealized(book_mid)  # informational only
        spot_pos = sb.signed_position()

        hedge_mid = self._hedge_book_mid()
        hedge_unreal = hb.unrealized(hedge_mid)
        hedge_pos = hb.signed_position()
        hedge_net = hb.realized_pnl - hb.fees
        if hedge_unreal is not None:
            hedge_net += hedge_unreal

        total_net = sb.realized_pnl - sb.fees + hedge_net

        return {
            "spread_capture": sb.realized_pnl,
            "fees": sb.fees,
            "inventory_pnl": inventory_pnl,
            "hedge_pnl": hedge_net,
            "total_net": total_net,
            "lvr_estimate": self._flair_lifetime_lvr,
            "spot_position": spot_pos,
            "spot_avg_cost": sb.avg_cost,
            "hedge_realized": hb.realized_pnl,
            "hedge_unrealized": hedge_unreal,
            "hedge_fees": hb.fees,
            "hedge_position": hedge_pos,
            "hedge_avg_cost": hb.avg_cost,
        }

    def _get_pnl(self) -> Optional[Decimal]:
        """Backwards-compat shim for total net P&L."""
        return self._get_pnl_breakdown()["total_net"]

    # ---- Market data helpers ----------------------------------------------

    def _get_book_mid(self) -> Optional[Decimal]:
        try:
            mid = self.connectors[self.config.exchange].get_mid_price(self.config.trading_pair)
            return mid if mid is not None and mid > ZERO else None
        except Exception:
            return None

    def _get_anchor_drift_pct(self) -> Decimal:
        """How far the anchor has drifted from the range center, in %."""
        range_center = (self.pool.p_lower + self.pool.p_upper) / D(2)
        if range_center <= ZERO:
            return ZERO
        return abs(self.pool.anchor_price - range_center) / range_center * D(100)

    def _format_pipeline_status(self) -> str:
        """One-line summary of the DynamicRangeController decision pipeline."""
        d = self._range_controller.last_decision
        if d is None:
            return "    Pipeline: not yet computed"
        return (
            f"    Pipeline: natr_mult={d.natr_mult:.2f} → adj={d.natr_adjusted_pct:.1f}% | "
            f"adx_norm={d.adx_normalized:.2f} × persist={d.persistence:+.2f} → eff_trend={d.effective_trend:.3f} | "
            f"hmm={d.hmm_override} | raw={d.raw_pct:.1f}% → smooth={d.smoothed_pct:.1f}%"
        )

    # ---- Pool auto-scaling ------------------------------------------------

    def _init_pool_from_balance(self) -> bool:
        real_base, real_quote = self.balance_gate.get_real_balances()
        if real_base <= ZERO or real_quote <= ZERO:
            return False
        mid = self._get_book_mid()
        if mid is None or mid <= ZERO:
            mid = self.config.initial_price
        pool_depth = real_base * mid + real_quote
        self.pool = ConcentratedPool(mid, pool_depth, self.config.base_concentration_pct)
        self._pool_scaled = True
        self._save_state()
        self._pair_logger.info(
            f"Initialized CL pool: {self.config.trading_pair} @ {mid} "
            f"depth={pool_depth:.0f} range=+-{self.config.base_concentration_pct}%"
        )
        return True

    def _auto_scale_pool(self):
        self._pool_scaled = True
        real_base, real_quote = self.balance_gate.get_real_balances()
        if real_base <= ZERO or real_quote <= ZERO:
            return
        book_mid = self._get_book_mid()
        if book_mid:
            self.pool.anchor_price = book_mid
        mid = self.pool.get_mid_price()
        if mid is None or mid <= ZERO:
            return

        real_total = real_base * mid + real_quote
        # Estimate pool total from L and current range
        sqrt_p = _sqrt(mid)
        sqrt_pa = _sqrt(self.pool.p_lower)
        sqrt_pb = _sqrt(self.pool.p_upper)
        pool_quote = self.pool.L * (sqrt_p - sqrt_pa)
        if pool_quote <= ZERO:
            return

        scale = real_total / (pool_quote * D(2))  # approximate: base_val + quote ~= 2*quote at mid
        if abs(scale - D(1)) < D("0.05"):
            return

        old_L = self.pool.L
        self.pool.L *= scale
        # Re-derive reserves from scaled L
        self.pool.base = self.pool.L * (D(1) / sqrt_p - D(1) / sqrt_pb)
        self.pool.quote = self.pool.L * (sqrt_p - sqrt_pa)
        self.pool.initial_base = self.pool.base
        self.pool.initial_quote = self.pool.quote
        self._base_flow *= scale
        self._quote_flow *= scale
        self._save_state()
        self._pair_logger.info(f"Auto-scaled CL pool: L {old_L:.0f} -> {self.pool.L:.0f} ({scale:.2f}x)")

    # ---- State persistence ------------------------------------------------

    def _save_state(self):
        state = self.pool.to_state()
        state["base_flow"] = str(self._base_flow)
        state["quote_flow"] = str(self._quote_flow)
        state.update(self._spot_book.to_dict(prefix="spot_"))
        state.update(self._hedge_book.to_dict(prefix="hedge_"))
        state["hedge_fills"] = self._hedge_fills
        with open(self._state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> dict:
        if not os.path.exists(self._state_file):
            return {}
        try:
            with open(self._state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, IOError) as e:
            self._pair_logger.warning(f"Failed to load state from {self._state_file}: {e}")
            return {}

    # ---- Status display ---------------------------------------------------

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        w = self.config.pool_price_weight
        pool_mid = self.pool.get_mid_price()
        blended_mid = (D(1) - w) * self.pool.anchor_price + w * pool_mid
        book_mid = self._get_book_mid()
        anchor_str = f"{self.pool.anchor_price:.6f}"
        book_str = f"{book_mid:.6f}" if book_mid else "n/a"
        pool_str = f"{pool_mid:.6f}" if pool_mid else "n/a"

        base_pct = self.pool.base / self.pool.initial_base * D(100) if self.pool.initial_base > ZERO else ZERO
        quote_pct = self.pool.quote / self.pool.initial_quote * D(100) if self.pool.initial_quote > ZERO else ZERO
        skew = self.pool.get_inventory_skew()

        real_base, real_quote = self.balance_gate.get_real_balances()
        base_token, quote_token = self.config.trading_pair.split("-")
        total_capital = real_base * blended_mid + real_quote

        pnl = self._get_pnl_breakdown()
        pnl_spread_str = f"{pnl['spread_capture']:+.4f}"
        pnl_fees_str = f"{pnl['fees']:.4f}"
        pnl_lvr_str = f"{pnl['lvr_estimate']:.4f}"
        pnl_hedge_str = f"{pnl['hedge_pnl']:+.4f}"
        pnl_net_str = f"{pnl['total_net']:+.4f}"
        pnl_pos_str = (f"{pnl['spot_position']:+.2f} @ {pnl['spot_avg_cost']:.6f}"
                       if pnl['spot_position'] != ZERO else "flat")
        hedge_pos_str = (
            f"{pnl['hedge_position']:+.2f} @ {pnl['hedge_avg_cost']:.6f}"
            if pnl['hedge_position'] != ZERO else "flat")

        max_safe_str = f"{self._last_max_safe:.1f}" if self._last_max_safe > ZERO else "n/a"
        order_base = self._last_max_safe * self.config.order_safe_ratio if self._last_max_safe > ZERO else ZERO
        order_value = order_base * blended_mid

        # Indicator status
        natr_str = f"{float(self._last_natr) * 100:.3f}%" if self._last_natr is not None else "warming up"
        adx_str = f"{self._last_adx:.1f}" if self._last_adx is not None else "warming up"
        hurst_str = f"{self._last_hurst:.3f}" if self._last_hurst is not None else "warming up"
        if self._last_hmm is not None:
            top = max(self._last_hmm, key=self._last_hmm.get)
            hmm_str = f"{top} ({self._last_hmm[top]:.0%})"
        else:
            hmm_str = "warming up" if HMM_AVAILABLE else "unavailable"

        # Candle warmup progress
        connector = self.connectors[self.config.exchange]
        candle_builder = getattr(connector, "candle_builder", None)
        candle_count = candle_builder.candle_count if candle_builder else 0

        momentum_bid_extra, momentum_ask_extra = self._momentum_spread_adjustment()
        toxicity = self._get_toxicity_profile()
        inventory = self._inventory_adjustments()
        flair = self._flair_summary()
        now = time.time()
        m_window = self.config.momentum_window_sec
        m_buys = sum(1 for ts, t in self._recent_fills if t == TradeType.BUY and now - ts <= m_window)
        m_sells = sum(1 for ts, t in self._recent_fills if t == TradeType.SELL and now - ts <= m_window)
        m_net = m_sells - m_buys
        if m_net > 0:
            momentum_str = f"+{m_net} sells -> ASK +{momentum_ask_extra * D('10000'):.0f}bps"
        elif m_net < 0:
            momentum_str = f"+{abs(m_net)} buys -> BID +{momentum_bid_extra * D('10000'):.0f}bps"
        else:
            momentum_str = "neutral"

        def _toxicity_label(state: Decimal) -> str:
            if state >= D(2):
                return "hard"
            if state >= D(1):
                return "soft"
            return "off"

        tox_buy = _toxicity_label(toxicity["buy_state"])
        tox_sell = _toxicity_label(toxicity["sell_state"])
        inv_state = _toxicity_label(inventory["state"])
        inv_side = inventory["accumulation_side"] or "none"

        lines = [
            f"  [CL-AMM {self.config.trading_pair}] Mid: {blended_mid:.6f} (w={w})",
            f"  Anchor: {anchor_str} | Book: {book_str} | Pool: {pool_str}",
            f"  Range: [{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}] "
            f"(+-{self.pool.concentration_pct:.2f}%) | L: {self.pool.L:.0f}"
            f" | drift: {self._get_anchor_drift_pct():.2f}%"
            f" (soft@{self.config.soft_recenter_drift_pct}%)",
            f"  Zones: inner {(D(1)-self.config.outer_capital_fraction)*100:.0f}% "
            f"({self.config.base_spread_bps:.0f}bps) | "
            f"outer {self.config.outer_capital_fraction*100:.0f}% "
            f"({self.config.base_spread_bps*self.config.outer_spread_mult:.0f}bps) | "
            f"trigger: {self._outer_trigger_pct()*100:.1f}% from center",
            f"  Adverse selection (avg move post-fill): {self._adverse_status_lines()}",
            f"  VPool: {base_pct:.1f}%B / {quote_pct:.1f}%Q | Skew: {skew:+.3f}",
            f"  Real: {real_base:.2f} {base_token} / {real_quote:.2f} {quote_token}"
            f" | Capital: {total_capital:.1f} {quote_token}",
            f"  Order: {self.config.order_safe_ratio * 100:.0f}% of max_safe"
            f" = {order_base:.1f} {base_token} ({order_value:.1f} {quote_token})"
            f" | max_safe: {max_safe_str} {base_token}",
            f"  Indicators ({candle_count} candles):",
            f"    NATR: {natr_str} | ADX: {adx_str} | Hurst: {hurst_str} | HMM: {hmm_str}",
            self._format_pipeline_status(),
            f"  Momentum ({m_window}s): {momentum_str}",
            f"  Toxicity ({self.config.toxicity_window_sec}s, n={int(toxicity['sample_count'])}): "
            f"buy={tox_buy} ({toxicity['buy_ratio']:.2f}) | sell={tox_sell} ({toxicity['sell_ratio']:.2f})",
            f"  Inventory guard: state={inv_state} | accumulation_side={inv_side} | skew={inventory['skew']:+.3f}",
            f"  FLAIR ({self.config.flair_window_sec}s, n={int(flair['samples'])}, pending={int(flair['pending'])}): "
            f"fee={flair['roll_fee']:.4f} | LVR={flair['roll_lvr']:.4f} | net={flair['roll_net']:+.4f} | ratio={flair['roll_ratio']:.2f}x",
            f"  FLAIR lifetime: fee={flair['lifetime_fee']:.4f} | LVR={flair['lifetime_lvr']:.4f} | "
            f"net={flair['lifetime_net']:+.4f} | ratio={flair['lifetime_ratio']:.2f}x",
            f"  Rebalance: {'enabled' if self.config.enable_rebalance else 'disabled'}"
            f"{' | last: ' + self._last_rebalance_action if self._last_rebalance_action else ''}",
            f"  P&L: net={pnl_net_str} {quote_token} | spread={pnl_spread_str} "
            f"- fees={pnl_fees_str} + hedge={pnl_hedge_str} "
            f"(limit -{self.config.max_cumulative_loss})",
            f"  LVR estimate (FLAIR ref, not in net): {pnl_lvr_str}",
            f"  Spot:  {pnl_pos_str}  |  Hedge: {hedge_pos_str}"
            f"{' (' + self.config.hedge_trading_pair + ' on ' + self.config.hedge_exchange + ')' if self.config.enable_hedge else ' (disabled)'}",
        ]
        if self.config.enable_hedge:
            ready, status = self._hedge_status()
            ready_mark = "✓" if ready else "✗"
            lines.append(f"  Hedge subsystem: {ready_mark} {status}")

        if self._stopped:
            lines.append("  *** STOPPED - loss limit exceeded ***")

        active = self.get_active_orders(connector_name=self.config.exchange)
        if active:
            lines.append(f"  Orders ({len(active)}):")
            for o in sorted(active, key=lambda x: x.price, reverse=True):
                side = "ASK" if not o.is_buy else "BID"
                lines.append(f"    {side} {o.quantity:.4f} @ {o.price:.6f}")
        else:
            lines.append("  Orders: none")

        return "\n".join(lines)
