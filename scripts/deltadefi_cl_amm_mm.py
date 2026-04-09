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
from hummingbot.core.data_type.common import OrderType, TradeType
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

    # Soft recenter: re-anchor range when anchor drifts from range center
    soft_recenter_drift_pct: Decimal = Field(D("2.0"))

    # Conditional rebalance after recenter
    enable_rebalance: bool = Field(True)
    rebalance_max_slippage_bps: Decimal = Field(D("30"))
    rebalance_partial_fraction: Decimal = Field(D("0.5"))

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
        if self.toxicity_activation_count <= 0:
            raise ValueError("toxicity_activation_count must be > 0")
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

    def _build_observations(self, candles) -> Optional[np.ndarray]:
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
            obs.append([ret, abs_ret, roll_vol])
        return np.array(obs) if obs else None

    def _fit(self, candles):
        if not self._available:
            return
        obs = self._build_observations(candles[-self.window:])
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

    def predict(self, candles) -> Optional[Dict[str, float]]:
        if not self._available:
            return None
        if len(candles) < self.min_candles:
            return None

        now = time.time()
        if self._model is None or (now - self._last_fit_time) >= self.refit_interval_sec:
            self._fit(candles)

        if self._model is None or self._label_map is None:
            return None

        obs = self._build_observations(candles[-20:])
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
        cls.markets = {config.exchange: {config.trading_pair}}

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
        self._last_max_safe: Decimal = ZERO
        self._rebalancing: bool = False
        self._last_rebalance_action: str = ""
        self._indicator_log_counter: int = 0

    # ---- Main loop --------------------------------------------------------

    def on_tick(self):
        if self._stopped:
            return

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

        # Hard recenter: book mid exits bounds
        if book_mid and not self.pool.is_in_range(book_mid):
            self._pair_logger.info(
                f"Hard recenter: book_mid {book_mid:.6f} outside "
                f"[{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}]"
            )
            self.pool.recenter(book_mid)
            self._save_state()
            safe_ensure_future(self._conditional_rebalance(book_mid))

        # Soft recenter: anchor drifted from range center (price still in range)
        elif book_mid and self.config.soft_recenter_drift_pct > ZERO:
            range_center = (self.pool.p_lower + self.pool.p_upper) / D(2)
            if range_center > ZERO:
                drift_pct = abs(self.pool.anchor_price - range_center) / range_center * D(100)
                if drift_pct >= self.config.soft_recenter_drift_pct:
                    self._pair_logger.info(
                        f"Soft recenter: anchor {self.pool.anchor_price:.6f} drifted "
                        f"{drift_pct:.2f}% from range center {range_center:.6f}"
                    )
                    self.pool.recenter(self.pool.anchor_price)
                    self._save_state()
                    safe_ensure_future(self._conditional_rebalance(self.pool.anchor_price))

        # Dynamic range update from indicators
        self._update_dynamic_range(connector)

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
        self._last_hmm = self._hmm_detector.predict(candles)

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
            return orders  # empty

        scale_factor = float(self.config.trend_order_scale_factor)
        trend_scale = max(0.0, 1.0 - scale_factor * eff_trend)
        if trend_scale <= 0.0:
            return orders  # empty

        max_safe_base = self._max_safe_order_base()
        order_base = max_safe_base * self.config.order_safe_ratio * D(str(round(trend_scale, 6)))
        order_value = order_base * mid_price

        weights = [self.config.size_decay ** i for i in range(self.config.num_levels)]
        total_weight = sum(weights)
        if total_weight <= ZERO:
            return orders

        connector = self.connectors[self.config.exchange]
        pair = self.config.trading_pair

        toxicity = self._get_toxicity_profile()
        inventory = self._inventory_adjustments()

        momentum_bid_extra, momentum_ask_extra = self._momentum_spread_adjustment()

        for i in range(self.config.num_levels):
            base_spread = self.config.base_spread_bps * (self.config.spread_multiplier ** i) / D("10000")
            w = weights[i] / total_weight
            layer_value = order_value * w

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

            ask_size = layer_value / ask_price if ask_price > ZERO else ZERO
            bid_size = layer_value / bid_price if bid_price > ZERO else ZERO

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

            if ask_size > ZERO:
                orders.append(OrderProposal(TradeType.SELL, ask_price, ask_size))
            if bid_size > ZERO:
                orders.append(OrderProposal(TradeType.BUY, bid_price, bid_size))

        return orders

    def _asymmetric_spreads(self, base_spread: Decimal):
        skew = D(str(self.pool.get_inventory_skew()))
        sens = self.config.skew_sensitivity
        floor = self.config.min_spread_bps / D("10000")
        bid_spread = max(base_spread * (D(1) + skew * sens), floor)
        ask_spread = max(base_spread * (D(1) - skew * sens), floor)
        return bid_spread, ask_spread

    def _get_toxicity_profile(self) -> Dict[str, Decimal]:
        now = time.time()
        window = self.config.toxicity_window_sec
        while self._toxicity_recent_fills and now - self._toxicity_recent_fills[0][0] > window:
            self._toxicity_recent_fills.popleft()

        buy_count = sum(1 for _, t in self._toxicity_recent_fills if t == TradeType.BUY)
        sell_count = sum(1 for _, t in self._toxicity_recent_fills if t == TradeType.SELL)
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

    def _inventory_adjustments(self) -> Dict[str, object]:
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
        """Decide rebalance fraction based on HMM regime.

        Returns (fraction, reason):
          fraction=0   → skip rebalance
          fraction=0.5 → partial (uncertain regime)
          fraction=1.0 → full rebalance
        """
        if not self.config.enable_rebalance:
            return 0.0, "disabled"

        hmm = self._last_hmm
        threshold = float(self.config.hmm_confidence_threshold)

        if hmm is None:
            # HMM not warmed up — don't rebalance without regime confidence
            return 0.0, "HMM warming up"

        if hmm.get("ranging", 0) > threshold:
            return 1.0, f"HMM ranging {hmm['ranging']:.0%}"

        # Trending or uncertain — don't rebalance
        if hmm.get("trending", 0) > threshold:
            return 0.0, f"HMM trending {hmm['trending']:.0%}"

        return 0.0, "HMM uncertain"

    async def _conditional_rebalance(self, book_mid: Decimal):
        """After recenter, optionally place a market order to realign real
        balance with the virtual pool's balanced reserves."""
        fraction, reason = self._should_rebalance()
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

    # ---- Fill handling ----------------------------------------------------

    def did_fill_order(self, event: OrderFilledEvent):
        if event.trade_type == TradeType.SELL:
            self._base_flow -= event.amount
            self._quote_flow += event.price * event.amount
        else:
            self._base_flow += event.amount
            self._quote_flow -= event.price * event.amount

        self.pool.update_on_fill(event.trade_type, event.amount)

        self._queue_flair_markout(event)
        self._toxicity_recent_fills.append((time.time(), event.trade_type))
        if self.config.enable_fill_velocity_detector or self.config.enable_momentum_spread:
            self._recent_fills.append((time.time(), event.trade_type))

        self._save_state()

        pnl = self._get_pnl()
        pnl_str = f"{pnl:+.4f}" if pnl is not None else "n/a"
        pool_price = self.pool.get_pool_price()
        pp_str = f"{pool_price:.6f}" if pool_price else "n/a"
        msg = (
            f"CL-AMM {event.trade_type.name} {event.amount:.4f} @ {event.price:.6f} | "
            f"P&L: {pnl_str} | pool: {pp_str} | range: [{self.pool.p_lower:.6f}, {self.pool.p_upper:.6f}]"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        safe_ensure_future(self._refresh_after_fill())

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

    def _get_pnl(self) -> Optional[Decimal]:
        book_mid = self._get_book_mid()
        if book_mid is None or book_mid <= ZERO:
            return None
        return self._base_flow * book_mid + self._quote_flow

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

        pnl = self._get_pnl()
        pnl_str = f"{pnl:+.4f}" if pnl is not None else "n/a"

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
            f"  P&L: {pnl_str} {quote_token} (limit: -{self.config.max_cumulative_loss})",
        ]

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
