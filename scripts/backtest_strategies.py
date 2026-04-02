"""Strategy adapters for the generic backtester.

Imports pool/indicator components from the live strategy files.
No hummingbot installation required — lightweight stubs are injected
for the hummingbot types that strategy scripts import at module level.

Usage:
    Imported by backtest_engine.py — not run directly.
"""

import math
import os
import sys
from decimal import Decimal
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Dict, List, NamedTuple, Optional

# ---------------------------------------------------------------------------
# Hummingbot stubs — the strategy scripts (deltadefi_cl_amm_mm.py etc.)
# import hummingbot types at module level, but the backtest only uses the
# pool/indicator classes which have zero runtime dependency on hummingbot.
# We inject minimal stubs so those imports succeed without the full package.
# ---------------------------------------------------------------------------


class _TradeType(Enum):
    BUY = 1
    SELL = 2
    RANGE = 3


class _OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    LIMIT_MAKER = 3
    AMM_SWAP = 4

    def is_limit_type(self):
        return self in (_OrderType.LIMIT, _OrderType.LIMIT_MAKER)


class _OrderFilledEvent(NamedTuple):
    timestamp: float = 0
    order_id: str = ""
    trading_pair: str = ""
    trade_type: _TradeType = _TradeType.BUY
    order_type: _OrderType = _OrderType.LIMIT
    price: Decimal = Decimal("0")
    amount: Decimal = Decimal("0")
    trade_fee: object = None


def _install_hummingbot_stubs():
    """Inject fake hummingbot modules into sys.modules."""
    if "hummingbot" in sys.modules:
        return  # real hummingbot available, skip stubs

    def _make_module(name, attrs=None):
        mod = ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    # hummingbot (root)
    _make_module("hummingbot")

    # hummingbot.core.data_type.common — TradeType, OrderType
    _make_module("hummingbot.core")
    _make_module("hummingbot.core.data_type")
    _make_module("hummingbot.core.data_type.common", {
        "TradeType": _TradeType,
        "OrderType": _OrderType,
    })

    # hummingbot.core.event.events — OrderFilledEvent
    _make_module("hummingbot.core.event")
    _make_module("hummingbot.core.event.events", {
        "OrderFilledEvent": _OrderFilledEvent,
    })

    # hummingbot.core.rate_oracle.rate_oracle — RateOracle
    _make_module("hummingbot.core.rate_oracle")
    _make_module("hummingbot.core.rate_oracle.rate_oracle", {
        "RateOracle": type("RateOracle", (), {}),
    })

    # hummingbot.core.utils.async_utils — safe_ensure_future
    _make_module("hummingbot.core.utils")
    _make_module("hummingbot.core.utils.async_utils", {
        "safe_ensure_future": lambda coro, *a, **kw: None,
    })

    # hummingbot.client.settings — DEFAULT_LOG_FILE_PATH
    _make_module("hummingbot.client")
    _make_module("hummingbot.client.settings", {
        "DEFAULT_LOG_FILE_PATH": "/tmp/hummingbot_logs/",
    })

    # hummingbot.connector.connector_base — ConnectorBase
    _make_module("hummingbot.connector")
    _make_module("hummingbot.connector.connector_base", {
        "ConnectorBase": type("ConnectorBase", (), {}),
    })

    # hummingbot.strategy.strategy_v2_base — StrategyV2Base, StrategyV2ConfigBase
    _make_module("hummingbot.strategy")
    _make_module("hummingbot.strategy.strategy_v2_base", {
        "StrategyV2Base": type("StrategyV2Base", (), {}),
        "StrategyV2ConfigBase": type("StrategyV2ConfigBase", (), {}),
    })


_install_hummingbot_stubs()
TradeType = _TradeType

# Ensure scripts dir is importable
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    from deltadefi_cl_amm_mm import (
        ConcentratedPool,
        NATRIndicator,
        ADXIndicator,
        HurstExponent,
        HMMRegimeDetector,
        DynamicRangeController,
        _sqrt,
    )
    from deltadefi_amm_mm import VirtualPool
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure strategy scripts are in the same directory as this file.")
    sys.exit(1)

from backtest_engine import BacktestStrategy, SimOrder, Candle

D = Decimal
ZERO = D("0")


# ---------------------------------------------------------------------------
# CL-AMM Backtest Strategy
# ---------------------------------------------------------------------------


class CLAMMBacktestStrategy(BacktestStrategy):
    """Concentrated liquidity AMM with NATR/ADX/Hurst/HMM dynamic range."""

    @property
    def name(self) -> str:
        return "CL-AMM"

    def __init__(self, **config):
        self.spread_bps = D(str(config.get("spread_bps", "40")))
        self.pool_price_weight = D(str(config.get("pool_price_weight", "0.70")))
        self.anchor_ema_alpha = D(str(config.get("anchor_ema_alpha", "0.05")))
        self.num_levels = int(config.get("num_levels", 1))
        self.size_decay = D(str(config.get("size_decay", "0.85")))
        self.spread_multiplier = D(str(config.get("spread_multiplier", "1.5")))
        self.order_safe_ratio = D(str(config.get("order_safe_ratio", "0.5")))
        self.pool_depth_override = config.get("pool_depth")
        self.enable_asymmetric_spread = config.get(
            "enable_asymmetric_spread", True)
        self.skew_sensitivity = D(str(config.get("skew_sensitivity", "0.5")))
        self.min_spread_bps = D(str(config.get("min_spread_bps", "20")))
        self.soft_recenter_drift_pct = D(str(
            config.get("soft_recenter_drift_pct", "2.0")))

        # Concentration
        self.base_concentration_pct = D(str(config.get("concentration", "5")))
        self.min_concentration_pct = D(str(
            config.get("min_concentration", "5")))
        self.max_concentration_pct = D(str(
            config.get("max_concentration", "20")))

        # Indicator params
        self.natr_period = int(config.get("natr_period", 14))
        self.natr_baseline = D(str(config.get("natr_baseline", "0.005")))
        self.natr_range_scale = D(str(config.get("natr_range_scale", "1.0")))
        self.adx_period = int(config.get("adx_period", 14))
        self.hurst_min_candles = int(config.get("hurst_min_candles", 100))
        self.hmm_n_states = int(config.get("hmm_n_states", 3))
        self.hmm_min_candles = int(config.get("hmm_min_candles", 200))
        self.hmm_refit_interval_sec = int(
            config.get("hmm_refit_interval_sec", 1800))
        self.hmm_window = int(config.get("hmm_window", 500))
        self.hmm_confidence_threshold = D(str(
            config.get("hmm_confidence_threshold", "0.80")))
        self.trend_sensitivity = D(str(config.get("trend_sensitivity", "0.5")))
        self.range_ema_alpha = D(str(config.get("range_ema_alpha", "0.1")))
        self.range_update_dead_band_pct = D(str(
            config.get("range_update_dead_band_pct", "0.5")))

        # Trend protection
        self.trend_order_scale_factor = D(str(config.get("trend_order_scale_factor", "0.0")))
        self.trend_halt_threshold = D(str(config.get("trend_halt_threshold", "0.0")))

        # State (set during initialize)
        self.pool: Optional[ConcentratedPool] = None
        self.base_balance = ZERO
        self.quote_balance = ZERO

        # Indicators (set during initialize)
        self._natr: Optional[NATRIndicator] = None
        self._adx: Optional[ADXIndicator] = None
        self._hurst: Optional[HurstExponent] = None
        self._hmm: Optional[HMMRegimeDetector] = None
        self._range_ctrl: Optional[DynamicRangeController] = None

        # Cached indicator values for snapshot
        self._last_natr: Optional[Decimal] = None
        self._last_adx: Optional[Decimal] = None
        self._last_hurst: Optional[Decimal] = None
        self._last_hmm: Optional[Dict[str, float]] = None
        self._recentered_this_candle = False
        self._range_changed_this_candle = False
        self._recenter_count = 0
        self._range_change_count = 0
        self._trend_halted_this_candle = False
        self._trend_order_scale = 1.0
        self._trend_halted_count = 0

    def initialize(self, initial_price: Decimal, base_balance: Decimal,
                   quote_balance: Decimal):
        self.base_balance = base_balance
        self.quote_balance = quote_balance

        pool_depth = self.pool_depth_override
        if pool_depth is None:
            pool_depth = base_balance * initial_price + quote_balance

        self.pool = ConcentratedPool(
            initial_price, pool_depth, self.base_concentration_pct)

        # Initialize indicators
        self._natr = NATRIndicator(self.natr_period)
        self._adx = ADXIndicator(self.adx_period)
        self._hurst = HurstExponent(
            self.hurst_min_candles, update_interval_sec=60)
        self._hmm = HMMRegimeDetector(
            self.hmm_n_states, self.hmm_min_candles,
            self.hmm_refit_interval_sec, self.hmm_window)

        # DynamicRangeController needs a config-like namespace
        range_config = SimpleNamespace(
            base_concentration_pct=self.base_concentration_pct,
            min_concentration_pct=self.min_concentration_pct,
            max_concentration_pct=self.max_concentration_pct,
            natr_baseline=self.natr_baseline,
            natr_range_scale=self.natr_range_scale,
            hmm_confidence_threshold=self.hmm_confidence_threshold,
            trend_sensitivity=self.trend_sensitivity,
            range_ema_alpha=self.range_ema_alpha,
        )
        self._range_ctrl = DynamicRangeController(range_config)

    def on_candle(self, candle: Candle, history: List[Candle],
                  sim_time: float) -> List[SimOrder]:
        self._recentered_this_candle = False
        self._range_changed_this_candle = False
        self._trend_halted_this_candle = False
        self._trend_order_scale = 1.0
        price = candle.close

        # Update anchor EMA
        self._update_anchor(price)

        # Hard recenter: price exits range bounds
        if not self.pool.is_in_range(price):
            self.pool.recenter(price)
            self._recentered_this_candle = True
            self._recenter_count += 1

        # Soft recenter: anchor drifted from range center
        elif self.soft_recenter_drift_pct > ZERO:
            range_center = (self.pool.p_lower + self.pool.p_upper) / D(2)
            if range_center > ZERO:
                drift_pct = (abs(self.pool.anchor_price - range_center)
                             / range_center * D(100))
                if drift_pct >= self.soft_recenter_drift_pct:
                    self.pool.recenter(self.pool.anchor_price)
                    self._recentered_this_candle = True
                    self._recenter_count += 1

        # Update indicators and dynamic range
        self._update_indicators(history)
        self._update_dynamic_range()

        # Compute effective mid: (1-w)*anchor + w*pool_mid
        w = self.pool_price_weight
        pool_mid = self.pool.get_mid_price()
        mid = (D(1) - w) * self.pool.anchor_price + w * pool_mid

        # Generate orders
        return self._generate_orders(mid)

    def on_fill(self, side: str, price: Decimal, amount: Decimal,
                sim_time: float):
        trade_type = TradeType.SELL if side == "sell" else TradeType.BUY
        self.pool.update_on_fill(trade_type, amount)

        if side == "sell":
            self.base_balance -= amount
            self.quote_balance += amount * price
        else:
            self.base_balance += amount
            self.quote_balance -= amount * price

    def get_snapshot(self) -> dict:
        hmm_regime = "n/a"
        if self._last_hmm is not None:
            top = max(self._last_hmm, key=self._last_hmm.get)
            hmm_regime = f"{top}({self._last_hmm[top]:.0%})"

        decision = self._range_ctrl.last_decision
        eff_trend_val = decision.effective_trend if decision is not None else 0.0

        return {
            "mid_price": self.pool.get_mid_price(),
            "base_balance": self.base_balance,
            "quote_balance": self.quote_balance,
            "concentration_pct": float(self.pool.concentration_pct),
            "p_lower": float(self.pool.p_lower),
            "p_upper": float(self.pool.p_upper),
            "pool_mid": float(self.pool.get_mid_price()),
            "anchor": float(self.pool.anchor_price),
            "natr": float(self._last_natr) if self._last_natr else 0,
            "adx": float(self._last_adx) if self._last_adx else 0,
            "hurst": float(self._last_hurst) if self._last_hurst else 0,
            "hmm_regime": hmm_regime,
            "inventory_skew": self.pool.get_inventory_skew(),
            "recentered": self._recentered_this_candle,
            "range_changed": self._range_changed_this_candle,
            "total_recenters": self._recenter_count,
            "total_range_changes": self._range_change_count,
            "effective_trend": eff_trend_val,
            "trend_order_scale": self._trend_order_scale,
            "trend_halted": self._trend_halted_this_candle,
            "total_trend_halts": self._trend_halted_count,
        }

    def get_portfolio_value(self, price: Decimal) -> Decimal:
        return self.base_balance * price + self.quote_balance

    # -- Internal methods --

    def _update_anchor(self, price: Decimal):
        if self.pool.anchor_price > ZERO:
            alpha = self.anchor_ema_alpha
            self.pool.anchor_price = (
                alpha * price + (D(1) - alpha) * self.pool.anchor_price)
        else:
            self.pool.anchor_price = price

    def _update_indicators(self, history: List[Candle]):
        self._last_natr = self._natr.compute(history)
        self._last_adx = self._adx.compute(history)
        self._last_hurst = self._hurst.compute(history)
        self._last_hmm = self._hmm.predict(history)

    def _update_dynamic_range(self):
        new_pct = self._range_ctrl.compute_concentration_pct(
            self._last_natr, self._last_adx,
            self._last_hurst, self._last_hmm,
        )

        if abs(new_pct - self.pool.concentration_pct) >= self.range_update_dead_band_pct:
            self.pool.set_concentration(new_pct)
            self._range_changed_this_candle = True
            self._range_change_count += 1

    def _max_safe_order_base(self) -> Decimal:
        """Max order (base) that won't trigger ping-pong.

        dp/p = dB * sqrt(p) / L
        w * dp/p < spread => dB < spread * L / (w * sqrt(p))
        """
        w = self.pool_price_weight
        spread = self.spread_bps / D("10000")
        if w <= ZERO or self.pool.L <= ZERO:
            return D("999999999")
        sqrt_p = _sqrt(self.pool.anchor_price)
        if sqrt_p <= ZERO:
            return D("999999999")
        return spread * self.pool.L / (w * sqrt_p)

    def _generate_orders(self, mid_price: Decimal) -> List[SimOrder]:
        orders: List[SimOrder] = []

        # Trend protection
        decision = self._range_ctrl.last_decision
        eff_trend = decision.effective_trend if decision is not None else 0.0

        halt_thresh = float(self.trend_halt_threshold)
        if halt_thresh > 0 and eff_trend >= halt_thresh:
            self._trend_halted_this_candle = True
            self._trend_halted_count += 1
            return orders  # empty

        scale_factor = float(self.trend_order_scale_factor)
        trend_scale = max(0.0, 1.0 - scale_factor * eff_trend)
        if trend_scale <= 0.0:
            self._trend_halted_this_candle = True
            self._trend_halted_count += 1
            return orders  # empty

        self._trend_order_scale = trend_scale

        max_safe_base = self._max_safe_order_base()
        order_base = max_safe_base * self.order_safe_ratio * D(str(round(trend_scale, 6)))
        order_value = order_base * mid_price

        weights = [self.size_decay ** i for i in range(self.num_levels)]
        total_weight = sum(weights)
        if total_weight <= ZERO:
            return orders

        for i in range(self.num_levels):
            base_spread = (self.spread_bps * (self.spread_multiplier ** i)
                           / D("10000"))
            w = weights[i] / total_weight
            layer_value = order_value * w

            if self.enable_asymmetric_spread:
                bid_spread, ask_spread = self._asymmetric_spreads(base_spread)
            else:
                bid_spread, ask_spread = base_spread, base_spread

            ask_price = mid_price * (D(1) + ask_spread)
            bid_price = mid_price * (D(1) - bid_spread)

            ask_size = layer_value / ask_price if ask_price > ZERO else ZERO
            bid_size = layer_value / bid_price if bid_price > ZERO else ZERO

            # Cap by available balance
            if ask_size > self.base_balance:
                ask_size = self.base_balance
            if bid_price > ZERO and bid_size * bid_price > self.quote_balance:
                bid_size = self.quote_balance / bid_price

            # Round to reasonable precision (no connector quantization)
            ask_price = D(str(round(float(ask_price), 8)))
            bid_price = D(str(round(float(bid_price), 8)))
            ask_size = D(str(round(float(ask_size), 4)))
            bid_size = D(str(round(float(bid_size), 4)))

            if ask_size > ZERO:
                orders.append(SimOrder("sell", ask_price, ask_size))
            if bid_size > ZERO:
                orders.append(SimOrder("buy", bid_price, bid_size))

        return orders

    def _asymmetric_spreads(self, base_spread: Decimal):
        skew = D(str(self.pool.get_inventory_skew()))
        sens = self.skew_sensitivity
        floor = self.min_spread_bps / D("10000")
        bid_spread = max(base_spread * (D(1) + skew * sens), floor)
        ask_spread = max(base_spread * (D(1) - skew * sens), floor)
        return bid_spread, ask_spread


# ---------------------------------------------------------------------------
# AMM Backtest Strategy
# ---------------------------------------------------------------------------


class AMMBacktestStrategy(BacktestStrategy):
    """Amplified x*y=k virtual AMM market maker."""

    @property
    def name(self) -> str:
        return "AMM"

    def __init__(self, **config):
        self.spread_bps = D(str(config.get("spread_bps", "40")))
        self.pool_price_weight = D(str(config.get("pool_price_weight", "0.70")))
        self.anchor_ema_alpha = D(str(config.get("anchor_ema_alpha", "0.05")))
        self.amplification = D(str(config.get("amplification", "5")))
        self.num_levels = int(config.get("num_levels", 1))
        self.size_decay = D(str(config.get("size_decay", "0.85")))
        self.spread_multiplier = D(str(config.get("spread_multiplier", "1.5")))
        self.order_safe_ratio = D(str(config.get("order_safe_ratio", "0.5")))
        self.pool_depth_override = config.get("pool_depth")
        self.enable_asymmetric_spread = config.get(
            "enable_asymmetric_spread", True)
        self.skew_sensitivity = D(str(config.get("skew_sensitivity", "0.5")))
        self.min_spread_bps = D(str(config.get("min_spread_bps", "20")))

        self.pool: Optional[VirtualPool] = None
        self.base_balance = ZERO
        self.quote_balance = ZERO

    def initialize(self, initial_price: Decimal, base_balance: Decimal,
                   quote_balance: Decimal):
        self.base_balance = base_balance
        self.quote_balance = quote_balance

        pool_depth = self.pool_depth_override
        if pool_depth is None:
            pool_depth = base_balance * initial_price + quote_balance

        self.pool = VirtualPool(initial_price, pool_depth, self.amplification)

    def on_candle(self, candle: Candle, history: List[Candle],
                  sim_time: float) -> List[SimOrder]:
        price = candle.close

        # Update anchor EMA
        self._update_anchor(price)

        # Compute blended mid (VirtualPool handles internally)
        mid = self.pool.get_mid_price(self.pool_price_weight)

        return self._generate_orders(mid)

    def on_fill(self, side: str, price: Decimal, amount: Decimal,
                sim_time: float):
        trade_type = TradeType.SELL if side == "sell" else TradeType.BUY
        self.pool.update_on_fill(trade_type, amount)

        if side == "sell":
            self.base_balance -= amount
            self.quote_balance += amount * price
        else:
            self.base_balance += amount
            self.quote_balance -= amount * price

    def get_snapshot(self) -> dict:
        raw_kp = self.pool.get_pool_price()
        return {
            "mid_price": self.pool.get_mid_price(self.pool_price_weight),
            "base_balance": self.base_balance,
            "quote_balance": self.quote_balance,
            "pool_mid": float(self.pool.get_mid_price(self.pool_price_weight)),
            "raw_k_price": float(raw_kp) if raw_kp else 0,
            "anchor": float(self.pool.anchor_price),
            "inventory_skew": self.pool.get_inventory_skew(),
            "amplification": float(self.amplification),
        }

    def get_portfolio_value(self, price: Decimal) -> Decimal:
        return self.base_balance * price + self.quote_balance

    # -- Internal methods --

    def _update_anchor(self, price: Decimal):
        if self.pool.anchor_price > ZERO:
            alpha = self.anchor_ema_alpha
            self.pool.anchor_price = (
                alpha * price + (D(1) - alpha) * self.pool.anchor_price)
        else:
            self.pool.anchor_price = price

    def _max_safe_order_base(self) -> Decimal:
        """Max order that won't trigger ping-pong.

        blended_shift = w * 2 * (dB/B) / A
        => dB < B * spread * A / (2w)
        """
        w = self.pool_price_weight
        A = self.amplification
        spread = self.spread_bps / D("10000")

        shift_factor = D(2) * w / A
        if shift_factor <= ZERO:
            return D("999999999")

        max_dB_over_B = spread / shift_factor
        return self.pool.base * max_dB_over_B

    def _generate_orders(self, mid_price: Decimal) -> List[SimOrder]:
        orders: List[SimOrder] = []

        max_safe_base = self._max_safe_order_base()
        order_base = max_safe_base * self.order_safe_ratio
        order_value = order_base * mid_price

        weights = [self.size_decay ** i for i in range(self.num_levels)]
        total_weight = sum(weights)
        if total_weight <= ZERO:
            return orders

        for i in range(self.num_levels):
            base_spread = (self.spread_bps * (self.spread_multiplier ** i)
                           / D("10000"))
            w = weights[i] / total_weight
            layer_value = order_value * w

            if self.enable_asymmetric_spread:
                bid_spread, ask_spread = self._asymmetric_spreads(base_spread)
            else:
                bid_spread, ask_spread = base_spread, base_spread

            ask_price = mid_price * (D(1) + ask_spread)
            bid_price = mid_price * (D(1) - bid_spread)

            ask_size = layer_value / ask_price if ask_price > ZERO else ZERO
            bid_size = layer_value / bid_price if bid_price > ZERO else ZERO

            # Cap by available balance
            if ask_size > self.base_balance:
                ask_size = self.base_balance
            if bid_price > ZERO and bid_size * bid_price > self.quote_balance:
                bid_size = self.quote_balance / bid_price

            ask_price = D(str(round(float(ask_price), 8)))
            bid_price = D(str(round(float(bid_price), 8)))
            ask_size = D(str(round(float(ask_size), 4)))
            bid_size = D(str(round(float(bid_size), 4)))

            if ask_size > ZERO:
                orders.append(SimOrder("sell", ask_price, ask_size))
            if bid_size > ZERO:
                orders.append(SimOrder("buy", bid_price, bid_size))

        return orders

    def _asymmetric_spreads(self, base_spread: Decimal):
        skew = D(str(self.pool.get_inventory_skew()))
        sens = self.skew_sensitivity
        floor = self.min_spread_bps / D("10000")
        bid_spread = max(base_spread * (D(1) + skew * sens), floor)
        ask_spread = max(base_spread * (D(1) - skew * sens), floor)
        return bid_spread, ask_spread
