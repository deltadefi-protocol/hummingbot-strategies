import asyncio
import json
import logging
import logging.handlers
import os
import random
import time
from collections import deque
from decimal import Decimal
from typing import Dict, List, NamedTuple, Optional

from pydantic import Field, model_validator

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.strategy_v2_base import StrategyV2Base

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


class DeltaDefiAMMConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    # Required (must set per pair)
    exchange: str = Field("deltadefi")
    trading_pair: str = Field(default="ADA-USDM")
    initial_price: Optional[Decimal] = Field(default=None)
    base_spread_bps: Decimal = Field(D("40"))
    max_cumulative_loss: Decimal = Field(D("500"))
    min_base_balance: Decimal = Field(D("1000"))
    min_quote_balance: Decimal = Field(D("500"))
    # Amplification: flattens k-price sensitivity per fill.
    # A=1: standard x*y=k, A=20: price shifts 20x slower.
    # Pool is initialized to match real capital; A controls how
    # much each fill moves the effective k-price.
    amplification: Decimal = Field(D("5"))
    num_levels: int = Field(1)
    size_decay: Decimal = Field(D("0.85"))
    spread_multiplier: Decimal = Field(D("1.5"))
    order_amount_pct: Decimal = Field(D("0.005"))
    order_refresh_time: int = Field(5)
    refresh_on_fill_only: bool = Field(True)
    floor_ratio: Decimal = Field(D("0.30"))
    balance_buffer_pct: Decimal = Field(D("0.90"))
    rebalance_threshold: Decimal = Field(D("0.02"))
    rebalance_cooldown: int = Field(60)
    min_both_sides_pct: Decimal = Field(D("0.40"))
    # Hybrid pricing: 0=pure PMM (anchor), 1=pure AMM (k-price)
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

    @model_validator(mode="after")
    def apply_pair_presets(self):
        preset = PAIR_PRESETS.get(self.trading_pair, {})
        if self.initial_price is None:
            self.initial_price = preset.get("initial_price", D("0.01"))
        if not (ZERO <= self.pool_price_weight <= D(1)):
            raise ValueError("pool_price_weight must be in [0, 1]")
        if not (ZERO < self.anchor_ema_alpha <= D(1)):
            raise ValueError("anchor_ema_alpha must be in (0, 1]")
        if self.amplification < D(1):
            raise ValueError("amplification must be >= 1")
        return self


# ---------------------------------------------------------------------------
# VirtualPool — pricing via amplified x*y=k
# ---------------------------------------------------------------------------


class VirtualPool:
    """Virtual AMM reserves. Updates only on fills. Ignores deposits,
    withdrawals, and other bots on the same account."""

    def __init__(self, initial_price: Decimal, pool_depth: Decimal, amplification: Decimal):
        self.initial_quote = D(str(pool_depth))
        self.initial_base = self.initial_quote / D(str(initial_price))
        self.base = self.initial_base
        self.quote = self.initial_quote
        self.k = self.base * self.quote
        self.amplification = D(str(amplification))
        self.anchor_price = D(str(initial_price))

    @classmethod
    def from_state(cls, state: dict) -> "VirtualPool":
        pool = cls.__new__(cls)
        pool.initial_base = D(str(state["initial_base"]))
        pool.initial_quote = D(str(state["initial_quote"]))
        pool.base = D(str(state["base"]))
        pool.quote = D(str(state["quote"]))
        pool.k = D(str(state["k"]))
        pool.amplification = D(str(state["amplification"]))
        # Migration: old state files lack anchor_price — fall back to initial_price
        if "anchor_price" in state:
            pool.anchor_price = D(str(state["anchor_price"]))
        else:
            pool.anchor_price = pool.initial_quote / pool.initial_base
        return pool

    def to_state(self) -> dict:
        return {
            "initial_base": str(self.initial_base),
            "initial_quote": str(self.initial_quote),
            "base": str(self.base),
            "quote": str(self.quote),
            "k": str(self.k),
            "amplification": str(self.amplification),
            "anchor_price": str(self.anchor_price),
        }

    def get_mid_price(self, pool_price_weight: Decimal = ZERO) -> Decimal:
        """Blended mid: (1-w)*anchor + w*dampened_k_price.
        Anchor = pure market reference (EMA of book mid).
        k-price deviation from anchor is dampened by amplification:
          dampened = anchor + (raw_k_price - anchor) / A
        w=0 → pure anchor, w=1 → pure dampened k-price."""
        if self.base <= ZERO:
            return self.anchor_price
        raw_pool_price = self.quote / self.base
        # Dampen: reduce k-price deviation from anchor by amplification factor
        deviation = raw_pool_price - self.anchor_price
        dampened_pool_price = self.anchor_price + deviation / self.amplification
        if pool_price_weight >= D(1):
            return dampened_pool_price
        if pool_price_weight <= ZERO:
            return self.anchor_price
        return (D(1) - pool_price_weight) * self.anchor_price + pool_price_weight * dampened_pool_price

    def get_pool_price(self) -> Optional[Decimal]:
        """Raw k-price (quote/base) without amplification dampening."""
        if self.base <= ZERO:
            return None
        return self.quote / self.base

    def get_dampened_pool_price(self) -> Optional[Decimal]:
        """K-price with amplification dampening applied."""
        if self.base <= ZERO:
            return None
        raw = self.quote / self.base
        return self.anchor_price + (raw - self.anchor_price) / self.amplification

    def update_on_fill(self, side: TradeType, amount: Decimal):
        filled = D(str(amount))
        if side == TradeType.SELL:      # we sold base → reserves shrink
            self.base -= filled
        elif side == TradeType.BUY:     # we bought base → reserves grow
            self.base += filled
        if self.base > ZERO:
            self.quote = self.k / self.base

    def get_inventory_skew(self) -> float:
        base_ratio = float(self.base / self.initial_base)
        quote_ratio = float(self.quote / self.initial_quote)
        denom = base_ratio + quote_ratio
        if denom == 0:
            return 0.0
        return (base_ratio - quote_ratio) / denom

    def get_available_reserves(self, floor_ratio: Decimal):
        floor = D(str(floor_ratio))
        base_avail = max(ZERO, self.base - self.initial_base * floor)
        quote_avail = max(ZERO, self.quote - self.initial_quote * floor)
        return base_avail, quote_avail


# ---------------------------------------------------------------------------
# BalanceGate — scales orders to real account balance
# ---------------------------------------------------------------------------


class BalanceGate:
    """Reads real balance from exchange connector each cycle. Scales or
    removes orders that exceed what the account can support."""

    def __init__(self, connector: ConnectorBase, config: DeltaDefiAMMConfig):
        self.connector = connector
        self.config = config

    def get_real_balances(self):
        base_token, quote_token = self.config.trading_pair.split("-")
        return (
            self.connector.get_available_balance(base_token),
            self.connector.get_available_balance(quote_token),
        )

    def scale_orders(self, orders: List[OrderProposal]) -> List[OrderProposal]:
        real_base, real_quote = self.get_real_balances()
        usable_base = real_base * self.config.balance_buffer_pct
        usable_quote = real_quote * self.config.balance_buffer_pct

        # Remove sides below minimum
        if real_base < self.config.min_base_balance:
            orders = [o for o in orders if o.side != TradeType.SELL]
        if real_quote < self.config.min_quote_balance:
            orders = [o for o in orders if o.side != TradeType.BUY]

        if not orders:
            return orders

        # Scale asks to fit real base
        total_ask = sum(o.size for o in orders if o.side == TradeType.SELL)
        if total_ask > ZERO and total_ask > usable_base:
            scale = usable_base / total_ask
            orders = [
                OrderProposal(o.side, o.price, o.size * scale) if o.side == TradeType.SELL else o
                for o in orders
            ]

        # Scale bids to fit real quote
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


class DeltaDefiAMM(StrategyV2Base):
    _default_config = DeltaDefiAMMConfig()
    markets = {_default_config.exchange: {_default_config.trading_pair}}
    _refresh_timestamp: float = 0
    _last_rebalance: float = 0
    _stopped: bool = False
    _refreshing: bool = False
    _pool_scaled: bool = False

    @classmethod
    def init_markets(cls, config: DeltaDefiAMMConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: Optional[DeltaDefiAMMConfig] = None):
        super().__init__(connectors, config)
        if self.config is None:
            self.config = DeltaDefiAMMConfig()

        # Per-pair logger with dedicated file handler → logs/logs_deltadefi_amm_mm_ADA-USDM.log
        pair_tag = self.config.trading_pair
        self._pair_logger = logging.getLogger(f"{__name__}.{pair_tag}")
        self._pair_logger.propagate = False  # Don't duplicate to root/parent handler

        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        pair_log_file = os.path.join(logs_dir, f"logs_deltadefi_amm_mm_{pair_tag}.log")
        if not self._pair_logger.handlers:
            fh = logging.handlers.TimedRotatingFileHandler(
                pair_log_file, when="D", interval=1, backupCount=7, encoding="utf8"
            )
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._pair_logger.addHandler(fh)
            # Also log to console for Hummingbot UI
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._pair_logger.addHandler(ch)
            self._pair_logger.setLevel(logging.DEBUG)

        self._state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
        os.makedirs(self._state_dir, exist_ok=True)
        self._state_file = os.path.join(self._state_dir, f"{self.config.trading_pair}_pool_state.json")

        state = self._load_state()
        if state:
            self.pool = VirtualPool.from_state(state)
            self._base_flow = D(str(state.get("base_flow", "0")))
            self._quote_flow = D(str(state.get("quote_flow", "0")))
            self._pair_logger.info(f"Restored pool state from {self._state_file}")
        else:
            # First run: pool will be initialized from real balance on first tick
            self.pool = None
            self._base_flow = ZERO
            self._quote_flow = ZERO

        self.balance_gate = BalanceGate(connectors[self.config.exchange], self.config)
        self._recent_fills: deque = deque()
        self._last_max_safe: Decimal = ZERO  # updated each order cycle

    # ---- Main loop --------------------------------------------------------

    def on_tick(self):
        if self._stopped:
            return

        # Wait for trading rules to be loaded before generating orders
        connector = self.connectors[self.config.exchange]
        if self.config.trading_pair not in connector.trading_rules:
            return

        # Initialize or auto-scale pool on first tick (balances available now)
        if self.pool is None:
            if not self._init_pool_from_balance():
                return
        elif not self._pool_scaled:
            self._auto_scale_pool()

        if self._check_circuit_breakers():
            return

        # Always keep anchor fresh — even when orders are active and we
        # won't place new ones this tick.  Without this, fill-only mode
        # freezes the anchor for the entire time orders sit on the book.
        book_mid = self._get_book_mid()
        if book_mid:
            self._update_anchor_ema(book_mid)

        # Don't place orders while _refresh_after_fill is in progress —
        # doing so creates duplicate sets that can cross and self-trade.
        if self._refreshing:
            return

        has_active = bool(self.get_active_orders(connector_name=self.config.exchange))

        if self.config.refresh_on_fill_only:
            # Fill-only mode: _refresh_after_fill handles requote on fills.
            # on_tick only places initial orders when nothing is on the book.
            if has_active:
                return
        else:
            # Timer mode: wait for order_refresh_time before refreshing.
            if self.current_timestamp < self._refresh_timestamp:
                return
            if has_active:
                self._cancel_all_orders()
                return

        mid = self.pool.get_mid_price(self.config.pool_price_weight)
        RateOracle.get_instance().set_price(self.config.trading_pair, mid)

        orders = self._generate_orders(mid)
        orders = self.balance_gate.scale_orders(orders)
        self._place_orders(orders)

        safe_ensure_future(self._check_rebalance())
        self._refresh_timestamp = self.current_timestamp + self.config.order_refresh_time

    # ---- Order generation -------------------------------------------------

    def _max_safe_order_base(self) -> Decimal:
        """Max order size (in base) that won't trigger ping-pong.

        No-ping-pong condition: blended_shift < spread
          k-price deviation is dampened by A, so:
          blended_shift = w × 2 × (dB/B) / A
          => dB < B × spread × A / (2w)
        """
        w = self.config.pool_price_weight
        A = self.config.amplification
        spread = self.config.base_spread_bps / D("10000")

        shift_factor = D(2) * w / A
        if shift_factor <= ZERO:
            return D("999999999")

        max_dB_over_B = spread / shift_factor
        max_base = self.pool.base * max_dB_over_B
        # Also express as quote limit for the bid side
        self._last_max_safe = max_base  # cache for status display
        return max_base

    def _generate_orders(self, mid_price: Decimal) -> List[OrderProposal]:
        orders: List[OrderProposal] = []

        # Single balance lookup for both order sizing and capping
        real_base, real_quote = self.balance_gate.get_real_balances()
        total_capital = real_base * mid_price + real_quote
        order_value = total_capital * self.config.order_amount_pct

        max_safe_base = self._max_safe_order_base()
        max_safe_value = max_safe_base * mid_price

        # Cap order value to ping-pong-safe maximum
        if order_value > max_safe_value:
            self._pair_logger.info(
                f"Ping-pong guard: order {order_value:.2f} → {max_safe_value:.2f} "
                f"(max safe {max_safe_base:.1f} base)"
            )
            order_value = max_safe_value

        weights = [self.config.size_decay ** i for i in range(self.config.num_levels)]
        total_weight = sum(weights)
        if total_weight <= ZERO:
            return orders

        connector = self.connectors[self.config.exchange]
        pair = self.config.trading_pair

        for i in range(self.config.num_levels):
            base_spread = self.config.base_spread_bps * (self.config.spread_multiplier ** i) / D("10000")
            w = weights[i] / total_weight
            layer_value = order_value * w

            if self.config.enable_asymmetric_spread:
                bid_spread, ask_spread = self._asymmetric_spreads(base_spread)
            else:
                bid_spread, ask_spread = base_spread, base_spread

            ask_price = mid_price * (D(1) + ask_spread)
            bid_price = mid_price * (D(1) - bid_spread)

            ask_size = layer_value / ask_price if ask_price > ZERO else ZERO
            bid_size = layer_value / bid_price if bid_price > ZERO else ZERO

            if self.config.enable_order_randomization:
                ask_size = self._randomize(ask_size)
                bid_size = self._randomize(bid_size)

            ask_price = connector.quantize_order_price(pair, ask_price)
            bid_price = connector.quantize_order_price(pair, bid_price)
            ask_size = connector.quantize_order_amount(pair, ask_size)
            bid_size = connector.quantize_order_amount(pair, bid_size)

            if ask_size > ZERO:
                orders.append(OrderProposal(TradeType.SELL, ask_price, ask_size))
            if bid_size > ZERO:
                orders.append(OrderProposal(TradeType.BUY, bid_price, bid_size))

        return orders

    def _asymmetric_spreads(self, base_spread: Decimal):
        skew = D(str(self.pool.get_inventory_skew()))
        sens = self.config.skew_sensitivity
        floor = self.config.min_spread_bps / D("10000")
        # Positive skew (excess base) → widen bid, tighten ask
        bid_spread = max(base_spread * (D(1) + skew * sens), floor)
        ask_spread = max(base_spread * (D(1) - skew * sens), floor)
        return bid_spread, ask_spread

    def _randomize(self, size: Decimal) -> Decimal:
        pct = float(self.config.randomization_pct)
        jitter = D(str(1.0 + random.uniform(-pct, pct)))
        return max(ZERO, size * jitter)

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
        """Wait until all orders have exchange_order_ids (submitted to exchange)."""
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
        # 1. Loss limit exceeded → full stop
        pnl = self._get_pnl()
        if pnl is not None and pnl < -self.config.max_cumulative_loss:
            self._pair_logger.error(f"Loss limit exceeded: P&L {pnl:.2f}. Shutting down.")
            self._cancel_all_orders()
            self._stopped = True
            return True

        # 2. Both virtual sides depleted → pause
        base_pct = self.pool.base / self.pool.initial_base
        quote_pct = self.pool.quote / self.pool.initial_quote
        if base_pct < self.config.min_both_sides_pct and quote_pct < self.config.min_both_sides_pct:
            self._pair_logger.warning("Both virtual sides depleted. Pausing.")
            self._cancel_all_orders()
            return True

        # 3. Fill velocity (enhancement)
        if self.config.enable_fill_velocity_detector:
            now = time.time()
            window = self.config.fill_velocity_window_sec
            while self._recent_fills and now - self._recent_fills[0][0] > window:
                self._recent_fills.popleft()
            buy_count = sum(1 for _, t in self._recent_fills if t == TradeType.BUY)
            sell_count = sum(1 for _, t in self._recent_fills if t == TradeType.SELL)
            if max(buy_count, sell_count) >= self.config.fill_velocity_max_same_side:
                self._pair_logger.warning(f"Fill velocity: {buy_count}B/{sell_count}S in {window}s. Pausing.")
                self._cancel_all_orders()
                return True

        return False

    # ---- Rebalance --------------------------------------------------------

    async def _check_rebalance(self):
        if time.time() - self._last_rebalance < self.config.rebalance_cooldown:
            return

        book_mid = self._get_book_mid()
        if book_mid is None or book_mid <= ZERO:
            return

        amm_mid = self.pool.get_mid_price(self.config.pool_price_weight)
        divergence = abs(amm_mid - book_mid) / book_mid

        if divergence <= self.config.rebalance_threshold:
            return

        target_base = (self.pool.k / book_mid).sqrt()
        rebalance_amount = abs(target_base - self.pool.base)
        side = TradeType.BUY if self.pool.base < target_base else TradeType.SELL

        # Cap to real balance
        real_base, real_quote = self.balance_gate.get_real_balances()
        if side == TradeType.BUY:
            max_amount = real_quote * self.config.balance_buffer_pct / book_mid
        else:
            max_amount = real_base * self.config.balance_buffer_pct
        rebalance_amount = min(rebalance_amount, max_amount)

        connector = self.connectors[self.config.exchange]
        rebalance_amount = connector.quantize_order_amount(self.config.trading_pair, rebalance_amount)
        if rebalance_amount <= ZERO:
            return

        # Ensure all existing orders are cancelled before placing rebalance MARKET order.
        # Without this, the MARKET order can land while stale LIMIT orders are still live.
        active_orders = self.get_active_orders(connector_name=self.config.exchange)
        if active_orders:
            results = await connector.cancel_all(timeout_seconds=10.0)
            if results and any(not r.success for r in results):
                self._pair_logger.warning("Rebalance aborted — cancel_all failed, orders still live on exchange.")
                return

        self._pair_logger.info(f"Rebalance: {side.name} {rebalance_amount} (divergence: {divergence:.4f})")
        if side == TradeType.BUY:
            self.buy(self.config.exchange, self.config.trading_pair, rebalance_amount, OrderType.MARKET)
        else:
            self.sell(self.config.exchange, self.config.trading_pair, rebalance_amount, OrderType.MARKET)
        self._last_rebalance = time.time()

    # ---- Anchor EMA -------------------------------------------------------

    def _update_anchor_ema(self, book_mid: Decimal):
        """Smooth anchor tracking via EMA to prevent jitter."""
        if self.pool.anchor_price and self.pool.anchor_price > ZERO:
            alpha = self.config.anchor_ema_alpha
            self.pool.anchor_price = alpha * book_mid + (D(1) - alpha) * self.pool.anchor_price
        else:
            self.pool.anchor_price = book_mid

    # ---- Fill handling ----------------------------------------------------

    def did_fill_order(self, event: OrderFilledEvent):
        # Track real flows
        if event.trade_type == TradeType.SELL:
            self._base_flow -= event.amount
            self._quote_flow += event.price * event.amount
        else:
            self._base_flow += event.amount
            self._quote_flow -= event.price * event.amount

        # Update virtual pool
        self.pool.update_on_fill(event.trade_type, event.amount)

        # Track fill velocity
        if self.config.enable_fill_velocity_detector:
            self._recent_fills.append((time.time(), event.trade_type))

        # Persist state
        self._save_state()

        # Log
        pnl = self._get_pnl()
        pnl_str = f"{pnl:+.4f}" if pnl is not None else "n/a"
        raw_kp = self.pool.get_pool_price()
        dampened_kp = self.pool.get_dampened_pool_price()
        raw_str = f"{raw_kp:.6f}" if raw_kp else "n/a"
        damp_str = f"{dampened_kp:.6f}" if dampened_kp else "n/a"
        msg = (
            f"AMM {event.trade_type.name} {event.amount:.4f} @ {event.price:.6f} | "
            f"P&L: {pnl_str} | k-raw: {raw_str} | k-damp: {damp_str}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

        # Cancel remaining orders and requote immediately with updated pool
        safe_ensure_future(self._refresh_after_fill())

    async def _refresh_after_fill(self):
        """Await batch cancel, then immediately place fresh orders from updated pool."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            connector = self.connectors[self.config.exchange]

            # Update anchor BEFORE cancelling — after cancel, book may be thin
            book_mid = self._get_book_mid()
            if book_mid:
                self._update_anchor_ema(book_mid)

            results = await connector.cancel_all(timeout_seconds=10.0)

            # Abort if cancel failed — old orders are still live on exchange;
            # placing new orders would create duplicates that can self-trade.
            if results and any(not r.success for r in results):
                self._pair_logger.warning("cancel_all had failures — skipping requote to avoid duplicate orders.")
                return

            mid = self.pool.get_mid_price(self.config.pool_price_weight)
            RateOracle.get_instance().set_price(self.config.trading_pair, mid)
            orders = self._generate_orders(mid)
            orders = self.balance_gate.scale_orders(orders)
            order_ids = self._place_orders(orders)
            self._refresh_timestamp = self.current_timestamp + self.config.order_refresh_time

            # Wait for orders to reach the exchange before releasing _refreshing.
            # Without this, a rapid fill can trigger a new refresh cycle while
            # these orders are still in the build/sign/submit pipeline, causing
            # duplicate orders that cross and self-trade.
            await self._await_order_confirmations(order_ids)
        except Exception:
            self._pair_logger.error("Error in refresh after fill.", exc_info=True)
        finally:
            self._refreshing = False

    # ---- P&L --------------------------------------------------------------

    def _get_pnl(self) -> Optional[Decimal]:
        """Mark-to-market P&L: value of all base exchanged at book mid + net quote flow."""
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

    # ---- Pool auto-scaling ------------------------------------------------

    def _init_pool_from_balance(self) -> bool:
        """First run: create pool from real account balance (pool = real capital)."""
        real_base, real_quote = self.balance_gate.get_real_balances()
        if real_base <= ZERO or real_quote <= ZERO:
            return False
        mid = self._get_book_mid()
        if mid is None or mid <= ZERO:
            mid = self.config.initial_price
        pool_depth = real_base * mid + real_quote
        self.pool = VirtualPool(mid, pool_depth, self.config.amplification)
        self._pool_scaled = True
        self._save_state()
        self._pair_logger.info(
            f"Initialized pool from account balance: "
            f"{self.config.trading_pair} @ {mid} depth={pool_depth:.0f} A={self.config.amplification}"
        )
        return True

    def _auto_scale_pool(self):
        """Scale virtual pool to match real account balance on restart.
        Preserves inventory ratio (skew) and P&L flows."""
        self._pool_scaled = True
        real_base, real_quote = self.balance_gate.get_real_balances()
        if real_base <= ZERO or real_quote <= ZERO:
            return
        book_mid = self._get_book_mid()
        if book_mid:
            self.pool.anchor_price = book_mid
        mid = self.pool.get_mid_price()  # no weight — use anchor for scaling
        if mid is None or mid <= ZERO:
            return

        real_total = real_base * mid + real_quote
        target_depth = real_total  # pool matches real capital
        pool_total = self.pool.initial_base * mid + self.pool.initial_quote
        if pool_total <= ZERO:
            return

        scale = target_depth / pool_total
        # Only scale if difference is meaningful (>5%)
        if abs(scale - D(1)) < D("0.05"):
            return

        old_depth = self.pool.initial_quote
        self.pool.initial_base *= scale
        self.pool.initial_quote *= scale
        self.pool.base *= scale
        self.pool.quote *= scale
        self.pool.k = self.pool.base * self.pool.quote
        self._base_flow *= scale
        self._quote_flow *= scale
        self._save_state()
        self._pair_logger.info(
            f"Auto-scaled pool to account balance: "
            f"{old_depth:.0f} -> {self.pool.initial_quote:.0f} ({scale:.2f}x)"
        )

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
        A = self.config.amplification
        blended_mid = self.pool.get_mid_price(w)
        raw_kp = self.pool.get_pool_price()
        dampened_kp = self.pool.get_dampened_pool_price()
        book_mid = self._get_book_mid()
        anchor_str = f"{self.pool.anchor_price:.6f}"
        book_str = f"{book_mid:.6f}" if book_mid else "n/a"
        raw_kp_str = f"{raw_kp:.6f}" if raw_kp else "n/a"
        damp_kp_str = f"{dampened_kp:.6f}" if dampened_kp else "n/a"

        base_pct = self.pool.base / self.pool.initial_base * D(100)
        quote_pct = self.pool.quote / self.pool.initial_quote * D(100)
        skew = self.pool.get_inventory_skew()

        real_base, real_quote = self.balance_gate.get_real_balances()
        base_token, quote_token = self.config.trading_pair.split("-")

        pnl = self._get_pnl()
        pnl_str = f"{pnl:+.4f}" if pnl is not None else "n/a"

        total_capital = real_base * blended_mid + real_quote
        order_value = total_capital * self.config.order_amount_pct
        max_safe_str = f"{self._last_max_safe:.1f}" if self._last_max_safe > ZERO else "n/a"
        max_safe_value = self._last_max_safe * blended_mid if self._last_max_safe > ZERO else ZERO
        capped = order_value > max_safe_value and max_safe_value > ZERO
        guard_str = f"CAPPED to {max_safe_value:.1f}" if capped else "ok"

        lines = [
            f"  [AMM {self.config.trading_pair}] Mid: {blended_mid:.6f} "
            f"(w={w} A={A})",
            f"  Anchor: {anchor_str} | Book: {book_str} | k-raw: {raw_kp_str} | k-damp: {damp_kp_str}",
            f"  VPool: {base_pct:.1f}%B / {quote_pct:.1f}%Q | Skew: {skew:+.3f}",
            f"  Real: {real_base:.2f} {base_token} / {real_quote:.2f} {quote_token}"
            f" | Capital: {total_capital:.1f} {quote_token}",
            f"  Order: {self.config.order_amount_pct * 100:.1f}% = {order_value:.1f} {quote_token}"
            f" | Guard: {max_safe_str} {base_token} ({guard_str})",
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
