import logging
import os
import time
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.exchange.deltadefi.deltadefi_health import ConnectorHealth
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base

class DeltaDefiMACrossoverConfig(BaseClientModel):
    script_file_name: str = os.path.basename(__file__)
    exchange: str = Field("deltadefi")
    trading_pair: str = Field("ADA-USDM")
    fast_ma_period: int = Field(7)
    slow_ma_period: int = Field(25)
    order_size: Decimal = Field(Decimal("10"))
    cooldown_after_trade: int = Field(60)
    max_position: Decimal = Field(Decimal("100"))
    stop_loss_pct: Decimal = Field(Decimal("0.05"))
    order_type: str = Field("MARKET", description="MARKET or LIMIT")
    limit_offset_pct: Decimal = Field(Decimal("0.001"), description="Offset from mid price for LIMIT orders (e.g. 0.001 = 0.1%)")


class DeltaDefiMACrossover(StrategyV2Base):

    create_timestamp = 0
    _last_trade_time: float = 0.0
    _last_fast_above_slow: bool = False
    _total_cost: Decimal = Decimal("0")
    _total_bought: Decimal = Decimal("0")
    _avg_entry_price: Decimal = Decimal("0")
    _last_warmup_log: float = 0.0

    @classmethod
    def init_markets(cls, config: DeltaDefiMACrossoverConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: DeltaDefiMACrossoverConfig):
        super().__init__(connectors)
        self.config = config

    def on_tick(self):
        connector = self.connectors[self.config.exchange]

        # Check connector health
        if hasattr(connector, "health_monitor"):
            health = connector.health_monitor.state
            if health != ConnectorHealth.NORMAL:
                if health == ConnectorHealth.DEGRADED:
                    self._cancel_all()
                elif health == ConnectorHealth.MAINTENANCE:
                    self._cancel_all()
                return

        # Check stop loss
        self._check_stop_loss()

        # Check candle builder data availability
        if not hasattr(connector, "candle_builder"):
            return

        candle_builder = connector.candle_builder

        if not candle_builder.has_enough_data(self.config.slow_ma_period):
            self._log_warmup_progress(candle_builder)
            return

        # Compute MAs
        closes = candle_builder.get_closes(self.config.slow_ma_period)
        fast_ma = self._compute_ma(closes, self.config.fast_ma_period)
        slow_ma = self._compute_ma(closes, self.config.slow_ma_period)

        if fast_ma is None or slow_ma is None:
            return

        fast_above_slow = fast_ma > slow_ma

        # Detect crossover
        if fast_above_slow and not self._last_fast_above_slow:
            # Bullish crossover - buy signal
            if self._can_trade(TradeType.BUY):
                self._place_order(TradeType.BUY)
        elif not fast_above_slow and self._last_fast_above_slow:
            # Bearish crossover - sell signal
            if self._can_trade(TradeType.SELL):
                self._place_order(TradeType.SELL)

        self._last_fast_above_slow = fast_above_slow

    def _compute_ma(self, closes: List[Decimal], period: int) -> Optional[Decimal]:
        if len(closes) < period:
            return None
        relevant = closes[-period:]
        return sum(relevant) / Decimal(str(period))

    def _can_trade(self, side: TradeType) -> bool:
        # Check cooldown
        now = time.time()
        if now - self._last_trade_time < self.config.cooldown_after_trade:
            return False

        # Check position size
        connector = self.connectors[self.config.exchange]
        base, quote = self.config.trading_pair.split("-")
        base_balance = connector.get_balance(base)

        if side == TradeType.BUY:
            if base_balance >= self.config.max_position:
                return False
        elif side == TradeType.SELL:
            if base_balance < self.config.order_size:
                return False

        return True

    def _place_order(self, side: TradeType):
        use_limit = self.config.order_type.upper() == "LIMIT"

        if use_limit:
            connector = self.connectors[self.config.exchange]
            try:
                mid_price = connector.get_mid_price(self.config.trading_pair)
            except Exception:
                self.logger().warning("Could not get mid price for LIMIT order, skipping.")
                return

            if mid_price is None or mid_price <= Decimal("0"):
                self.logger().warning("Invalid mid price for LIMIT order, skipping.")
                return

            if side == TradeType.BUY:
                price = mid_price * (Decimal("1") + self.config.limit_offset_pct)
            else:
                price = mid_price * (Decimal("1") - self.config.limit_offset_pct)

            if side == TradeType.BUY:
                self.buy(
                    connector_name=self.config.exchange,
                    trading_pair=self.config.trading_pair,
                    amount=self.config.order_size,
                    order_type=OrderType.LIMIT,
                    price=price,
                )
            else:
                self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=self.config.trading_pair,
                    amount=self.config.order_size,
                    order_type=OrderType.LIMIT,
                    price=price,
                )
        else:
            if side == TradeType.BUY:
                self.buy(
                    connector_name=self.config.exchange,
                    trading_pair=self.config.trading_pair,
                    amount=self.config.order_size,
                    order_type=OrderType.MARKET,
                )
            else:
                self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=self.config.trading_pair,
                    amount=self.config.order_size,
                    order_type=OrderType.MARKET,
                )

    def _check_stop_loss(self):
        if self._avg_entry_price <= Decimal("0"):
            return

        connector = self.connectors[self.config.exchange]
        try:
            mid_price = connector.get_mid_price(self.config.trading_pair)
        except Exception:
            return

        if mid_price is None or mid_price <= Decimal("0"):
            return

        loss_pct = (self._avg_entry_price - mid_price) / self._avg_entry_price
        if loss_pct >= self.config.stop_loss_pct:
            self.logger().warning(
                f"Stop loss triggered: loss {loss_pct:.2%} >= threshold {self.config.stop_loss_pct:.2%}"
            )
            # Flatten position
            base, quote = self.config.trading_pair.split("-")
            base_balance = connector.get_balance(base)
            if base_balance > Decimal("0"):
                self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=self.config.trading_pair,
                    amount=base_balance,
                    order_type=OrderType.MARKET,
                )
            self._reset_entry_tracking()

    def _cancel_all(self):
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def _log_warmup_progress(self, candle_builder):
        now = time.time()
        if now - self._last_warmup_log < 30:
            return
        self._last_warmup_log = now
        progress = candle_builder.warmup_progress(self.config.slow_ma_period)
        self.logger().info(
            f"Warming up: {progress:.0%} ({candle_builder.candle_count}/{self.config.slow_ma_period} candles)"
        )

    def _reset_entry_tracking(self):
        self._total_cost = Decimal("0")
        self._total_bought = Decimal("0")
        self._avg_entry_price = Decimal("0")

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 2)} "
            f"{event.trading_pair} {self.config.exchange} at {round(event.price, 2)}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
        self._last_trade_time = time.time()

        if event.trade_type == TradeType.BUY:
            self._total_cost += event.price * event.amount
            self._total_bought += event.amount
            self._avg_entry_price = self._total_cost / self._total_bought
        elif event.trade_type == TradeType.SELL:
            # Check if position is fully closed
            connector = self.connectors[self.config.exchange]
            base, _ = self.config.trading_pair.split("-")
            remaining = connector.get_balance(base)
            if remaining <= Decimal("0"):
                self._reset_entry_tracking()

    def format_status(self) -> str:
        lines = []
        connector = self.connectors[self.config.exchange]

        # Connector health
        if hasattr(connector, "health_monitor"):
            health = connector.health_monitor.state
            lines.append(f"  Health: {health.name}")
        else:
            lines.append("  Health: unknown")

        # Warmup / MA values
        if hasattr(connector, "candle_builder"):
            cb = connector.candle_builder
            if not cb.has_enough_data(self.config.slow_ma_period):
                progress = cb.warmup_progress(self.config.slow_ma_period)
                lines.append(f"  Warmup: {progress:.0%} ({cb.candle_count}/{self.config.slow_ma_period} candles)")
            else:
                closes = cb.get_closes(self.config.slow_ma_period)
                fast_ma = self._compute_ma(closes, self.config.fast_ma_period)
                slow_ma = self._compute_ma(closes, self.config.slow_ma_period)
                if fast_ma is not None and slow_ma is not None:
                    signal = "BULLISH" if fast_ma > slow_ma else "BEARISH"
                    lines.append(f"  Fast MA({self.config.fast_ma_period}): {fast_ma:.6f}")
                    lines.append(f"  Slow MA({self.config.slow_ma_period}): {slow_ma:.6f}")
                    lines.append(f"  Signal: {signal}")
        else:
            lines.append("  Candle builder: not available")

        # Position info
        base, quote = self.config.trading_pair.split("-")
        base_balance = connector.get_balance(base)
        quote_balance = connector.get_balance(quote)
        lines.append(f"  Position: {base_balance:.4f} {base}")

        if self._avg_entry_price > Decimal("0") and base_balance > Decimal("0"):
            lines.append(f"  Avg entry: {self._avg_entry_price:.6f}")
            try:
                mid_price = connector.get_mid_price(self.config.trading_pair)
                if mid_price is not None and mid_price > Decimal("0"):
                    unrealized = (mid_price - self._avg_entry_price) * base_balance
                    pnl_pct = (mid_price - self._avg_entry_price) / self._avg_entry_price
                    lines.append(f"  Unrealized P&L: {unrealized:.4f} {quote} ({pnl_pct:.2%})")
            except Exception:
                pass

        # Cooldown
        elapsed = time.time() - self._last_trade_time
        remaining_cd = self.config.cooldown_after_trade - elapsed
        if self._last_trade_time > 0 and remaining_cd > 0:
            lines.append(f"  Cooldown: {remaining_cd:.0f}s remaining")

        # Balances
        lines.append(f"  Balances: {base_balance:.4f} {base} / {quote_balance:.4f} {quote}")

        # Active orders
        active_orders = self.get_active_orders(connector_name=self.config.exchange)
        if active_orders:
            lines.append(f"  Active orders: {len(active_orders)}")
            for order in active_orders:
                lines.append(f"    {order.trading_pair} {order.order_type.name} {order.trade_type.name} "
                             f"{order.quantity:.4f} @ {order.price:.6f}")
        else:
            lines.append("  Active orders: none")

        # Order type config
        lines.append(f"  Order type: {self.config.order_type.upper()}")
        if self.config.order_type.upper() == "LIMIT":
            lines.append(f"  Limit offset: {self.config.limit_offset_pct:.4%}")

        return "\n".join(lines)
