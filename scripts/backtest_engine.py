#!/usr/bin/env python3
"""Generic strategy backtester with Binance historical data.

Usage:
    python backtest_engine.py --strategy cl-amm --symbol ADAUSDT \\
        --start 2025-01-01 --end 2025-03-01

Fill model: cross-based -- if candle high >= ask price, ask fills;
if candle low <= bid price, bid fills.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

D = Decimal
ZERO = D("0")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Candle:
    """OHLCV candle compatible with indicator compute methods."""
    open_time: float
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = ZERO


@dataclass
class SimOrder:
    side: str       # "buy" or "sell"
    price: Decimal
    size: Decimal


@dataclass
class Snapshot:
    timestamp: float
    candle_close: Decimal
    portfolio_value: Decimal
    base_balance: Decimal
    quote_balance: Decimal
    mid_price: Decimal
    num_fills_this_candle: int
    strategy_data: dict


# ---------------------------------------------------------------------------
# BacktestStrategy ABC
# ---------------------------------------------------------------------------


class BacktestStrategy(ABC):
    @abstractmethod
    def initialize(self, initial_price: Decimal, base_balance: Decimal,
                   quote_balance: Decimal):
        """Set up pool, indicators, initial state."""

    @abstractmethod
    def on_candle(self, candle: Candle, history: List[Candle],
                  sim_time: float) -> List[SimOrder]:
        """Process a new candle. Return new orders to place."""

    @abstractmethod
    def on_fill(self, side: str, price: Decimal, amount: Decimal,
                sim_time: float):
        """Handle a fill: update pool reserves, track balances."""

    @abstractmethod
    def get_snapshot(self) -> dict:
        """Return strategy-specific state for performance tracking.
        Must include: mid_price, base_balance, quote_balance."""

    @abstractmethod
    def get_portfolio_value(self, price: Decimal) -> Decimal:
        """Total portfolio value = base * price + quote."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy display name for reports."""


# ---------------------------------------------------------------------------
# CandleDataLoader
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


class CandleDataLoader:

    @staticmethod
    def load(symbol: str, interval: str, start: str, end: str,
             csv_path: Optional[str] = None) -> List[Candle]:
        if csv_path:
            return CandleDataLoader._load_csv(csv_path)

        start_ms = CandleDataLoader._date_to_ms(start)
        end_ms = CandleDataLoader._date_to_ms(end)

        cached = CandleDataLoader._check_cache(symbol, interval, start_ms, end_ms)
        if cached:
            print(f"Loaded {len(cached)} candles from cache")
            return cached

        candles = CandleDataLoader._fetch_binance(symbol, interval, start_ms, end_ms)
        if candles:
            CandleDataLoader._save_cache(symbol, interval, start_ms, end_ms, candles)
        return candles

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        import datetime
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _fetch_binance(symbol: str, interval: str,
                       start_ms: int, end_ms: int) -> List[Candle]:
        candles: List[Candle] = []
        current_start = start_ms

        while current_start < end_ms:
            url = (
                f"{BINANCE_KLINES_URL}"
                f"?symbol={symbol}&interval={interval}"
                f"&startTime={current_start}&endTime={end_ms}&limit=1000"
            )
            print(f"  Fetching {symbol} from {current_start}... "
                  f"({len(candles)} candles so far)")

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            if not data:
                break

            for row in data:
                candles.append(Candle(
                    open_time=row[0] / 1000.0,
                    open=D(str(row[1])),
                    high=D(str(row[2])),
                    low=D(str(row[3])),
                    close=D(str(row[4])),
                    volume=D(str(row[5])),
                ))

            current_start = int(data[-1][0]) + 1
            if len(data) < 1000:
                break

            time.sleep(0.1)

        print(f"  Fetched {len(candles)} candles from Binance")
        return candles

    @staticmethod
    def _load_csv(path: str) -> List[Candle]:
        candles: List[Candle] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                candles.append(Candle(
                    open_time=float(row["timestamp"]),
                    open=D(row["open"]),
                    high=D(row["high"]),
                    low=D(row["low"]),
                    close=D(row["close"]),
                    volume=D(row.get("volume", "0")),
                ))
        print(f"  Loaded {len(candles)} candles from {path}")
        return candles

    @staticmethod
    def _cache_path(symbol: str, interval: str,
                    start_ms: int, end_ms: int) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return os.path.join(
            CACHE_DIR,
            f"{symbol}_{interval}_{start_ms}_{end_ms}.csv",
        )

    @staticmethod
    def _check_cache(symbol: str, interval: str,
                     start_ms: int, end_ms: int) -> Optional[List[Candle]]:
        # Exact match
        path = CandleDataLoader._cache_path(symbol, interval, start_ms, end_ms)
        if os.path.exists(path):
            return CandleDataLoader._load_csv(path)

        # Search for a cached file that covers our range (superset)
        if not os.path.isdir(CACHE_DIR):
            return None
        prefix = f"{symbol}_{interval}_"
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        best = None
        best_size = float("inf")
        for fname in os.listdir(CACHE_DIR):
            if not fname.startswith(prefix) or not fname.endswith(".csv"):
                continue
            parts = fname[len(prefix):-4].split("_")
            if len(parts) != 2:
                continue
            try:
                f_start = int(parts[0])
                f_end = int(parts[1])
            except ValueError:
                continue
            if f_start <= start_ms and f_end >= end_ms:
                size = f_end - f_start
                if size < best_size:
                    best = os.path.join(CACHE_DIR, fname)
                    best_size = size

        if best is not None:
            all_candles = CandleDataLoader._load_csv(best)
            sliced = [c for c in all_candles
                      if start_sec <= c.open_time < end_sec]
            if sliced:
                print(f"  Sliced {len(sliced)} candles from superset cache")
                return sliced

        return None

    @staticmethod
    def _save_cache(symbol: str, interval: str,
                    start_ms: int, end_ms: int, candles: List[Candle]):
        path = CandleDataLoader._cache_path(symbol, interval, start_ms, end_ms)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for c in candles:
                writer.writerow([c.open_time, c.open, c.high, c.low, c.close, c.volume])
        print(f"  Cached {len(candles)} candles to {path}")


# ---------------------------------------------------------------------------
# FillSimulator
# ---------------------------------------------------------------------------


def check_fills(candle: Candle,
                orders: List[SimOrder]) -> Tuple[List[SimOrder], List[SimOrder]]:
    """Cross-based fill detection. Returns (remaining, filled)."""
    filled: List[SimOrder] = []
    remaining: List[SimOrder] = []
    for order in orders:
        if order.side == "sell" and candle.high >= order.price:
            filled.append(order)
        elif order.side == "buy" and candle.low <= order.price:
            filled.append(order)
        else:
            remaining.append(order)
    return remaining, filled


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------


class LightweightTracker:
    """Streaming metrics tracker — O(1) memory per candle.

    Computes all summary metrics incrementally without storing snapshots.
    Use this in sweep/HPC mode where memory matters (80 workers x 500K candles).
    """

    def __init__(self):
        self.initial_value: Optional[Decimal] = None
        self.initial_base: Optional[Decimal] = None
        self.initial_quote: Optional[Decimal] = None
        self._peak = ZERO
        self._max_dd = ZERO
        self._candle_count = 0
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._last_close = ZERO
        self._last_pv = ZERO
        # Sharpe: store hourly PV samples (1 per 60 candles for 1m data)
        self._hourly_pvs: List[float] = []
        self._candles_since_hourly = 0
        # Skew
        self._skew_sum = 0.0
        self._skew_count = 0
        self._max_abs_skew = 0.0
        # Fills
        self.buy_fills: List[Tuple[float, Decimal, Decimal]] = []
        self.sell_fills: List[Tuple[float, Decimal, Decimal]] = []
        # Strategy counters
        self._recenters = 0
        self._range_changes = 0
        self._trend_halts = 0

    def set_initial(self, value: Decimal, base: Decimal, quote: Decimal):
        self.initial_value = value
        self.initial_base = base
        self.initial_quote = quote
        self._peak = value

    def record_fill(self, side: str, price: Decimal, size: Decimal,
                    timestamp: float):
        if side == "buy":
            self.buy_fills.append((timestamp, price, size))
        else:
            self.sell_fills.append((timestamp, price, size))

    def record(self, snapshot: Snapshot):
        pv = snapshot.portfolio_value
        self._candle_count += 1
        self._last_close = snapshot.candle_close
        self._last_pv = pv

        if self._first_ts is None:
            self._first_ts = snapshot.timestamp
        self._last_ts = snapshot.timestamp

        # Max drawdown
        if pv > self._peak:
            self._peak = pv
        if self._peak > ZERO:
            dd = (self._peak - pv) / self._peak * D(100)
            if dd > self._max_dd:
                self._max_dd = dd

        # Hourly PV sample
        self._candles_since_hourly += 1
        if self._candles_since_hourly >= 60:
            self._hourly_pvs.append(float(pv))
            self._candles_since_hourly = 0

        # Skew
        skew = snapshot.strategy_data.get("inventory_skew")
        if skew is not None:
            self._skew_sum += skew
            self._skew_count += 1
            abs_skew = abs(skew)
            if abs_skew > self._max_abs_skew:
                self._max_abs_skew = abs_skew

        # Strategy counters
        if snapshot.strategy_data.get("recentered", False):
            self._recenters += 1
        if snapshot.strategy_data.get("range_changed", False):
            self._range_changes += 1
        if snapshot.strategy_data.get("trend_halted", False):
            self._trend_halts += 1

    def get_metrics(self) -> dict:
        if self._candle_count == 0:
            return {}

        initial_val = self.initial_value or ZERO
        final_val = self._last_pv

        total_return = (float((final_val - initial_val) / initial_val * D(100))
                        if initial_val > ZERO else 0.0)

        hold_value = ZERO
        if self.initial_base is not None and self.initial_quote is not None:
            hold_value = self.initial_base * self._last_close + self.initial_quote
        hold_return = (float((hold_value - initial_val) / initial_val * D(100))
                       if initial_val > ZERO else 0.0)

        sharpe = self._compute_sharpe()

        total_buys = len(self.buy_fills)
        total_sells = len(self.sell_fills)
        total_buy_vol = sum(s for _, _, s in self.buy_fills)
        total_sell_vol = sum(s for _, _, s in self.sell_fills)
        avg_spread_bps = self._avg_spread_captured()

        avg_skew = self._skew_sum / self._skew_count if self._skew_count else 0
        duration_hours = ((self._last_ts - self._first_ts) / 3600
                          if self._first_ts and self._last_ts else 0)

        return {
            "duration_hours": round(duration_hours, 1),
            "candles": self._candle_count,
            "initial_value": float(initial_val),
            "final_value": float(final_val),
            "total_return_pct": round(total_return, 3),
            "hold_return_pct": round(hold_return, 3),
            "excess_return_pct": round(total_return - hold_return, 3),
            "max_drawdown_pct": round(float(self._max_dd), 3),
            "sharpe": round(sharpe, 3),
            "total_fills": total_buys + total_sells,
            "buy_fills": total_buys,
            "sell_fills": total_sells,
            "buy_volume": float(total_buy_vol),
            "sell_volume": float(total_sell_vol),
            "avg_spread_bps": round(avg_spread_bps, 1),
            "avg_inv_skew": round(avg_skew, 4),
            "max_abs_inv_skew": round(self._max_abs_skew, 4),
            "recenters": self._recenters,
            "range_changes": self._range_changes,
            "trend_halts": self._trend_halts,
        }

    def _compute_sharpe(self) -> float:
        if len(self._hourly_pvs) < 2:
            return 0.0
        returns = []
        for i in range(1, len(self._hourly_pvs)):
            if self._hourly_pvs[i - 1] > 0:
                returns.append(
                    self._hourly_pvs[i] / self._hourly_pvs[i - 1] - 1.0)
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.0
        if std_r == 0:
            return 0.0
        return (mean_r / std_r) * math.sqrt(8760)

    def _avg_spread_captured(self) -> float:
        if not self.buy_fills or not self.sell_fills:
            return 0.0
        all_fills = (
            [(t, "buy", p, s) for t, p, s in self.buy_fills]
            + [(t, "sell", p, s) for t, p, s in self.sell_fills]
        )
        all_fills.sort(key=lambda x: x[0])
        spreads: List[float] = []
        last_buy: Optional[Decimal] = None
        last_sell: Optional[Decimal] = None
        for _, side, price, _ in all_fills:
            if side == "buy":
                if last_sell is not None:
                    mid = (last_sell + price) / D(2)
                    if mid > ZERO:
                        spreads.append(
                            float((last_sell - price) / mid * D(10000)))
                last_buy = price
            else:
                if last_buy is not None:
                    mid = (price + last_buy) / D(2)
                    if mid > ZERO:
                        spreads.append(
                            float((price - last_buy) / mid * D(10000)))
                last_sell = price
        return sum(spreads) / len(spreads) if spreads else 0.0


class PerformanceTracker:

    def __init__(self):
        self.snapshots: List[Snapshot] = []
        self.buy_fills: List[Tuple[float, Decimal, Decimal]] = []
        self.sell_fills: List[Tuple[float, Decimal, Decimal]] = []
        self.initial_value: Optional[Decimal] = None
        self.initial_base: Optional[Decimal] = None
        self.initial_quote: Optional[Decimal] = None

    def set_initial(self, value: Decimal, base: Decimal, quote: Decimal):
        self.initial_value = value
        self.initial_base = base
        self.initial_quote = quote

    def record_fill(self, side: str, price: Decimal, size: Decimal,
                    timestamp: float):
        if side == "buy":
            self.buy_fills.append((timestamp, price, size))
        else:
            self.sell_fills.append((timestamp, price, size))

    def record(self, snapshot: Snapshot):
        self.snapshots.append(snapshot)

    def get_metrics(self) -> dict:
        """Compute all summary metrics and return as a dict."""
        if not self.snapshots:
            return {}

        first = self.snapshots[0]
        last = self.snapshots[-1]
        initial_val = self.initial_value or first.portfolio_value
        final_val = last.portfolio_value

        total_return = (float((final_val - initial_val) / initial_val * D(100))
                        if initial_val > ZERO else 0.0)

        hold_value = ZERO
        if self.initial_base is not None and self.initial_quote is not None:
            hold_value = self.initial_base * last.candle_close + self.initial_quote
        hold_return = (float((hold_value - initial_val) / initial_val * D(100))
                       if initial_val > ZERO else 0.0)

        peak = initial_val
        max_dd = ZERO
        for s in self.snapshots:
            if s.portfolio_value > peak:
                peak = s.portfolio_value
            dd = ((peak - s.portfolio_value) / peak * D(100)
                  if peak > ZERO else ZERO)
            if dd > max_dd:
                max_dd = dd

        sharpe = self._compute_sharpe()

        total_buys = len(self.buy_fills)
        total_sells = len(self.sell_fills)
        total_buy_vol = sum(s for _, _, s in self.buy_fills)
        total_sell_vol = sum(s for _, _, s in self.sell_fills)

        avg_spread_bps = self._avg_spread_captured()

        skews = [s.strategy_data.get("inventory_skew", 0)
                 for s in self.snapshots
                 if "inventory_skew" in s.strategy_data]
        avg_skew = sum(skews) / len(skews) if skews else 0
        max_abs_skew = max((abs(s) for s in skews), default=0)

        duration_hours = (last.timestamp - first.timestamp) / 3600

        recenters = sum(1 for s in self.snapshots
                        if s.strategy_data.get("recentered", False))
        range_changes = sum(1 for s in self.snapshots
                            if s.strategy_data.get("range_changed", False))
        trend_halts = sum(1 for s in self.snapshots
                          if s.strategy_data.get("trend_halted", False))

        return {
            "duration_hours": round(duration_hours, 1),
            "candles": len(self.snapshots),
            "initial_value": float(initial_val),
            "final_value": float(final_val),
            "total_return_pct": round(total_return, 3),
            "hold_return_pct": round(hold_return, 3),
            "excess_return_pct": round(total_return - hold_return, 3),
            "max_drawdown_pct": round(float(max_dd), 3),
            "sharpe": round(sharpe, 3),
            "total_fills": total_buys + total_sells,
            "buy_fills": total_buys,
            "sell_fills": total_sells,
            "buy_volume": float(total_buy_vol),
            "sell_volume": float(total_sell_vol),
            "avg_spread_bps": round(avg_spread_bps, 1),
            "avg_inv_skew": round(avg_skew, 4),
            "max_abs_inv_skew": round(max_abs_skew, 4),
            "recenters": recenters,
            "range_changes": range_changes,
            "trend_halts": trend_halts,
        }

    def print_summary(self):
        m = self.get_metrics()
        if not m:
            print("No data to summarize.")
            return

        print("\n" + "=" * 70)
        print("  BACKTEST SUMMARY")
        print("=" * 70)
        print(f"  Duration:           {m['duration_hours']:.1f} hours "
              f"({m['duration_hours'] / 24:.1f} days)")
        print(f"  Candles:            {m['candles']}")
        print(f"  Initial value:      {m['initial_value']:.2f}")
        print(f"  Final value:        {m['final_value']:.2f}")
        print(f"  Strategy return:    {m['total_return_pct']:+.2f}%")
        print(f"  Hold-only return:   {m['hold_return_pct']:+.2f}%")
        print(f"  Excess return:      {m['excess_return_pct']:+.2f}%")
        print(f"  Max drawdown:       {m['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe (annual):    {m['sharpe']:.3f}")
        print("-" * 70)
        print(f"  Total fills:        {m['total_fills']} "
              f"({m['buy_fills']}B / {m['sell_fills']}S)")
        print(f"  Buy volume:         {m['buy_volume']:.2f}")
        print(f"  Sell volume:        {m['sell_volume']:.2f}")
        print(f"  Avg spread (bps):   {m['avg_spread_bps']:.1f}")
        print(f"  Avg inv. skew:      {m['avg_inv_skew']:+.3f}")
        print(f"  Max |inv. skew|:    {m['max_abs_inv_skew']:.3f}")

        if m['recenters'] > 0 or m['range_changes'] > 0:
            print(f"  Recenters:          {m['recenters']}")
            print(f"  Range changes:      {m['range_changes']}")

        if m.get('trend_halts', 0) > 0:
            print(f"  Trend halts:        {m['trend_halts']}")

        print("=" * 70)

    def _compute_sharpe(self) -> float:
        if len(self.snapshots) < 61:
            return 0.0

        hourly_returns: List[float] = []
        step = 60
        for i in range(step, len(self.snapshots), step):
            prev_val = float(self.snapshots[i - step].portfolio_value)
            curr_val = float(self.snapshots[i].portfolio_value)
            if prev_val > 0:
                hourly_returns.append(curr_val / prev_val - 1.0)

        if len(hourly_returns) < 2:
            return 0.0

        mean_r = sum(hourly_returns) / len(hourly_returns)
        var_r = sum((r - mean_r) ** 2 for r in hourly_returns) / (len(hourly_returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 0.0

        if std_r == 0:
            return 0.0
        return (mean_r / std_r) * math.sqrt(8760)

    def _avg_spread_captured(self) -> float:
        """Estimate avg spread by pairing buy/sell fills chronologically."""
        if not self.buy_fills or not self.sell_fills:
            return 0.0

        all_fills = (
            [(t, "buy", p, s) for t, p, s in self.buy_fills]
            + [(t, "sell", p, s) for t, p, s in self.sell_fills]
        )
        all_fills.sort(key=lambda x: x[0])

        spreads: List[float] = []
        last_buy_price: Optional[Decimal] = None
        last_sell_price: Optional[Decimal] = None

        for _, side, price, _ in all_fills:
            if side == "buy":
                if last_sell_price is not None:
                    mid = (last_sell_price + price) / D(2)
                    if mid > ZERO:
                        spreads.append(
                            float((last_sell_price - price) / mid * D(10000)))
                last_buy_price = price
            else:
                if last_buy_price is not None:
                    mid = (price + last_buy_price) / D(2)
                    if mid > ZERO:
                        spreads.append(
                            float((price - last_buy_price) / mid * D(10000)))
                last_sell_price = price

        return sum(spreads) / len(spreads) if spreads else 0.0

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            # Exclude keys that duplicate snapshot-level fields
            skip = {"base_balance", "quote_balance", "mid_price"}
            all_keys: set = set()
            for s in self.snapshots:
                all_keys.update(s.strategy_data.keys())
            extra_keys = sorted(all_keys - skip)

            writer = csv.writer(f)
            header = [
                "timestamp", "candle_close", "portfolio_value",
                "base_balance", "quote_balance", "mid_price", "fills",
            ] + extra_keys
            writer.writerow(header)

            for s in self.snapshots:
                row = [
                    s.timestamp, s.candle_close, s.portfolio_value,
                    s.base_balance, s.quote_balance, s.mid_price,
                    s.num_fills_this_candle,
                ]
                for k in extra_keys:
                    row.append(s.strategy_data.get(k, ""))
                writer.writerow(row)

        print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# TrendValidator — evaluates indicator prediction quality
# ---------------------------------------------------------------------------


class TrendValidator:
    """Collects per-candle indicator readings and validates them against
    realised forward returns at multiple horizons.

    After the backtest loop, call ``compute()`` to produce accuracy and
    correlation metrics.
    """

    # Forward-return horizons (in candles — 1-minute data)
    HORIZONS = {"15m": 15, "1h": 60, "4h": 240}

    def __init__(self):
        # Parallel lists — one entry per candle
        self._closes: List[float] = []
        self._effective_trend: List[float] = []
        self._adx: List[float] = []
        self._hurst: List[float] = []
        self._natr: List[float] = []
        self._hmm_regime: List[str] = []  # "ranging", "trending", "volatile", "n/a"
        self._timestamps: List[float] = []

    # -- recording --

    def record(self, timestamp: float, close: float, strategy_data: dict):
        self._timestamps.append(timestamp)
        self._closes.append(close)
        self._effective_trend.append(
            float(strategy_data.get("effective_trend", 0)))
        self._adx.append(float(strategy_data.get("adx", 0)))
        self._hurst.append(float(strategy_data.get("hurst", 0)))
        self._natr.append(float(strategy_data.get("natr", 0)))

        hmm_raw = strategy_data.get("hmm_regime", "n/a")
        # hmm_regime format: "ranging(85%)" — extract label
        regime = hmm_raw.split("(")[0] if "(" in str(hmm_raw) else str(hmm_raw)
        self._hmm_regime.append(regime)

    # -- analysis --

    def compute(self) -> dict:
        """Compute validation metrics. Returns a dict suitable for printing
        and/or merging into the main metrics dict."""
        n = len(self._closes)
        if n < 100:
            return {}

        results: dict = {}

        for label, horizon in self.HORIZONS.items():
            if n <= horizon:
                continue

            # Forward absolute return (%) for each candle
            fwd_abs = []  # |r|
            fwd_dir = []  # sign of return: +1 / -1
            for i in range(n - horizon):
                r = (self._closes[i + horizon] - self._closes[i]) / self._closes[i]
                fwd_abs.append(abs(r) * 100)
                fwd_dir.append(1.0 if r >= 0 else -1.0)

            m = len(fwd_abs)

            # 1. effective_trend vs forward |return| — Pearson correlation
            et = self._effective_trend[:m]
            corr_et = self._pearson(et, fwd_abs)
            results[f"corr_eff_trend_vs_fwd_abs_{label}"] = round(corr_et, 4)

            # 2. ADX vs forward |return| correlation
            corr_adx = self._pearson(self._adx[:m], fwd_abs)
            results[f"corr_adx_vs_fwd_abs_{label}"] = round(corr_adx, 4)

            # 3. Hurst accuracy — when hurst > 0.5 (trending), is forward
            #    |return| higher than when hurst < 0.5 (mean-reverting)?
            hurst_trending_abs = [
                fwd_abs[i] for i in range(m) if self._hurst[i] > 0.5]
            hurst_reverting_abs = [
                fwd_abs[i] for i in range(m) if 0 < self._hurst[i] <= 0.5]
            if hurst_trending_abs and hurst_reverting_abs:
                avg_t = sum(hurst_trending_abs) / len(hurst_trending_abs)
                avg_r = sum(hurst_reverting_abs) / len(hurst_reverting_abs)
                results[f"hurst_trend_avg_fwd_abs_{label}"] = round(avg_t, 4)
                results[f"hurst_revert_avg_fwd_abs_{label}"] = round(avg_r, 4)
                results[f"hurst_ratio_{label}"] = (
                    round(avg_t / avg_r, 3) if avg_r > 0 else 0)

            # 4. HMM regime accuracy — avg forward |return| per regime
            regime_buckets: Dict[str, List[float]] = {}
            for i in range(m):
                reg = self._hmm_regime[i]
                if reg == "n/a":
                    continue
                regime_buckets.setdefault(reg, []).append(fwd_abs[i])
            for reg, vals in sorted(regime_buckets.items()):
                avg = sum(vals) / len(vals) if vals else 0
                results[f"hmm_{reg}_avg_fwd_abs_{label}"] = round(avg, 4)
                results[f"hmm_{reg}_count_{label}"] = len(vals)

            # 5. effective_trend buckets — low/mid/high
            buckets = {"low": [], "mid": [], "high": []}
            for i in range(m):
                v = et[i]
                if v < 0.3:
                    buckets["low"].append(fwd_abs[i])
                elif v < 0.7:
                    buckets["mid"].append(fwd_abs[i])
                else:
                    buckets["high"].append(fwd_abs[i])
            for bk, vals in buckets.items():
                avg = sum(vals) / len(vals) if vals else 0
                results[f"eff_trend_{bk}_avg_fwd_abs_{label}"] = round(avg, 4)
                results[f"eff_trend_{bk}_count_{label}"] = len(vals)

            # 6. NATR vs realised volatility (forward std of returns)
            if m > 1:
                fwd_std = []
                for i in range(min(m, n - horizon)):
                    chunk = self._closes[i:i + horizon]
                    if len(chunk) < 2:
                        continue
                    rets = [(chunk[j] - chunk[j - 1]) / chunk[j - 1]
                            for j in range(1, len(chunk))]
                    mu = sum(rets) / len(rets)
                    var = sum((r - mu) ** 2 for r in rets) / len(rets)
                    fwd_std.append(math.sqrt(var))
                if fwd_std:
                    corr_natr = self._pearson(
                        self._natr[:len(fwd_std)], fwd_std)
                    results[f"corr_natr_vs_fwd_vol_{label}"] = round(
                        corr_natr, 4)

        return results

    def print_summary(self, metrics: dict):
        if not metrics:
            print("\nInsufficient data for indicator validation.")
            return

        print("\n" + "=" * 70)
        print("  INDICATOR VALIDATION")
        print("=" * 70)

        for label in self.HORIZONS:
            # Correlations
            keys_for_horizon = [k for k in metrics if k.endswith(f"_{label}")]
            if not keys_for_horizon:
                continue
            print(f"\n  --- Forward horizon: {label} ---")

            corr_et = metrics.get(f"corr_eff_trend_vs_fwd_abs_{label}")
            corr_adx = metrics.get(f"corr_adx_vs_fwd_abs_{label}")
            corr_natr = metrics.get(f"corr_natr_vs_fwd_vol_{label}")
            if corr_et is not None:
                print(f"  effective_trend ↔ |fwd return|:  r={corr_et:+.4f}")
            if corr_adx is not None:
                print(f"  ADX ↔ |fwd return|:              r={corr_adx:+.4f}")
            if corr_natr is not None:
                print(f"  NATR ↔ fwd volatility:           r={corr_natr:+.4f}")

            # Hurst ratio
            ratio = metrics.get(f"hurst_ratio_{label}")
            if ratio is not None:
                ht = metrics.get(f"hurst_trend_avg_fwd_abs_{label}", 0)
                hr = metrics.get(f"hurst_revert_avg_fwd_abs_{label}", 0)
                print(f"  Hurst trending avg |fwd|:        {ht:.4f}%")
                print(f"  Hurst reverting avg |fwd|:       {hr:.4f}%")
                print(f"  Hurst ratio (trend/revert):      {ratio:.3f}x")

            # effective_trend buckets
            for bk in ("low", "mid", "high"):
                avg = metrics.get(f"eff_trend_{bk}_avg_fwd_abs_{label}")
                cnt = metrics.get(f"eff_trend_{bk}_count_{label}", 0)
                if avg is not None:
                    print(f"  eff_trend {bk:4s}  avg|fwd|={avg:.4f}%  "
                          f"n={cnt}")

            # HMM regimes
            hmm_keys = sorted(
                k for k in keys_for_horizon
                if k.startswith("hmm_") and "_avg_" in k)
            for k in hmm_keys:
                reg = k.replace("hmm_", "").replace(
                    f"_avg_fwd_abs_{label}", "")
                avg = metrics[k]
                cnt = metrics.get(f"hmm_{reg}_count_{label}", 0)
                print(f"  HMM {reg:10s}  avg|fwd|={avg:.4f}%  n={cnt}")

        print("=" * 70)

    def save_csv(self, path: str):
        """Save per-candle indicator + forward return data for external analysis."""
        n = len(self._closes)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp", "close", "effective_trend", "adx",
                      "hurst", "natr", "hmm_regime"]
            for label, h in self.HORIZONS.items():
                header.append(f"fwd_return_{label}")
                header.append(f"fwd_abs_return_{label}")
            writer.writerow(header)

            for i in range(n):
                row = [
                    self._timestamps[i], self._closes[i],
                    self._effective_trend[i], self._adx[i],
                    self._hurst[i], self._natr[i], self._hmm_regime[i],
                ]
                for label, h in self.HORIZONS.items():
                    if i + h < n:
                        r = ((self._closes[i + h] - self._closes[i])
                             / self._closes[i] * 100)
                        row.append(round(r, 6))
                        row.append(round(abs(r), 6))
                    else:
                        row.extend(["", ""])
                writer.writerow(row)

        print(f"Indicator validation data saved to {path}")

    # -- helpers --

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> float:
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        sx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
        sy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
        if sx == 0 or sy == 0:
            return 0.0
        return cov / (sx * sy)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

_real_time_fn = time.time


class BacktestEngine:

    def __init__(self, strategy: BacktestStrategy, candles: List[Candle],
                 base_balance: Decimal, quote_balance: Decimal,
                 output_dir: str = "results", charts: bool = True,
                 quiet: bool = False, lightweight: bool = False,
                 validate_indicators: bool = False):
        self.strategy = strategy
        self.candles = candles
        self.base_balance = base_balance
        self.quote_balance = quote_balance
        self.output_dir = output_dir
        self.charts = charts
        self.quiet = quiet
        self.tracker = LightweightTracker() if lightweight else PerformanceTracker()
        self.active_orders: List[SimOrder] = []
        self.validator = TrendValidator() if validate_indicators else None

    def run(self) -> dict:
        """Run backtest. Returns metrics dict."""
        if not self.candles:
            if not self.quiet:
                print("No candles to backtest.")
            return {}

        initial_price = self.candles[0].close
        self.strategy.initialize(initial_price, self.base_balance,
                                 self.quote_balance)

        initial_value = self.strategy.get_portfolio_value(initial_price)
        self.tracker.set_initial(initial_value, self.base_balance,
                                 self.quote_balance)

        history: List[Candle] = []
        total = len(self.candles)
        report_interval = max(1, total // 20)

        for i, candle in enumerate(self.candles):
            sim_time = float(candle.open_time)
            history.append(candle)

            # Monkey-patch time.time for indicator rate-limiting
            time.time = lambda _t=sim_time: _t

            try:
                # 1. Check fills against active orders
                self.active_orders, filled = check_fills(
                    candle, self.active_orders)
                for fill in filled:
                    self.strategy.on_fill(fill.side, fill.price, fill.size,
                                          sim_time)
                    self.tracker.record_fill(fill.side, fill.price, fill.size,
                                             sim_time)

                # 2. Strategy processes candle -> returns new orders
                new_orders = self.strategy.on_candle(candle, history, sim_time)
                self.active_orders = new_orders

                # 3. Record snapshot
                snap_data = self.strategy.get_snapshot()
                pv = self.strategy.get_portfolio_value(candle.close)
                snapshot = Snapshot(
                    timestamp=sim_time,
                    candle_close=candle.close,
                    portfolio_value=pv,
                    base_balance=snap_data.get("base_balance", ZERO),
                    quote_balance=snap_data.get("quote_balance", ZERO),
                    mid_price=snap_data.get("mid_price", candle.close),
                    num_fills_this_candle=len(filled),
                    strategy_data=snap_data,
                )
                self.tracker.record(snapshot)

                # Record indicator values for validation
                if self.validator is not None:
                    self.validator.record(
                        sim_time, float(candle.close), snap_data)

                # Early stop: portfolio wiped out
                if pv <= ZERO:
                    if not self.quiet:
                        print(f"  STOPPED: portfolio value <= 0 at "
                              f"candle {i + 1}/{total}")
                    break
            finally:
                time.time = _real_time_fn

            if not self.quiet and (i + 1) % report_interval == 0:
                pct = (i + 1) / total * 100
                print(f"  [{pct:5.1f}%] candle {i + 1}/{total} "
                      f"| price={candle.close} | PV={pv:.2f}")

        metrics = self.tracker.get_metrics()

        if not self.quiet:
            self.tracker.print_summary()

            # Save CSV
            start_date = self._ts_to_date(self.candles[0].open_time)
            end_date = self._ts_to_date(self.candles[-1].open_time)
            csv_name = (f"backtest_{self.strategy.name}"
                        f"_{start_date}_{end_date}.csv")
            csv_path = os.path.join(self.output_dir, csv_name)
            self.tracker.save_csv(csv_path)

            if self.charts:
                self._plot_charts()

        # Indicator validation
        if self.validator is not None:
            val_metrics = self.validator.compute()
            if val_metrics:
                metrics.update(val_metrics)
                if not self.quiet:
                    self.validator.print_summary(val_metrics)
                    start_date = self._ts_to_date(self.candles[0].open_time)
                    end_date = self._ts_to_date(self.candles[-1].open_time)
                    val_csv = os.path.join(
                        self.output_dir,
                        f"indicator_validation_{self.strategy.name}"
                        f"_{start_date}_{end_date}.csv")
                    self.validator.save_csv(val_csv)

        return metrics

    def _ts_to_date(self, ts: float) -> str:
        import datetime
        return datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc).strftime("%Y%m%d")

    def _plot_charts(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import datetime
        except ImportError:
            print("matplotlib not available, skipping charts.")
            return

        timestamps = [datetime.datetime.fromtimestamp(
                          s.timestamp, tz=datetime.timezone.utc)
                      for s in self.tracker.snapshots]
        portfolio_values = [float(s.portfolio_value)
                            for s in self.tracker.snapshots]

        # Hold-only baseline
        hold_values = None
        if self.tracker.initial_base is not None:
            hold_values = [
                float(self.tracker.initial_base * s.candle_close
                      + self.tracker.initial_quote)
                for s in self.tracker.snapshots
            ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Backtest: {self.strategy.name}", fontsize=14)

        # 1. Portfolio value
        ax1 = axes[0]
        ax1.plot(timestamps, portfolio_values, label="Strategy", linewidth=1)
        if hold_values:
            ax1.plot(timestamps, hold_values, label="Hold-only",
                     linewidth=1, alpha=0.7)
        ax1.set_ylabel("Portfolio Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Price with concentration range overlay
        ax2 = axes[1]
        closes = [float(s.candle_close) for s in self.tracker.snapshots]
        ax2.plot(timestamps, closes, label="Close", linewidth=0.8, color="gray")
        p_lowers = [float(s.strategy_data.get("p_lower", 0))
                     for s in self.tracker.snapshots]
        p_uppers = [float(s.strategy_data.get("p_upper", 0))
                     for s in self.tracker.snapshots]
        if any(v > 0 for v in p_lowers):
            ax2.fill_between(timestamps, p_lowers, p_uppers,
                             alpha=0.2, color="blue", label="Range")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Indicators
        ax3 = axes[2]
        natrs = [float(s.strategy_data.get("natr", 0)) * 100
                 for s in self.tracker.snapshots]
        if any(v > 0 for v in natrs):
            ax3.plot(timestamps, natrs, label="NATR %",
                     linewidth=0.8, color="orange")
        skews = [float(s.strategy_data.get("inventory_skew", 0))
                 for s in self.tracker.snapshots]
        ax3.plot(timestamps, skews, label="Inv. Skew",
                 linewidth=0.8, color="purple")
        ax3.set_ylabel("Indicators")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        plt.tight_layout()
        chart_path = os.path.join(
            self.output_dir,
            f"backtest_{self.strategy.name}_"
            f"{self._ts_to_date(self.tracker.snapshots[0].timestamp)}.png",
        )
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generic Strategy Backtester with Binance Historical Data")

    parser.add_argument("--strategy", required=True, choices=["cl-amm", "amm"],
                        help="Strategy to backtest")
    parser.add_argument("--symbol", default="ADAUSDT",
                        help="Binance symbol (default: ADAUSDT)")
    parser.add_argument("--interval", default="1m",
                        help="Candle interval (default: 1m)")
    parser.add_argument("--start", required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--base-balance", type=Decimal, default=D("100000"),
                        help="Initial base balance (default: 100000)")
    parser.add_argument("--quote-balance", type=Decimal, default=D("27000"),
                        help="Initial quote balance (default: 27000)")
    parser.add_argument("--csv", default=None,
                        help="Load candles from CSV instead of Binance API")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip matplotlib chart generation")
    parser.add_argument("--validate-indicators", action="store_true",
                        help="Evaluate indicator prediction accuracy")

    # Shared strategy params
    parser.add_argument("--spread-bps", type=Decimal, default=D("40"),
                        help="Base spread in basis points (default: 40)")
    parser.add_argument("--pool-depth", type=Decimal, default=None,
                        help="Pool depth override (default: base*price + quote)")
    parser.add_argument("--pool-price-weight", type=Decimal, default=D("0.70"),
                        help="Pool vs anchor blend weight (default: 0.70)")
    parser.add_argument("--anchor-ema-alpha", type=Decimal, default=D("0.05"),
                        help="Anchor EMA smoothing (default: 0.05)")
    parser.add_argument("--num-levels", type=int, default=1,
                        help="Number of order levels (default: 1)")
    parser.add_argument("--size-decay", type=Decimal, default=D("0.85"),
                        help="Size decay per level (default: 0.85)")
    parser.add_argument("--spread-multiplier", type=Decimal, default=D("1.5"),
                        help="Spread multiplier per level (default: 1.5)")
    parser.add_argument("--order-safe-ratio", type=Decimal, default=D("0.5"),
                        help="Order size as fraction of max safe (default: 0.5)")
    parser.add_argument("--enable-asymmetric-spread",
                        action="store_true", default=True)
    parser.add_argument("--no-asymmetric-spread",
                        dest="enable_asymmetric_spread", action="store_false")
    parser.add_argument("--skew-sensitivity", type=Decimal, default=D("0.5"))
    parser.add_argument("--min-spread-bps", type=Decimal, default=D("20"))

    # CL-AMM specific
    cl = parser.add_argument_group("CL-AMM options")
    cl.add_argument("--concentration", type=Decimal, default=D("5"),
                    help="Base concentration %% (default: 5)")
    cl.add_argument("--min-concentration", type=Decimal, default=D("5"))
    cl.add_argument("--max-concentration", type=Decimal, default=D("20"))
    cl.add_argument("--natr-period", type=int, default=14)
    cl.add_argument("--natr-baseline", type=Decimal, default=D("0.005"))
    cl.add_argument("--natr-range-scale", type=Decimal, default=D("1.0"))
    cl.add_argument("--adx-period", type=int, default=14)
    cl.add_argument("--hurst-min-candles", type=int, default=100)
    cl.add_argument("--hmm-n-states", type=int, default=3)
    cl.add_argument("--hmm-min-candles", type=int, default=200)
    cl.add_argument("--hmm-refit-interval-sec", type=int, default=1800)
    cl.add_argument("--hmm-window", type=int, default=500)
    cl.add_argument("--hmm-confidence-threshold", type=Decimal, default=D("0.80"))
    cl.add_argument("--trend-sensitivity", type=Decimal, default=D("0.5"))
    cl.add_argument("--range-ema-alpha", type=Decimal, default=D("0.1"))
    cl.add_argument("--range-update-dead-band-pct", type=Decimal, default=D("0.5"))
    cl.add_argument("--soft-recenter-drift-pct", type=Decimal, default=D("2.0"))
    cl.add_argument("--trend-order-scale-factor", type=Decimal, default=D("0.0"))
    cl.add_argument("--trend-halt-threshold", type=Decimal, default=D("0.0"))

    # AMM specific
    amm = parser.add_argument_group("AMM options")
    amm.add_argument("--amplification", type=Decimal, default=D("5"),
                     help="AMM amplification factor (default: 5)")

    args = parser.parse_args()

    # Load candle data
    print(f"\nLoading {args.symbol} {args.interval} candles: "
          f"{args.start} to {args.end}")
    candles = CandleDataLoader.load(
        args.symbol, args.interval, args.start, args.end, args.csv)
    if not candles:
        print("No candles loaded. Exiting.")
        sys.exit(1)
    print(f"  {len(candles)} candles "
          f"({candles[0].close} -> {candles[-1].close})")

    # Build strategy
    from backtest_strategies import CLAMMBacktestStrategy, AMMBacktestStrategy

    STRATEGIES = {
        "cl-amm": CLAMMBacktestStrategy,
        "amm": AMMBacktestStrategy,
    }

    config = vars(args)
    strategy = STRATEGIES[args.strategy](**config)

    print(f"\nRunning {strategy.name} backtest...")
    engine = BacktestEngine(
        strategy=strategy,
        candles=candles,
        base_balance=args.base_balance,
        quote_balance=args.quote_balance,
        output_dir=args.output,
        charts=not args.no_charts,
        validate_indicators=args.validate_indicators,
    )
    engine.run()


if __name__ == "__main__":
    main()
