#!/usr/bin/env python3
"""Pre-download candle data from Binance and cache locally.

Usage:
    python download_candles.py ADAUSDT 2020-01-01 2025-03-19
    python download_candles.py ADAUSDC 2020-01-01 2025-03-19
    python download_candles.py BTCUSDT 2020-01-01 2025-03-19 --interval 1m
"""

import argparse
import os
import sys

# Disable SSL verification if needed (VPN/proxy issues)
if os.environ.get("PYTHONHTTPSVERIFY") == "0":
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

from backtest_engine import CandleDataLoader, CACHE_DIR


def main():
    parser = argparse.ArgumentParser(description="Download Binance candle data")
    parser.add_argument("symbol", help="e.g. ADAUSDT, ADAUSDC, BTCUSDT")
    parser.add_argument("start", help="Start date YYYY-MM-DD")
    parser.add_argument("end", help="End date YYYY-MM-DD")
    parser.add_argument("--interval", default="1m")
    args = parser.parse_args()

    print(f"Downloading {args.symbol} {args.interval} candles: "
          f"{args.start} to {args.end}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    candles = CandleDataLoader.load(
        args.symbol, args.interval, args.start, args.end)

    if candles:
        print(f"\nDone: {len(candles)} candles")
        print(f"  First: {candles[0].open} @ {candles[0].open_time}")
        print(f"  Last:  {candles[-1].close} @ {candles[-1].open_time}")

        size_mb = os.path.getsize(
            CandleDataLoader._cache_path(
                args.symbol, args.interval,
                CandleDataLoader._date_to_ms(args.start),
                CandleDataLoader._date_to_ms(args.end),
            )) / 1024 / 1024
        print(f"  Cache file: {size_mb:.1f} MB")
    else:
        print("No candles returned. Check symbol and date range.")


if __name__ == "__main__":
    main()
