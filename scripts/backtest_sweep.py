#!/usr/bin/env python3
"""HPC-optimized parameter sweep for strategy backtesting.

Designed for 80-core HPC: shared-memory candle data, lightweight metrics,
range syntax for large grids, progress with ETA.

Usage:
    # Explicit values
    python backtest_sweep.py --strategy cl-amm --symbol ADAUSDT \
        --start 2025-01-01 --end 2025-03-01 \
        --sweep spread_bps=20,40,60,80 concentration=3,5,10,15 \
        --workers 80

    # Range syntax:  param=start:stop:step
    python backtest_sweep.py --strategy amm --symbol ADAUSDT \
        --start 2025-01-01 --end 2025-03-01 \
        --sweep spread_bps=10:100:5 amplification=2:30:2 \
        --workers 80

    # Mixed: range + explicit + fixed overrides
    python backtest_sweep.py --strategy cl-amm --symbol ADAUSDT \
        --start 2025-01-01 --end 2025-03-01 \
        --sweep spread_bps=10:100:10 concentration=3,5,10,15,20 \
        --set pool_price_weight=0.5 anchor_ema_alpha=0.03 \
        --workers 80

    # Dry run: preview grid without running
    python backtest_sweep.py --strategy cl-amm ... --dry-run

    # Sort and filter
    python backtest_sweep.py ... --sort-by sharpe --top 20
"""

import argparse
import csv
import itertools
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

D = Decimal


# ---------------------------------------------------------------------------
# Shared state: candle cache path loaded once per worker via initializer
# ---------------------------------------------------------------------------

_WORKER_CANDLES = None
_WORKER_CACHE_ARGS = None


def _worker_init(symbol: str, interval: str, start: str, end: str,
                 csv_path: Optional[str]):
    """Called once per worker process. Loads candles from disk cache."""
    global _WORKER_CANDLES, _WORKER_CACHE_ARGS
    _WORKER_CACHE_ARGS = (symbol, interval, start, end, csv_path)

    # Suppress print output during worker load
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from backtest_engine import CandleDataLoader
        _WORKER_CANDLES = CandleDataLoader.load(
            symbol, interval, start, end, csv_path)
    finally:
        sys.stdout = old_stdout


def _run_single(task: Tuple[int, str, dict, Decimal, Decimal]) -> dict:
    """Run a single backtest using shared candle data. Returns params + metrics."""
    task_idx, strategy_name, params, base_bal, quote_bal = task

    from backtest_engine import BacktestEngine
    from backtest_strategies import CLAMMBacktestStrategy, AMMBacktestStrategy
    import time as time_mod

    STRATEGIES = {
        "cl-amm": CLAMMBacktestStrategy,
        "amm": AMMBacktestStrategy,
    }

    strategy = STRATEGIES[strategy_name](**params)
    engine = BacktestEngine(
        strategy=strategy,
        candles=_WORKER_CANDLES,
        base_balance=base_bal,
        quote_balance=quote_bal,
        quiet=True,
        lightweight=True,
    )

    t0 = time_mod.time()
    metrics = engine.run()
    elapsed = time_mod.time() - t0

    result = {"_task_idx": task_idx, "_elapsed_sec": round(elapsed, 1)}
    # Only include swept/interesting params, not all defaults
    skip = {"strategy", "symbol", "interval", "start", "end",
            "csv", "output", "no_charts"}
    for k, v in params.items():
        if k not in skip:
            result[f"p_{k}"] = v
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------


def _parse_value(v: str):
    """Auto-detect int / Decimal / str."""
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return D(v)
    except Exception:
        return v


def parse_sweep_args(sweep_strs: List[str]) -> Dict[str, List]:
    """Parse sweep specs. Supports:
      - param=val1,val2,val3       (explicit values)
      - param=start:stop:step      (numeric range, inclusive)
    """
    sweeps = {}
    for s in sweep_strs:
        if "=" not in s:
            print(f"Invalid sweep format: {s}")
            print("  Expected: param=val1,val2,... or param=start:stop:step")
            sys.exit(1)
        key, vals_str = s.split("=", 1)

        # Range syntax: start:stop:step
        if ":" in vals_str and "," not in vals_str:
            parts = vals_str.split(":")
            if len(parts) != 3:
                print(f"Invalid range for {key}: {vals_str}")
                print("  Expected: start:stop:step (e.g., 10:100:10)")
                sys.exit(1)
            start, stop, step = [float(p) for p in parts]
            vals = []
            v = start
            while v <= stop + 1e-9:
                # Preserve int if all parts are int-like
                if all(float(p) == int(float(p)) for p in parts):
                    vals.append(int(round(v)))
                else:
                    vals.append(D(str(round(v, 10))))
                v += step
            sweeps[key] = vals
        else:
            # Explicit values
            sweeps[key] = [_parse_value(v) for v in vals_str.split(",")]

    return sweeps


def parse_set_args(set_strs: List[str]) -> dict:
    """Parse 'param=value' fixed overrides."""
    overrides = {}
    for s in set_strs:
        if "=" not in s:
            print(f"Invalid set format: {s} (expected param=value)")
            sys.exit(1)
        key, val = s.split("=", 1)
        overrides[key] = _parse_value(val)
    return overrides


def build_param_grid(sweeps: Dict[str, List],
                     base_config: dict) -> List[dict]:
    """Cartesian product of sweep parameters merged with base config."""
    keys = list(sweeps.keys())
    values = [sweeps[k] for k in keys]

    grid = []
    for combo in itertools.product(*values):
        config = dict(base_config)
        for k, v in zip(keys, combo):
            config[k] = v
        grid.append(config)
    return grid


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_results_table(results: List[dict], sweep_keys: List[str],
                         sort_by: str) -> str:
    """Format as aligned text table."""
    if not results:
        return "No results."

    reverse = sort_by not in ("max_drawdown_pct", "max_abs_inv_skew")
    results.sort(key=lambda r: r.get(sort_by, 0), reverse=reverse)

    param_cols = [f"p_{k}" for k in sweep_keys]
    metric_cols = [
        "total_return_pct", "hold_return_pct", "excess_return_pct",
        "max_drawdown_pct", "sharpe", "total_fills",
        "avg_spread_bps", "max_abs_inv_skew",
    ]
    if any("recenters" in r and r["recenters"] > 0 for r in results):
        metric_cols.extend(["recenters", "range_changes"])
    if any(r.get("trend_halts", 0) > 0 for r in results):
        metric_cols.append("trend_halts")

    display = {
        "total_return_pct": "Return%",
        "hold_return_pct": "Hold%",
        "excess_return_pct": "Excess%",
        "max_drawdown_pct": "MaxDD%",
        "sharpe": "Sharpe",
        "total_fills": "Fills",
        "avg_spread_bps": "AvgSprd",
        "max_abs_inv_skew": "MaxSkew",
        "recenters": "Recntr",
        "range_changes": "RngChg",
        "trend_halts": "THalts",
        "_elapsed_sec": "Time(s)",
    }
    for k in sweep_keys:
        display[f"p_{k}"] = k

    rows = []
    for i, r in enumerate(results):
        row = [str(i + 1)]
        for col in param_cols + metric_cols + ["_elapsed_sec"]:
            val = r.get(col, "")
            if isinstance(val, float):
                if "pct" in col or col == "sharpe":
                    row.append(f"{val:+.2f}" if "return" in col
                               or "excess" in col else f"{val:.2f}")
                else:
                    row.append(f"{val:.1f}")
            elif isinstance(val, Decimal):
                row.append(str(val))
            else:
                row.append(str(val))
        rows.append(row)

    headers = ["#"] + [display.get(c, c) for c in
                       param_cols + metric_cols + ["_elapsed_sec"]]
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]

    lines = []
    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        lines.append("  ".join(v.rjust(w) for v, w in zip(row, widths)))

    return "\n".join(lines)


def save_results_csv(results: List[dict], path: str):
    """Save all results to CSV."""
    if not results:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    all_keys = []
    seen = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CL_AMM_CONFIG = {
    "spread_bps": D("40"),
    "pool_price_weight": D("0.70"),
    "anchor_ema_alpha": D("0.05"),
    "num_levels": 1,
    "size_decay": D("0.85"),
    "spread_multiplier": D("1.5"),
    "order_safe_ratio": D("0.5"),
    "enable_asymmetric_spread": True,
    "skew_sensitivity": D("0.5"),
    "min_spread_bps": D("20"),
    "concentration": D("5"),
    "min_concentration": D("5"),
    "max_concentration": D("20"),
    "natr_period": 14,
    "natr_baseline": D("0.005"),
    "natr_range_scale": D("1.0"),
    "adx_period": 14,
    "hurst_min_candles": 100,
    "hmm_n_states": 3,
    "hmm_min_candles": 200,
    "hmm_refit_interval_sec": 1800,
    "hmm_window": 500,
    "hmm_confidence_threshold": D("0.80"),
    "trend_sensitivity": D("0.5"),
    "range_ema_alpha": D("0.1"),
    "range_update_dead_band_pct": D("0.5"),
    "soft_recenter_drift_pct": D("2.0"),
    "trend_order_scale_factor": D("0.0"),
    "trend_halt_threshold": D("0.0"),
}

DEFAULT_AMM_CONFIG = {
    "spread_bps": D("40"),
    "pool_price_weight": D("0.70"),
    "anchor_ema_alpha": D("0.05"),
    "num_levels": 1,
    "size_decay": D("0.85"),
    "spread_multiplier": D("1.5"),
    "order_safe_ratio": D("0.5"),
    "enable_asymmetric_spread": True,
    "skew_sensitivity": D("0.5"),
    "min_spread_bps": D("20"),
    "amplification": D("5"),
}


def main():
    parser = argparse.ArgumentParser(
        description="HPC Parameter Sweep for Strategy Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweep syntax:
  param=val1,val2,val3       Explicit values
  param=start:stop:step      Numeric range (inclusive)

Examples:
  # 80-core HPC: large grid with range syntax
  python backtest_sweep.py --strategy cl-amm --symbol ADAUSDT \\
      --start 2025-01-01 --end 2025-03-01 \\
      --sweep spread_bps=10:100:5 concentration=2:20:1 \\
      --workers 80

  # AMM: sweep 2 params across full range
  python backtest_sweep.py --strategy amm --symbol ADAUSDT \\
      --start 2025-01-01 --end 2025-03-01 \\
      --sweep spread_bps=10:100:10 amplification=2:30:2 \\
      --workers 80

  # Mixed explicit + range + fixed overrides
  python backtest_sweep.py --strategy cl-amm --symbol ADAUSDT \\
      --start 2025-01-01 --end 2025-03-01 \\
      --sweep spread_bps=10:80:10 concentration=3,5,10,15,20 \\
      --set pool_price_weight=0.5 anchor_ema_alpha=0.03 \\
      --workers 80

  # Dry run: preview grid size without executing
  python backtest_sweep.py --strategy cl-amm ... --dry-run

  # Sort by Sharpe, show top 20
  python backtest_sweep.py ... --sort-by sharpe --top 20
""")

    parser.add_argument("--strategy", required=True, choices=["cl-amm", "amm"])
    parser.add_argument("--symbol", default="ADAUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--base-balance", type=Decimal, default=D("100000"))
    parser.add_argument("--quote-balance", type=Decimal, default=D("27000"))
    parser.add_argument("--balanced", action="store_true",
                        help="Auto-compute 50/50 base/quote split from first "
                             "candle price. Uses --portfolio-usd for total size.")
    parser.add_argument("--portfolio-usd", type=Decimal, default=D("100000"),
                        help="Total portfolio in USD when --balanced (default: 100000)")
    parser.add_argument("--csv", default=None,
                        help="Load candles from CSV file")
    parser.add_argument("--output", default="results",
                        help="Output directory")

    parser.add_argument("--sweep", nargs="+", required=True,
                        help="param=val1,val2 or param=start:stop:step")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Fixed parameter overrides: param=value")
    parser.add_argument("--sort-by", default="excess_return_pct",
                        choices=["total_return_pct", "excess_return_pct",
                                 "sharpe", "max_drawdown_pct",
                                 "total_fills", "avg_spread_bps"],
                        help="Sort metric (default: excess_return_pct)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    parser.add_argument("--top", type=int, default=None,
                        help="Show top N results only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview parameter grid without running")

    args = parser.parse_args()

    # Parse sweep and overrides
    sweeps = parse_sweep_args(args.sweep)
    overrides = parse_set_args(args.set)

    base_config = dict(
        DEFAULT_CL_AMM_CONFIG if args.strategy == "cl-amm"
        else DEFAULT_AMM_CONFIG
    )
    base_config.update(overrides)

    grid = build_param_grid(sweeps, base_config)
    sweep_keys = list(sweeps.keys())
    total_combos = len(grid)

    # Summary
    cpu_count = os.cpu_count() or 1
    workers = args.workers or min(total_combos, cpu_count)

    print(f"\n{'=' * 60}")
    print(f"  PARAMETER SWEEP — {args.strategy.upper()}")
    print(f"{'=' * 60}")
    print(f"  Symbol:       {args.symbol} ({args.interval})")
    print(f"  Period:       {args.start} to {args.end}")
    print(f"  Balances:     {args.base_balance} base / "
          f"{args.quote_balance} quote"
          f"{' (--balanced will adjust)' if args.balanced else ''}")
    print(f"  CPU cores:    {cpu_count}")
    print(f"  Workers:      {workers}")
    print()
    print(f"  Sweep parameters:")
    for k, v in sweeps.items():
        if len(v) > 10:
            print(f"    {k}: {v[0]} .. {v[-1]} "
                  f"({len(v)} values, step={v[1]-v[0]})")
        else:
            print(f"    {k}: {v}")
    if overrides:
        print(f"  Fixed overrides:")
        for k, v in overrides.items():
            print(f"    {k}: {v}")
    print(f"\n  Total combinations: {total_combos}")
    print(f"  Est. parallelism:  {min(workers, total_combos)} concurrent")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("Dry run — showing first 10 combinations:\n")
        for i, params in enumerate(grid[:10]):
            vals = ", ".join(f"{k}={params[k]}" for k in sweep_keys)
            print(f"  [{i + 1:4d}] {vals}")
        if total_combos > 10:
            print(f"  ... ({total_combos - 10} more)")
        return

    # Load candle data (main process — triggers cache for workers)
    from backtest_engine import CandleDataLoader

    print(f"Loading candle data...")
    candles = CandleDataLoader.load(
        args.symbol, args.interval, args.start, args.end, args.csv)
    if not candles:
        print("No candles loaded. Exiting.")
        sys.exit(1)
    candle_mem_mb = len(candles) * 800 / 1024 / 1024
    print(f"  {len(candles)} candles "
          f"({candles[0].close} -> {candles[-1].close})")
    print(f"  Memory per worker: ~{candle_mem_mb:.0f}MB "
          f"(total: ~{candle_mem_mb * workers:.0f}MB for {workers} workers)")

    # Compute balanced 50/50 split if requested
    base_balance = args.base_balance
    quote_balance = args.quote_balance
    if args.balanced:
        start_price = candles[0].close
        half_usd = args.portfolio_usd / D(2)
        quote_balance = half_usd
        base_balance = half_usd / start_price
        print(f"\n  Balanced 50/50 @ start price {start_price}:")
        print(f"    base:  {base_balance:.2f} ({half_usd} USD)")
        print(f"    quote: {quote_balance:.2f} USD")
        print(f"    total: {args.portfolio_usd} USD")
    print()

    # Build tasks (lightweight: just index + params, no candle data)
    tasks = [
        (i, args.strategy, params, base_balance, quote_balance)
        for i, params in enumerate(grid)
    ]

    # Run with shared-memory workers
    results: List[dict] = []
    t0 = time.time()
    completed = 0
    failed = 0

    # Use fork context on Linux for zero-copy candle sharing.
    # On macOS, workers load from cache file (small overhead).
    mp_context = None
    if sys.platform == "linux":
        mp_context = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(args.symbol, args.interval, args.start, args.end, args.csv),
        mp_context=mp_context,
    ) as executor:
        futures = {executor.submit(_run_single, task): task[0]
                   for task in tasks}

        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            elapsed_total = time.time() - t0
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            remaining = total_combos - completed
            eta = remaining / rate if rate > 0 else 0

            try:
                result = future.result()
                results.append(result)
                params_str = ", ".join(
                    f"{k}={grid[idx][k]}" for k in sweep_keys)
                ret = result.get("total_return_pct", 0)
                t_run = result.get("_elapsed_sec", 0)
                print(f"  [{completed:4d}/{total_combos}] {params_str} "
                      f"-> {ret:+.2f}% ({t_run}s) "
                      f"[ETA {eta:.0f}s]")
            except Exception as e:
                failed += 1
                params_str = ", ".join(
                    f"{k}={grid[idx][k]}" for k in sweep_keys)
                print(f"  [{completed:4d}/{total_combos}] {params_str} "
                      f"-> FAILED: {e}")

    total_time = time.time() - t0
    total_cpu_time = sum(r.get("_elapsed_sec", 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"  SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Runs:         {len(results)} OK, {failed} failed "
          f"/ {total_combos} total")
    print(f"  Wall time:    {total_time:.1f}s")
    print(f"  CPU time:     {total_cpu_time:.1f}s")
    print(f"  Parallelism:  {total_cpu_time / total_time:.1f}x "
          f"(of {workers} workers)")
    print(f"  Throughput:   {len(results) / total_time:.1f} runs/sec")
    print(f"{'=' * 60}")

    if not results:
        print("No results to display.")
        return

    # Sort before display/save
    reverse = args.sort_by not in ("max_drawdown_pct", "max_abs_inv_skew")
    results.sort(key=lambda r: r.get(args.sort_by, 0), reverse=reverse)

    display_results = results[:args.top] if args.top else results

    print(f"\n  RESULTS (sorted by {args.sort_by})"
          f"{f', top {args.top}' if args.top else ''}\n")
    print(format_results_table(display_results, sweep_keys, args.sort_by))
    print()

    # Save CSV
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        args.output,
        f"sweep_{args.strategy}_{args.symbol}_{ts}.csv",
    )
    save_results_csv(results, csv_path)

    # Highlight best
    best = results[0]
    print(f"\nBest ({args.sort_by}):")
    for k in sweep_keys:
        print(f"  {k}: {best.get(f'p_{k}', '?')}")
    print(f"  return: {best.get('total_return_pct', 0):+.2f}%"
          f"  excess: {best.get('excess_return_pct', 0):+.2f}%"
          f"  sharpe: {best.get('sharpe', 0):.2f}"
          f"  maxDD: {best.get('max_drawdown_pct', 0):.2f}%")


if __name__ == "__main__":
    main()
