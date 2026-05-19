#!/usr/bin/env python3
"""Step 2-4 of the walk-forward workflow: best-tuned candidate config.

Combines per-axis winners from the coarse sweeps into a single config and
runs it across all six annual windows + the 5yr aggregate. Returns a single
CSV with one row per window so generalisation is visible at a glance.

The expectation is NOT "every window is profitable" — strong bull years
(2020 +225%, 2021 +331%) will almost certainly lose to buy-and-hold without
a hedge. Pass criteria, applied jointly to the OOS set (2020/21/22/23/25):
    * total_return_pct > 0 in 5 of 5 windows, OR
    * excess_return_pct > 0 in 4 of 5 windows, AND
    * sharpe > 0.5 in 4 of 5 windows, AND
    * max_drawdown_pct < 60% in all windows

Usage:
    cd scripts
    python run_best_tuned.py                          # all windows
    python run_best_tuned.py --windows 2024 2022      # subset
    python run_best_tuned.py --override skew_sensitivity=2.0  # ad-hoc tweak
"""

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal

D = Decimal
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Best-tuned config — derived from coarse-sweep winners (results/coarse/*).
# Trend layer DEFAULT OFF (Option B): bear-2022 evidence + phase6 indicator
# validation showing direction is unpredictable. Switch on per regime if a
# vol-based gate is later wired (Option A in walk-forward doc).
# ---------------------------------------------------------------------------

# ALL_ON baseline = the actual config that gave +46.6% on 2024 in the
# ablation. Useful as control / generalisation check. If THIS doesn't
# walk-forward, no amount of per-axis tuning will save the strategy.
ALL_ON = {
    "spread_bps":                        D("50"),
    "concentration":                     D("15"),
    "min_concentration":                 D("3"),
    "max_concentration":                 D("40"),
    "pool_price_weight":                 D("0.70"),
    "anchor_ema_alpha":                  D("0.05"),
    "order_safe_ratio":                  D("0.5"),
    "natr_period":                       14,
    "natr_baseline":                     D("0.005"),
    "natr_range_scale":                  D("1.0"),
    "outer_capital_fraction":            D("0.30"),
    "outer_spread_mult":                 D("2.5"),
    "outer_spread_pct_of_range":         D("0.60"),
    "outer_recenter_trigger_pct":        D("0.50"),
    "outer_range_mult":                  D("2.5"),
    "enable_asymmetric_spread":          True,
    "skew_sensitivity":                  D("0.5"),
    "min_spread_bps":                    D("20"),
    "range_ema_alpha":                   D("0.1"),
    "range_update_dead_band_pct":        D("0.5"),
    "soft_recenter_drift_pct":           D("2.0"),
    "trend_sensitivity":                 D("0.5"),
    "trend_order_scale_factor":          D("0.0"),
    "trend_halt_threshold":              D("0.0"),
    "adx_period":                        14,
    "hurst_min_candles":                 100,
    "hmm_n_states":                      3,
    "hmm_min_candles":                   999999,        # effectively off
    "hmm_refit_interval_sec":            1800,
    "hmm_window":                        500,
    "hmm_confidence_threshold":          D("0.80"),
    "toxicity_window_sec":               300,
    "toxicity_window_fills":             20,
    "toxicity_activation_count":         8,
    "toxicity_buy_ratio_soft":           D("0.65"),
    "toxicity_buy_ratio_hard":           D("0.80"),
    "toxicity_sell_ratio_soft":          D("0.65"),
    "toxicity_sell_ratio_hard":          D("0.80"),
    "toxicity_soft_spread_mult":         D("1.30"),
    "toxicity_hard_spread_mult":         D("1.80"),
    "toxicity_soft_size_mult":           D("0.70"),
    "toxicity_hard_size_mult":           D("0.40"),
    "inventory_skew_soft_limit":         D("0.60"),
    "inventory_skew_hard_limit":         D("0.80"),
    "inventory_soft_size_mult":          D("0.60"),
    "inventory_hard_size_mult":          D("0.20"),
    "inventory_soft_spread_mult":        D("1.30"),
    "inventory_hard_spread_mult":        D("1.80"),
    "inventory_hard_disable_accumulation_side": True,
    "enable_flair_monitor":              True,
    "flair_markout_sec":                 30,
    "flair_window_sec":                  1800,
    "flair_fee_bps":                     D("10"),
}


# BEST_TUNED = aggressive combination of per-axis coarse-sweep winners.
# WARNING: walk-forward showed this loses to ALL_ON on 2024 — per-axis
# winners do not stack additively due to interactions (especially
# skew_sensitivity=2.0 × min_spread_bps=40 amplifying floor clipping).
# Kept for reference; do NOT use as the launch config.
BEST_TUNED = {
    # Inner zone — kept at 2024 yearly-sweep winner
    "spread_bps":                        D("50"),
    "concentration":                     D("15"),
    "min_concentration":                 D("3"),
    "max_concentration":                 D("20"),       # was 40 — narrowing back per 3d sweep

    # Mid quote / anchoring
    "pool_price_weight":                 D("0.70"),
    "anchor_ema_alpha":                  D("0.05"),
    "order_safe_ratio":                  D("0.5"),

    # Dynamic range — 0.5 is sweep 3a winner (vs default 1.0)
    "natr_period":                       14,
    "natr_baseline":                     D("0.005"),
    "natr_range_scale":                  D("0.5"),

    # Outer dual-range — 4a winner 0.40, 4d winner 3.5
    "outer_capital_fraction":            D("0.40"),
    "outer_spread_mult":                 D("3.5"),
    "outer_spread_pct_of_range":         D("0.60"),
    "outer_recenter_trigger_pct":        D("0.65"),
    "outer_range_mult":                  D("2.5"),      # deprecated but still set

    # Asymmetric spread — sweep 6a winner sens=2.0 + 6b winner floor=40
    "enable_asymmetric_spread":          True,
    "skew_sensitivity":                  D("2.0"),
    "min_spread_bps":                    D("40"),

    # Range/trend EMA
    "range_ema_alpha":                   D("0.1"),
    "range_update_dead_band_pct":        D("0.5"),
    "soft_recenter_drift_pct":           D("2.0"),

    # Trend layer — OFF per Option B
    "trend_sensitivity":                 D("0.0"),
    "trend_order_scale_factor":          D("0.0"),
    "trend_halt_threshold":              D("0.0"),

    # ADX / Hurst — unused with trend off, but configured for safety
    "adx_period":                        14,
    "hurst_min_candles":                 100,

    # HMM — kept off (phase5 showed no measurable effect)
    "hmm_n_states":                      3,
    "hmm_min_candles":                   999999,
    "hmm_refit_interval_sec":            1800,
    "hmm_window":                        500,
    "hmm_confidence_threshold":          D("0.80"),

    # Toxicity guard — 5b winner ratio_hard=0.85, 5d winner size_mult=0.6
    "toxicity_window_sec":               300,
    "toxicity_window_fills":             20,
    "toxicity_activation_count":         8,
    "toxicity_buy_ratio_soft":           D("0.65"),
    "toxicity_buy_ratio_hard":           D("0.85"),
    "toxicity_sell_ratio_soft":          D("0.65"),
    "toxicity_sell_ratio_hard":          D("0.85"),
    "toxicity_soft_spread_mult":         D("1.30"),
    "toxicity_hard_spread_mult":         D("1.80"),
    "toxicity_soft_size_mult":           D("0.70"),
    "toxicity_hard_size_mult":           D("0.60"),

    # Inventory guard — kept as-is; ablation showed soft is firing usefully
    "inventory_skew_soft_limit":         D("0.60"),
    "inventory_skew_hard_limit":         D("0.80"),
    "inventory_soft_size_mult":          D("0.60"),
    "inventory_hard_size_mult":          D("0.20"),
    "inventory_soft_spread_mult":        D("1.30"),
    "inventory_hard_spread_mult":        D("1.80"),
    "inventory_hard_disable_accumulation_side": True,

    # FLAIR
    "enable_flair_monitor":              True,
    "flair_markout_sec":                 30,
    "flair_window_sec":                  1800,
    "flair_fee_bps":                     D("10"),
}


WINDOWS = {
    "2020":    ("2020-01-01", "2020-12-31"),
    "2021":    ("2021-01-01", "2021-12-31"),
    "2022":    ("2022-01-01", "2022-12-31"),
    "2023":    ("2023-01-01", "2023-12-31"),
    "2024":    ("2024-01-01", "2024-12-31"),   # in-sample
    "2025q1":  ("2025-01-01", "2025-03-19"),
    "5yr":     ("2020-01-01", "2025-03-19"),
}


# ---------------------------------------------------------------------------
# Worker — same shared-candle pattern, but each task gets its own (start, end)
# ---------------------------------------------------------------------------

def _run_window(args):
    label, symbol, start, end, csv_path, base_bal, quote_bal, params = args
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    import time as _t
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from backtest_engine import CandleDataLoader, BacktestEngine
        from backtest_strategies import CLAMMBacktestStrategy
        candles = CandleDataLoader.load(symbol, "1m", start, end, csv_path)
    finally:
        sys.stdout = old_stdout

    # Per-window balanced 50/50 — start_price differs across years.
    start_price = candles[0].close
    half_usd = D("50000")
    quote = half_usd
    base = half_usd / start_price

    t0 = _t.time()
    try:
        strat = CLAMMBacktestStrategy(**params)
        engine = BacktestEngine(
            strategy=strat, candles=candles,
            base_balance=base, quote_balance=quote,
            quiet=True, lightweight=True)
        metrics = engine.run()
    except Exception as e:
        metrics = {"_error": f"{type(e).__name__}: {e}"}

    metrics["_label"]       = label
    metrics["_start"]       = start
    metrics["_end"]         = end
    metrics["_elapsed_sec"] = round(_t.time() - t0, 1)
    metrics["_start_price"] = float(start_price)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["all_on", "best_tuned"], default="all_on",
                   help="Which preset config to walk-forward. Default: all_on "
                        "(known to give +46.6%% on ADAUSDC 2024). best_tuned is "
                        "kept for reference but loses to all_on.")
    p.add_argument("--symbol", default="ADAUSDC",
                   help="Trading pair, e.g. ADAUSDC, ETHUSDC, BTCUSDC. "
                        "Candle CSV cache must exist or be downloadable "
                        "(see download_candles.py).")
    p.add_argument("--windows", nargs="+", default=list(WINDOWS.keys()),
                   help=f"Subset of windows to run. Default: all. Choices: {list(WINDOWS.keys())}")
    p.add_argument("--override", nargs="*", default=[],
                   help="param=value pairs to override the chosen preset for this run")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--csv", default=None)
    p.add_argument("--output", default="results/best_tuned")
    args = p.parse_args()

    # Choose preset
    preset = ALL_ON if args.preset == "all_on" else BEST_TUNED
    params = dict(preset)
    for ov in args.override:
        if "=" not in ov:
            print(f"Invalid override: {ov}"); sys.exit(1)
        k, v = ov.split("=", 1)
        # Bool first (string "False" is truthy), then Decimal/int, then str
        if v in ("True", "true"):       params[k] = True
        elif v in ("False", "false"):   params[k] = False
        else:
            try: params[k] = int(v)
            except ValueError:
                try: params[k] = D(v)
                except Exception: params[k] = v

    print(f"\n{'='*68}\n  WALK-FORWARD  preset={args.preset}\n{'='*68}")
    print("  Windows:", ", ".join(args.windows))
    print(f"  Workers: {args.workers}")
    if args.override:
        print("  Overrides:", " ".join(args.override))

    tasks = []
    for label in args.windows:
        if label not in WINDOWS:
            print(f"Unknown window: {label} (choices: {list(WINDOWS.keys())})")
            sys.exit(1)
        start, end = WINDOWS[label]
        tasks.append((label, args.symbol, start, end, args.csv, None, None, params))
    print(f"  Symbol:  {args.symbol}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        futures = {ex.submit(_run_window, t): t[0] for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            print(f"  [{r['_label']:7s}] ret={r.get('total_return_pct','?')} "
                  f"hold={r.get('hold_return_pct','?')} "
                  f"excess={r.get('excess_return_pct','?')} "
                  f"sharpe={r.get('sharpe','?')} "
                  f"DD={r.get('max_drawdown_pct','?')} "
                  f"({r.get('_elapsed_sec','?')}s)")

    # Sort by window order
    order = {w: i for i, w in enumerate(args.windows)}
    results.sort(key=lambda r: order.get(r["_label"], 999))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # CSV
    os.makedirs(args.output, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output,
                            f"walk_forward_{args.symbol}_{args.preset}_{ts}.csv")
    keys: list = []
    seen = set()
    for r in results:
        for k in r:
            if k not in seen: keys.append(k); seen.add(k)
    # Also embed swept params in the CSV
    for k, v in params.items():
        col = f"p_{k}"
        if col not in seen: keys.append(col); seen.add(col)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            for k, v in params.items():
                row[f"p_{k}"] = v
            w.writerow(row)
    print(f"\nResults → {csv_path}")

    # Walk-forward gate summary
    print(f"\n{'─'*68}\n  WALK-FORWARD GATE\n{'─'*68}")
    oos = [r for r in results if r["_label"] not in ("2024", "5yr")]
    if not oos:
        print("  No OOS windows ran; skip gate.")
        return
    pos_return  = sum(1 for r in oos if (r.get("total_return_pct") or -1) > 0)
    pos_excess  = sum(1 for r in oos if (r.get("excess_return_pct") or -1) > 0)
    good_sharpe = sum(1 for r in oos if (r.get("sharpe") or -1) > 0.5)
    bad_dd      = sum(1 for r in oos if (r.get("max_drawdown_pct") or 0) >= 60)
    print(f"  Out-of-sample windows: {len(oos)} ({', '.join(r['_label'] for r in oos)})")
    print(f"    positive return:        {pos_return}/{len(oos)}")
    print(f"    positive excess vs hold: {pos_excess}/{len(oos)}")
    print(f"    sharpe > 0.5:           {good_sharpe}/{len(oos)}")
    print(f"    max DD ≥ 60% (bad):     {bad_dd}/{len(oos)}")

    passes = (pos_return == len(oos)) or (
        pos_excess >= len(oos) - 1
        and good_sharpe >= len(oos) - 1
        and bad_dd == 0
    )
    verdict = "✓ PASS — proceed to canary" if passes else "✗ FAIL — config likely overfit to 2024"
    print(f"\n  Verdict: {verdict}")


if __name__ == "__main__":
    main()
