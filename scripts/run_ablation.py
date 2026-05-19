#!/usr/bin/env python3
"""Guard-ablation experiment for CL-AMM.

Answers: "Do the active features (asymmetric spread, dynamic range, toxicity
guard, inventory guard, outer band) add value over a passive ('dumb') LP that
just sits in a fixed range?"

Runs 6 configs on the same window, all sharing the same (spread, concentration)
baseline:
    1. dumb_lp       — all active features OFF (static range, symmetric spread,
                       no guards, no outer band)
    2. +inv          — dumb_lp + inventory guards only
    3. +tox          — dumb_lp + toxicity guards only
    4. +asym         — dumb_lp + asymmetric spread only
    5. +dyn_range    — dumb_lp + NATR-driven dynamic concentration only
    6. all_on        — current full-feature config (the "production" tuning)

Output: one CSV with all 6 rows, sortable by total_return_pct or
fee/inventory-decomposed metrics. The interesting comparison is `all_on` vs
`dumb_lp` — if all_on doesn't beat dumb_lp on a risk-adjusted basis, the
active machinery is decoration.

Usage:
    cd /Users/yuyanyuk/Git/hummingbot-strategies
    python3 scripts/run_ablation.py --start 2024-01-01 --end 2024-12-31 \\
        --spread-bps 50 --concentration 15

Defaults to the 2024-yearly-sweep winner (spread_bps=50, concentration=15)
on full year 2024.
"""

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from typing import List, Optional

D = Decimal

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Baseline (dumb LP): every adaptive feature neutralised. Sweep params are
# applied on top.
# ---------------------------------------------------------------------------

DUMB_LP_BASE = {
    "pool_price_weight":               D("0.70"),
    "anchor_ema_alpha":                D("0.05"),
    "num_levels":                      1,
    "size_decay":                      D("0.85"),
    "spread_multiplier":               D("1.5"),
    "order_safe_ratio":                D("0.5"),

    # --- ASYM SPREAD: OFF ---
    "enable_asymmetric_spread":        False,
    "skew_sensitivity":                D("0.0"),
    "min_spread_bps":                  D("20"),

    # --- DYNAMIC RANGE: OFF (static concentration) ---
    # natr_range_scale=0 zeroes the NATR multiplier deviation.
    # trend_sensitivity=0 zeroes the ADX/Hurst trend boost.
    # range_ema_alpha=1 disables EMA smoothing (raw_pct passes through).
    "natr_period":                     14,
    "natr_baseline":                   D("0.005"),
    "natr_range_scale":                D("0.0"),
    "adx_period":                      14,
    "hurst_min_candles":               999999,   # never warms up
    "trend_sensitivity":               D("0.0"),
    "range_ema_alpha":                 D("1.0"),
    "range_update_dead_band_pct":      D("0.5"),
    "soft_recenter_drift_pct":         D("2.0"),

    # --- TREND GUARD: OFF ---
    "trend_order_scale_factor":        D("0.0"),
    "trend_halt_threshold":            D("0.0"),

    # --- HMM: effectively off (never warms up) ---
    "hmm_n_states":                    3,
    "hmm_min_candles":                 999999,
    "hmm_refit_interval_sec":          1800,
    "hmm_window":                      500,
    "hmm_confidence_threshold":        D("0.80"),

    # --- TOXICITY GUARD: OFF (activation_count so high it never fires) ---
    "toxicity_window_sec":             300,
    "toxicity_window_fills":           20,
    "toxicity_activation_count":       999999,
    "toxicity_buy_ratio_soft":         D("0.65"),
    "toxicity_buy_ratio_hard":         D("0.80"),
    "toxicity_sell_ratio_soft":        D("0.65"),
    "toxicity_sell_ratio_hard":        D("0.80"),
    "toxicity_soft_spread_mult":       D("1.00"),
    "toxicity_hard_spread_mult":       D("1.00"),
    "toxicity_soft_size_mult":         D("1.00"),
    "toxicity_hard_size_mult":         D("1.00"),

    # --- INVENTORY GUARD: OFF (limits at 0.99 / 0.999 so they never trip) ---
    "inventory_skew_soft_limit":         D("0.99"),
    "inventory_skew_hard_limit":         D("0.999"),
    "inventory_soft_size_mult":          D("1.00"),
    "inventory_hard_size_mult":          D("1.00"),
    "inventory_soft_spread_mult":        D("1.00"),
    "inventory_hard_spread_mult":        D("1.00"),
    "inventory_hard_disable_accumulation_side": False,

    # --- FLAIR monitor: ON for accounting (read-only) ---
    "enable_flair_monitor":            True,
    "flair_markout_sec":               30,
    "flair_window_sec":                1800,
    "flair_fee_bps":                   D("10"),
}


# Feature presets — patches applied on top of DUMB_LP_BASE.
def feature_presets():
    """Returns dict {label: patch_dict} for each ablation config.

    Patches override DUMB_LP_BASE keys to turn specific features on.
    """
    INV_ON = {
        "inventory_skew_soft_limit":         D("0.60"),
        "inventory_skew_hard_limit":         D("0.80"),
        "inventory_soft_size_mult":          D("0.60"),
        "inventory_hard_size_mult":          D("0.20"),
        "inventory_soft_spread_mult":        D("1.30"),
        "inventory_hard_spread_mult":        D("1.80"),
        "inventory_hard_disable_accumulation_side": True,
    }

    TOX_ON = {
        "toxicity_activation_count":     8,
        "toxicity_buy_ratio_soft":       D("0.65"),
        "toxicity_buy_ratio_hard":       D("0.80"),
        "toxicity_sell_ratio_soft":      D("0.65"),
        "toxicity_sell_ratio_hard":      D("0.80"),
        "toxicity_soft_spread_mult":     D("1.30"),
        "toxicity_hard_spread_mult":     D("1.80"),
        "toxicity_soft_size_mult":       D("0.70"),
        "toxicity_hard_size_mult":       D("0.40"),
    }

    ASYM_ON = {
        "enable_asymmetric_spread":      True,
        "skew_sensitivity":              D("0.5"),
    }

    DYN_RANGE_ON = {
        "natr_range_scale":              D("1.0"),
        "trend_sensitivity":             D("0.5"),
        "range_ema_alpha":               D("0.1"),
        "hurst_min_candles":             100,
    }

    return {
        "1_dumb_lp":    {},
        "2_plus_inv":   {**INV_ON},
        "3_plus_tox":   {**TOX_ON},
        "4_plus_asym":  {**ASYM_ON},
        "5_plus_dyn":   {**DYN_RANGE_ON},
        "6_all_on":     {**INV_ON, **TOX_ON, **ASYM_ON, **DYN_RANGE_ON},
    }


# ---------------------------------------------------------------------------
# Worker — same shared-candle pattern as run_experiment.py
# ---------------------------------------------------------------------------

_W_CANDLES = None
_W_BASE = D("100000")
_W_QUOTE = D("27000")


def _worker_init(symbol, start, end, csv_path, base_bal, quote_bal):
    global _W_CANDLES, _W_BASE, _W_QUOTE
    _W_BASE = D(base_bal)
    _W_QUOTE = D(quote_bal)
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        from backtest_engine import CandleDataLoader
        _W_CANDLES = CandleDataLoader.load(symbol, "1m", start, end, csv_path)
    finally:
        sys.stdout = old_stdout


def _run_one(task: dict) -> dict:
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    import time as _t
    from backtest_engine import BacktestEngine
    from backtest_strategies import CLAMMBacktestStrategy

    label = task.pop("_label")
    strategy_params = {k: v for k, v in task.items() if not k.startswith("_")}
    t0 = _t.time()
    try:
        strat = CLAMMBacktestStrategy(**strategy_params)
        engine = BacktestEngine(
            strategy=strat,
            candles=_W_CANDLES,
            base_balance=_W_BASE,
            quote_balance=_W_QUOTE,
            quiet=True,
            lightweight=True,
        )
        metrics = engine.run()
    except Exception as e:
        metrics = {"_error": f"{type(e).__name__}: {e}"}

    elapsed = round(_t.time() - t0, 1)
    result = {"_label": label, "_elapsed": elapsed}
    result.update({f"p_{k}": v for k, v in strategy_params.items()})
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="ADAUSDC")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--csv", default=None,
                   help="Path to candle CSV (optional, uses cache otherwise)")
    p.add_argument("--spread-bps", type=Decimal, default=D("50"),
                   help="Fixed spread in bps shared by all configs (default 50)")
    p.add_argument("--concentration", type=Decimal, default=D("15"),
                   help="Fixed concentration shared by all configs (default 15)")
    p.add_argument("--min-concentration", type=Decimal, default=D("3"),
                   help="Range floor for dyn_range mode (default 3)")
    p.add_argument("--max-concentration", type=Decimal, default=D("40"),
                   help="Range ceiling for dyn_range mode (default 40)")
    p.add_argument("--portfolio-usd", type=Decimal, default=D("100000"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", default="results/ablation",
                   help="Output directory (default: results/ablation)")
    args = p.parse_args()

    # Build the 6 tasks.
    presets = feature_presets()
    common = dict(DUMB_LP_BASE)
    common["spread_bps"] = args.spread_bps
    common["concentration"] = args.concentration
    common["min_concentration"] = args.min_concentration
    common["max_concentration"] = args.max_concentration

    tasks = []
    for label, patch in presets.items():
        cfg = dict(common)
        cfg.update(patch)
        cfg["_label"] = label
        tasks.append(cfg)

    # Pre-load candles in main process to populate cache, also derive
    # balanced 50/50 split.
    sys.path.insert(0, SCRIPTS_DIR)
    from backtest_engine import CandleDataLoader
    print(f"Loading candles for {args.symbol} {args.start} → {args.end} ...")
    candles = CandleDataLoader.load(args.symbol, "1m", args.start, args.end, args.csv)
    if not candles:
        print("No candles loaded."); sys.exit(1)
    start_price = candles[0].close
    half_usd = args.portfolio_usd / D(2)
    base_bal = half_usd / start_price
    quote_bal = half_usd
    print(f"  {len(candles)} candles, start price {start_price}")
    print(f"  Balanced 50/50: base={base_bal:.2f}, quote={quote_bal:.2f}")

    print(f"\nRunning {len(tasks)} ablation configs with {args.workers} workers...")
    print(f"  Shared: spread_bps={args.spread_bps}, concentration={args.concentration}, "
          f"min/max={args.min_concentration}/{args.max_concentration}\n")

    results: List[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks)),
        initializer=_worker_init,
        initargs=(args.symbol, args.start, args.end, args.csv,
                  str(base_bal), str(quote_bal)),
    ) as ex:
        futures = {ex.submit(_run_one, t): t["_label"] for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            ret = r.get("total_return_pct", "?")
            hold = r.get("hold_return_pct", "?")
            sh = r.get("sharpe", "?")
            dd = r.get("max_drawdown_pct", "?")
            fills = r.get("total_fills", "?")
            fee_pct = r.get("flair_fee_pct", "?")
            lvr_pct = r.get("flair_lvr_pct", "?")
            print(f"  [{r['_label']:14s}] ret={ret} hold={hold} sharpe={sh} "
                  f"DD={dd} fills={fills} fee%={fee_pct} lvr%={lvr_pct}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Save sorted by label so config order is stable across runs.
    results.sort(key=lambda r: r["_label"])

    os.makedirs(args.output, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output,
                            f"ablation_{args.symbol}_{args.start}_{args.end}_{ts}.csv")
    all_keys: List[str] = []
    seen = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k); seen.add(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nResults → {csv_path}")

    # Compact summary table.
    print("\nSUMMARY")
    print(f"  {'label':14s}  {'ret%':>8s}  {'hold%':>8s}  {'excess%':>8s}  "
          f"{'sharpe':>7s}  {'DD%':>6s}  {'fills':>6s}  {'fee%':>6s}  {'lvr%':>7s}")
    for r in results:
        def fmt(k, w, prec=2):
            v = r.get(k, "")
            if isinstance(v, (int, float)):
                return f"{v:>{w}.{prec}f}"
            return f"{str(v):>{w}s}"
        print(f"  {r['_label']:14s}  {fmt('total_return_pct',8)}  "
              f"{fmt('hold_return_pct',8)}  {fmt('excess_return_pct',8)}  "
              f"{fmt('sharpe',7)}  {fmt('max_drawdown_pct',6,1)}  "
              f"{str(r.get('total_fills','')):>6s}  "
              f"{fmt('flair_fee_pct',6,1)}  {fmt('flair_lvr_pct',7,1)}")


if __name__ == "__main__":
    main()
