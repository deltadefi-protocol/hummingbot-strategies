#!/usr/bin/env python3
"""
CL-AMM Strategy Experiment Runner
===================================
Structured 6-phase parameter search designed for M4 Pro (12 cores, 48 GB).

Phases
------
  1. Core grid      — spread × concentration on 2022 (16 runs)
  2. Execution grid — rebalance trigger × order depth on 2022 (9 runs, best P1 fixed)
  3. Signal grid    — fill-asymmetry window × thresholds on 2022 (9 runs)
  4. Validation     — top-5 configs cross-validated on 2023 (5 runs)
  5. HMM experiment — baseline vs HMM-v2 on 2025, with F1/MCC output (4 runs)
  6. Final test     — best config on 2025 (1 run)

Usage
-----
    cd /Users/yuyanyuk/Git/hummingbot-strategies
    python3 scripts/run_experiment.py

    # Skip to later phases (if earlier phases already ran):
    python3 scripts/run_experiment.py --from-phase 4

    # Use 2021 data if available:
    python3 scripts/run_experiment.py --train-year 2021

    # Custom workers (default: 10):
    python3 scripts/run_experiment.py --workers 8
"""

import argparse
import csv
import io
import itertools
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

D = Decimal
ZERO = D("0")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, "results", "experiment")

# Candle cache: symbol → {year → (start_date, end_date)}
CANDLE_PERIODS = {
    "ADAUSDC": {
        # Regime windows for Phase 1-3 sweep (HMM disabled)
        "regime_crash_1":   ("2022-01-01", "2022-01-31"),   # ADA -33% sharp crash
        "regime_crash_2":   ("2022-05-01", "2022-05-31"),   # LUNA collapse
        "regime_ranging_1": ("2022-08-01", "2022-08-31"),   # ranging at lows
        "regime_ranging_2": ("2022-10-01", "2022-10-31"),   # pre-FTX lull
        "regime_bull_1":    ("2021-03-01", "2021-03-31"),   # ADA ATH run
        "regime_bull_2":    ("2023-07-01", "2023-07-31"),   # mini recovery rally
        # Default sweep: Jan-Feb 2022 (crash + recovery, 2 months)
        "sweep": ("2022-01-01", "2022-02-28"),
        # Full-year periods
        2021: ("2021-01-01", "2021-12-31"),
        2022: ("2022-01-01", "2022-12-31"),
        2023: ("2023-01-01", "2023-12-31"),
        2024: ("2024-01-01", "2024-12-30"),
        2025: ("2025-01-01", "2025-03-18"),
    }
}

# Phase 1-3 override: disable slow indicators irrelevant to spread/concentration search.
# HMM refits every 30 candles × 131k candles = ~4400 refits per run → minutes each.
# Setting hmm_min_candles=999999 means it never warms up; same for Hurst.
SWEEP_OVERRIDE = {
    "hmm_min_candles": 999999,
    "hurst_min_candles": 999999,
}


# ---------------------------------------------------------------------------
# Worker shared state — loaded once per worker via initializer
# ---------------------------------------------------------------------------

_W_CANDLES: Optional[list] = None
_W_BASE: Decimal = D("100000")
_W_QUOTE: Decimal = D("27000")


def _worker_init(symbol: str, start: str, end: str, csv_path: Optional[str],
                 base_bal: str, quote_bal: str):
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


def _run_task(task: dict) -> dict:
    """Execute one backtest. task = strategy params + special _* control keys."""
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)

    import time as _t
    from backtest_engine import BacktestEngine, TrendValidator
    from backtest_strategies import CLAMMBacktestStrategy

    validate = task.pop("_validate", False)
    val_csv_path = task.pop("_val_csv", None)
    label = task.pop("_label", "")

    strategy_params = {k: v for k, v in task.items() if not k.startswith("_")}

    t0 = _t.time()
    try:
        strat = CLAMMBacktestStrategy(**strategy_params)
        validator = TrendValidator() if validate else None

        engine = BacktestEngine(
            strategy=strat,
            candles=_W_CANDLES,
            base_balance=_W_BASE,
            quote_balance=_W_QUOTE,
            quiet=True,
            lightweight=not validate,
        )
        if validator is not None:
            engine.validator = validator

        metrics = engine.run()

        if validator is not None:
            val_metrics = validator.compute()
            metrics.update(val_metrics)
            if val_csv_path:
                validator.save_csv(val_csv_path)

    except Exception as e:
        metrics = {"_error": str(e)}

    elapsed = round(_t.time() - t0, 1)
    result = {"_label": label, "_elapsed": elapsed}
    result.update({f"p_{k}": v for k, v in strategy_params.items()})
    result.update(metrics)
    return result


# ---------------------------------------------------------------------------
# Base config — defaults for all runs
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "spread_bps": D("40"),
    "pool_price_weight": D("0.70"),
    "anchor_ema_alpha": D("0.05"),
    "order_safe_ratio": D("0.5"),
    "enable_asymmetric_spread": True,
    "skew_sensitivity": D("0.5"),
    "min_spread_bps": D("20"),
    # Outer range
    "outer_capital_fraction": D("0.30"),
    "outer_spread_mult": D("2.5"),
    "outer_range_mult": D("2.5"),
    "outer_recenter_trigger_pct": D("0.50"),
    # Dynamic range
    "concentration": D("5"),
    "min_concentration": D("3"),
    "max_concentration": D("30"),
    "natr_period": 14,
    "natr_baseline": D("0.005"),
    "natr_range_scale": D("1.0"),
    "adx_period": 14,
    "hurst_min_candles": 100,
    # HMM
    "hmm_n_states": 3,
    "hmm_min_candles": 200,
    "hmm_refit_interval_sec": 1800,
    "hmm_window": 500,
    "hmm_confidence_threshold": D("0.80"),
    "hmm_use_fill_asymmetry": False,
    # Regime control
    "trend_sensitivity": D("0.5"),
    "range_ema_alpha": D("0.1"),
    "range_update_dead_band_pct": D("0.5"),
    "trend_order_scale_factor": D("0.0"),
    "trend_halt_threshold": D("0.0"),
    # Toxicity
    "toxicity_window_sec": 300,
    "toxicity_window_fills": 20,
    "toxicity_activation_count": 8,
    "toxicity_buy_ratio_soft": D("0.65"),
    "toxicity_buy_ratio_hard": D("0.80"),
    "toxicity_sell_ratio_soft": D("0.65"),
    "toxicity_sell_ratio_hard": D("0.80"),
    "toxicity_soft_spread_mult": D("1.30"),
    "toxicity_hard_spread_mult": D("1.80"),
    "toxicity_soft_size_mult": D("0.70"),
    "toxicity_hard_size_mult": D("0.40"),
    # Inventory skew
    "inventory_skew_soft_limit": D("0.60"),
    "inventory_skew_hard_limit": D("0.80"),
    "inventory_soft_size_mult": D("0.60"),
    "inventory_hard_size_mult": D("0.20"),
    "inventory_soft_spread_mult": D("1.30"),
    "inventory_hard_spread_mult": D("1.80"),
    "inventory_hard_disable_accumulation_side": True,
    # FLAIR
    "enable_flair_monitor": True,
    "flair_markout_sec": 30,
    "flair_window_sec": 1800,
    "flair_fee_bps": D("10"),
    # Mock hedge (off by default)
    "enable_hedge": False,
    "hedge_size_cap_pct": D("0.30"),
    "hedge_taker_fee_bps": D("5"),
    "hedge_funding_rate_per_hr": D("0.0001"),
}

# ---------------------------------------------------------------------------
# Phase grids
# ---------------------------------------------------------------------------

# Phase 1: spread × concentration  (25 combos)
PHASE1_SWEEP = {
    "spread_bps": [D("30"), D("55"), D("70"), D("100"), D("130")],
    "concentration": [D("5"), D("8"), D("12"), D("20"), D("30")],
}

# Phase 2: outer trigger × capital fraction  (9 combos, best P1 fixed)
PHASE2_SWEEP = {
    "outer_recenter_trigger_pct": [D("0.35"), D("0.50"), D("0.65")],
    "outer_capital_fraction": [D("0.20"), D("0.30"), D("0.40")],
}

# Phase 3: fill-asymmetry signal params  (9 combos, best P1+P2 fixed)
PHASE3_SWEEP = {
    "toxicity_window_fills": [10, 20, 40],
    "toxicity_buy_ratio_soft": [D("0.60"), D("0.65"), D("0.70")],
}

# Phase 5: HMM experiment  (4 combos: baseline + 3 v2 variants)
PHASE5_HMM = [
    {"hmm_use_fill_asymmetry": False, "hmm_refit_interval_sec": 1800},  # V0 baseline
    {"hmm_use_fill_asymmetry": True,  "hmm_refit_interval_sec": 300},   # V2-fast
    {"hmm_use_fill_asymmetry": True,  "hmm_refit_interval_sec": 600},   # V2-mid
    {"hmm_use_fill_asymmetry": True,  "hmm_refit_interval_sec": 1800},  # V2-slow
]

# Phase 7: outer range architecture — spread multiplier and concentration ratio
PHASE7_SWEEP = {
    "outer_spread_mult": [D("2.0"), D("2.5"), D("3.0")],
    "outer_range_mult": [D("2.0"), D("2.5"), D("3.0")],
}

# Phase 8: hedging overlay (enable_hedge=True)
PHASE8_HEDGE = [
    {"enable_hedge": True, "hedge_size_cap_pct": D("0.20"), "hedge_taker_fee_bps": D("5")},
    {"enable_hedge": True, "hedge_size_cap_pct": D("0.30"), "hedge_taker_fee_bps": D("5")},
    {"enable_hedge": True, "hedge_size_cap_pct": D("0.40"), "hedge_taker_fee_bps": D("5")},
    {"enable_hedge": False},  # baseline no-hedge for comparison
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_grid(sweep: Dict[str, list], fixed: dict) -> List[dict]:
    """Cartesian product of sweep values merged with fixed params."""
    keys = list(sweep.keys())
    grid = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        cfg = dict(fixed)
        for k, v in zip(keys, combo):
            cfg[k] = v
        grid.append(cfg)
    return grid


def run_phase(
    phase_name: str,
    tasks: List[dict],
    symbol: str,
    start: str,
    end: str,
    workers: int,
    base_bal: Decimal,
    quote_bal: Decimal,
    sort_by: str = "sharpe",
) -> List[dict]:
    """Run all tasks in parallel and return sorted results."""

    n = len(tasks)
    print(f"\n{'─' * 60}")
    print(f"  {phase_name}  ({n} runs | {start} → {end} | {workers}w)")
    print(f"{'─' * 60}")

    results: List[dict] = []
    t0 = time.time()
    completed = 0
    failed = 0

    mp_ctx = None
    if sys.platform == "linux":
        mp_ctx = multiprocessing.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(symbol, start, end, None, str(base_bal), str(quote_bal)),
        mp_context=mp_ctx,
    ) as exe:
        future_map = {exe.submit(_run_task, t): i for i, t in enumerate(tasks)}

        for fut in as_completed(future_map):
            completed += 1
            elapsed_total = time.time() - t0
            rate = completed / elapsed_total if elapsed_total > 0 else 1e-9
            eta = (n - completed) / rate

            try:
                res = fut.result()
                results.append(res)
                ret = res.get("total_return_pct", 0)
                shr = res.get("sharpe", 0)
                exc = res.get("excess_return_pct", 0)
                ela = res.get("_elapsed", 0)
                lbl = res.get("_label", "")
                print(f"  [{completed:3d}/{n}] {lbl or _short_params(res)}"
                      f"  ret={ret:+.2f}%  exc={exc:+.2f}%  sharpe={shr:.2f}"
                      f"  ({ela}s)  ETA={eta:.0f}s")
            except Exception as e:
                failed += 1
                print(f"  [{completed:3d}/{n}] FAILED: {e}")

    wall = time.time() - t0
    cpu = sum(r.get("_elapsed", 0) for r in results)
    print(f"\n  Done: {len(results)} OK / {failed} failed — "
          f"wall {wall:.0f}s, CPU {cpu:.0f}s, "
          f"parallelism {cpu/wall:.1f}x")

    reverse = sort_by not in ("max_drawdown_pct",)
    results.sort(key=lambda r: r.get(sort_by, -9999), reverse=reverse)
    return results


def _short_params(result: dict) -> str:
    """One-line summary of swept params in a result."""
    parts = []
    for k, v in result.items():
        if k.startswith("p_") and k not in ("p__validate", "p__label"):
            parts.append(f"{k[2:]}={v}")
    return " ".join(parts[:5])


def best_params(results: List[dict], fixed_keys: List[str]) -> dict:
    """Extract fixed params from the best result (rank 0)."""
    if not results:
        return {}
    best = results[0]
    return {k[2:]: best[k] for k in fixed_keys if k in best}


def print_top(results: List[dict], n: int = 10,
              metric_cols: Optional[List[str]] = None):
    if not results:
        return
    if metric_cols is None:
        metric_cols = ["total_return_pct", "excess_return_pct",
                       "max_drawdown_pct", "sharpe", "total_fills",
                       "recenters"]
    sweep_keys = sorted({k for r in results for k in r
                         if k.startswith("p_")
                         and k not in ("p__validate", "p__label")})
    header = ["#"] + [k[2:] for k in sweep_keys] + metric_cols
    rows = []
    for i, r in enumerate(results[:n]):
        row = [str(i + 1)]
        for k in sweep_keys:
            row.append(str(r.get(k, "")))
        for m in metric_cols:
            v = r.get(m, "")
            row.append(f"{v:+.2f}" if isinstance(v, float)
                       and "pct" in m else f"{v:.3f}"
                       if isinstance(v, float) else str(v))
        rows.append(row)

    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(header)]
    sep = "  "
    print("\n" + sep.join(h.rjust(w) for h, w in zip(header, widths)))
    print("─" * sum(widths + [2 * len(widths)]))
    for row in rows:
        print(sep.join(v.rjust(w) for v, w in zip(row, widths)))


def save_phase_csv(results: List[dict], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = []
    seen: set = set()
    for r in results:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# F1 / classification summary for Phase 5
# ---------------------------------------------------------------------------

def print_hmm_experiment(results: List[dict]):
    signals = ["fill_asymmetry", "autocorr_lag1", "vol_ratio"]
    horizons = ["15m", "1h", "4h"]
    variants = [("V0 baseline", False, 1800),
                ("V2 fast(5m)", True,  300),
                ("V2 mid(10m)", True,  600),
                ("V2 slow(30m)", True, 1800)]

    print("\n  HMM Experiment — Regime Detection F1 Scores")
    print("  " + "─" * 70)
    # Header
    cols = [f"{s}_{h}_f1" for s in signals for h in horizons]
    col_labels = [f"{s[:7]}/{h}" for s in signals for h in horizons]
    print(f"  {'Variant':22s} | " + " | ".join(f"{c:9s}" for c in col_labels))
    print("  " + "─" * 70)

    for name, use_fa, refit in variants:
        row_data = None
        for r in results:
            if (r.get("p_hmm_use_fill_asymmetry") == use_fa
                    and int(r.get("p_hmm_refit_interval_sec", 0)) == refit):
                row_data = r
                break
        if row_data is None:
            continue
        vals = []
        for col in cols:
            v = row_data.get(col, None)
            vals.append(f"{v:.3f}" if isinstance(v, float) else "  n/a ")
        print(f"  {name:22s} | " + " | ".join(f"{v:9s}" for v in vals))

    print("  " + "─" * 70)
    print("  Target: F1 > 0.65 and recall > 0.70 to include signal in composite")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CL-AMM Experiment Runner")
    parser.add_argument("--symbol", default="ADAUSDC")
    parser.add_argument("--train-year", type=int, default=2022)
    parser.add_argument("--validate-year", type=int, default=2023)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--base-balance", type=Decimal, default=D("100000"))
    parser.add_argument("--quote-balance", type=Decimal, default=D("27000"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--from-phase", type=int, default=1,
                        choices=list(range(1, 10)),
                        help="Skip to phase N (requires phase_N_results.json)")
    parser.add_argument("--sort-by", default="sharpe",
                        choices=["sharpe", "total_return_pct",
                                 "excess_return_pct", "max_drawdown_pct"])
    parser.add_argument("--top", type=int, default=5,
                        help="Number of best configs to carry forward")
    args = parser.parse_args()

    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    sym = args.symbol
    periods = CANDLE_PERIODS.get(sym, {})
    sweep_s, sweep_e = periods.get("sweep", ("2022-01-01", "2022-03-31"))
    train_s, train_e = periods.get(args.train_year, ("2022-01-01", "2022-12-31"))
    val_s, val_e     = periods.get(args.validate_year, ("2023-01-01", "2023-12-31"))
    test_s, test_e   = periods.get(args.test_year, ("2025-01-01", "2025-03-18"))
    W = args.workers
    BB = args.base_balance
    QB = args.quote_balance
    TOP = args.top
    SRT = args.sort_by

    print(f"\n{'=' * 60}")
    print(f"  CL-AMM EXPERIMENT RUNNER")
    print(f"{'=' * 60}")
    print(f"  Symbol:         {sym}")
    print(f"  Sweep (P1-3):   {sweep_s} → {sweep_e}  [HMM+Hurst disabled, ~6min/run]")
    print(f"  Train (P4):     {train_s} → {train_e}")
    print(f"  Validate (P4):  {val_s} → {val_e}")
    print(f"  Test (P5-6):    {test_s} → {test_e}")
    print(f"  Workers:        {W} (of 12 cores)")
    print(f"  Balance:        {BB} base / {QB} quote")
    print(f"  Sort by:        {SRT}")
    print(f"  Top-N carry:    {TOP}")
    print(f"  Output:         {RESULTS_DIR}")
    print(f"{'=' * 60}")

    # Track best params across phases
    best = dict(BASE_CONFIG)
    all_phase_results: Dict[int, List[dict]] = {}

    def _cache_path(phase: int) -> str:
        return os.path.join(RESULTS_DIR, f"phase{phase}_results.json")

    def _save_phase(phase: int, results: List[dict]):
        path = _cache_path(phase)
        with open(path, "w") as f:
            json.dump(results, f, default=str)

    def _load_phase(phase: int) -> Optional[List[dict]]:
        path = _cache_path(phase)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # -----------------------------------------------------------------------
    # Phase 1: spread × concentration on sweep quarter (HMM disabled)
    # -----------------------------------------------------------------------
    if args.from_phase <= 1:
        sweep_base = dict(BASE_CONFIG)
        sweep_base.update(SWEEP_OVERRIDE)
        grid = build_grid(PHASE1_SWEEP, sweep_base)
        for t in grid:
            t["_label"] = f"sprd={t['spread_bps']} conc={t['concentration']}"
        results1 = run_phase(
            "PHASE 1 — Spread × Concentration", grid,
            sym, sweep_s, sweep_e, W, BB, QB, SRT)
        save_phase_csv(results1,
                       os.path.join(RESULTS_DIR, "phase1_train.csv"))
        _save_phase(1, results1)
        all_phase_results[1] = results1
        print(f"\n  Top {TOP} — Phase 1:")
        print_top(results1, TOP)
    else:
        results1 = _load_phase(1) or []
        all_phase_results[1] = results1
        print(f"\n  Phase 1: loaded {len(results1)} cached results")

    # Extract best spread + concentration
    p1_keys = ["p_spread_bps", "p_concentration"]
    best.update({k[2:]: results1[0][k] for k in p1_keys if results1})

    # -----------------------------------------------------------------------
    # Phase 2: outer trigger × capital fraction  (best P1 fixed, sweep quarter)
    # -----------------------------------------------------------------------
    if args.from_phase <= 2:
        fixed2 = dict(BASE_CONFIG)
        fixed2.update(SWEEP_OVERRIDE)
        fixed2.update({k[2:]: results1[0][k]
                       for k in p1_keys if results1})
        grid2 = build_grid(PHASE2_SWEEP, fixed2)
        for t in grid2:
            t["_label"] = (f"outer_trigger={t['outer_recenter_trigger_pct']} "
                           f"outer_frac={t['outer_capital_fraction']}")
        results2 = run_phase(
            "PHASE 2 — Outer Trigger × Capital Fraction", grid2,
            sym, sweep_s, sweep_e, W, BB, QB, SRT)
        save_phase_csv(results2,
                       os.path.join(RESULTS_DIR, "phase2_train.csv"))
        _save_phase(2, results2)
        all_phase_results[2] = results2
        print(f"\n  Top {TOP} — Phase 2:")
        print_top(results2, TOP)
    else:
        results2 = _load_phase(2) or []
        all_phase_results[2] = results2
        print(f"\n  Phase 2: loaded {len(results2)} cached results")

    p2_keys = ["p_outer_recenter_trigger_pct", "p_outer_capital_fraction"]
    best.update({k[2:]: results2[0][k] for k in p2_keys if results2})

    # -----------------------------------------------------------------------
    # Phase 3: signal tuning  (best P1+P2 fixed, sweep quarter)
    # -----------------------------------------------------------------------
    if args.from_phase <= 3:
        fixed3 = dict(BASE_CONFIG)
        fixed3.update(SWEEP_OVERRIDE)
        fixed3.update({k[2:]: results1[0][k] for k in p1_keys if results1})
        fixed3.update({k[2:]: results2[0][k] for k in p2_keys if results2})
        grid3 = build_grid(PHASE3_SWEEP, fixed3)
        for t in grid3:
            t["_label"] = (f"wfills={t['toxicity_window_fills']} "
                           f"soft={t['toxicity_buy_ratio_soft']}")
            t["toxicity_sell_ratio_soft"] = t["toxicity_buy_ratio_soft"]
        results3 = run_phase(
            "PHASE 3 — Fill-Asymmetry Signal Tuning", grid3,
            sym, sweep_s, sweep_e, W, BB, QB, SRT)
        save_phase_csv(results3,
                       os.path.join(RESULTS_DIR, "phase3_train.csv"))
        _save_phase(3, results3)
        all_phase_results[3] = results3
        print(f"\n  Top {TOP} — Phase 3:")
        print_top(results3, TOP)
    else:
        results3 = _load_phase(3) or []
        all_phase_results[3] = results3
        print(f"\n  Phase 3: loaded {len(results3)} cached results")

    p3_keys = ["p_toxicity_window_fills", "p_toxicity_buy_ratio_soft",
               "p_toxicity_sell_ratio_soft"]
    best.update({k[2:]: results3[0][k] for k in p3_keys if results3})

    # -----------------------------------------------------------------------
    # Phase 4: cross-validate top-N on 2023
    # -----------------------------------------------------------------------
    if args.from_phase <= 4:
        # Pick top-N distinct configs from the Phase 3 winner plus
        # top-4 from Phase 1 and Phase 2 that differed in core params
        candidates = (results3[:TOP] if results3 else [])
        if not candidates and results1:
            candidates = results1[:TOP]

        val_tasks = []
        for i, r in enumerate(candidates):
            cfg = dict(BASE_CONFIG)
            for k, v in r.items():
                if k.startswith("p_") and not k.startswith("p__"):
                    key = k[2:]
                    if key not in SWEEP_OVERRIDE:  # don't carry sweep-only overrides
                        cfg[key] = v
            cfg.update(SWEEP_OVERRIDE)  # disable HMM/Hurst for speed
            cfg["_label"] = f"cfg{i+1}"
            val_tasks.append(cfg)

        results4 = run_phase(
            "PHASE 4 — Cross-Validate Top Configs (2023)", val_tasks,
            sym, val_s, val_e, min(W, len(val_tasks)), BB, QB, SRT)
        save_phase_csv(results4,
                       os.path.join(RESULTS_DIR, "phase4_validate.csv"))
        _save_phase(4, results4)
        all_phase_results[4] = results4
        print(f"\n  Validation results (2023):")
        print_top(results4, len(results4))
    else:
        results4 = _load_phase(4) or []
        all_phase_results[4] = results4
        print(f"\n  Phase 4: loaded {len(results4)} cached results")

    # Best validated config — exclude SWEEP_OVERRIDE keys so HMM runs normally in P5/P6
    validated_best = dict(BASE_CONFIG)
    src = results4[0] if results4 else (results3[0] if results3 else {})
    for k, v in src.items():
        if k.startswith("p_") and not k.startswith("p__"):
            key = k[2:]
            if key not in SWEEP_OVERRIDE:
                validated_best[key] = v

    # -----------------------------------------------------------------------
    # Phase 5: HMM experiment (2025 test year, with indicator validation)
    # -----------------------------------------------------------------------
    if args.from_phase <= 5:
        hmm_tasks = []
        for variant in PHASE5_HMM:
            cfg = dict(validated_best)
            cfg.update(variant)
            use_fa = variant["hmm_use_fill_asymmetry"]
            refit = variant["hmm_refit_interval_sec"]
            label = (f"HMM-V0 (baseline)"
                     if not use_fa
                     else f"HMM-V2 refit={refit}s")
            cfg["_label"] = label
            cfg["_validate"] = True
            vcsv = os.path.join(
                RESULTS_DIR,
                f"hmm_{'v2' if use_fa else 'v0'}_{refit}s_indicator_val.csv")
            cfg["_val_csv"] = vcsv
            hmm_tasks.append(cfg)

        results5 = run_phase(
            "PHASE 5 — HMM Experiment (2025 + F1/MCC)", hmm_tasks,
            sym, test_s, test_e, min(W, len(hmm_tasks)), BB, QB, SRT)
        save_phase_csv(results5,
                       os.path.join(RESULTS_DIR, "phase5_hmm.csv"))
        _save_phase(5, results5)
        all_phase_results[5] = results5
        print_hmm_experiment(results5)
    else:
        results5 = _load_phase(5) or []
        all_phase_results[5] = results5
        print(f"\n  Phase 5: loaded {len(results5)} cached results")
        print_hmm_experiment(results5)

    # -----------------------------------------------------------------------
    # Phase 6: Final test — best config on 2025
    # -----------------------------------------------------------------------
    if args.from_phase <= 6:
        final_cfg = dict(validated_best)
        final_cfg["_label"] = "FINAL"
        final_cfg["_validate"] = True
        final_csv = os.path.join(RESULTS_DIR, "final_indicator_val.csv")
        final_cfg["_val_csv"] = final_csv

        results6 = run_phase(
            "PHASE 6 — Final Test (2025)", [final_cfg],
            sym, test_s, test_e, 1, BB, QB, SRT)
        save_phase_csv(results6,
                       os.path.join(RESULTS_DIR, "phase6_final.csv"))
        _save_phase(6, results6)
        all_phase_results[6] = results6
    else:
        results6 = _load_phase(6) or []
        all_phase_results[6] = results6

    # -----------------------------------------------------------------------
    # Phase 7: outer range architecture (full 2022 training year)
    # -----------------------------------------------------------------------
    if args.from_phase <= 7:
        fixed7 = dict(validated_best)
        fixed7.update(SWEEP_OVERRIDE)
        grid7 = build_grid(PHASE7_SWEEP, fixed7)
        for t in grid7:
            t["_label"] = (f"outer_sprd_mult={t['outer_spread_mult']} "
                           f"outer_rng_mult={t['outer_range_mult']}")
        results7 = run_phase(
            "PHASE 7 — Outer Range Architecture (2022)", grid7,
            sym, train_s, train_e, W, BB, QB, SRT)
        save_phase_csv(results7, os.path.join(RESULTS_DIR, "phase7_outer_range.csv"))
        _save_phase(7, results7)
        all_phase_results[7] = results7
        print(f"\n  Top {TOP} — Phase 7:")
        print_top(results7, TOP)
    else:
        results7 = _load_phase(7) or []
        all_phase_results[7] = results7
        print(f"\n  Phase 7: loaded {len(results7)} cached results")

    p7_keys = ["p_outer_spread_mult", "p_outer_range_mult"]
    if results7:
        validated_best.update({k[2:]: results7[0][k] for k in p7_keys if k in results7[0]})

    # -----------------------------------------------------------------------
    # Phase 8: hedge overlay comparison (2022 training year)
    # -----------------------------------------------------------------------
    if args.from_phase <= 8:
        hedge_tasks = []
        for variant in PHASE8_HEDGE:
            cfg = dict(validated_best)
            cfg.update(SWEEP_OVERRIDE)
            cfg.update(variant)
            enabled = variant.get("enable_hedge", False)
            cap = variant.get("hedge_size_cap_pct", D("0"))
            cfg["_label"] = (f"hedge ON cap={cap}" if enabled else "hedge OFF (baseline)")
            cfg["_validate"] = True
            vcsv = os.path.join(RESULTS_DIR, f"hedge_{'on' if enabled else 'off'}_{cap}_val.csv")
            cfg["_val_csv"] = vcsv
            hedge_tasks.append(cfg)

        results8 = run_phase(
            "PHASE 8 — Hedge Overlay (2022)", hedge_tasks,
            sym, train_s, train_e, min(W, len(hedge_tasks)), BB, QB, SRT)
        save_phase_csv(results8, os.path.join(RESULTS_DIR, "phase8_hedge.csv"))
        _save_phase(8, results8)
        all_phase_results[8] = results8
        print(f"\n  Phase 8 — Hedge vs No-Hedge:")
        print_top(results8, len(results8))
    else:
        results8 = _load_phase(8) or []
        all_phase_results[8] = results8
        print(f"\n  Phase 8: loaded {len(results8)} cached results")

    # -----------------------------------------------------------------------
    # Phase 9: regime stability — best config across crash/ranging/bull + multi-year
    # -----------------------------------------------------------------------
    if args.from_phase <= 9:
        final_cfg = dict(validated_best)
        final_cfg.update(SWEEP_OVERRIDE)

        regime_windows = [
            ("regime_crash_1",   "crash_jan2022"),
            ("regime_crash_2",   "crash_may2022"),
            ("regime_ranging_1", "ranging_aug2022"),
            ("regime_ranging_2", "ranging_oct2022"),
            ("regime_bull_1",    "bull_mar2021"),
            ("regime_bull_2",    "bull_jul2023"),
        ]
        stability_years = [2021, 2023]

        all9_results: List[dict] = []

        for window_key, label in regime_windows:
            ws, we = periods.get(window_key, ("2022-01-01", "2022-01-31"))
            cfg = dict(final_cfg)
            cfg["_label"] = label
            res = run_phase(
                f"PHASE 9 — {label}", [cfg],
                sym, ws, we, 1, BB, QB, SRT)
            all9_results.extend(res)

        for yr in stability_years:
            ys, ye = periods.get(yr, ("2022-01-01", "2022-12-31"))
            cfg = dict(final_cfg)
            cfg["_label"] = f"full_{yr}"
            res = run_phase(
                f"PHASE 9 — full {yr}", [cfg],
                sym, ys, ye, 1, BB, QB, SRT)
            all9_results.extend(res)

        save_phase_csv(all9_results,
                       os.path.join(RESULTS_DIR, "phase9_regime_stability.csv"))
        _save_phase(9, all9_results)
        all_phase_results[9] = all9_results
        print(f"\n  Regime Stability Results:")
        print_top(all9_results, len(all9_results))
    else:
        results9 = _load_phase(9) or []
        all_phase_results[9] = results9
        print(f"\n  Phase 9: loaded {len(results9)} cached results")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")

    if results6:
        r = results6[0]
        print(f"\n  FINAL RESULT (2025 out-of-sample):")
        print(f"    Total return:        {r.get('total_return_pct', 0):+.2f}%")
        print(f"    vs HODL:             {r.get('excess_return_pct', 0):+.2f}%")
        print(f"    FLAIR fee captured:  {r.get('flair_fee_pct', 0):+.2f}% "
              f"({r.get('flair_lifetime_fee_quote', 0):.2f} quote)")
        print(f"    FLAIR LVR paid:      {r.get('flair_lvr_pct', 0):+.2f}% "
              f"({r.get('flair_lifetime_lvr_quote', 0):.2f} quote)")
        print(f"    FLAIR net (fee-LVR): {r.get('flair_net_pct', 0):+.2f}% "
              f"(positive = fees > adverse selection)")
        print(f"    Strategy vs FLAIR:   "
              f"{r.get('return_minus_flair_net_pct', 0):+.2f}% "
              f"(extra alpha beyond modelled LVR)")
        print(f"    Max drawdown:        {r.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Sharpe:              {r.get('sharpe', 0):.3f}")
        print(f"    Total fills:         {r.get('total_fills', 0)}")
        print(f"    Recenters:           {r.get('recenters', 0)}")

    print(f"\n  BEST PARAMETERS:")
    key_params = [
        "spread_bps", "concentration",
        "outer_recenter_trigger_pct", "outer_capital_fraction",
        "outer_spread_mult", "outer_range_mult",
        "toxicity_window_fills", "toxicity_buy_ratio_soft",
        "hmm_use_fill_asymmetry", "hmm_refit_interval_sec",
    ]
    for k in key_params:
        src = validated_best.get(k, BASE_CONFIG.get(k, "?"))
        print(f"    {k:35s} = {src}")

    print(f"\n  DECISION CRITERIA (from Phase 5):")
    print(f"    HMM v2 worth adding if F1 improvement > 0.05 on 2025 test")
    print(f"    New signals (autocorr/vol_ratio) worth adding if F1 > 0.65")
    print(f"    Regime-adaptive worth complexity if Sharpe delta > 0.2 vs fixed")

    print(f"\n  Output files: {RESULTS_DIR}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
