#!/bin/bash
# Coarse one-at-a-time sweep for axis #3: dynamic concentration (NATR-driven range).
# Baseline = 2024 yearly-sweep winner (spread=50, concentration=15), full year 2024.
# Each sub-sweep varies ONE axis at a time around the baseline.
#
# Hypothesis (#3): the dynamic range machinery (NATR scale, baseline, corridor
# bounds) should beat a static range. natr_range_scale=0.0 is the static control.

set -e

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"
SYMBOL=ADAUSDC
WORKERS=8
START=2024-01-01
END=2024-12-31
OUTDIR=results/coarse/dyn_concentration

# Baseline (all "_all_on" defaults, varied axis is the only thing different)
COMMON_SET=(
    --set
    enable_asymmetric_spread=True
    skew_sensitivity=0.5
    trend_sensitivity=0.5
    trend_order_scale_factor=0.0
    trend_halt_threshold=0.0
    range_ema_alpha=0.1
    toxicity_activation_count=8
    inventory_skew_soft_limit=0.60
    inventory_skew_hard_limit=0.80
    outer_capital_fraction=0.30
    outer_recenter_trigger_pct=0.65
    concentration=15
    min_concentration=3
    max_concentration=40
    natr_range_scale=1.0
    natr_baseline=0.005
)

mkdir -p "$OUTDIR"
echo "========================================"
echo "  COARSE SWEEP — dynamic concentration (#3)"
echo "  Window: $START → $END   Workers: $WORKERS"
echo "========================================"

# --- 3a. NATR range scale (0.0 = static control) ---
echo "\n[3a] natr_range_scale"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 natr_range_scale=0.0,0.5,1.0,2.0,3.0 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/3a_natr_range_scale"

# --- 3b. NATR baseline ---
echo "\n[3b] natr_baseline"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 natr_baseline=0.003,0.005,0.010,0.020 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/3b_natr_baseline"

# --- 3c. Corridor floor (min_concentration) ---
echo "\n[3c] min_concentration"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 min_concentration=3,5,10,14 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/3c_min_concentration"

# --- 3d. Corridor ceiling (max_concentration) ---
echo "\n[3d] max_concentration"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 max_concentration=20,30,40,60 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/3d_max_concentration"

echo "\nCoarse dyn-concentration sweep done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/*/sweep_*.csv 2>/dev/null
