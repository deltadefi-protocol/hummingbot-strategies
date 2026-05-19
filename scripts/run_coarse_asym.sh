#!/bin/bash
# Coarse one-at-a-time sweep for axis #6: asymmetric spread / skew.
# Baseline = 2024 yearly-sweep winner (spread=50, concentration=15), full year 2024.
#
# Hypothesis (#6): inventory-aware quoting (asymmetric spread) should reduce
# inventory drift compared to symmetric quoting at the cost of fewer fills on
# the accumulating side. Expected smallest impact among the axes but cheap.

set -e

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"
SYMBOL=ADAUSDC
WORKERS=8
START=2024-01-01
END=2024-12-31
OUTDIR=results/coarse/asym

COMMON_SET=(
    --set
    enable_asymmetric_spread=True
    skew_sensitivity=0.5
    trend_sensitivity=0.5
    trend_order_scale_factor=0.0
    trend_halt_threshold=0.0
    range_ema_alpha=0.1
    outer_capital_fraction=0.30
    outer_recenter_trigger_pct=0.65
    concentration=15
    min_concentration=3
    max_concentration=40
    natr_range_scale=1.0
    toxicity_activation_count=8
    inventory_skew_soft_limit=0.60
    inventory_skew_hard_limit=0.80
    min_spread_bps=20
)

mkdir -p "$OUTDIR"
echo "========================================"
echo "  COARSE SWEEP — asymmetric spread (#6)"
echo "  Window: $START → $END   Workers: $WORKERS"
echo "========================================"

# --- 6a. Skew sensitivity (0.0 = symmetric, higher = more inventory-skewed) ---
echo "\n[6a] skew_sensitivity"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 skew_sensitivity=0.0,0.5,1.0,1.5,2.0 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/6a_skew_sensitivity"

# --- 6b. Min spread floor (matters in calm regimes) ---
echo "\n[6b] min_spread_bps"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 min_spread_bps=5,10,20,40 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/6b_min_spread_bps"

# --- 6c. Enable/disable asymmetric ablation ---
echo "\n[6c] enable_asymmetric_spread"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 enable_asymmetric_spread=True,False \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/6c_enable_asym"

echo "\nCoarse asym sweep done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/*/sweep_*.csv 2>/dev/null
