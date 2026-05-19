#!/bin/bash
# Coarse one-at-a-time sweep for axis #4: outer dual-range band.
# Baseline = 2024 yearly-sweep winner (spread=50, concentration=15), full year 2024.
#
# Hypothesis (#4): without hedge, the outer band is your shock absorber for
# trends — catches price excursions at a worse price but stops you running
# out of inventory mid-recenter. outer_capital_fraction=0.0 disables the band
# entirely (single-band control).

set -e

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"
SYMBOL=ADAUSDC
WORKERS=8
START=2024-01-01
END=2024-12-31
OUTDIR=results/coarse/outer

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
    concentration=15
    min_concentration=3
    max_concentration=40
    natr_range_scale=1.0
    outer_spread_mult=2.5
    outer_recenter_trigger_pct=0.65
    outer_capital_fraction=0.30
    outer_spread_pct_of_range=0.60
)

mkdir -p "$OUTDIR"
echo "========================================"
echo "  COARSE SWEEP — outer dual-range (#4)"
echo "  Window: $START → $END   Workers: $WORKERS"
echo "========================================"

# --- 4a. Outer capital fraction (0.0 = no outer band) ---
echo "\n[4a] outer_capital_fraction"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 outer_capital_fraction=0.0,0.20,0.40,0.60 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/4a_capital_fraction"

# --- 4b. Outer spread position within range ---
echo "\n[4b] outer_spread_pct_of_range"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 outer_spread_pct_of_range=0.40,0.60,0.80 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/4b_spread_pct"

# --- 4c. Recenter trigger ---
echo "\n[4c] outer_recenter_trigger_pct"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 outer_recenter_trigger_pct=0.50,0.65,0.80 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/4c_recenter_trigger"

# --- 4d. Outer spread multiplier (floor) ---
echo "\n[4d] outer_spread_mult"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 outer_spread_mult=2.0,3.5,5.0 \
    "${COMMON_SET[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/4d_spread_mult"

echo "\nCoarse outer sweep done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/*/sweep_*.csv 2>/dev/null
