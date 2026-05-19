#!/bin/bash
# Coarse one-at-a-time sweep for axis #5: toxicity & inventory guards.
# Baseline = 2024 yearly-sweep winner (spread=50, concentration=15), full year 2024.
#
# Hypothesis (#5): without hedge, the inventory guard is the only structural
# defense against directional drift. Toxicity guard widens spreads when fills
# imbalance signals adverse selection. Both have aggressive settings worth
# testing beyond phase6 defaults.
#
# Sweeps are split into TOX (4 axes) and INV (4 axes). Inventory soft/hard
# limits are paired since hard > soft is enforced.

set -e

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"
SYMBOL=ADAUSDC
WORKERS=8
START=2024-01-01
END=2024-12-31
OUTDIR=results/coarse/guards

# NOTE: COMMON_FLAT has NO `--set` prefix so each invocation merges these
# defaults with its loop-specific overrides under a SINGLE --set flag. Using
# two `--set` flags in one argparse call drops the first set entirely
# (nargs="*" + repeated flag = last occurrence wins).
COMMON_FLAT=(
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
    toxicity_buy_ratio_hard=0.80
    toxicity_hard_spread_mult=1.80
    toxicity_hard_size_mult=0.40
    inventory_skew_soft_limit=0.60
    inventory_skew_hard_limit=0.80
    inventory_hard_size_mult=0.20
    inventory_soft_spread_mult=1.30
)

mkdir -p "$OUTDIR"
echo "========================================"
echo "  COARSE SWEEP — toxicity & inventory guards (#5)"
echo "  Window: $START → $END   Workers: $WORKERS"
echo "========================================"

# ---------- TOXICITY ----------

# --- 5a. Toxicity activation count (lower = react faster) ---
echo "\n[5a] toxicity_activation_count"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 toxicity_activation_count=3,5,8,12,30 \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5a_tox_activation"

# --- 5b. Toxicity buy_ratio_hard (lower = trip earlier) ---
echo "\n[5b] toxicity_buy_ratio_hard"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 toxicity_buy_ratio_hard=0.65,0.75,0.85 \
    --set "${COMMON_FLAT[@]}" toxicity_sell_ratio_hard=0.85 \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5b_tox_buy_ratio_hard"

# --- 5c. Toxicity hard spread multiplier (widening magnitude) ---
echo "\n[5c] toxicity_hard_spread_mult"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 toxicity_hard_spread_mult=1.8,3.0,5.0 \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5c_tox_hard_spread_mult"

# --- 5d. Toxicity hard size multiplier (0.0 = full disengage on toxic side) ---
echo "\n[5d] toxicity_hard_size_mult"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 toxicity_hard_size_mult=0.0,0.1,0.3,0.6 \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5d_tox_hard_size_mult"

# ---------- INVENTORY (the bigger lever — no hedge) ----------

# --- 5e. Inventory soft/hard limit pair (hard = soft + 0.20) ---
echo "\n[5e] inventory soft/hard limits (paired)"
for PAIR in "0.30:0.55" "0.45:0.70" "0.60:0.85"; do
    IFS=":" read -r SOFT HARD <<< "$PAIR"
    LBL=$(printf "soft%02d_hard%02d" "$(printf '%.0f' $(echo "$SOFT * 100" | bc -l))" "$(printf '%.0f' $(echo "$HARD * 100" | bc -l))")
    echo "  → soft=$SOFT, hard=$HARD"
    $PYTHON backtest_sweep.py \
        --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
        --balanced --portfolio-usd 100000 --workers $WORKERS \
        --sweep spread_bps=50 \
        --set "${COMMON_FLAT[@]}" inventory_skew_soft_limit=$SOFT inventory_skew_hard_limit=$HARD \
        --sort-by excess_return_pct \
        --output "$OUTDIR/5e_inv_limits_$LBL"
done

# --- 5f. Inventory hard size multiplier (0.0 = full stop on accumulation side) ---
echo "\n[5f] inventory_hard_size_mult"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 inventory_hard_size_mult=0.0,0.10,0.25,0.50 \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5f_inv_hard_size_mult"

# --- 5g. Inventory soft spread multiplier (push quote to incentivize unwind) ---
echo "\n[5g] inventory_soft_spread_mult"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 inventory_soft_spread_mult=1.0,1.3,1.8,2.5 \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5g_inv_soft_spread_mult"

# --- 5h. Inventory hard_disable_accumulation_side ablation ---
echo "\n[5h] inventory_hard_disable_accumulation_side"
$PYTHON backtest_sweep.py \
    --strategy cl-amm --symbol $SYMBOL --start $START --end $END \
    --balanced --portfolio-usd 100000 --workers $WORKERS \
    --sweep spread_bps=50 inventory_hard_disable_accumulation_side=True,False \
    --set "${COMMON_FLAT[@]}" \
    --sort-by excess_return_pct \
    --output "$OUTDIR/5h_inv_disable_accum"

echo "\nCoarse guards sweep done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/*/sweep_*.csv 2>/dev/null
