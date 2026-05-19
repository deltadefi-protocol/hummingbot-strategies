#!/bin/bash
# Coarse sweep for trend layer (#7) across distinct market-regime windows.
# Sweeps trend_halt_threshold and trend_order_scale_factor (Cartesian, 4x4)
# at fixed trend_sensitivity=0.5, then a separate scan over trend_sensitivity.
#
# Hypothesis: phase6 indicator validation showed near-zero correlation between
# trend signals and forward |return| (~0.03-0.05) at 15m/1h/4h. So trend params
# probably can't help by PREDICTING direction. But they might help by RISK
# REDUCTION — halting quotes during trends so the bot doesn't become exit
# liquidity. Expected: hurt slightly in rangebound (false positives), help
# significantly in trends. If it doesn't help in trends, kill the feature.
#
# Each axis runs across 4 regime windows so per-regime effect is visible:
#   - bull_2021q1  : 2021-02-01 → 2021-05-01  (strong up-trend)
#   - bear_2022    : 2022-05-01 → 2022-07-01  (Luna collapse, sharp down)
#   - range_2023   : 2023-04-01 → 2023-09-01  (rangebound mid-2023)
#   - shock_2020   : 2020-03-01 → 2020-04-01  (COVID crash, vol spike)

set -e

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"
SYMBOL=ADAUSDC
WORKERS=8
OUTDIR=results/coarse/trend

REGIMES=(
    "bull_2021q1:2021-02-01:2021-05-01"
    "bear_2022:2022-05-01:2022-07-01"
    # range_2023 removed: Binance has no continuous ADAUSDC 1m data for 2023
    # (only ~3.4k candles for Dec 28-31). Replaced with a calmer 2024 chunk.
    "range_2024q3:2024-07-01:2024-09-01"
    "shock_2020:2020-03-01:2020-04-01"
)

# NOTE: COMMON_FLAT carries no `--set` prefix — every invocation must merge
# it with its loop-specific overrides under a SINGLE --set flag. Repeating
# --set on the same command drops the earlier occurrence (argparse nargs="*"
# + repeated flag = last wins).
COMMON_FLAT=(
    enable_asymmetric_spread=True
    skew_sensitivity=0.5
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
)

mkdir -p "$OUTDIR"
echo "========================================"
echo "  COARSE SWEEP — trend layer (#7) × regimes"
echo "  Regimes: bull / bear / range / shock"
echo "========================================"

for ENTRY in "${REGIMES[@]}"; do
    IFS=":" read -r LABEL START END <<< "$ENTRY"
    echo ""
    echo "----------------------------------------"
    echo "  Regime: $LABEL  ($START → $END)"
    echo "----------------------------------------"

    # --- 7a. Cartesian: trend_halt_threshold × trend_order_scale_factor ---
    echo "  [7a] halt_threshold × order_scale_factor (Cartesian 4×4)"
    $PYTHON backtest_sweep.py \
        --strategy cl-amm --symbol $SYMBOL --start "$START" --end "$END" \
        --balanced --portfolio-usd 100000 --workers $WORKERS \
        --sweep spread_bps=50 \
                trend_halt_threshold=0.0,0.3,0.5,0.7 \
                trend_order_scale_factor=0.0,0.3,0.6,1.0 \
        --set "${COMMON_FLAT[@]}" trend_sensitivity=0.5 \
        --sort-by excess_return_pct \
        --output "$OUTDIR/${LABEL}_7a_halt_x_scale"

    # --- 7b. trend_sensitivity at default halt/scale ---
    echo "  [7b] trend_sensitivity"
    $PYTHON backtest_sweep.py \
        --strategy cl-amm --symbol $SYMBOL --start "$START" --end "$END" \
        --balanced --portfolio-usd 100000 --workers $WORKERS \
        --sweep spread_bps=50 trend_sensitivity=0.0,0.3,0.5,1.0,1.5 \
        --set "${COMMON_FLAT[@]}" trend_halt_threshold=0.5 trend_order_scale_factor=0.5 \
        --sort-by excess_return_pct \
        --output "$OUTDIR/${LABEL}_7b_sensitivity"
done

echo "\nCoarse trend sweep done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/*/sweep_*.csv 2>/dev/null
