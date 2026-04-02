#!/bin/bash
# Remaining sweeps: CL-AMM years + AMM 5yr + CL-AMM 5yr
# (AMM 2020-2025 already done, but need re-run with balance cap fix)

set -e

PYTHON=/Users/yyy/miniforge3/envs/hummingbot/envs/hummingbot/bin/python
SYMBOL=ADAUSDC
WORKERS=8

YEARS=("2020-01-01:2020-12-31:2020"
       "2021-01-01:2021-12-31:2021"
       "2022-01-01:2022-12-31:2022"
       "2023-01-01:2023-12-31:2023"
       "2024-01-01:2024-12-31:2024"
       "2025-01-01:2025-03-19:2025"
       "2020-01-01:2025-03-19:5yr")

TOTAL=$((${#YEARS[@]} * 2))
RUN=0
START_TIME=$(date +%s)

echo "========================================"
echo "  REMAINING SWEEPS (balance-cap fix applied)"
echo "  Symbol: $SYMBOL | Workers: $WORKERS"
echo "  Started: $(date)"
echo "========================================"

# Re-run AMM with balance cap fix
for entry in "${YEARS[@]}"; do
    IFS=":" read -r START END LABEL <<< "$entry"
    RUN=$((RUN + 1))
    echo ""
    echo "[$RUN/$TOTAL] AMM $LABEL ($START to $END)"
    echo "----------------------------------------"
    $PYTHON backtest_sweep.py \
        --strategy amm --symbol $SYMBOL \
        --start "$START" --end "$END" \
        --sweep spread_bps=20:100:10 amplification=3,5,10,20 \
        --workers $WORKERS \
        --sort-by excess_return_pct \
        --output "results/amm_${LABEL}"
done

# CL-AMM sweeps
for entry in "${YEARS[@]}"; do
    IFS=":" read -r START END LABEL <<< "$entry"
    RUN=$((RUN + 1))
    echo ""
    echo "[$RUN/$TOTAL] CL-AMM $LABEL ($START to $END)"
    echo "----------------------------------------"
    $PYTHON backtest_sweep.py \
        --strategy cl-amm --symbol $SYMBOL \
        --start "$START" --end "$END" \
        --sweep spread_bps=20:100:10 concentration=3,5,10,15,20 \
        --workers $WORKERS \
        --sort-by excess_return_pct \
        --output "results/cl-amm_${LABEL}"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "========================================"
echo "  ALL SWEEPS COMPLETE"
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "Results:"
ls -lh results/*/sweep_*.csv 2>/dev/null
