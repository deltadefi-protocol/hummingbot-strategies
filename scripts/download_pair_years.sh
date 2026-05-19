#!/bin/bash
# Download candles for a symbol, one year at a time, with retry on failure.
# Per-year chunking survives transient SSL/timeout errors that crash the
# full-range fetch — each yearly window has its own cache file, and the
# walk-forward (run_best_tuned.py) loads each year independently via
# CandleDataLoader's superset-cache slicing.
#
# Usage:
#   bash download_pair_years.sh ETHUSDC
#   bash download_pair_years.sh BTCUSDC
#   bash download_pair_years.sh BTCUSDC 2022 2023      # specific years only

set -e
cd "$(dirname "$0")"

SYMBOL="${1:?symbol required (e.g. ETHUSDC)}"
shift || true
PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
RETRIES=3

# Default to 2020..2025q1
YEARS=("2020-01-01:2020-12-31"
       "2021-01-01:2021-12-31"
       "2022-01-01:2022-12-31"
       "2023-01-01:2023-12-31"
       "2024-01-01:2024-12-31"
       "2025-01-01:2025-03-19")

# If extra args provided, override with explicit year ranges
if [ $# -gt 0 ]; then
    YEARS=()
    for y in "$@"; do
        if [ "$y" = "2025" ]; then
            YEARS+=("2025-01-01:2025-03-19")
        else
            YEARS+=("${y}-01-01:${y}-12-31")
        fi
    done
fi

echo "════════════════════════════════════════════════════════════"
echo "  Downloading $SYMBOL year-by-year (${#YEARS[@]} chunks)"
echo "════════════════════════════════════════════════════════════"

for ENTRY in "${YEARS[@]}"; do
    IFS=":" read -r START END <<< "$ENTRY"
    echo ""
    echo "──── $SYMBOL  $START → $END ────"

    OK=0
    for ATTEMPT in $(seq 1 $RETRIES); do
        if $PYTHON download_candles.py "$SYMBOL" "$START" "$END" 2>&1 | tail -8; then
            OK=1
            break
        fi
        echo "  Attempt $ATTEMPT failed, retrying after 5s..."
        sleep 5
    done
    if [ $OK -ne 1 ]; then
        echo "  ✗ FAILED after $RETRIES retries: $SYMBOL $START → $END"
        echo "  Moving on to next year; rerun this script later for missing years."
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done. Cache files:"
ls -lh cache/${SYMBOL}_*.csv 2>/dev/null
echo "════════════════════════════════════════════════════════════"
