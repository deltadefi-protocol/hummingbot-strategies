#!/bin/bash
#SBATCH --job-name=bt-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --account=normal
#SBATCH --cpus-per-task=80
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================
# SLURM Job Array: Backtest Parameter Sweep
# ============================================
#
# Array index mapping:
#   0 = 2020    3 = 2023    6 = full 5yr (2020-2025)
#   1 = 2021    4 = 2024
#   2 = 2022    5 = 2025
#
# Usage:
#   sbatch --array=0-6 slurm/sweep_backtest.sh amm ADAUSDT
#   sbatch --array=0-6 slurm/sweep_backtest.sh cl-amm ADAUSDT
#   sbatch --array=3   slurm/sweep_backtest.sh amm ADAUSDT   # 2023 only
#   sbatch --array=6   slurm/sweep_backtest.sh amm ADAUSDT   # 5yr only
#   sbatch --array=0-6 --cpus-per-task=40 slurm/sweep_backtest.sh amm ADAUSDT
#
# ============================================

# ---------- Arguments ----------
STRATEGY="${1:?Usage: sbatch --array=0-6 slurm/sweep_backtest.sh <amm|cl-amm> [SYMBOL]}"
SYMBOL="${2:-ADAUSDT}"

# ---------- Paths (adjust these) ----------
PROJECT_DIR="/research/d7/fyp25/yyyu2/hummingbot-strategies"
export UV_TOOL_DIR="/research/d7/fyp25/yyyu2/.uv/tools"
export UV_TOOL_BIN_DIR="/research/d7/fyp25/yyyu2/.uv/bin"
export UV_CACHE_DIR="/research/d7/fyp25/yyyu2/.uv/cache"

# ---------- Period mapping ----------
STARTS=("2020-01-01" "2021-01-01" "2022-01-01" "2023-01-01" "2024-01-01" "2025-01-01" "2020-01-01")
ENDS=(  "2020-12-31" "2021-12-31" "2022-12-31" "2023-12-31" "2024-12-31" "2025-03-19" "2025-03-19")
LABELS=("2020" "2021" "2022" "2023" "2024" "2025" "5yr")

IDX=${SLURM_ARRAY_TASK_ID}
START=${STARTS[$IDX]}
END=${ENDS[$IDX]}
LABEL=${LABELS[$IDX]}
WORKERS=${SLURM_CPUS_PER_TASK:-80}

echo "========================================"
echo "Job ID:     $SLURM_JOB_ID (array $IDX)"
echo "Node:       $SLURM_NODELIST"
echo "CPUs:       $WORKERS"
echo "Strategy:   $STRATEGY"
echo "Symbol:     $SYMBOL"
echo "Period:     $START to $END ($LABEL)"
echo "Start time: $(date)"
echo "========================================"

# ---------- Environment ----------
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate

mkdir -p logs results

# ---------- Sweep parameters ----------
if [ "$STRATEGY" = "amm" ]; then
    SWEEP_PARAMS=(
        "spread_bps=20:100:10"
        "amplification=3,5,10,20"
    )
    OUTPUT_DIR="results/amm_${SYMBOL}_${LABEL}"
elif [ "$STRATEGY" = "cl-amm" ]; then
    SWEEP_PARAMS=(
        "spread_bps=20:100:10"
        "concentration=3,5,10,15,20"
    )
    OUTPUT_DIR="results/cl-amm_${SYMBOL}_${LABEL}"
else
    echo "ERROR: Unknown strategy: $STRATEGY"
    exit 1
fi

# ---------- Run sweep ----------
echo ""
echo "Sweep params: ${SWEEP_PARAMS[*]}"
echo "Output:       $OUTPUT_DIR"
echo "Workers:      $WORKERS"
echo ""

cd "$PROJECT_DIR/scripts"

python3 backtest_sweep.py \
    --strategy "$STRATEGY" \
    --symbol "$SYMBOL" \
    --start "$START" \
    --end "$END" \
    --sweep "${SWEEP_PARAMS[@]}" \
    --workers "$WORKERS" \
    --sort-by excess_return_pct \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: $STRATEGY $SYMBOL $LABEL"
    echo "Results: $OUTPUT_DIR/"
    ls -lh "$OUTPUT_DIR/"
else
    echo "FAILED: $STRATEGY $SYMBOL $LABEL (exit code $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "========================================"

exit $EXIT_CODE
