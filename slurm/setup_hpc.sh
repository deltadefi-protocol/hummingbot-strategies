#!/bin/bash
# ============================================
# One-time HPC Setup for Backtest Sweep
# Run on login node (has internet for data download)
#
# Usage:
#   ssh hpc
#   cd /research/d7/fyp25/yyyu2/hummingbot-strategies
#   bash slurm/setup_hpc.sh [SYMBOL]
# ============================================

set -e

echo "========================================"
echo "Setting up backtest environment on HPC"
echo "========================================"

# ---------- Paths (adjust these) ----------
PROJECT_DIR="/research/d7/fyp25/yyyu2/hummingbot-strategies"

export UV_TOOL_DIR="/research/d7/fyp25/yyyu2/.uv/tools"
export UV_TOOL_BIN_DIR="/research/d7/fyp25/yyyu2/.uv/bin"
export UV_CACHE_DIR="/research/d7/fyp25/yyyu2/.uv/cache"

cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

# ============================================
# 1. Create venv with uv
# ============================================
echo ""
echo "Step 1: Setting up Python environment with uv..."

if [ ! -d ".venv" ]; then
    uv venv --python 3.10
    echo "Created .venv"
else
    echo ".venv already exists"
fi

source .venv/bin/activate

echo "Installing dependencies..."
uv pip install numpy scikit-learn hmmlearn

# ============================================
# 2. Verify imports
# ============================================
echo ""
echo "Step 2: Verifying imports..."

cd "$PROJECT_DIR/scripts"

python3 -c "
from backtest_engine import BacktestEngine, CandleDataLoader, Candle
from backtest_strategies import CLAMMBacktestStrategy, AMMBacktestStrategy, TradeType
print('All imports OK')
import sys
print(f'Python: {sys.version}')
print(f'TradeType.BUY = {TradeType.BUY}')
print(f'TradeType.SELL = {TradeType.SELL}')
"

# ============================================
# 3. Pre-download candle data (login node has internet)
# ============================================
echo ""
echo "Step 3: Pre-downloading candle data..."
echo "(Compute nodes may not have internet access)"

SYMBOL="${1:-ADAUSDT}"

for YEAR in 2020 2021 2022 2023 2024; do
    START="${YEAR}-01-01"
    END="${YEAR}-12-31"
    echo "  Downloading $SYMBOL $START to $END ..."
    python3 -c "
from backtest_engine import CandleDataLoader
candles = CandleDataLoader.load('$SYMBOL', '1m', '$START', '$END')
print(f'    {len(candles)} candles cached')
"
done

echo "  Downloading $SYMBOL 2025-01-01 to 2025-03-19 ..."
python3 -c "
from backtest_engine import CandleDataLoader
candles = CandleDataLoader.load('$SYMBOL', '1m', '2025-01-01', '2025-03-19')
print(f'    {len(candles)} candles cached')
"

echo "  Downloading $SYMBOL 2020-01-01 to 2025-03-19 (full 5yr) ..."
python3 -c "
from backtest_engine import CandleDataLoader
candles = CandleDataLoader.load('$SYMBOL', '1m', '2020-01-01', '2025-03-19')
print(f'    {len(candles)} candles cached')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Cache directory:"
ls -lh ~/.hummingbot_backtest_cache/ | head -20
echo ""
echo "Submit jobs:"
echo "  sbatch --array=0-6 slurm/sweep_backtest.sh amm $SYMBOL"
echo "  sbatch --array=0-6 slurm/sweep_backtest.sh cl-amm $SYMBOL"
