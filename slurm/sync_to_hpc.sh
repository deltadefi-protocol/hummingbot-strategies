#!/bin/bash
# ============================================
# Sync project files to HPC
# Only hummingbot-strategies needed (stubs replace hummingbot package)
#
# Usage:
#   ./slurm/sync_to_hpc.sh yyyu2 hpc.university.edu
#   ./slurm/sync_to_hpc.sh yyyu2 hpc.university.edu /scratch/yyyu2
# ============================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <username> <hostname> [base_dir]"
    echo "Example: $0 yyyu2 hpc.university.edu"
    echo "Example: $0 yyyu2 hpc.university.edu /scratch/yyyu2"
    exit 1
fi

REMOTE_USER="$1"
REMOTE_HOST="$2"
BASE_DIR="${3:-/research/d7/fyp25/$REMOTE_USER}"

LOCAL_STRATEGIES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================"
echo "Syncing to ${REMOTE_USER}@${REMOTE_HOST}"
echo "  hummingbot-strategies -> ${BASE_DIR}/hummingbot-strategies/"
echo "========================================"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${BASE_DIR}/hummingbot-strategies"

echo ""
echo "Syncing hummingbot-strategies..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.venv' \
    --exclude 'results' \
    --exclude 'logs' \
    "$LOCAL_STRATEGIES/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${BASE_DIR}/hummingbot-strategies/"

echo ""
echo "========================================"
echo "Sync complete!"
echo "========================================"
echo ""
echo "Next steps on HPC:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${BASE_DIR}/hummingbot-strategies"
echo "  bash slurm/setup_hpc.sh"
echo ""
echo "Then submit jobs:"
echo "  sbatch --array=0-6 slurm/sweep_backtest.sh amm ADAUSDT"
echo "  sbatch --array=0-6 slurm/sweep_backtest.sh cl-amm ADAUSDT"
