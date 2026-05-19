#!/bin/bash
# Master runner — executes all coarse one-at-a-time sweeps + the ablation
# experiment. Each component can also be run standalone.
#
# Layout produced:
#   results/ablation/                     ← run_ablation.py output
#   results/coarse/dyn_concentration/...  ← #3 (axis 3a-3d)
#   results/coarse/outer/...              ← #4 (axis 4a-4d)
#   results/coarse/guards/...             ← #5 (axis 5a-5h)
#   results/coarse/asym/...               ← #6 (axis 6a-6c)
#   results/coarse/trend/<regime>_...     ← #7 (per regime × axis)
#
# Estimated wall time on 8 workers / M-series mac:
#   ablation         ~3 min   (6 configs, 1 year)
#   dyn concentr.    ~6 min   (~16 configs, 1 year)
#   outer            ~6 min   (~14 configs, 1 year)
#   guards          ~12 min   (~28 configs, 1 year)
#   asym             ~4 min   (~11 configs, 1 year)
#   trend           ~10 min   (4 regimes × ~21 configs, short windows)
#   TOTAL           ~40 min
#
# Run individual subsets with the SKIP_* env vars:
#   SKIP_TREND=1 bash run_coarse_all.sh
#   SKIP_ABLATION=1 SKIP_OUTER=1 bash run_coarse_all.sh

set -e
cd "$(dirname "$0")"

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"

START_TS=$(date +%s)
echo "════════════════════════════════════════════════════════════"
echo "  CL-AMM COARSE SWEEPS — MASTER RUNNER"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

run() {
    local NAME="$1"; shift
    local SKIP_VAR="$1"; shift
    if [ "${!SKIP_VAR:-0}" = "1" ]; then
        echo "  ⊘ SKIP $NAME (${SKIP_VAR}=1)"
        return
    fi
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  ▶ $NAME"
    echo "────────────────────────────────────────────────────────────"
    "$@"
}

# 1. Ablation: highest-information experiment, run first
run "Ablation (Phase 7)" "SKIP_ABLATION" \
    $PYTHON run_ablation.py \
        --start 2024-01-01 --end 2024-12-31 \
        --spread-bps 50 --concentration 15 \
        --workers 6 --output results/ablation

# 2-6. Coarse per-axis sweeps
run "Dyn concentration (#3)"  "SKIP_DYN"     bash run_coarse_dyn.sh
run "Outer dual-range (#4)"   "SKIP_OUTER"   bash run_coarse_outer.sh
run "Guards (#5)"             "SKIP_GUARDS"  bash run_coarse_guards.sh
run "Asymmetric spread (#6)"  "SKIP_ASYM"    bash run_coarse_asym.sh
run "Trend × regimes (#7)"    "SKIP_TREND"   bash run_coarse_trend.sh

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL DONE in ${MINS}m ${SECS}s"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Artifacts (most recent):"
find results/ablation results/coarse -name "*.csv" -newer /tmp 2>/dev/null \
    | head -30 || true
