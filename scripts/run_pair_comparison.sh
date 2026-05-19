#!/bin/bash
# Multi-pair walk-forward: run the all_on preset on ADA/ETH/BTC against USDC.
# Goal: test the "pivot the pair" hypothesis from docs/walkforward_findings.md
# — does the same strategy survive on lower-vol assets?
#
# Each pair runs the same 6 yearly windows (2020/21/22/23/24/25q1).
# Output: results/best_tuned/walk_forward_<SYMBOL>_all_on_*.csv per pair.
# Cross-pair summary printed at end.

set -e
cd "$(dirname "$0")"

PYTHON="/Users/yuyanyuk/miniforge3/envs/hummingbot/bin/python -u"
export PYTHONPATH="/Users/yuyanyuk/Git/hummingbot:${PYTHONPATH}"

PAIRS=("ADAUSDC" "ETHUSDC" "BTCUSDC")
WORKERS=6
START_TS=$(date +%s)

echo "════════════════════════════════════════════════════════════"
echo "  MULTI-PAIR WALK-FORWARD (preset=all_on)"
echo "  Pairs:   ${PAIRS[*]}"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

for SYMBOL in "${PAIRS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  ▶ $SYMBOL"
    echo "────────────────────────────────────────────────────────────"
    $PYTHON run_best_tuned.py \
        --preset all_on \
        --symbol "$SYMBOL" \
        --workers "$WORKERS" \
        --windows 2020 2021 2022 2023 2024 2025q1
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL PAIRS DONE in ${MINS}m ${SECS}s"
echo "════════════════════════════════════════════════════════════"

# Cross-pair summary: pull the latest walk-forward CSV per pair and print
# a comparison table.
$PYTHON - <<'PYEOF'
import csv, glob, os
results_dir = "results/best_tuned"
pairs = ["ADAUSDC", "ETHUSDC", "BTCUSDC"]
print("\nCross-pair summary (all_on preset, yearly windows):\n")
header = f"  {'window':9s} | " + " | ".join(f"{p:>12s}" for p in pairs)
print(header); print("  " + "-" * (len(header) - 2))

per_pair = {}
for p in pairs:
    files = sorted(glob.glob(f"{results_dir}/walk_forward_{p}_all_on_*.csv"),
                   key=os.path.getmtime, reverse=True)
    if not files:
        per_pair[p] = {}
        continue
    with open(files[0]) as f:
        rows = list(csv.DictReader(f))
    per_pair[p] = {r["_label"]: r for r in rows}

windows = ["2020", "2021", "2022", "2023", "2024", "2025q1"]
for w in windows:
    cells = []
    for p in pairs:
        r = per_pair.get(p, {}).get(w)
        if r is None:
            cells.append("       —    ")
        else:
            ret = float(r.get("total_return_pct", "0") or 0)
            hold = float(r.get("hold_return_pct", "0") or 0)
            excess = ret - hold
            marker = "✓" if excess > 0 else "✗"
            cells.append(f"{marker}{ret:+7.1f}/{excess:+5.0f}")
    print(f"  {w:9s} | " + " | ".join(f"{c:>12s}" for c in cells))

print("\n  Legend: ret%/excess%  (✓ = beats hold; ✗ = loses to hold)")
PYEOF
