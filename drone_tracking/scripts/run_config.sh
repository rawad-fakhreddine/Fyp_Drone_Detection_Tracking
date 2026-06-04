#!/bin/bash
# run_config.sh — Loop N simulations for a given CONFIG
# Usage: ./run_config.sh CONFIG --runs N [--zone Z] [--traj T] [--duration S] [--seed-start S]
# Examples:
#   ./run_config.sh 2 --runs 10
#   ./run_config.sh 1 --runs 5 --zone 7 --traj 4
#   ./run_config.sh 3 --runs 3 --duration 180
set -u

CONFIG="${1:-}"
shift || { echo "Usage: $0 CONFIG [--runs N] [--zone Z] [--traj T] [--duration S] [--seed-start S]"; exit 1; }

RUNS=1; ZONE="random"; TRAJ="random"; DURATION=300; SEED_START=42

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)              RUNS="$2";       shift 2 ;;
        --zone)              ZONE="$2";       shift 2 ;;
        --traj|--trajectory) TRAJ="$2";       shift 2 ;;
        --duration)          DURATION="$2";   shift 2 ;;
        --seed-start)        SEED_START="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$CONFIG" =~ ^[123]$ ]] || { echo "[ERROR] CONFIG must be 1, 2 or 3"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/results
SUMMARY_LOG=~/results/_batch_$(date +%Y-%m-%d_%H-%M).log

echo "================================================================"
echo "  BATCH  Config=$CONFIG  Runs=$RUNS  Zone=$ZONE  Traj=$TRAJ"
echo "  Duration=${DURATION}s each  SeedStart=$SEED_START"
echo "================================================================"
{ echo "Batch start: $(date)"; echo "Config=$CONFIG Runs=$RUNS Zone=$ZONE Traj=$TRAJ Duration=${DURATION}s"; echo ""; } > "$SUMMARY_LOG"

SUCCESS=0; FAILED=0
ZONES=(5 6 7 9)

for i in $(seq 1 $RUNS); do
    SEED=$((SEED_START + i - 1))
    [ "$ZONE"  = "random" ] && Z=${ZONES[$RANDOM % 4]}  || Z=$ZONE
    [ "$TRAJ"  = "random" ] && T=$((RANDOM % 8 + 1))    || T=$TRAJ

    echo ""
    echo ">>> Run $i/$RUNS  cfg=$CONFIG traj=$T zone=$Z seed=$SEED <<<"
    echo ""

    if bash "$SCRIPT_DIR/launch_stack.sh" $CONFIG $T $Z $SEED $DURATION; then
        SUCCESS=$((SUCCESS+1))
        echo "  ✓ Run $i OK: cfg=$CONFIG traj=$T zone=$Z seed=$SEED" >> "$SUMMARY_LOG"
    else
        FAILED=$((FAILED+1))
        echo "  ✗ Run $i FAIL: cfg=$CONFIG traj=$T zone=$Z seed=$SEED" >> "$SUMMARY_LOG"
    fi
done

echo ""
echo "================================================================"
echo "  BATCH DONE — Success=$SUCCESS/$RUNS  Failed=$FAILED/$RUNS"
echo "  Logs:    ~/results/"
echo "  Summary: $SUMMARY_LOG"
echo "================================================================"
{ echo ""; echo "Batch end: $(date)"; echo "Success=$SUCCESS/$RUNS  Failed=$FAILED/$RUNS"; } >> "$SUMMARY_LOG"
