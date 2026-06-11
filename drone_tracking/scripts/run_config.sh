#!/bin/bash
# run_config.sh — Loop N simulations for a given CONFIG
# Usage: ./run_config.sh CONFIG --runs N [--zone Z] [--traj T] [--duration S] [--seed-start S]
#        ./run_config.sh CONFIG --matrix [--repeats N] [--duration S] [--seed-start S]
# Examples:
#   ./run_config.sh 2 --runs 10
#   ./run_config.sh 1 --runs 5 --zone 7 --traj 4
#   ./run_config.sh 3 --runs 3 --duration 180
#   ./run_config.sh 2 --matrix              # T1-8 x zones {5,6,7,9} = 32 runs
#   ./run_config.sh 2 --matrix --repeats 3  # 3 disjoint-seed passes = 96 runs
set -u

CONFIG="${1:-}"
shift || { echo "Usage: $0 CONFIG [--runs N | --matrix [--repeats N]] [--zone Z] [--traj T] [--duration S] [--seed-start S]"; exit 1; }

RUNS=1; ZONE="random"; TRAJ="random"; DURATION=300; SEED_START=42; MATRIX=false; REPEATS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)              RUNS="$2";       shift 2 ;;
        --zone)              ZONE="$2";       shift 2 ;;
        --traj|--trajectory) TRAJ="$2";       shift 2 ;;
        --duration)          DURATION="$2";   shift 2 ;;
        --seed-start)        SEED_START="$2"; shift 2 ;;
        --matrix)            MATRIX=true;     shift   ;;
        --repeats)           REPEATS="$2";    shift 2 ;;
        *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$CONFIG"  =~ ^[123]$ ]]  || { echo "[ERROR] CONFIG must be 1, 2 or 3"; exit 1; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] || { echo "[ERROR] --repeats must be an integer"; exit 1; }

# B7: deterministic matrix mode overrides --runs/--zone/--traj
if [ "$MATRIX" = "true" ]; then
    RUNS=$((32 * REPEATS)); ZONE="matrix{5,6,7,9}"; TRAJ="1-8"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE=~/fyp/Results            # M9.6 step 2: matches launch_stack.sh
mkdir -p "$RESULTS_BASE"
SUMMARY_LOG="$RESULTS_BASE/_batch_$(date +%Y-%m-%d_%H-%M).log"

echo "================================================================"
echo "  BATCH  Config=$CONFIG  Runs=$RUNS  Zone=$ZONE  Traj=$TRAJ"
echo "  Duration=${DURATION}s each  SeedStart=$SEED_START"
echo "================================================================"
{ echo "Batch start: $(date)"; echo "Config=$CONFIG Runs=$RUNS Zone=$ZONE Traj=$TRAJ Duration=${DURATION}s"; echo ""; } > "$SUMMARY_LOG"

SUCCESS=0; FAILED=0
ZONES=(5 6 7 9)

if [ "$MATRIX" = "true" ]; then
    # B7: deterministic grid — T1..8 x zones {5,6,7,9} = 32 runs per repeat.
    # seed = SEED_START + (within-grid index K); +100 per repeat keeps the seed
    # ranges disjoint across repeats. --runs/--zone/--traj are ignored here.
    GLOBAL=0
    for r in $(seq 0 $((REPEATS - 1))); do
        K=0
        for T in 1 2 3 4 5 6 7 8; do
            for Z in "${ZONES[@]}"; do
                SEED=$((SEED_START + r * 100 + K))
                GLOBAL=$((GLOBAL + 1))
                echo ""
                echo ">>> Run $GLOBAL/$RUNS  rep$((r+1)) cfg=$CONFIG traj=$T zone=$Z seed=$SEED <<<"
                echo ""
                if bash "$SCRIPT_DIR/launch_stack.sh" $CONFIG $T $Z $SEED $DURATION; then
                    SUCCESS=$((SUCCESS+1))
                    echo "  ✓ Run $GLOBAL OK: cfg=$CONFIG traj=$T zone=$Z seed=$SEED" >> "$SUMMARY_LOG"
                else
                    FAILED=$((FAILED+1))
                    echo "  ✗ Run $GLOBAL FAIL: cfg=$CONFIG traj=$T zone=$Z seed=$SEED" >> "$SUMMARY_LOG"
                fi
                K=$((K + 1))
            done
        done
    done
else
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
fi

echo ""
echo "================================================================"
echo "  BATCH DONE — Success=$SUCCESS/$RUNS  Failed=$FAILED/$RUNS"
echo "  Logs:    $RESULTS_BASE/"
echo "  Summary: $SUMMARY_LOG"
echo "================================================================"
{ echo ""; echo "Batch end: $(date)"; echo "Success=$SUCCESS/$RUNS  Failed=$FAILED/$RUNS"; } >> "$SUMMARY_LOG"

# Mirror results to the Windows OneDrive FYP folder (kept out of the hot path)
bash "$SCRIPT_DIR/sync_results.sh"
