#!/bin/bash
# m12_phaseD_ext.sh — M12 Phase D-ext: extend hard-block from n=6 to n=8.
#   1) FREEZE GATE: catkin flight code must be byte-identical to ea75f96 (else abort).
#   2) SEED SCREEN: 49,50,51,52 (Config1,T1,Zone1,70s) -> pick first 2 clean.
#   3) FLY 12: chosen2 x T3/T6/T7 x C1/C2, identical spec to the sealed campaign,
#      appended to the SAME campaign_summary.csv + per-run folders + manifest.
# Banner freeze-checked every run (HALT on violation).
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
REPO=~/Fyp_Drone_Detection_Tracking
FREEZE=ea75f9601e2c9607ea8779767116c8860d4d7c68
COMMIT=$(git -C "$REPO" rev-parse HEAD)
ZONE=1; DURATION=200
TRAJS=(3 6 7); CFGS=(1 2); SCREEN_SEEDS=(49 50 51 52)
RESULTS=~/fyp/Results; CDIR="$RESULTS/M12_campaign"
CSUM="$CDIR/campaign_summary.csv"; MAN="$CDIR/m12_campaign_manifest.md"
LOG="$CDIR/ext_run.log"; : > "$LOG"
EXP_LAM="lambda=(1.00,1.00,1.00,1.00)"; EXP_HZ="Kp_wz=0.90 Kp_y=1.80"
EXP_KPZ="Kp_z=3.00 max_vz=1.50"; EXP_LATCH="search_latch=False"
declare -A SEEN
say(){ echo "$@" | tee -a "$LOG"; }

# ── 1) FREEZE GATE ──────────────────────────────────────────────────────────
if [ "$COMMIT" != "$FREEZE" ]; then say "!! repo HEAD $COMMIT != frozen $FREEZE — ABORT"; echo "=== EXT_ABORT_FREEZE ==="; exit 1; fi
drift=0
for f in ibvs_controller_node.py flight_logger.py extract_metrics.py launch_stack.sh \
         kalman_filter_node.py target_mover.py random_spawn_target.py takeoff_both.py yolo_detection_node.py; do
  git -C "$REPO" show HEAD:drone_tracking/scripts/$f >/tmp/_rf 2>/dev/null
  diff -q /tmp/_rf "$S/$f" >/dev/null 2>&1 || { say "!! DRIFT in $f"; drift=1; }
done
rm -f /tmp/_rf
if [ "$drift" != "0" ]; then say "!! flight code drifted from ea75f96 — ABORT (no flights)"; echo "=== EXT_ABORT_FREEZE ==="; exit 1; fi
say "FREEZE GATE OK — flight code byte-identical to ea75f96"

settle(){
  bash "$S/cleanup.sh" >/dev/null 2>&1
  pkill -9 -f 'px4-simulator' 2>/dev/null; pkill -9 -f 'px4-' 2>/dev/null
  for port in 4560 4561 11311; do lsof -ti :$port 2>/dev/null | xargs -r kill -9 2>/dev/null; done
  local i procs avail
  for i in $(seq 1 90); do
    procs=$(pgrep -c -f 'gzserver|gzclient|px4|mavros' 2>/dev/null||true)
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "${procs:-0}" -eq 0 ] && [ "${avail:-0}" -ge 5000 ]; then break; fi
    sleep 2
  done
  sleep 6
}
run_one(){ # cfg traj seed dur -> echoes csv path
  local cfg=$1 traj=$2 seed=$3 dur=$4 rdir before csv
  rdir=~/fyp/Results/Config${cfg}      # separate line: cfg must be bound before use (set -u)
  before=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null|head -1); csv=""
  for attempt in 1 2 3; do
    settle; [ "$attempt" -gt 1 ] && say "    (retry $attempt/3)"
    SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 bash "$LS" "$cfg" "$traj" "$ZONE" "$seed" "$dur" >>"$LOG" 2>&1
    csv=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null|head -1)
    [ -n "$csv" ] && [ "$csv" != "$before" ] && break; csv=""
  done
  echo "$csv"
}
banner_check(){ local traj=$1 t7log b
  t7log=$(ls -t /tmp/T7_traj${traj}_zone${ZONE}_*.log 2>/dev/null|head -1)
  b=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null|head -1)
  [ -z "$b" ] && { echo "BAD:no-banner"; return; }
  echo "$b"|grep -q "$EXP_LAM"||{ echo "BAD:lambda"; return; }
  echo "$b"|grep -q "$EXP_HZ"||{ echo "BAD:horiz"; return; }
  echo "$b"|grep -q "$EXP_KPZ"||{ echo "BAD:kpz"; return; }
  echo "$b"|grep -q "$EXP_LATCH"||{ echo "BAD:latch"; return; }
  echo "ok"; }
postko(){ [ -f "$1" ]||{ echo 0; return; }; awk -F, 'NR>1&&($2=="SEARCH"||$2=="APPROACH"||$2=="HOLD"){n++}END{print n+0}' "$1"; }

# ── 2) SEED SCREEN ──────────────────────────────────────────────────────────
say ""; say "===== SEED SCREEN (Config1,T1,Zone1,70s)  $(date) ====="
CHOSEN=()
for seed in "${SCREEN_SEEDS[@]}"; do
  say ">> screen seed $seed"; csv=$(run_one 1 1 "$seed" 70)
  if [ -z "$csv" ]; then say "   seed $seed: FAIL (no csv)"; continue; fi
  n=$(postko "$csv")
  if [ "$n" -gt 0 ]; then say "   seed $seed: PASS (rows=$n)"; CHOSEN+=("$seed");
  else say "   seed $seed: FAIL (stuck takeoff)"; fi
done
say "screen PASS seeds: ${CHOSEN[*]:-none}"
if [ "${#CHOSEN[@]}" -lt 2 ]; then say "!! fewer than 2 clean seeds — HALT before flying"; echo "=== EXT_HALT_SEEDS ==="; exit 1; fi
FLY_SEEDS=("${CHOSEN[@]:0:2}")
say "chosen (first 2 clean): ${FLY_SEEDS[*]}"

# manifest extension header
{
  echo ""
  echo "---"
  echo ""
  echo "## n=8 extension (Phase D-ext)"
  echo ""
  echo "- **Commit (frozen, re-verified byte-identical):** \`$(git -C "$REPO" rev-parse --short HEAD)\`"
  echo "- **Screened:** ${SCREEN_SEEDS[*]} -> PASS: ${CHOSEN[*]} -> flown: ${FLY_SEEDS[*]}"
  echo "- **Started:** $(date)"
  echo ""
  echo "| # | cfg | traj | seed | outcome | attempts | md5 | banner | run_dir |"
  echo "|---|-----|------|------|---------|----------|-----|--------|---------|"
} >> "$MAN"

# ── 3) FLY 12 ───────────────────────────────────────────────────────────────
say ""; say "===== EXT FLY 12 (seeds ${FLY_SEEDS[*]})  $(date) ====="
E=0; NOK=0; NFAIL=0; HALT=0
for traj in "${TRAJS[@]}"; do
  for seed in "${FLY_SEEDS[@]}"; do
    for cfg in "${CFGS[@]}"; do
      E=$((E+1)); say ""; say ">>> ext [$E/12] Config $cfg T$traj seed $seed"
      csv=$(run_one "$cfg" "$traj" "$seed" "$DURATION")
      if [ -z "$csv" ]; then NFAIL=$((NFAIL+1)); say "   !! FAIL bringup"
        echo "| E$E | $cfg | $traj | $seed | FAIL-bringup | 3 | - | - | - |" >>"$MAN"; continue; fi
      bch=$(banner_check "$traj")
      if [ "$bch" != "ok" ]; then HALT=1; say "   !!!! BANNER FREEZE VIOLATION: $bch -> HALT"
        echo "| E$E | $cfg | $traj | $seed | HALT-banner | - | - | $bch | - |" >>"$MAN"; break 3; fi
      tag=$(basename "$csv" .csv); md5=$(md5sum "$csv"|cut -d' ' -f1)
      dup=""; [ -n "${SEEN[$md5]:-}" ] && dup="DUP"; SEEN[$md5]=1
      rd=~/fyp/Results/Config${cfg}/cfg${cfg}_T${traj}_z1_s${seed}; mkdir -p "$rd"
      cp "$csv" "$rd/"; cp /tmp/T*_${tag}.log "$rd/" 2>/dev/null
      cat > "$rd/meta.json" <<META
{"config":$cfg,"trajectory":$traj,"seed":$seed,"zone":$ZONE,"duration_s":$DURATION,
 "run_tag":"$tag","csv":"$(basename "$csv")","md5":"$md5","commit":"$COMMIT",
 "banner":"ok","phase":"D-ext-n8"}
META
      python3 "$S/extract_metrics.py" --csv "$csv" --config "$cfg" --zone "$ZONE" \
        --traj "$traj" --seed "$seed" --duration "$DURATION" --summary "$CSUM" >/dev/null 2>&1
      NOK=$((NOK+1)); say "   OK -> cfg${cfg}_T${traj}_z1_s${seed}/ md5=${md5:0:8} $dup"
      echo "| E$E | $cfg | $traj | $seed | OK$([ -n "$dup" ]&&echo -$dup) | - | ${md5:0:8} | ok | cfg${cfg}_T${traj}_z1_s${seed} |" >>"$MAN"
    done
  done
done
settle
{ echo ""; echo "**Ext finished:** $(date) | OK=$NOK FAIL=$NFAIL HALT=$HALT (of 12)"; } >>"$MAN"
say ""; say "===== EXT DONE OK=$NOK FAIL=$NFAIL HALT=$HALT $(date) ====="
[ "$HALT" = "1" ] && echo "=== EXT_HALTED ===" || echo "=== EXT_DONE ==="
