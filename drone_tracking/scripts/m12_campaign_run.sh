#!/bin/bash
# m12_campaign_run.sh — M12 Phase D HARD-BLOCK campaign (36 runs).
# T3/T6/T7 x seeds {42,43,45,46,47,48} x Config {1,2}, Zone 1, 200 s, lambda=1,
# search latch OFF, paired per seed (C1 then C2 back-to-back).
# FREEZE DISCIPLINE: banner is checked every run; a banner mismatch (wrong
# gain/lambda/latch) HALTS the campaign immediately (freeze violation). Transient
# bringup failures retry 3x then are logged FAIL and the campaign continues.
# Preserves EVERYTHING per run in Config{N}/cfg{N}_T{X}_z1_s{SEED}/.
set -u
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
S=~/catkin_ws/src/drone_tracking/scripts
LS="$S/launch_stack.sh"
REPO=~/Fyp_Drone_Detection_Tracking
COMMIT=$(git -C "$REPO" rev-parse HEAD)
COMMIT_SHORT=$(git -C "$REPO" rev-parse --short HEAD)
ZONE=1; DURATION=200
TRAJS=(3 6 7); SEEDS=(42 43 45 46 47 48); CFGS=(1 2)
RESULTS=~/fyp/Results
CDIR="$RESULTS/M12_campaign"
CSUM="$CDIR/campaign_summary.csv"
MAN="$CDIR/m12_campaign_manifest.md"
LOG="$CDIR/campaign_run.log"
mkdir -p "$CDIR"; : > "$LOG"
# banner freeze expectations
EXP_LAM="lambda=(1.00,1.00,1.00,1.00)"
EXP_HZ="Kp_wz=0.90 Kp_y=1.80"
EXP_KPZ="Kp_z=3.00 max_vz=1.50"
EXP_LATCH="search_latch=False"
declare -A SEEN_MD5
say(){ echo "$@" | tee -a "$LOG"; }

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

# manifest header
{
  echo "# M12 Phase D hard-block campaign manifest"
  echo ""
  echo "- **Commit (frozen code):** \`$COMMIT_SHORT\` ($COMMIT)"
  echo "- **Grid:** T3/T6/T7 x seeds {42,43,45,46,47,48} x Config{1,2}, Zone 1, 200 s, lambda=1, latch OFF, paired per seed."
  echo "- **Started:** $(date)"
  echo ""
  echo "| # | cfg | traj | seed | outcome | attempts | md5 | banner | run_dir |"
  echo "|---|-----|------|------|---------|----------|-----|--------|---------|"
} > "$MAN"

banner_check(){ # $1=traj -> echoes ok / BAD:...
  local traj=$1 t7log b
  t7log=$(ls -t /tmp/T7_traj${traj}_zone${ZONE}_*.log 2>/dev/null|head -1)
  b=$(grep -o "v6\.30 .*step_clamp=[0-9]*px" "$t7log" 2>/dev/null|head -1)
  [ -z "$b" ] && { echo "BAD:no-banner"; return; }
  echo "$b"|grep -q "$EXP_LAM"   || { echo "BAD:lambda[$b]"; return; }
  echo "$b"|grep -q "$EXP_HZ"    || { echo "BAD:horiz[$b]"; return; }
  echo "$b"|grep -q "$EXP_KPZ"   || { echo "BAD:kpz[$b]"; return; }
  echo "$b"|grep -q "$EXP_LATCH" || { echo "BAD:latch[$b]"; return; }
  echo "ok"
}

N=0; NOK=0; NFAIL=0; HALT=0
for traj in "${TRAJS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for cfg in "${CFGS[@]}"; do
      N=$((N+1))
      say ""; say ">>> [$N/36] Config $cfg  T$traj  seed $seed"
      rdir="$RESULTS/Config${cfg}"
      before=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
      csv=""; attempts=0
      for attempt in 1 2 3; do
        attempts=$attempt
        settle
        [ "$attempt" -gt 1 ] && say "    (retry $attempt/3)"
        SEARCH_LATCH=false VIEWER=0 LOSS_TIMEOUT=60 \
          bash "$LS" "$cfg" "$traj" "$ZONE" "$seed" "$DURATION" >> "$LOG" 2>&1
        csv=$(ls -t ${rdir}/traj${traj}_zone${ZONE}_*.csv 2>/dev/null | head -1)
        [ -n "$csv" ] && [ "$csv" != "$before" ] && break; csv=""
      done

      if [ -z "$csv" ]; then
        NFAIL=$((NFAIL+1))
        say "    !! FAIL (bringup, 3 attempts) C$cfg T$traj s$seed"
        echo "| $N | $cfg | $traj | $seed | FAIL-bringup | $attempts | - | - | - |" >> "$MAN"
        continue
      fi

      bch=$(banner_check "$traj")
      if [ "$bch" != "ok" ]; then
        HALT=1
        say "    !!!! BANNER FREEZE VIOLATION: $bch  -> HALTING CAMPAIGN"
        echo "| $N | $cfg | $traj | $seed | HALT-banner | $attempts | - | $bch | - |" >> "$MAN"
        break 3
      fi

      tag=$(basename "$csv" .csv)                 # traj{X}_zone1_{ts}
      md5=$(md5sum "$csv"|cut -d' ' -f1)
      dup=""; [ -n "${SEEN_MD5[$md5]:-}" ] && dup="DUP"; SEEN_MD5[$md5]=1

      # per-run folder — preserve EVERYTHING
      run_dir="$rdir/cfg${cfg}_T${traj}_z1_s${seed}"
      mkdir -p "$run_dir"
      cp "$csv" "$run_dir/"
      # ROS node logs for this exact run tag
      cp /tmp/T*_${tag}.log "$run_dir/" 2>/dev/null
      # metadata
      cat > "$run_dir/meta.json" <<META
{"config":$cfg,"trajectory":$traj,"seed":$seed,"zone":$ZONE,"duration_s":$DURATION,
 "timestamp":"${tag##*_zone${ZONE}_}","run_tag":"$tag","csv":"$(basename "$csv")",
 "md5":"$md5","commit":"$COMMIT","banner":"ok","attempts":$attempts,"dup":"$dup"}
META
      # authoritative campaign summary row (config-correct centroid source)
      python3 "$S/extract_metrics.py" --csv "$csv" --config "$cfg" --zone "$ZONE" \
        --traj "$traj" --seed "$seed" --duration "$DURATION" --summary "$CSUM" >/dev/null 2>&1

      NOK=$((NOK+1))
      say "    OK saved -> $(basename "$run_dir")/  md5=${md5:0:8} $dup banner=ok"
      echo "| $N | $cfg | $traj | $seed | OK$([ -n "$dup" ] && echo "-$dup") | $attempts | ${md5:0:8} | ok | cfg${cfg}_T${traj}_z1_s${seed} |" >> "$MAN"
    done
  done
done

settle
{
  echo ""
  echo "**Finished:** $(date)  |  OK=$NOK  FAIL=$NFAIL  HALT=$HALT  (of 36)"
} >> "$MAN"
say ""
say "===== CAMPAIGN DONE  OK=$NOK FAIL=$NFAIL HALT=$HALT  $(date) ====="
if [ "$HALT" = "1" ]; then echo "=== M12_CAMPAIGN_HALTED ==="; else echo "=== M12_CAMPAIGN_DONE ==="; fi
