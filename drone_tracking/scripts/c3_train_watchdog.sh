#!/usr/bin/env bash
# c3_train_watchdog.sh — detection canary for baylands TD3+CAPS training.
# Watches detection_frac in the training log; if it drops (camera render leak),
# HALTS training before the policy learns blind. Checkpoints auto-save every 5k
# steps to td3_baylands/, so we resume from the last clean one after wsl restart.
LOG="${LOG:-/tmp/c3_baylands_train.log}"
DST="${DST:-$HOME/fyp/rl/models/td3_baylands}"
WLOG="$DST/watchdog.log"
LOW=0
echo "$(date '+%m-%d %H:%M') watchdog started" >> "$WLOG"
# Wait for the training process to actually appear (sim boot takes ~2 min) before
# we start treating its absence as "finished" — otherwise we exit during boot.
SEEN=0
for i in $(seq 1 60); do
  if pgrep -f rl_train_td3 >/dev/null 2>&1; then SEEN=1; echo "$(date '+%m-%d %H:%M') training process detected -> monitoring" >> "$WLOG"; break; fi
  sleep 10
done
[ "$SEEN" = "0" ] && { echo "$(date '+%m-%d %H:%M') training never appeared in 10min -> watchdog exit" >> "$WLOG"; exit 0; }
while true; do
  sleep 60
  if ! pgrep -f rl_train_td3 >/dev/null 2>&1; then
    echo "$(date '+%m-%d %H:%M') training gone -> watchdog exit (normal end or crash)" >> "$WLOG"
    break
  fi
  DET=$(grep 'detection_frac' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1)
  [ -z "$DET" ] && continue
  if python3 -c "import sys;sys.exit(0 if float('$DET')<0.6 else 1)" 2>/dev/null; then
    LOW=$((LOW+1))
  else
    LOW=0
  fi
  echo "$(date '+%m-%d %H:%M') det=$DET low_streak=$LOW" >> "$WLOG"
  if [ "$LOW" -ge 2 ]; then
    echo "$(date '+%m-%d %H:%M') DET-CANARY HALT det=$DET — killing to avoid blind training (needs wsl --shutdown to continue)" >> "$WLOG"
    mkdir -p "$DST/clean_halt"; cp "$DST"/td3_ckpt_*.zip "$DST/clean_halt/" 2>/dev/null
    kill -9 $(pgrep -f rl_train_td3) 2>/dev/null
    kill -9 $(ps -eo pid,args | grep -E 'gzserver|px4|mavros|rosmaster|roslaunch|launch_stack|rl_td3_train_launch' | grep -v grep | awk '{print $1}') 2>/dev/null
    break
  fi
done
