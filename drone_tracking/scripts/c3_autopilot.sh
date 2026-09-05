#!/usr/bin/env bash
# c3_autopilot.sh — UNATTENDED supervisor for the C3 pure-scratch run (overnight).
# Jobs:
#   1. Keep a training run alive (adopts the one already running, or launches).
#   2. Bank the LAST CLEAN checkpoint (only advanced while the camera is confirmed healthy).
#   3. On training death (c3's inline leak-canary kills it, or a crash): clear everything,
#      take a real BREAK (camera-recovery chance), then RESUME from the last clean checkpoint
#      — never from a contaminated one, so a dead-camera stretch can't corrupt the good model.
# The camera degradation usually needs `wsl --shutdown` (can't do that here) — but break+relaunch
# is harmless and preserves progress, and if the camera ever recovers training continues productively.
# Stop cleanly:  touch ~/fyp/rl/AUTOPILOT_STOP
set -u
SAVE="${SAVE:-td3_scratch_def}"
DST="$HOME/fyp/rl/models/$SAVE"
FLOG="$HOME/flight_log_latest.csv"
SCR="$HOME/catkin_ws/src/drone_tracking/scripts"
STOP="$HOME/fyp/rl/AUTOPILOT_STOP"
ALOG="$DST/autopilot.log"
GOODF="$DST/last_good_ckpt.txt"
BREAK_SECS="${BREAK_SECS:-720}"      # 12-min break after clearing (let the sim/camera settle)
MAX_RECOVER="${MAX_RECOVER:-30}"     # ~all-night at ~16 min/failed-cycle
TOTAL_STEPS="${TOTAL_STEPS:-60000}"
MIX="${MIX:-1,4}"
mkdir -p "$DST"

# folded-in fixes (identical to the definitive manual launch)
# 2026-08-31 anti-thrash: CAPS_LAMBDA_S 0.10->0.20 + W_S 0.70->1.2 penalise jerky action so the
# policy learns to be STILL on a held target (74k eval thrashed: vel std ~1.5 on a static target,
# HOLD 0.6%). LEAK_HALT 4->3 tightens the canary to protect the buffer from contaminated frames.
export CHUNK="${CHUNK:-2000}" P_LOST=2.0 W_VEL=0 W_VY_PEN=0.5 W_APPROACH=1.0 APPROACH_CAP=1.5 \
       W_PURSUIT_BLIND=1.0 PURSUIT_BLIND_SECS=3.0 PURSUIT_BLIND_CAP=1.5 \
       MAX_VX=4 MAX_VZ=2.5 ALT_CAP=8 \
       CAPS_LAMBDA_S="${CAPS_LAMBDA_S:-0.20}" W_S="${W_S:-1.2}" ACTION_NOISE="${ACTION_NOISE:-0.10}" \
       LEAK_HALT="${LEAK_HALT:-3}" \
       GRADIENT_STEPS="${GRADIENT_STEPS:-2}" \
       LEARNING_RATE="${LEARNING_RATE:-3e-4}"
       # 2026-08-31 anti-divergence resume (after the mix run over-trained past ~180k, see
       # [[rl-td3-diverges-after-180k-peak]]): launch with LEARNING_RATE=1e-4 GRADIENT_STEPS=1
       # to fine-tune the 164k peak stably. Both are resume-safe training hyperparams (re-applied
       # on resume in rl_train_td3.py, since TD3.load ignores the CLI values).
       # 2026-08-31 (Rawad asked to speed learning): gradient_steps 1->2 = 2x learning per env-step.
       # Env-steps (camera frames) are the scarce resource (camera-limited), so a higher update-to-data
       # ratio squeezes more learning from each precious clean step. Resume-safe (training hyperparam,
       # not a velocity cap). TD3 twin-critics + delayed updates tolerate the higher UTD ratio.

log(){ echo "$(date '+%m-%d %H:%M:%S') $*" >> "$ALOG"; }

health(){  # prints "P_EASY N_EASY" — detection rate on EASY targets (well-framed & close), the
           # ROBUST camera proxy (same logic as cam_health.py). P_EASY=-1 = too few easy samples
           # (policy not framing) => NOT a camera problem. 2026-08-31: replaced the crude
           # P(det|in_fov>0.5), which conflated policy framing-quality with camera health and
           # false-fired the watchdog while cam_health read 0.95 (Rawad caught it).
  python3 - "$FLOG" <<'PY'
import sys,csv,numpy as np,math
try:
    r=list(csv.DictReader(open(sys.argv[1])))[-500:]
    det=np.array([1.0 if x.get('raw_det')=='REAL' else 0.0 for x in r])
    fov=np.array([float(x.get('in_fov','nan') or 'nan') for x in r])
    td =np.array([float(x.get('true_dist_3d','nan') or 'nan') for x in r])
    easy=(fov>0.80)&(td<15.0)&(~np.isnan(td))
    pe=det[easy].mean() if easy.sum()>=15 else float('nan')
    print(f"{-1 if math.isnan(pe) else round(pe,3)} {int(easy.sum())}")
except Exception:
    print("-1 0")
PY
}

teardown(){
  kill -9 $(pgrep -f rl_train_td3) 2>/dev/null
  kill -9 $(pgrep -f c3_train_baylands) 2>/dev/null
  kill -9 $(ps -eo pid,args | grep -E 'gzserver|gzclient|px4|mavros|rosmaster|roslaunch|launch_stack' | grep -v grep | awk '{print $1}') 2>/dev/null
  sleep 8
}

latest_ckpt(){ ls -t "$DST"/td3_ckpt_*_steps.zip 2>/dev/null | head -1; }
ckpt_steps(){ echo "$1" | grep -oE '[0-9]+_steps' | grep -oE '^[0-9]+'; }

launch(){  # $1 = RESUME_FROM (checkpoint path or SCRATCH)
  cd "$SCR"
  nohup bash c3_train_baylands.sh "$TOTAL_STEPS" "$SAVE" "$MIX" "$1" > "/tmp/${SAVE}_wrapper.log" 2>&1 &
  log "LAUNCH c3_train_baylands resume_from=$1 wrapper_pid=$!"
}

log "=== AUTOPILOT START break=${BREAK_SECS}s max_recover=$MAX_RECOVER total=$TOTAL_STEPS mix=$MIX ==="
recover=0
deadmin=0
DEAD_LIMIT="${DEAD_LIMIT:-4}"   # consecutive unhealthy minutes -> force a recovery cycle
while true; do
  [ -f "$STOP" ] && { log "STOP file -> exit. best clean ckpt: $(cat "$GOODF" 2>/dev/null)"; break; }

  if pgrep -f rl_train_td3 >/dev/null 2>&1; then
    read PEASY NEASY <<<"$(health)"
    # advance the last-good checkpoint ONLY while the camera is confirmed healthy (easy-target det high)
    if awk "BEGIN{exit !($PEASY>=0.80)}" 2>/dev/null; then
      lc=$(latest_ckpt); [ -n "$lc" ] && echo "$lc" > "$GOODF"
    fi
    # DEAD-SIM-WHILE-ALIVE watchdog: on the contained curriculum (T1 static + T4 orbit) with
    # _recover re-framing every episode, a HEALTHY sim can never sit with the target un-framable
    # for minutes. Sustained in_fov~0 (target not in the GT frustum AND _recover can't get it back)
    # OR the classic leak (in_fov high + detector blind) = the sim/camera degraded. The leak-canary
    # only kills the in_fov-HIGH leak; this catches the in_fov~0 dead-sim the canary misses, so we
    # don't burn the night training blind on garbage (2026-08-31: sim went dead 35min in, in_fov=0
    # + true_dist=nan for 5min while training stayed alive).
    dead=0
    # 2026-08-31 FIX (Rawad caught it): halt ONLY on a real camera leak = EASY targets (well-framed
    # & close) being missed (P_easy < 0.40). This is the same robust metric as cam_health.py and it
    # does NOT conflate policy framing-quality with camera health. P_EASY=-1 (too few easy samples =
    # cold policy not framing) is NOT dead — that's the normal cold-start it must train through.
    awk "BEGIN{exit !($PEASY>=0 && $PEASY<0.40)}" 2>/dev/null && dead=1
    if [ "$dead" = "1" ]; then deadmin=$((deadmin+1)); else deadmin=0; fi
    log "alive P_easy=$PEASY n_easy=$NEASY good=$(basename "$(cat "$GOODF" 2>/dev/null)" 2>/dev/null) deadmin=$deadmin recover=$recover"
    if [ "$deadmin" -ge "$DEAD_LIMIT" ]; then
      # NOTE: the c3 canary now also halts on dead-sim (in_fov<0.08 x4) and its kill is proven —
      # it normally fires first. This watchdog is the BACKUP: use the SAME comprehensive teardown
      # (training + full sim), not the training-only kill that got stuck 2026-08-31 (fired but
      # training survived because the sim/wrapper kept it alive).
      log "DEAD-SIM watchdog: $deadmin min unhealthy while alive -> full teardown to force clear+break+resume"
      teardown
      deadmin=0
    fi
    sleep 60
    continue
  fi

  # ---- training is DOWN ----
  lc=$(latest_ckpt); steps=0; [ -n "$lc" ] && steps=$(ckpt_steps "$lc")
  if [ "${steps:-0}" -ge "$TOTAL_STEPS" ]; then
    log "STEP TARGET reached ($steps) -> DONE. final ckpt: $lc"; break
  fi
  recover=$((recover+1))
  if [ "$recover" -gt "$MAX_RECOVER" ]; then
    log "MAX_RECOVER exceeded -> STOP. Camera likely needs wsl --shutdown. best clean ckpt: $(cat "$GOODF" 2>/dev/null)"; break
  fi
  good=$(cat "$GOODF" 2>/dev/null)
  { [ -z "$good" ] || [ ! -f "$good" ]; } && good="SCRATCH"
  log "training DOWN (recover $recover/$MAX_RECOVER) -> clear + ${BREAK_SECS}s break -> resume from $good"
  teardown
  sleep "$BREAK_SECS"
  launch "$good"
  sleep 150          # let sim + prefill/recover come up before health polling resumes
done
log "=== AUTOPILOT END ==="
