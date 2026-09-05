#!/usr/bin/env bash
# c3_train_baylands.sh — ONE command: resume-train C3 (TD3+CAPS) on baylands with
# an INLINE detection canary (no separate watchdog process / no env-var path bugs).
# Usage: bash c3_train_baylands.sh <STEPS> <SAVEDIR> <MIX_TRAJS> [RESUME_FROM.zip]
# Example: bash c3_train_baylands.sh 30000 td3_baylands2 3,4,5,6,7,8 ~/fyp/rl/models/td3_baylands_325k_clean.zip
set -u
STEPS="${1:-30000}"
SAVENAME="${2:-td3_baylands2}"
MIX="${3:-3,4,5,6,7,8}"
RESUME_FROM="${4:-$HOME/fyp/rl/models/td3_baylands_325k_clean.zip}"
DST="$HOME/fyp/rl/models/$SAVENAME"
LOG="/tmp/${SAVENAME}_train.log"
CLOG="$DST/canary.log"
SCR="$HOME/catkin_ws/src/drone_tracking/scripts"
mkdir -p "$DST"
# SCRATCH sentinel: RESUME_FROM=SCRATCH -> cold start (RESUME=0, no checkpoint staged).
if [ "$RESUME_FROM" = "SCRATCH" ]; then
  RESUMEVAL=0
  echo "$(date '+%m-%d %H:%M') START steps=$STEPS mix=$MIX resume=SCRATCH(cold) -> $DST" >> "$CLOG"
else
  RESUMEVAL=1
  # stage resume source (frozen originals untouched)
  cp "$RESUME_FROM" "$DST/td3_policy.zip"
  # matching replay buffer — WITHOUT it TD3 resumes with an EMPTY critic memory → divergence.
  # Two naming schemes: (a) plain "<name>_replay.pkl" (frozen deliverables like 228k_retreatfix),
  # (b) SB3 CheckpointCallback "td3_ckpt_replay_buffer_<N>_steps.pkl" for a "td3_ckpt_<N>_steps.zip".
  # (2026-09-03 BUGFIX: (b) was silently unmatched → every ckpt-resume this project diverged.)
  _plain_replay="${RESUME_FROM%.zip}_replay.pkl"
  _ckpt_replay="$(echo "$RESUME_FROM" | sed -E 's#td3_ckpt_([0-9]+_steps)\.zip#td3_ckpt_replay_buffer_\1.pkl#')"
  if [ -f "$_plain_replay" ]; then
    cp "$_plain_replay" "$DST/td3_replay.pkl"; echo "  staged replay (plain): $_plain_replay"
  elif [ "$_ckpt_replay" != "$RESUME_FROM" ] && [ -f "$_ckpt_replay" ]; then
    cp "$_ckpt_replay" "$DST/td3_replay.pkl"; echo "  staged replay (ckpt):  $_ckpt_replay"
  else
    echo "  WARNING: no replay buffer found for $RESUME_FROM — TD3 will resume with empty critic memory!"
  fi
  echo "$(date '+%m-%d %H:%M') START steps=$STEPS mix=$MIX resume=$RESUME_FROM -> $DST" >> "$CLOG"
fi

# launch training
cd "$SCR"
WORLD="${WORLD:-baylands}" RESUME=$RESUMEVAL TD3_SAVE="$DST" \
MIX_TRAJS="$MIX" ACCEL_LIMIT="${ACCEL_LIMIT:-6}" ACCEL_LIMIT_WZ="${ACCEL_LIMIT_WZ:-6}" W_RETREAT="${W_RETREAT:-4}" CLOSE_THRESH="${CLOSE_THRESH:-6.0}" BAND_PEN_CAP=3 BAND_BONUS=1.5 \
ACTION_NOISE="${ACTION_NOISE:-0.10}" CAPS_LAMBDA_S="${CAPS_LAMBDA_S:-0.10}" CAPS_SIGMA_S=0.05 W_S="${W_S:-0.70}" \
N_STACK=1 MAX_VX="${MAX_VX:-4}" MAX_WZ=1.0 W_APPROACH="${W_APPROACH:-1.0}" APPROACH_CAP="${APPROACH_CAP:-0.5}" \
MAX_VZ="${MAX_VZ:-4.0}" ACCEL_LIMIT_VZ="${ACCEL_LIMIT_VZ:-10}" ALT_CAP="${ALT_CAP:-12.0}" \
SIGMA="${SIGMA:-0.6}" CENTER_A="${CENTER_A:-3.0}" W_EX="${W_EX:-0.0}" W_VY_PEN="${W_VY_PEN:-0.0}" \
W_D_CLOSE="${W_D_CLOSE:-0.5}" BAND_PEN_CAP_CLOSE="${BAND_PEN_CAP_CLOSE:-1.0}" W_DDOT="${W_DDOT:-0.0}" DDOT_EX_GATE="${DDOT_EX_GATE:-0.30}" \
P_LOST="${P_LOST:-0.5}" W_VEL="${W_VEL:-0.05}" \
W_PURSUIT_BLIND="${W_PURSUIT_BLIND:-0.0}" PURSUIT_BLIND_SECS="${PURSUIT_BLIND_SECS:-1.0}" PURSUIT_BLIND_CAP="${PURSUIT_BLIND_CAP:-1.0}" \
W_SPEEDMATCH="${W_SPEEDMATCH:-0.0}" SPEEDMATCH_FLOOR="${SPEEDMATCH_FLOOR:-0.6}" SPEED_RATIO_CAP="${SPEED_RATIO_CAP:-1.2}" SPEEDMATCH_DMIN="${SPEEDMATCH_DMIN:-8.0}" SPEEDMATCH_TMIN="${SPEEDMATCH_TMIN:-0.8}" SPEEDMATCH_CENTER_SIGMA="${SPEEDMATCH_CENTER_SIGMA:-0.0}" \
W_MATCH="${W_MATCH:-0.0}" MATCH_RECEDE_REF="${MATCH_RECEDE_REF:-2.0}" W_ACC="${W_ACC:-0.0}" ACC_CAP="${ACC_CAP:-3.0}" W_FWD="${W_FWD:-0.0}" FWD_CAP="${FWD_CAP:-1.5}" MAX_VX="${MAX_VX:-4}" \
nohup bash rl_td3_train_launch.sh "$STEPS" 4 42 > "$LOG" 2>&1 &
TPID=$!
echo "$(date '+%m-%d %H:%M') training pid=$TPID log=$LOG" >> "$CLOG"

# INLINE canary: wait for training python to appear, then watch detection_frac in THIS log.
SEEN=0
for i in $(seq 1 60); do pgrep -f rl_train_td3 >/dev/null 2>&1 && { SEEN=1; break; }; sleep 10; done
[ "$SEEN" = "0" ] && { echo "$(date '+%m-%d %H:%M') training never started -> exit" >> "$CLOG"; exit 1; }
echo "$(date '+%m-%d %H:%M') training detected -> canary monitoring $LOG" >> "$CLOG"
# LEAK-AWARE canary. A REAL camera leak = detector blind even when the target IS in frame →
# P(detect|in_fov) crashes toward 0. The POLICY losing the target (freeze-release transition,
# hard T3 far-range episodes) drops raw det too, BUT keeps P(detect|in_fov) high (~0.75) — that
# is NOT a leak and must NOT halt the run (it's the policy learning). So gate on P(detect|in_fov)
# from the live flight log, never on raw det. (2026-08-30: old raw-det canary false-killed the
# freeze-fix run at det=0.17 while P(detect|in_fov) was 0.75 = healthy camera, policy transition.)
FLOG="$HOME/flight_log_latest.csv"
LEAK=0
while pgrep -f rl_train_td3 >/dev/null 2>&1; do
  sleep 60
  read PEASY NEASY DETF <<<"$(python3 - "$FLOG" <<'PY'
import sys,csv,numpy as np,math
try:
    r=list(csv.DictReader(open(sys.argv[1])))[-800:]
    det=np.array([1.0 if x.get('raw_det')=='REAL' else 0.0 for x in r])
    fov=np.array([float(x.get('in_fov','nan') or 'nan') for x in r])
    td =np.array([float(x.get('true_dist_3d','nan') or 'nan') for x in r])
    easy=(fov>0.80)&(td<15.0)&(~np.isnan(td))          # GT-based: well-framed & close = easy to detect
    pe=det[easy].mean() if easy.sum()>=15 else float('nan')
    print(f"{-1 if math.isnan(pe) else round(pe,3)} {int(easy.sum())} {det.mean():.3f}")
except Exception:
    print("-1 0 0")
PY
)"
  echo "$(date '+%m-%d %H:%M') P_easy=$PEASY n_easy=$NEASY det=$DETF leak=$LEAK" >> "$CLOG"
  # REAL leak = EASY targets (well-framed & close, GT-based) are being MISSED (P_easy < 0.40) — same
  # robust metric as cam_health.py. This does NOT conflate policy framing-quality with camera health.
  # P_easy=-1 (too few easy samples = cold policy not framing) is NOT a leak — train through it.
  # (2026-08-31: replaced P(det|in_fov>0.5) which false-fired while cam_health read 0.95 — Rawad caught.)
  python3 -c "import sys
try: pe=float('$PEASY')
except: pe=-1.0
sys.exit(0 if (pe>=0 and pe<0.40) else 1)" 2>/dev/null && LEAK=$((LEAK+1)) || LEAK=0
  if [ "$LEAK" -ge "${LEAK_HALT:-3}" ]; then
    echo "$(date '+%m-%d %H:%M') LEAK-CANARY HALT P_easy=$PEASY x${LEAK_HALT:-3} — easy targets missed = real camera leak, killing to protect the buffer (needs wsl --shutdown)" >> "$CLOG"
    kill -9 $(pgrep -f rl_train_td3) 2>/dev/null
    kill -9 $(ps -eo pid,args | grep -E 'gzserver|px4|mavros|rosmaster|roslaunch|launch_stack' | grep -v grep | awk '{print $1}') 2>/dev/null
    exit 3
  fi
done
echo "$(date '+%m-%d %H:%M') training ended (canary exit)" >> "$CLOG"
