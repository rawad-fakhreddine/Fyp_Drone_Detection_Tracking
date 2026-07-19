#!/bin/bash
# code_stamp.sh RUN_TAG_PATH — write a code-provenance record next to a run's
# CSV: md5 of every behavior-relevant script + git state + scenario env.
# Called by launch_stack per run (Rawad's directive 2026-07-17: results must
# record the code that produced them — revision by fact, not by guessing).
S=~/catkin_ws/src/drone_tracking/scripts
OUT="${1:?usage: code_stamp.sh /path/to/run_basename}"
{
  echo "stamped: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "git_head: $(git -C ~/Fyp_Drone_Detection_Tracking rev-parse --short HEAD 2>/dev/null)"
  echo "git_dirty: $(git -C ~/Fyp_Drone_Detection_Tracking status --porcelain 2>/dev/null | wc -l) files modified vs HEAD"
  echo "--- script md5 (behavior-relevant) ---"
  md5sum "$S/ibvs_controller_node.py" "$S/kalman_filter_node.py" \
         "$S/yolo_detection_node.py" "$S/target_mover.py" \
         "$S/launch_stack.sh" "$S/random_spawn_target.py" 2>/dev/null \
    | awk '{n=split($2,p,"/"); printf "%s  %s\n",$1,p[n]}'
  echo "--- scenario env ---"
  env | grep -E '^(VX_MODE|KP_|KI_|KD_|INT_|D_HOLD|D_STAR|D_SAFE|MIN_DIST|MAX_V|MAX_ACCEL|A_DEC|ALPHA_|PITCH_|DERIV_|EMERG_|BAND_KP|PRED_|SEARCH_|CHASER_ZDN|STRAIGHT_|SPAWN_YAW|START_DIST|LOSS_TIMEOUT)' | sort
} > "$OUT.code.txt" 2>/dev/null
