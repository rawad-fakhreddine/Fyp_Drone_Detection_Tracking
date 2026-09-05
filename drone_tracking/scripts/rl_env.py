#!/usr/bin/env python3
"""
rl_env.py — RL Milestone (Config 3) · Step 1: environment skeleton + observation builder.

DroneTrackingEnv (gymnasium.Env): the RL policy replaces the IBVS CONTROL block only —
YOLO perception stays. This file implements the LOCKED observation design
(FYP/RL/observation/Observation_Design.docx) and the action interface
(FYP/RL/action/Action_Design.docx). Reward/reset are Step 3/4 stubs.

Observation (single frame, 13 values, normalized; frame-stacking N=4 is done by the
SB3 VecFrameStack wrapper, NOT here):
    [ ex, ey_c, d_hat, dex, dey, dd, w, h, conf, t_since_det, pitch, roll,
      a_prev(4) ]                    -> total 16 floats (a_prev = 4 axes)
  - ex   = (cx-320)/320
  - ey_c = pitch-compensated vertical error (same math as ibvs_controller_node:
           beta = atan2(cy-240, F_PX);  ey_c = (F_PX/240)*(beta + pitch_comp*pitch))
  - d_hat = sqrt(alpha_dist_k / alpha), alpha = box_area/(640*480), k = 0.077 (calibrated)
  - dropout rule (locked): FREEZE last valid ex/ey/d/w/h; rates -> 0; conf -> 0;
    t_since_det rises (clipped at 1 s).
Action: [vx, vy, vz, wz] in [-1,1], scaled to caps -> /mavros/setpoint_raw/local
        (FRAME_BODY_NED velocities) — the exact IBVS output interface.

Modes (rosparam ~mode):
  probe  : READ-ONLY. Print the live observation ~2 Hz while the normal stack flies.
           Validates the obs against a real IBVS flight before any learning.
  record : READ-ONLY. Write (obs, IBVS-action) pairs at 20 Hz to CSV for the
           behaviour-cloning warm-start (captures IBVS cmds from setpoint_raw/local).
  env    : library use — import DroneTrackingEnv; step() publishes velocities.

Usage:
  rosrun drone_tracking rl_env.py _mode:=probe
  rosrun drone_tracking rl_env.py _mode:=record _out:=~/rl_demos/T3_C1_s42.csv
"""
import os, sys, csv, math, time, threading
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import String, Int32
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, CommandBool
from gazebo_msgs.msg import ModelStates

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:            # probe/record work without gymnasium installed
    _GYM = False

IMG_W, IMG_H = 640.0, 480.0
IMG_CX, IMG_CY = 320.0, 240.0
F_PX = 277.19                  # live camera_info fx (same constant as IBVS)
AREA_NORM = IMG_W * IMG_H

# 2026-08-18 (supervisor): EXPLICIT-RATE observation — the finite-difference rates
# dex/dey/dd carry the temporal info, so pitch/roll are DROPPED and frame-stacking
# is OFF (single frame). a_prev kept for action continuity + the smoothness reward.
OBS_NAMES = ["ex","ey","d_hat","dex","dey","dd","w","h","conf","t_nodet",
             "a_vx","a_vy","a_vz","a_wz"]
OBS_DIM = len(OBS_NAMES)

# Env steps at 20 Hz (rl_env _rate 20). The approach/retreat rewards use the GT distance
# CHANGE between consecutive steps, (d_prev - d_true), which at ~1 m/s closing is only ~0.05 m
# — so w_approach·Δd was ~25× too small vs the O(1) centering/band terms, and pursuit was barely
# rewarded (2026-08-30 audit: this is why the freeze fix was weak). Dividing by DT_NOMINAL converts
# the per-step delta to a per-SECOND closing RATE (m/s), so w_approach is now reward-per-(m/s) and
# the approach term is O(1) like every other term. Nominal (not exact) dt is fine — it's a shaped
# reward, not a physics constraint; the 20 Hz pacing is stable.
DT_NOMINAL = 0.05

# Reward params (FYP/RL/reward/Reward_Design.docx). Defaults are START values, all
# rosparam-overridable. Design principles baked in here:
#  (1) BOUNDED, COMPARABLE SCALE (2026-08-24): every per-step term is clipped to ~[-1,1]
#      so no single term dominates the critic's target. The OLD reward left the out-of-band
#      distance penalty UNBOUNDED (−16.5/step at d=40 m) and terminals at −100/−150 (~100×
#      the per-step scale) — that scale gap made Q-values hard to fit and stalled/degraded
#      learning. Now per-step total ≈ [−2,+2]; terminals are the largest but only ~7× that.
#      NB: this CHANGES reward semantics → a replay buffer built with the old reward must NOT
#      be reused; train fresh (--scratch) after this change.
#  (2) per-frame P_lost is small (0.5); the real loss deterrent is the SUSTAINED-loss terminal.
#  (3) collision terminal (P_safe) > sustained-loss terminal (P_lost_final): when a loss looks
#      inevitable the agent must NEVER score better by RAMMING (−P_safe) than by losing it
#      (−P_lost_final). The 2 m safety bar in reward form.
#  (4) the centering bonus (+A) is paid ONLY on a valid detection, so a frozen (lost) frame
#      can't collect it. A tracked in-band step (+2) dominates either terminal.
REWARD_DEFAULTS = dict(
    w_ex=0.0,                  # linear centering penalty −w_ex·(|ex|+|ey|), vision-gated (2026-08-30
                               # T3/T6 lateral-lag fix): the Gaussian centering bump (sigma=0.6) is
                               # nearly flat near the operating point — at ex=+0.31 the policy still
                               # collects 66% of max reward, so it SETTLES for a steady lateral lag
                               # (target parks off-centre right, |ex|~0.31, never driven to 0). A linear
                               # term adds a CONSTANT inward gradient at all offsets → drives ex/ey to 0
                               # (fixes the under-turn on fast-lateral T3 + inclined T6). Default 0 = OFF.
    A=3.0, sigma=0.6,          # centering (valid only): exp bump in [0,A], peaks +A when centered.
                               # sigma WIDENED 0.3→0.5→0.6; A RAISED 1.0→3.0 (2026-08-24 eval #2).
    sigma_far=0.0,             # >0 → use this WIDER sigma when d>band_hi (pursuing) so sprinting
                               # off-centre isn't punished (fixes sprint-vs-center contradiction). 0=OFF.
                               # Priority fix: with all other terms capped ~+1, centering at A=1 paid
                               # only +0.18 at the working point |ex|=0.65 while band/approach paid
                               # their full +1 each → policy farmed distance, ignored the frame (eval:
                               # 61% in-FOV, mean|ex|=0.65, 47% action saturation). A=3/sigma=0.6 make
                               # centering DOMINANT: +3 centered, +0.93 at |ex|=0.65 (real edge→center
                               # gradient). Centering-in-FOV are the same objective (Rawad 2026-08-24):
                               # a strongly-rewarded centered target cannot leave the frame.
    w_d=0.5, band_lo=6.0, band_hi=7.0,   # band: +band_bonus inside [6,7] m; linear penalty outside...
                               # S17b: reverted w_d 0.7→0.5. The 0.7 steepening (S17) closed distance
                               # (23→8.5m) but the aggressive maneuver climbed to the 22m ceiling and lost
                               # framing (det 0.93→0.005) — it overpowered the w_alt=0.25 recovery balance.
                               # Keep the GENTLE 0.5/m far slope (which w_alt was tuned against); the real
                               # fix was band_pen_cap_far 8→30 (extend the gradient), NOT steepen it.
    # S18 BAND-WIDTH ANNEALING (structural, off by default). The [6,7] band is unreachable in one
    # lunge: approaching drops framing → the policy retreats to the ~9m framing-safe equilibrium.
    # But BC1 proves 6-8m IS framable — the barrier is the APPROACH PATH, not the destination. So
    # start the band WIDE (band_hi_start, e.g. 9.0 — which the policy already holds) and shrink the
    # ceiling to band_hi over band_anneal_steps, forcing small framed approach steps instead of a
    # lunge. band_anneal_steps=0 → OFF (band_hi fixed, preserves 2b and all prior configs).
    band_hi_start=9.0, band_anneal_steps=0,
    band_pen_cap=1.0,          # ...clipped to −band_pen_cap.  [BC1-repro: mild band — sim-vs-config diagnostic]
    # CLOSE-SIDE band penalty (2026-08-29, T1/T5 too-close fix). The default close side
    # (w_d 0.5, cap 1.0) is too weak/asymmetric vs the far side (cap 30): sitting at 5.7m
    # costs only −0.15 so the equilibrium leans below band-center 6.5 and dips <5m (T1 30%,
    # T5 20% — CONFIRMED not estimate bias: d_hat tracks GT ±0.2m, policy KNEW it was close).
    # w_d_close/band_pen_cap_close give the close side its OWN steeper slope + higher cap so
    # <6m actively repels UP toward 6.5. Defaults = old behavior (0.5/1.0) → backward-compat.
    w_d_close=0.5, band_pen_cap_close=1.0,
    band_bonus=1.5,            # in-band bonus. S13: 1.0→1.5 (make [6,7] the clear reward peak). [was
                               # reverted to BC1's healthy config to test sim degradation
                               # ships back to 2b's 1.0/0.5/1.0 at sweep end unless a config wins]
                               # Band-shaping sweep 2026-08-24 (s3 cap 2.5 resume; s4 bonus 2.0 scratch):
                               # NEITHER moved the deterministic policy into [6,7] — it settles ~9m
                               # regardless (in-band ~0.14 in all of 2b/s3/s4). CONCLUSION: the tight
                               # [6,7] band is NOT reachable by reward shaping on the fast T4 orbit —
                               # close-range framing is structurally too risky, so ~9m (reliable
                               # framing) is the rational equilibrium. Stronger band pulls only
                               # destabilise (s3) or hurt smoothness (s4). See AUTONOMOUS_SESSION_SUMMARY.
                               # Structural levers for the band (future): raise max_wz cap; move band out.
    w_s=0.20, smooth_cap=1.0,  # smoothness −w_s·‖Δa‖², clipped to −smooth_cap. w_s 0.05→0.20
                               # (2026-08-24 eval #2): 47% of steps pinned an axis to the rail (bang-
                               # bang) → 14 m/s² velocity jerk. At 0.20 a full action swing costs the
                               # −1 cap, biting hard enough to smooth the control vs the +3 centering.
    w_approach=1.0, approach_cap=1.5,    # CASE 1 (far, d>band_hi): +w_approach·(closing RATE m/s), clip +approach_cap.
                               # 2026-08-30 audit fix: term is now RATE-based (÷DT_NOMINAL) so w_approach is
                               # reward-per-(m/s). At 1 m/s closing this pays ~1.0 (was ~0.05 with the old per-step
                               # delta — ~20× too weak, which starved pursuit and let the freeze persist). Cap 1.5
                               # keeps it BELOW the +3 centering term so strong pursuit can't yank the chaser laterally
                               # off-frame (the 2026-08-24 failure) — the strafe penalty channels pursuit into fwd+yaw.
    w_retreat=2.0, close_thresh=6.0, retreat_cap=1.0,   # CASE 2 (too close, d<close_thresh): GT-always
                               # S15 curriculum uses S13's config (close_thresh=6.0). S14's 5.5 overshot
                               # to too-close on T4 (deterministic); reverted. S13 (thresh=6) is the base.
                               # S13: close_thresh 5→6, w 1.5→2.0 — CONTINUOUS push off the close side up
                               # to the band FLOOR. S12 parked at 4.9m where retreat (thresh 5) barely
                               # engaged; now 4.9-6m is actively rewarded to back off toward the band.
                               # safety term +w_retreat·(d−d_prev): REWARDS backing off, PENALISES
                               # continuing to close when already <5 m. Symmetric to approach; stops
                               # the hug-too-close/collision behaviour vx=4 exposed. Bounded ±retreat_cap.
                               # S11 (2026-08-25): TRIED close_thresh 5→6 + w 1.5→2.0 (retreat-to-floor)
                               # paired with TARGET_ENTROPY=-8 → FAILED (oscillating limit-cycle, 105
                               # collisions, worst yet). REVERTED. Reward shaping cannot hold the band in
                               # pure scratch — needs a smooth prior (BC1). See SWEEP_RESULTS.md S11 note.
    w_vy_pen=0.0,              # strafe penalty −w_vy_pen·|vy| (2026-08-30 degenerate-strategy fix):
                               # pure-RL learned to STRAFE (vy saturated ~1.04) to track laterally
                               # instead of yawing like IBVS (vy~0.02). Penalising |vy| removes the
                               # strafe crutch → forces yaw(wz)+pursuit(vx), the IBVS strategy. OFF=0.
    w_vel=0.05,                # anti-hover: small reward for any nonzero action magnitude
    w_alt=0.25, alt_cap=12.0,  # altitude-match (GT, always-on): −w_alt·min(|cz−tz|,alt_cap).
                               # alt_cap 4.0→12.0 (2026-08-30 vertical-plateau fix): the log study of
                               # T3/T6/T7/T8 showed the target rising/inclining opens a 5-11 m vertical
                               # gap while the chaser lags — but the OLD 4 m cap made the penalty a FLAT
                               # plateau beyond 4 m (r_alt = −0.25·4 = −1.0 constant), so there was ZERO
                               # gradient to climb HARDER when far below → target exits the TOP of frame
                               # (el 30-55°, 39%/45% of T7/T8 spent >4 m below). Same plateau bug fixed
                               # on the band-far side (band_pen_cap_far); now graded across the whole
                               # 0-12 m vertical range so the chaser is pulled up continuously. Slope
                               # (0.25/m) UNCHANGED → level trajs (T3 gap ~0.6 m) see NO change (no
                               # regression); only the >4 m vertical losses gain gradient.
                               # RE-ENABLED 0.0→0.25 (2026-08-24 session-2 evidence): the ey-alone
                               # test (w_alt=0) FAILED — the policy drifted UP above the target (alt
                               # 11→22m ceiling), the target sank out the BOTTOM of frame (ey pinned
                               # +1.02), and once blind ey freezes with NO gradient to descend →
                               # detection_frac peaked 0.34 then regressed to 0.18, every late episode
                               # lost at the ceiling. Centering (vision-gated) cannot fix this: it only
                               # acts while the target is visible. w_alt is the ALWAYS-ON (works-blind)
                               # recovery gradient pulling the chaser to the target's altitude so the
                               # target re-enters frame. Complementary to centering-ey (aligned when
                               # visible), essential when blind. Caps at −1.0 for |Δalt|≥4m.
    P_lost=0.5,                # per-frame keep-in-view penalty (detection lost)
    d_min=2.0, P_safe=15.0,    # collision: d_true<d_min → terminal −P_safe (bounded; > loss terminal)
    loss_secs=5.0, P_lost_final=10.0,    # sustained loss>5s → terminal −P_lost_final
    # DAMPING (S12, 2026-08-25): reward LOW per-step distance-rate |Δd| when detected AND in the
    # settle zone [damp_zone_lo, damp_zone_hi] around the band. Every prior run rewarded the target's
    # POSITION (in-band) but not HOLDING STILL — so in-band-swinging scored the same as in-band-parked,
    # giving the policy no reason to stop the close↔far limit-cycle. This Gaussian bonus (peaks +w_damp
    # at Δd=0) makes PARKED-IN-BAND the single highest-value state (band_bonus + damp stack), directly
    # opposing the oscillation. Vision-gated (park only while you can see the target). damp_sigma is the
    # per-step Δd scale at 20 Hz (~0.06 m/step: parked ~0.01 → ~+w_damp; fast surge ~0.15 → near 0).
    band_pen_cap_far=30.0,     # S17: 8.0→30.0 — the T4curr deterministic policy PARKED at exactly 23m,
                               # the old cap edge (7 + 8/0.5): the cap re-created the flat far basin one
                               # more time. 30.0 pushes the gradient wall out to ~50m so NO reachable far
                               # distance is a stable optimum. (S16's 8.0 with w_d=0.5 saturated at d=23m
                               # → flat far basin → policy sat there.) The operating range now has a real
                               # inward gradient everywhere. Close side stays capped at band_pen_cap=1.0.
    # DISTANCE-RATE (ḋ) DAMPING (2026-08-29, T1/T5 anti-hunting). r_damp above only pays a
    # BONUS inside [6,7.5]; it does nothing to punish the swing that carries the policy OUT of
    # band (T1 static std=1.6m → dips <5 30%, out >9 12%). w_ddot adds a linear PENALTY on
    # |d−d_prev| that bites everywhere — but GATED to |ex|<ddot_ex_gate (the HOLDING regime) so
    # it never penalises legitimate pursuit closing/opening (uncentered = chasing → no penalty).
    # Default w_ddot=0 → OFF → backward-compat (pursuit trajectories unaffected).
    w_ddot=0.0, ddot_ex_gate=0.30, ddot_cap=1.0,
    # BLIND-PURSUIT (2026-08-30, default OFF): the diagnosed fix for T3/T6/T7. approach/band are
    # VISION-FIRST (paid only when the target is detected), so a receding target that slips out of
    # frame gives ZERO gradient -> the policy freezes at vx~0 and the target escapes (a 3x approach-
    # weight boost had no effect on cmd_vx -> wrong lever). This term rewards CLOSING GT distance
    # (penalises growing it) for a brief window t_lost<pursuit_blind_secs after loss, so the policy
    # pushes THROUGH a dropout instead of freezing. GT-legal (training only). w=0 => no-op.
    w_pursuit_blind=0.0, pursuit_blind_secs=1.0, pursuit_blind_cap=1.0,
    # SITUATIONAL FORWARD-PURSUIT (2026-09-02, default OFF): the diagnosed fix for T3 (GT: chaser
    # sustains only ~1.5 m/s vs a 3+ m/s receding target = behavioral under-commit, NOT a cap limit;
    # raising max_vx barely moved mean speed). Rewards commanding FORWARD vx (a[0]>0) ONLY when the
    # target is VISIBLE, FAR (d>band_hi) AND RECEDING (d_true>d_prev). This is SITUATIONAL: on
    # lateral trajectories (T4/T5/T8 orbit/lemniscate/helix) the target keeps ~constant distance
    # (not receding) so this NEVER fires → their vy-strafe centering is untouched. This replaces the
    # global w_vy_pen (which doubled |ex| and wrecked lateral centering: T4 cen% 78→26). Positive +
    # bounded → far lower divergence risk than penalty-heavy pushes. w_fwd=0 => no-op.
    w_fwd=0.0, fwd_cap=1.5,
    # VELOCITY-MATCHING / ANTI-ESCAPE (2026-09-02, default OFF): the OUTCOME-based T3 fix. The
    # action-based w_fwd got GAMED (chaser jabbed forward to collect it but mean dist got WORSE
    # 17→20.7m). This rewards the OUTCOME instead: when VISIBLE + FAR (d>band_hi), pay full w_match
    # for KEEPING UP (ḋ≤0, distance not growing) and taper to 0 as the target escapes at
    # match_recede_ref m/s. Uses the REAL distance-rate ḋ=(d_true−d_prev)/DT so it CANNOT be gamed
    # by faking forward commands. Positive + bounded [0,w_match] → divergence-safe. Complements
    # r_approach (which rewards actual closing ḋ<0): keep-up is the floor, closing earns extra.
    # SITUATIONAL like w_fwd (only fires far) so lateral-centering trajectories are untouched. w=0 => no-op.
    w_match=0.0, match_recede_ref=2.0,
    # ---- r_acc (ACCELERATION-PRIORITY, 2026-09-02, Rawad directive "prioritise the chaser's
    # acceleration; learn high-speed target → high-speed chaser") ----
    # The velocity gap while chasing is the closing rate ḋ (ḋ>0 = falling behind). "The chaser
    # accelerates to handle the target" == the chaser makes ḋ SHRINK over time. So reward the
    # REDUCTION in ḋ, i.e. −(ḋ_t − ḋ_prev), whenever VISIBLE + FAR (d>band_hi) + FALLING BEHIND
    # (ḋ_t>0). ḋ_t=(d−d_prev)/DT, ḋ_prev=(d_prev−d_prev2)/DT — needs the two-steps-back distance.
    # WHY THIS WORKS where r_fwd (gamed) / r_match (too sparse) did not:
    #   • DENSE + EARLY: pays the INSTANT the chaser starts accelerating to close the gap (does not
    #     wait for ḋ≤0 like r_match) → fixes exploration starvation (the deterministic-mean wall).
    #   • NON-GAMEABLE: built from the REAL ḋ trend (GT distance), not the commanded action.
    #   • It literally rewards the chaser's acceleration RELATIVE to the target = the requested lever.
    # Bounded ±acc_cap. Gated ḋ_t>0 so it never fires once matched/closing (r_approach owns that).
    # w_acc=0 => no-op (backward-compat). SITUATIONAL (far only) so lateral trajs are untouched.
    w_acc=0.0, acc_cap=3.0,
    # ---- r_speedmatch (SPEED-FLOOR, 2026-09-03, Rawad: "after transient, MIN chaser speed = target
    # speed; below it you lose the target — increase chaser speed") ----
    # Reward the chaser's SPEED RATIO directly: ratio = chaser_hspeed / target_hspeed. Dense +
    # MONOTONIC from 0→speed_ratio_cap → fills the reward DEAD-ZONE the diag found (cmd 2.2→3.4 all
    # earned 0 before because ḋ-based terms die when target escapes). Pays proportionally for EVERY
    # increment of speed toward matching the target, saturating just above 1.0 (speed_ratio_cap=1.2)
    # to still favour closing. GT speeds (twist) = training-only, legal. Positive+bounded → divergence
    # -safe. Gated VISIBLE + FAR (d>speedmatch_dmin) + TARGET-MOVING (tspd>speedmatch_tmin) so static/
    # in-band trajectories are untouched (transient/hold handled by r_center/r_band). w=0 => no-op.
    w_speedmatch=0.0, speed_ratio_cap=1.2, speedmatch_dmin=8.0, speedmatch_tmin=0.8,
    speedmatch_center_sigma=0.0,   # >0 → multiply speed reward by exp(-ex²/2σ²): pay speed ONLY while
                                   # centered (teaches sprint-WHILE-tracking; 0 = OFF, no gate).
    speedmatch_floor=0.6,   # pay ~0 below this speed-ratio, ramp to full at speed_ratio_cap. Keeps the
                            # offset tiny at the current timid ratio (~0.65) → no critic scale-shock on
                            # resume — the reward is a GRADIENT toward match, not a constant bonus.
    w_damp=1.5, damp_sigma=0.06, damp_zone_lo=6.0, damp_zone_hi=7.5,   # S13: zone 5→6/8→7.5 to
                               # OVERLAP the band. S12 parked at 4.9m = just below the old floor (5),
                               # so damping never engaged to hold the band. Now the parking bonus is
                               # earned ONLY inside [6,7.5] → the policy must come UP to collect it.
)

def compute_reward(p, a, prev_a, ex, ey, d_true, valid, t_lost, d_prev=None, alt_gap=None, d_prev2=None,
                   chaser_spd=None, target_spd=None):
    """Shaped reward + terminal flag. p = params dict (see REWARD_DEFAULTS). GT (d_true,
    t_lost) is legal here — reward is training-only. Returns (reward, terminated, reason).

    VISION-FIRST (2026-08-24 root fix): centering, band AND approach are ALL paid ONLY on a
    valid detection. The only way to earn reward is to keep the target in the camera. This
    closes the loophole where the GT band reward paid for the right distance even while BLIND
    — which let the policy sink below the target (target leaves the top of frame → no ey
    signal → stuck low, blind, but still collecting distance credit). Now a blind step scores
    only −P_lost, so keeping the target framed (hence matching altitude + yaw) is non-optional.
    All per-step terms are BOUNDED to ~[-1,1] so the reward stays on one scale."""
    if valid:
        # centering: exp bump in [0,1], peaks +A when centered.
        # DISTANCE-WEIGHTED SIGMA (2026-09-03, Rawad — resolves the sprint-vs-center contradiction):
        # r_center(A=3) DOMINATED r_approach(cap 1.5), so sprinting to catch up (which pushes the
        # target off-centre) LOST more centering reward (~-2.2) than the closing gain (+1.5) → the
        # reward literally paid the policy to stay slow-and-centred = the timid basin. FIX: when FAR
        # (d>band_hi = pursuing), widen sigma to sigma_far so off-centre is tolerated → sprinting is
        # no longer punished and the closing reward drives the catch-up. In-band keeps the tight sigma
        # for precise HOLD. sigma_far=0 → OFF (uses sigma everywhere = old behaviour).
        _sig = p['sigma']
        _sf = p.get('sigma_far', 0.0)
        if _sf > 0.0 and not math.isnan(d_true) and d_true > p['band_hi']:
            _sig = _sf
        r_center = p['A'] * math.exp(-(ex*ex + ey*ey) / (_sig*_sig))
        # linear centering pressure: constant inward gradient the Gaussian lacks near centre
        # (fixes the steady lateral lag on T3/T6 where the target parks off-centre). Default off.
        r_center -= p.get('w_ex', 0.0) * (abs(ex) + abs(ey))
        # band: +band_bonus inside [6,7]; linear penalty outside. S16 FIX: the FAR side uses a
        # much larger cap (band_pen_cap_far) so there is a CONTINUOUS inward gradient — the old
        # −1.0 cap made a FLAT plateau beyond ~9 m (zero gradient), so the policy sat far (S15 on
        # STATIC target parked at 18 m, in_band 0, drifting farther). Close side keeps the small
        # cap (safety terminal handles <2 m). Far cap high enough to grade the whole 7-24 m range.
        bcap = p.get('band_pen_cap', 1.0)
        bcap_far = p.get('band_pen_cap_far', 8.0)
        if math.isnan(d_true):        r_band = 0.0
        elif d_true < p['band_lo']:   r_band = max(-p.get('band_pen_cap_close', bcap), -p.get('w_d_close', p['w_d']) * (p['band_lo'] - d_true))
        elif d_true > p['band_hi']:   r_band = max(-bcap_far, -p['w_d'] * (d_true - p['band_hi']))
        else:                         r_band = p.get('band_bonus', 1.0)
        # approach: bounded positive signal for closing distance when outside band
        r_approach = 0.0
        if (d_prev is not None and not math.isnan(d_true) and not math.isnan(d_prev)
                and d_true > p['band_hi']):
            r_approach = min(p.get('approach_cap', 1.0), p['w_approach'] * max(0.0, (d_prev - d_true) / DT_NOMINAL))
    else:
        # BLIND: target not in the camera → NO centering/band/approach credit. GT distance must
        # not pay while the target is unseen, or the policy loiters at the right range blind.
        r_center = r_band = r_approach = 0.0
    # anti-hover + smoothness are action-shaping, applied every step.
    r_vel = p.get('w_vel', 0.0) * float(np.linalg.norm(np.asarray(a, dtype=np.float32)))
    # strafe penalty: −w_vy_pen·|vy_normalized| — removes the degenerate strafe crutch so the
    # policy must yaw+pursue (IBVS strategy) instead of over-strafing. a[1] = normalized vy.
    r_vy = -p.get('w_vy_pen', 0.0) * abs(float(a[1]))
    da = np.asarray(a) - np.asarray(prev_a)
    r_smooth = max(-p.get('smooth_cap', 1.0), -p['w_s'] * float(np.dot(da, da)))
    r_lost = -p['P_lost'] if valid == 0 else 0.0
    # altitude-match (GT, ALWAYS-ON incl. blind): the recovery gradient that pulls the chaser
    # back to the target's altitude so the target re-enters the frame. alt_gap = |cz − tz|.
    r_alt = 0.0
    if alt_gap is not None and not math.isnan(alt_gap):
        r_alt = -p.get('w_alt', 0.0) * min(abs(alt_gap), p.get('alt_cap', 4.0))
    # CASE 2 — too-close retreat (GT, ALWAYS-ON, safety): when d_true < close_thresh, reward the
    # signed distance change (d−d_prev): POSITIVE when opening (backing off), NEGATIVE when still
    # closing. Symmetric counterpart to the (far) approach term; teaches the policy to slow/reverse
    # before it collides. GT-always so it works even blind (safety must not need detection).
    r_retreat = 0.0
    if (d_prev is not None and not math.isnan(d_true) and not math.isnan(d_prev)
            and d_true < p.get('close_thresh', 5.0)):
        rcap = p.get('retreat_cap', 1.0)
        r_retreat = max(-rcap, min(rcap, p.get('w_retreat', 0.0) * ((d_true - d_prev) / DT_NOMINAL)))
    # CASE 3 — blind-pursuit (GT, brief-loss window, default OFF): the fix for the T3/T6/T7
    # vision-gated blind-freeze. When the target was JUST lost (t_lost small) and is FAR, reward
    # closing / penalise growing the GT distance so the policy pushes forward through the dropout
    # instead of freezing at vx~0. Bounded +/-pursuit_blind_cap. GT-legal (training only).
    r_pursuit = 0.0
    if (not valid and p.get('w_pursuit_blind', 0.0) > 0.0 and d_prev is not None
            and not math.isnan(d_true) and not math.isnan(d_prev)
            and t_lost < p.get('pursuit_blind_secs', 1.0)
            and d_true > p['band_hi']):
        pcap = p.get('pursuit_blind_cap', 1.0)
        # 2026-08-30 audit fix (acct2 continuing acct1): RATE-based (÷DT_NOMINAL) like r_approach/
        # r_retreat. Was raw per-step Δd (~0.05m) → ~0.05 reward vs +3 centering = starved. Now
        # closing at 1 m/s while briefly blind pays w_pursuit_blind·1.0 → forward re-acquire instead
        # of the blind-backoff freeze. Bounded ±pursuit_blind_cap (kept < centering so it can't yank).
        r_pursuit = max(-pcap, min(pcap, p['w_pursuit_blind'] * ((d_prev - d_true) / DT_NOMINAL)))
    # DAMPING (S12): Gaussian bonus for near-zero per-step distance change, paid only when the
    # target is visible AND distance is in the settle zone. Rewards PARKING in the band rather than
    # swinging through it — the direct counter to the close↔far limit-cycle. Bounded [0, w_damp].
    r_damp = 0.0
    if (valid and d_prev is not None and not math.isnan(d_true) and not math.isnan(d_prev)
            and p.get('damp_zone_lo', 5.0) <= d_true <= p.get('damp_zone_hi', 8.0)):
        dd_step = d_true - d_prev
        ds = p.get('damp_sigma', 0.06)
        r_damp = p.get('w_damp', 0.0) * math.exp(-(dd_step*dd_step) / (2.0 * ds * ds))
    # ḋ-DAMPING PENALTY (anti-hunting): linear cost on per-step |Δd|, ONLY in the holding
    # regime (target centered, |ex|<gate) so pursuit is never penalised. Bounded −ddot_cap.
    r_ddot = 0.0
    if (p.get('w_ddot', 0.0) > 0.0 and valid and d_prev is not None
            and not math.isnan(d_true) and not math.isnan(d_prev)
            and abs(ex) < p.get('ddot_ex_gate', 0.30)):
        r_ddot = max(-p.get('ddot_cap', 1.0), -p['w_ddot'] * abs(d_true - d_prev))
    # SITUATIONAL FORWARD-PURSUIT (2026-09-02): reward commanding forward vx (a[0]>0) ONLY when the
    # target is VISIBLE, FAR (d>band_hi) and RECEDING (d_true>d_prev) — the T3 chase. Scaled by the
    # recede rate so a faster-escaping target pays a bigger forward incentive; bounded +fwd_cap.
    # Positive-only, and gated to the receding case so lateral-centering trajectories are untouched.
    r_fwd = 0.0
    if (p.get('w_fwd', 0.0) > 0.0 and valid and d_prev is not None
            and not math.isnan(d_true) and not math.isnan(d_prev)
            and d_true > p['band_hi'] and d_true > d_prev):
        r_fwd = min(p.get('fwd_cap', 1.5),
                    p['w_fwd'] * max(0.0, float(a[0])) * ((d_true - d_prev) / DT_NOMINAL))
    # VELOCITY-MATCHING / ANTI-ESCAPE (outcome-based): when visible + far, pay for keeping up.
    # ddot = signed distance rate (m/s). ddot<=0 (keeping up / closing) → full w_match; tapers
    # linearly to 0 as ddot rises to match_recede_ref (target escaping). Real ḋ → not gameable.
    r_match = 0.0
    if (p.get('w_match', 0.0) > 0.0 and valid and d_prev is not None
            and not math.isnan(d_true) and not math.isnan(d_prev)
            and d_true > p['band_hi']):
        ddot = (d_true - d_prev) / DT_NOMINAL
        ref = max(1e-6, p.get('match_recede_ref', 2.0))
        keepup = 1.0 - max(0.0, ddot) / ref           # 1 when keeping up, 0 when escaping >= ref
        r_match = p['w_match'] * max(0.0, min(1.0, keepup))
    # r_acc: ACCELERATION-PRIORITY (2026-09-02). Reward the chaser for SHRINKING the closing rate ḋ
    # (out-accelerating the gap) when VISIBLE + FAR + FALLING BEHIND. ḋ_t=(d−d_prev)/DT,
    # ḋ_prev=(d_prev−d_prev2)/DT; reward −(ḋ_t−ḋ_prev) = the chaser's accel relative to the target.
    # Dense (fires the instant it accelerates to catch up) + non-gameable (real ḋ trend). ±acc_cap.
    r_acc = 0.0
    if (p.get('w_acc', 0.0) > 0.0 and valid and d_prev is not None and d_prev2 is not None
            and not math.isnan(d_true) and not math.isnan(d_prev) and not math.isnan(d_prev2)
            and d_true > p['band_hi']):
        ddot_t = (d_true - d_prev) / DT_NOMINAL
        if ddot_t > 0.0:                              # only while falling behind
            ddot_prev = (d_prev - d_prev2) / DT_NOMINAL
            cap = p.get('acc_cap', 3.0)
            # POSITIVE-ONLY (clip low at 0): reward accelerating-to-catch-up (ḋ shrinking) but do NOT
            # punish easing-off. The penalty side (2026-09-02) diverged training — it hit the timid
            # policy faster than it could learn (same collapse as prior T3 penalty pushes). Positive +
            # bounded [0,acc_cap] → divergence-safe like r_match, but DENSER (rewards the accel act).
            r_acc = max(0.0, min(cap, p['w_acc'] * (ddot_prev - ddot_t)))   # >0 when ḋ shrinking
    # r_speedmatch: SPEED-FLOOR (2026-09-03). Dense monotonic reward on chaser/target speed RATIO
    # when VISIBLE + FAR + target moving. Fills the dead-zone: pays for every m/s of chaser speed
    # up toward (and just past) the target speed → pulls the timid ~2.2 command up to break-even.
    r_speedmatch = 0.0
    if (p.get('w_speedmatch', 0.0) > 0.0 and valid and chaser_spd is not None and target_spd is not None
            and not math.isnan(d_true) and not math.isnan(chaser_spd) and not math.isnan(target_spd)
            and d_true > p.get('speedmatch_dmin', 8.0) and target_spd > p.get('speedmatch_tmin', 0.8)):
        ratio = chaser_spd / max(target_spd, 1e-6)
        flr = p.get('speedmatch_floor', 0.6); cap_r = p.get('speed_ratio_cap', 1.2)
        norm = (ratio - flr) / max(1e-6, cap_r - flr)   # 0 at floor, 1 at cap → gradient, no offset
        # CENTERING GATE (2026-09-03, Rawad "learn vx a little each step + don't lose target"):
        # multiply the speed reward by a Gaussian on |ex| so speed only pays WHILE CENTERED. Teaches
        # the sprint-WHILE-tracking coordination the critic needs (plain speed reward valued speed as
        # risky because sprinting broke tracking). From slow+centered the policy earns MORE for adding
        # speed only if it keeps the target centered → incremental vx climb, no reckless-sprint reward
        # (reward collapses the instant centering breaks → divergence-safe). csig=0 → OFF (backward compat).
        csig = p.get('speedmatch_center_sigma', 0.0)
        cen = math.exp(-(ex*ex) / (2.0 * csig*csig)) if csig > 0.0 else 1.0
        r_speedmatch = p['w_speedmatch'] * max(0.0, min(1.0, norm)) * cen
    reward = r_center + r_band + r_approach + r_vel + r_vy + r_smooth + r_lost + r_alt + r_retreat + r_damp + r_ddot + r_pursuit + r_fwd + r_match + r_acc + r_speedmatch
    terminated = False; reason = ""
    if (not math.isnan(d_true)) and d_true < p['d_min']:
        reward -= p['P_safe']; terminated = True; reason = "collision"   # GT safety, even if blind
    elif t_lost > p['loss_secs']:
        reward -= p['P_lost_final']; terminated = True; reason = "lost"
    return reward, terminated, reason


class ObsBuilder(object):
    """Assembles the locked observation vector from the live ROS topics.
    Read-only: subscribes, never publishes."""

    def __init__(self):
        self.k          = float(rospy.get_param("~alpha_dist_k", 0.077))
        self.pitch_comp = float(rospy.get_param("~pitch_comp", 1.3))   # launch default
        self.rate_lpf   = float(rospy.get_param("~rate_lpf", 0.6))     # EMA on rates
        # zero_aprev: feed ZEROS into the 4 a_prev obs slots so the policy CANNOT see
        # its own last action -> removes the positive-feedback runaway channel (the 2026-
        # 08-17 SAC finding). self.a_prev still tracks the REAL last action for the
        # smoothness reward (that path is unaffected). Default False = v1 obs unchanged;
        # SAC v2 passes ~zero_aprev:=true and uses a clone RETRAINED with a_prev zeroed.
        self.zero_aprev = bool(rospy.get_param("~zero_aprev", False))
        # raw detection state
        self.cx = self.cy = float('nan'); self.alpha = 0.0
        self.box_w = self.box_h = 0.0; self.conf = 0.0
        self.valid = False
        self.t_last_det = None
        # chaser attitude
        self.pitch = self.roll = 0.0
        # frozen-last-valid features (dropout rule) + rates
        self.f_ex = self.f_ey = 0.0; self.f_d = 0.0; self.f_w = self.f_h = 0.0
        self.dex = self.dey = self.dd = 0.0
        self._prev = None                                  # (t, ex, ey, d)
        # a_prev = the PREVIOUS action carried in the observation (policy action in
        # env mode; managed by the record loop in record mode). latest_cmd = the
        # freshest IBVS command captured by the tap — kept SEPARATE from a_prev so
        # the recorded label is never leaked into the observation (see _run_record).
        self.a_prev = np.zeros(4, dtype=np.float32)
        self.latest_cmd = np.zeros(4, dtype=np.float32)
        # max_vx LOWERED 8→4 (2026-08-24): the 8 m/s cap let the policy LUNGE forward and
        # overshoot the standoff band (d_true swung 3–52 m, per-step stuck ~−0.22, occasional
        # near-collisions), even though framing was solved. 4 m/s is still 2× the T4 target
        # speed (2 m/s) so pursuit/reacquisition still works, but distance is finer to regulate.
        # max_vy 1.2→2.5, max_wz 0.5→0.8 (session-5, 2026-08-24): saturation diagnostic on the
        # 2b/s4 evals showed the distance oscillation (2↔17 m, mean ~9 m) is CONTROL-LIMITED, not a
        # reward preference — the target orbits at ~2 m/s tangentially but max_vy=1.2 < 2 m/s, so the
        # chaser cannot hold station laterally (vy railed 15% of the time, wz railed 9%), while vx sat
        # idle (0.5/4). The policy strafes with its most-capped axis. Raising vy above the orbit speed
        # + more yaw authority lets it actually track the orbit and settle in the band. vx kept at 4.
        self.caps = np.array([float(rospy.get_param("~max_vx", 4.0)),
                              float(rospy.get_param("~max_vy", 1.2)),
                              float(rospy.get_param("~max_vz", 2.5)),
                              float(rospy.get_param("~max_wz", 0.5))], dtype=np.float32)
        # ground truth (record extras / reward / recovery scaffolding — NEVER in the obs)
        self.true_dist = float('nan')
        self.rel_w = (float('nan'), float('nan'), float('nan'))  # target−chaser, world XYZ
        self.chaser_hspeed = float('nan')   # GT chaser horizontal speed (world, m/s) — reward only
        self.target_hspeed = float('nan')   # GT target horizontal speed (world, m/s) — reward only
        self.chaser_yaw = 0.0                                    # chaser heading (world)
        self.chaser_alt = 0.0                                    # chaser ENU z (m), for altitude safety

        rospy.Subscriber('/drone_tracking/target_center', Point, self._center_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/target_box', Quaternion, self._box_cb, queue_size=1)
        rospy.Subscriber('/drone_tracking/detector_status', String, self._status_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._gz_cb, queue_size=2)

    # ---- callbacks -------------------------------------------------------
    def _center_cb(self, m):
        if math.isnan(m.x):                       # detector's explicit no-detection point
            self.valid = False; return
        self.cx, self.cy = float(m.x), float(m.y)
        self.alpha = abs(float(m.z)) / AREA_NORM
        self.valid = self.alpha > 1e-9
        if self.valid:
            self.t_last_det = rospy.Time.now()

    def _box_cb(self, m):
        self.box_w, self.box_h = float(m.z), float(m.w)

    def _status_cb(self, m):
        try:
            parts = m.data.split(',')
            self.conf = float(parts[1]) if len(parts) > 1 else 0.0
        except (ValueError, IndexError):
            self.conf = 0.0

    def _pose_cb(self, m):
        q = m.pose.orientation
        sp = 2.0*(q.w*q.y - q.z*q.x)
        self.pitch = math.copysign(math.pi/2, sp) if abs(sp) >= 1 else math.asin(sp)
        sr = 2.0*(q.w*q.x + q.y*q.z); cr = 1.0 - 2.0*(q.x*q.x + q.y*q.y)
        self.roll = math.atan2(sr, cr)

    def _gz_cb(self, m):
        # exact model names, matching flight_logger.gazebo_cb ('iris' / 'target_iris')
        try:
            ic = m.name.index('iris'); it = m.name.index('target_iris')
        except ValueError:
            return
        try:
            c, t = m.pose[ic].position, m.pose[it].position
            self.true_dist = math.sqrt((c.x-t.x)**2 + (c.y-t.y)**2 + (c.z-t.z)**2)
            # GT relative vector (world) + chaser yaw — TRAINING-ONLY, used by _recover to
            # re-acquire a target lost in ANY direction (never enters the observation).
            self.rel_w = (t.x - c.x, t.y - c.y, t.z - c.z)
            self.chaser_alt = float(c.z)
            # GT world-frame horizontal speeds (twist) — TRAINING-ONLY reward input (r_speedmatch);
            # NEVER enters the observation. Encodes Rawad's principle: chaser speed must be >= target.
            try:
                cv, tv = m.twist[ic].linear, m.twist[it].linear
                self.chaser_hspeed = math.hypot(cv.x, cv.y)
                self.target_hspeed = math.hypot(tv.x, tv.y)
            except (IndexError, AttributeError):
                pass
            q = m.pose[ic].orientation
            self.chaser_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y),
                                         1.0 - 2.0*(q.y*q.y + q.z*q.z))
        except IndexError:
            pass

    # ---- assembly --------------------------------------------------------
    def _t_since_det(self):
        if self.t_last_det is None: return 1.0
        return min((rospy.Time.now() - self.t_last_det).to_sec(), 1.0)

    def build(self):
        """Returns (obs16 normalized float32, raw dict for logging/printing)."""
        now = rospy.Time.now().to_sec()
        if self.valid:
            ex = (self.cx - IMG_CX) / IMG_CX
            beta = math.atan2(self.cy - IMG_CY, F_PX)
            ey = (F_PX / IMG_CY) * (beta + self.pitch_comp * self.pitch)
            d = math.sqrt(self.k / max(self.alpha, 1e-9))
            d = min(d, 40.0)
            if self._prev is not None:
                tp, pex, pey, pd = self._prev
                dt = max(now - tp, 1e-3)
                a = self.rate_lpf
                self.dex = a*self.dex + (1-a)*(ex - pex)/dt
                self.dey = a*self.dey + (1-a)*(ey - pey)/dt
                self.dd  = a*self.dd  + (1-a)*(d  - pd )/dt
            self._prev = (now, ex, ey, d)
            self.f_ex, self.f_ey, self.f_d = ex, ey, d
            self.f_w, self.f_h = self.box_w, self.box_h
            conf = self.conf if not math.isnan(self.conf) else 0.0
        else:
            # Explicit-rate dropout (supervisor 2026-08-23): freeze rates at last known
            # values instead of zeroing. Network retains last velocity direction during
            # YOLO blackout. _prev kept so rates resume correctly on re-acquisition.
            conf = 0.0

        if self.t_last_det is None:
            t_since_raw = 999.0
        else:
            t_since_raw = (rospy.Time.now() - self.t_last_det).to_sec()
        t_nd = min(t_since_raw, 1.0)
        obs = np.array([
            np.clip(self.f_ex, -1.5, 1.5),
            np.clip(self.f_ey, -1.5, 1.5),
            np.clip(self.f_d / 20.0, 0.0, 2.0),
            np.clip(self.dex / 2.0, -1.0, 1.0),
            np.clip(self.dey / 2.0, -1.0, 1.0),
            np.clip(self.dd  / 5.0, -1.0, 1.0),
            np.clip(self.f_w / 100.0, 0.0, 3.0),
            np.clip(self.f_h / 100.0, 0.0, 3.0),
            np.clip(conf, 0.0, 1.0),
            t_nd,
            # pitch/roll REMOVED (explicit-rate obs, 2026-08-18). a_prev slots: zeroed
            # when ~zero_aprev; the real self.a_prev still drives the smoothness reward.
            *(np.zeros(4, dtype=np.float32) if self.zero_aprev else self.a_prev[:4]),
        ], dtype=np.float32)
        raw = dict(cx=self.cx, cy=self.cy, alpha=self.alpha, d_hat=self.f_d,
                   valid=int(self.valid), conf=conf, t_nodet=t_nd, t_since_raw=t_since_raw,
                   pitch_deg=math.degrees(self.pitch), roll_deg=math.degrees(self.roll),
                   true_dist=self.true_dist,
                   # centering errors (ey = pitch-comp ey_c) + GT speeds, for TB monitoring
                   # (supervisor 2026-09-04: watch ex/ey off-center + chaser-vs-target speed gap)
                   ex=self.f_ex, ey_c=self.f_ey,
                   chaser_spd=self.chaser_hspeed, target_spd=self.target_hspeed)
        return obs, raw


class IbvsActionTap(object):
    """Captures the live IBVS velocity commands (probe/record modes) so a_prev and
    the BC action labels are the REAL controller output, normalized by the caps."""
    def __init__(self, builder):
        self.b = builder
        rospy.Subscriber('/mavros/setpoint_raw/local', PositionTarget, self._cb, queue_size=1)
    def _cb(self, m):
        a = np.array([m.velocity.x, m.velocity.y, m.velocity.z, m.yaw_rate], dtype=np.float32)
        self.b.latest_cmd = np.clip(a / self.b.caps, -1.0, 1.0)   # NOT a_prev (no label leak)


if _GYM:
    class DroneTrackingEnv(gym.Env):
        """Gymnasium wrapper. step() publishes body-frame velocities (the IBVS output
        interface). reward/terminated are Step-3/4 stubs; reset is reset-free for now
        (returns the current obs — target re-randomization comes in Step 3)."""
        metadata = {"render_modes": []}

        def __init__(self, ctrl_hz=20.0, episode_secs=None):
            super().__init__()
            self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.ob = ObsBuilder()
            self.dt = 1.0 / ctrl_hz
            # Episode length (Option A): truncate at max_episode_secs. RESET IS
            # RESET-FREE — no teleport of either drone (avoids the PX4 EKF instability
            # that teleports caused historically); the target keeps flying, an episode
            # is just a bounded time-window, and truncation is bootstrapped (not terminal).
            self.max_episode_secs = float(episode_secs if episode_secs is not None
                                          else rospy.get_param("~episode_secs", 100.0))
            self._ep_t0 = None
            self._ep_steps = 0
            # first-reset handoff window: HOVER (keepalive) for this long so IBVS can be
            # killed WITHOUT the policy ever stepping against it. The overlap-fight — the
            # RL policy stepping aggressively while a live IBVS also commands — drove the
            # target off-frame in every failed run; hovering during the kill removes it.
            self._first_reset = True
            self._handoff_wait = float(rospy.get_param("~handoff_wait", 6.0))
            self._prev_d_true = float('nan')   # for approach reward
            self._prev_d_true2 = float('nan')  # t-2 distance (for r_acc ḋ-trend)
            self._global_step = 0              # S18: persistent (non-episodic) step count for band annealing
            # ACCEL LIMIT (IBVS A_DEC analog, 2026-08-26): per-step slew cap on the APPLIED
            # action so the policy must RAMP velocity instead of stepping it. vx8 showed an
            # unbounded action → the deterministic mean swung 1.9↔51 m (near-collision + fly-out);
            # a bounded ramp is the IBVS ingredient that closes distance while keeping framing.
            # accel_limit in m/s^2 (vx/vy/vz), accel_limit_wz in rad/s^2. Δa_max(norm)=accel·dt/cap.
            # 0 = OFF (default, preserves 2b and every prior config).
            self._accel_lim = float(rospy.get_param("~accel_limit", 0.0))
            self._accel_lim_wz = float(rospy.get_param("~accel_limit_wz", self._accel_lim))
            # Altitude floor/ceiling (2026-09-04): configurable per-world. rl_empty is flat so
            # 11 m is safe; BAYLANDS has terrain/trees and the target flies at ~14 m+, so the
            # chaser must not sink below it → set alt_floor=14 for baylands (user directive).
            self._alt_floor = float(rospy.get_param("~alt_floor", 11.0))
            self._alt_ceil  = float(rospy.get_param("~alt_ceil", 22.0))
            # accel_limit_vz (2026-08-30 vertical fix): SEPARATE vertical-accel ceiling so the
            # chaser can react to the target's climb onset faster than the shared vx/vy ramp allows
            # (log study: chaser reacts ~1 s late to the rise-to-altitude → loses the target out the
            # top). Defaults to the shared accel_limit (no change unless overridden).
            self._accel_lim_vz = float(rospy.get_param("~accel_limit_vz", self._accel_lim))
            # ACTION EMA LOW-PASS (IBVS d_lpf analog, 2026-08-27): frequency-selective output
            # filter y_t = α·a_t + (1−α)·y_(t−1). Diagnostic showed the residual oscillation is
            # ~10 Hz chatter while the tracking signal is <1 Hz — a huge frequency gap, so a MILD
            # EMA (cutoff ~2-3 Hz) removes the dither with near-zero tracking lag (unlike the hard
            # slew cap accel_limit, which throttles tracking-band moves too). α=1.0 = OFF (default;
            # preserves 2b and every prior config). Lower α = more smoothing = more lag.
            self._act_lpf = float(rospy.get_param("~action_lpf", 1.0))
            # reward params (rosparam-overridable; defaults = REWARD_DEFAULTS)
            self._rp = {k: float(rospy.get_param("~rew_" + k, v))
                        for k, v in REWARD_DEFAULTS.items()}
            self._band_hi_final = self._rp['band_hi']   # anneal TARGET (final ceiling)
            # MIXED-TRAJECTORY training (2026-08-27, Rawad OK'd the target_mover change):
            # on each episode reset, command target_mover to switch to a RANDOM trajectory
            # from ~mix_trajs (e.g. "4,5,8"). Interleaving trajectories per-episode prevents
            # the single-traj overfit that made R9 strong on T4/T8 but weak on static/T5.
            # Empty = OFF (default). The publisher is created ONLY when mix_trajs is set, so
            # C1/C2 and single-traj RL are byte-identical (target_mover switches only when it
            # RECEIVES a message — a dormant subscriber with no publisher does nothing).
            _mt = str(rospy.get_param("~mix_trajs", "")).strip()
            self._mix_trajs = [int(x) for x in _mt.split(",") if x.strip()] if _mt else []
            self._traj_pub = (rospy.Publisher('/rl/set_target_traj', Int32, queue_size=1)
                              if self._mix_trajs else None)
            if self._mix_trajs:
                rospy.loginfo("[rl_env] MIXED-TRAJECTORY training: episodes drawn from %s",
                              self._mix_trajs)
            self.pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
            self._msg = PositionTarget()
            self._msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED  # world-z avoids tilt-drift
            self._msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                                   PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                                   PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                                   PositionTarget.IGNORE_YAW)
            # control-rate clock: rospy.Rate ABSORBS the SAC gradient-step time so the
            # effective loop stays ~20 Hz (a bare sleep(dt) would drift to ~13-15 Hz once
            # a gradient step is added on top of every env step).
            self._rate = rospy.Rate(ctrl_hz)
            # OFFBOARD keepalive: PX4 drops OFFBOARD after ~0.5 s without a setpoint.
            # A daemon thread republishes the last command at 20 Hz so the gaps during
            # SAC gradient steps, checkpoint/replay-buffer saves (seconds) and the
            # reset()-recovery wait never starve the stream. step()/reset() only UPDATE
            # self._msg (under the lock); the thread guarantees continuity.
            self._msg_lock = threading.Lock()
            self._alive = True
            # RL always runs with SKIP_IBVS=1 — no IBVS on the setpoint topic.
            # Start keepalive immediately so hover setpoints are published from env
            # creation onward. Without this, the gap between takeoff_both.py exiting
            # and reset() being called (~2-10s) causes PX4 to time out OFFBOARD and
            # land the drone before a single training step runs.
            self._ka_active = True
            self._ka = threading.Thread(target=self._keepalive, daemon=True)
            self._ka.start()

        def _keepalive(self):
            r = rospy.Rate(20)
            while self._alive and not rospy.is_shutdown():
                if self._ka_active:
                    with self._msg_lock:
                        self._msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
                        self.pub.publish(self._msg)
                r.sleep()

        def _set_cmd(self, vx, vy, vz, wz):
            """Atomically update + publish the setpoint the keepalive thread holds."""
            with self._msg_lock:
                self._msg.velocity.x = float(vx); self._msg.velocity.y = float(vy)
                self._msg.velocity.z = float(vz); self._msg.yaw_rate = float(wz)
                self._msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
                self.pub.publish(self._msg)      # immediate; thread republishes between steps

        def _set_world_cmd(self, vx_ned, vy_ned, vz_ned, wz):
            """Publish a one-shot FRAME_LOCAL_NED setpoint (used by _recover only).
            Bypasses EKF yaw dependency: _recover computes velocities in ENU world frame
            and converts to NED; PX4 applies them directly without body→world rotation."""
            m = PositionTarget()
            m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
            m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                           PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                           PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                           PositionTarget.IGNORE_YAW)
            m.velocity.x = float(vx_ned); m.velocity.y = float(vy_ned)
            m.velocity.z = float(vz_ned); m.yaw_rate = float(wz)
            m.header.stamp = rospy.Time(0)
            self.pub.publish(m)

        def close(self):
            self._alive = False

        def step(self, action):
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            prev_a = np.asarray(self.ob.a_prev, dtype=np.float32).copy()   # a_(t-1) for smoothness
            d_prev = self._prev_d_true                                       # for approach reward
            d_prev2 = self._prev_d_true2                                     # t-2 distance for r_acc
            # ACCEL LIMIT (IBVS A_DEC-style ramp): clamp per-step change in the applied action.
            # prev_a is the previously APPLIED action, so this is a true rate limiter — the policy
            # can pick a direction but not jump velocity, killing the unbounded deterministic swing.
            if self._accel_lim > 0.0:
                caps = self.ob.caps
                damax = np.array([self._accel_lim    * self.dt / max(float(caps[0]), 1e-6),
                                  self._accel_lim    * self.dt / max(float(caps[1]), 1e-6),
                                  self._accel_lim_vz * self.dt / max(float(caps[2]), 1e-6),
                                  self._accel_lim_wz * self.dt / max(float(caps[3]), 1e-6)],
                                 dtype=np.float32)
                a = np.clip(a, prev_a - damax, prev_a + damax).astype(np.float32)
            # ACTION EMA LOW-PASS (d_lpf analog): y_t = α·a_t + (1−α)·y_(t−1). prev_a is the last
            # APPLIED (filtered) action → this is a true first-order low-pass on the output. Mild α
            # kills the ~10 Hz chatter with minimal lag (tracking is <1 Hz). α=1.0 → no-op.
            if self._act_lpf < 1.0:
                a = (self._act_lpf * a + (1.0 - self._act_lpf) * prev_a).astype(np.float32)
            v = a * self.ob.caps
            # Convert body-frame action → world LOCAL frame so vz is world-z (avoids tilt-drift).
            # PROBE-VERIFIED 2026-08-24: setpoint_raw/local is ENU — velocity.x→world-x,
            # velocity.y→world-y, velocity.z→Up. Map body (fwd=v[0], right=v[1]) to world;
            # chaser_yaw is the ENU heading (CCW from world-x). The old code put the world-y
            # component into velocity.x (x/y swapped) → chaser drove ~90° off, flew away.
            yaw = self.ob.chaser_yaw
            c_y, s_y = math.cos(yaw), math.sin(yaw)
            vx_ned = v[0] * c_y + v[1] * s_y   # velocity.x (world-x) = fwd·cos + right·sin
            vy_ned = v[0] * s_y - v[1] * c_y   # velocity.y (world-y) = fwd·sin − right·cos
            # Altitude envelope [8m floor, 22m ceiling] — both clamp the policy action.
            # SIGN CONVENTION (verified live 2026-08-24): /mavros/setpoint_raw/local is ENU
            # on the ROS side — MAVROS converts ENU→NED internally, so vz POSITIVE = UP and
            # vz NEGATIVE = DOWN. The old code assumed NED (positive=down) and inverted every
            # altitude clamp: the ceiling forced vz=+2.5 to "descend" → drone climbed to 378 m.
            ALT_FLOOR = self._alt_floor   # configurable (default 11; baylands=14, see __init__)
                               # rl_empty flat=11 safe; baylands terrain/trees + target at ~14 m → 14
            ALT_CEIL  = self._alt_ceil
            alt = self.ob.chaser_alt
            if alt < ALT_FLOOR:
                climb = min(self.ob.caps[2], 1.5 * max(1.0, ALT_FLOOR - alt))    # +vz = UP
                if v[2] < climb:
                    v = v.copy(); v[2] = float(climb)
                rospy.logwarn_throttle(2.0, "[rl_env] alt-floor: z=%.2fm forcing vz=%.2f", alt, v[2])
            elif alt > ALT_CEIL:
                descend = max(-self.ob.caps[2], -1.0 * max(1.0, alt - ALT_CEIL))  # -vz = DOWN
                if v[2] > descend:
                    v = v.copy(); v[2] = float(descend)
                rospy.logwarn_throttle(2.0, "[rl_env] alt-ceiling: z=%.2fm forcing vz=%.2f", alt, v[2])
            self._set_cmd(vx_ned, vy_ned, v[2], v[3])
            self.ob.a_prev = a                       # closed loop: a_prev = own action
            self._rate.sleep()                       # paced 20 Hz (absorbs gradient-step time)
            obs, raw = self.ob.build()
            self._prev_d_true2 = self._prev_d_true   # shift t-1 → t-2 before updating (r_acc)
            self._prev_d_true = raw['true_dist']     # update for next step
            self._ep_steps += 1
            self._global_step += 1
            # S18: band-width annealing — shrink the ceiling band_hi_start → band_hi_final linearly
            # over band_anneal_steps (0 = off). Forces framed, incremental approach into the band.
            _asteps = self._rp.get('band_anneal_steps', 0)
            if _asteps > 0:
                _frac = min(1.0, self._global_step / _asteps)
                _hi0 = self._rp.get('band_hi_start', self._band_hi_final)
                self._rp['band_hi'] = _hi0 + (self._band_hi_final - _hi0) * _frac
                rospy.loginfo_throttle(10.0, "[rl_env] S18 band_hi=%.2f (step %d/%d)",
                                       self._rp['band_hi'], self._global_step, int(_asteps))
            elapsed = (rospy.Time.now() - self._ep_t0).to_sec() if self._ep_t0 else 0.0
            # altitude gap = |target_z − chaser_z| from GT rel vector (training-only reward input)
            _rz = self.ob.rel_w[2]
            _alt_gap = abs(_rz) if not math.isnan(_rz) else None
            reward, terminated, reason = compute_reward(
                self._rp, a, prev_a, float(obs[0]), float(obs[1]),
                raw['true_dist'], raw['valid'], raw['t_since_raw'], d_prev=d_prev, alt_gap=_alt_gap,
                d_prev2=d_prev2, chaser_spd=self.ob.chaser_hspeed, target_spd=self.ob.target_hspeed)
            truncated = (not terminated) and (elapsed >= self.max_episode_secs)  # Option A
            if alt < 10.0:
                rospy.logwarn_throttle(1.0, "[rl_env] LOW ALT in step: z=%.2fm vz_cmd=%.2f ep_step=%d",
                                       alt, v[2], self._ep_steps)
            if terminated or truncated:
                # Stop any descending action before SB3's gradient step runs.
                # The keepalive publishes the last _msg for the duration of learn(); hover avoids drift.
                self._set_cmd(0, 0, 0, 0)
            if terminated:                        # diagnostic: which terminal + where
                rospy.loginfo("[rl_env] TERMINAL %s after %d steps: d_true=%.2f t_lost=%.1f ex=%.2f ey=%.2f alt=%.2f",
                              reason, self._ep_steps, raw['true_dist'], raw['t_since_raw'],
                              float(obs[0]), float(obs[1]), alt)
            raw['elapsed'] = elapsed; raw['ep_steps'] = self._ep_steps
            raw['reward'] = reward; raw['term_reason'] = reason
            return obs, reward, terminated, truncated, raw

        def _request_offboard(self):
            """Request OFFBOARD mode via MAVROS. Needed when SKIP_IBVS=1: without a
            prior setpoint stream, PX4 drops OFFBOARD within 0.5s of last setpoint,
            so by the time reset() is called the drone is in HOLD. Stream setpoints
            first for 1s, THEN request mode change (PX4 rejects the switch if no
            recent setpoints exist)."""
            try:
                # pre-stream setpoints so PX4 accepts the mode request
                r = rospy.Rate(20)
                for _ in range(20):   # 1s @ 20 Hz
                    self._set_cmd(0, 0, 0, 0)
                    r.sleep()
                set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                resp = set_mode(custom_mode='OFFBOARD')
                if resp.mode_sent:
                    rospy.loginfo("[rl_env] OFFBOARD mode requested OK")
                else:
                    rospy.logwarn("[rl_env] OFFBOARD request returned mode_sent=False")
            except Exception as e:
                rospy.logwarn("[rl_env] OFFBOARD request failed: %s", e)

        def reset(self, seed=None, options=None):
            # RESET-FREE: no teleport. But EVERY episode must start from a VALID, SAFE
            # tracking state. Unlike the BC teacher, a learning policy ends episodes on a
            # terminal (collision / sustained-loss) — so without recovery the next
            # episode would start already-lost or already-too-close and instantly
            # re-terminate, flooding the replay buffer with 1-step −P episodes. _recover()
            # holds a hover, waits for a detection, gently retreats if too close, and
            # slow yaw-sweeps if still lost. It is TRAINING SCAFFOLDING (IBVS SEARCH is
            # offline during RL) that runs ONLY between episodes → it never enters an
            # observation the policy learns from, so the RL controller stays pure.
            super().reset(seed=seed)
            # MIXED-TRAJECTORY: draw this episode's target trajectory and command the switch
            # BEFORE _recover() (below) re-acquires — so recovery locks onto the NEW motion.
            # target_mover re-anchors the new trajectory to the target's current position
            # (smooth, no teleport). 0.3s lets the switch + re-anchor settle first.
            if self._mix_trajs and self._traj_pub is not None:
                _tj = int(np.random.choice(self._mix_trajs))
                self._traj_pub.publish(Int32(data=_tj))
                rospy.loginfo("[rl_env] mixed-traj: episode target trajectory -> T%d", _tj)
                rospy.sleep(0.3)
            self._ka_active = True            # training starts now -> take over the stream
            if self._first_reset:
                self._first_reset = False
                # Re-request OFFBOARD when SKIP_IBVS=1: OFFBOARD drops within 0.5s of last
                # setpoint; with no IBVS, the mode was lost by the time reset() is called.
                # The service call streams setpoints first (PX4 requires a live stream to
                # accept the OFFBOARD request), then switches the mode.
                self._request_offboard()
                # hold a pure hover while IBVS is killed — the policy must NOT step yet
                rospy.loginfo("[rl_env] HANDOFF: hovering %.1fs — kill IBVS now.", self._handoff_wait)
                t0 = rospy.Time.now()
                while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < self._handoff_wait:
                    self._set_cmd(0, 0, 0, 0)
                    rospy.sleep(0.05)
            self._recover()
            # Always reset the detection timer at episode start. Two failure modes:
            # (1) t_last_det=None (never detected): t_since_raw=999 → instant "lost" terminal.
            # (2) t_last_det=old (from prior episode that ended lost): episode 2+ starts
            #     with t_since_raw already > loss_secs → 1-step episode loop.
            # Resetting to now gives EVERY episode loss_secs wall-seconds to acquire.
            _det_age = (float('inf') if self.ob.t_last_det is None
                        else (rospy.Time.now() - self.ob.t_last_det).to_sec())
            if _det_age > 1.0:
                self.ob.t_last_det = rospy.Time.now()
                rospy.loginfo("[rl_env] reset t_last_det at episode start (was %.1fs old)", _det_age)
            self.ob.a_prev = np.zeros(4, dtype=np.float32)   # fresh smoothness baseline
            self._prev_d_true = float('nan')                  # reset approach reward baseline
            self._prev_d_true2 = float('nan')                 # reset r_acc ḋ-trend baseline
            self._ep_t0 = rospy.Time.now()
            self._ep_steps = 0
            obs, raw = self.ob.build()
            return obs, raw

        def _rearm_and_takeoff(self, target_alt=14.0):
            """Re-arm and climb back to target_alt using POSITION setpoints.

            Velocity setpoints work from ground when armed+OFFBOARD (verified live).
            The keepalive is already streaming at 20 Hz (_ka_active=True), so OFFBOARD
            is already satisfied. Sequence: OFFBOARD → arm → climb velocity → wait.
            """
            rospy.logwarn("[rl_env] CRASH RECOVERY: chaser at alt=%.2fm — re-arming", self.ob.chaser_alt)
            r = rospy.Rate(20)

            # 1. Request OFFBOARD (keepalive is already streaming hover setpoints).
            try:
                set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
                resp = set_mode(custom_mode='OFFBOARD')
                rospy.loginfo("[rl_env] OFFBOARD re-request: mode_sent=%s", resp.mode_sent)
            except Exception as e:
                rospy.logwarn("[rl_env] OFFBOARD re-request failed: %s", e)

            # 2. Arm.
            try:
                arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
                resp = arm_srv(True)
                rospy.loginfo("[rl_env] arm: success=%s", resp.success)
            except Exception as e:
                rospy.logwarn("[rl_env] arm failed: %s", e)

            # 3. Climb: update keepalive msg to vz=+1.5 (ENU via MAVROS: positive = upward).
            with self._msg_lock:
                self._msg.velocity.x = 0.0
                self._msg.velocity.y = 0.0
                self._msg.velocity.z = 1.5
                self._msg.yaw_rate = 0.0
                self._msg.header.stamp = rospy.Time(0)

            # 4. Wait for altitude (max 40 s).
            t0 = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < 40.0:
                if self.ob.chaser_alt >= target_alt - 2.0:
                    rospy.loginfo("[rl_env] crash recovery done: alt=%.1fm", self.ob.chaser_alt)
                    break
                r.sleep()
            else:
                rospy.logwarn("[rl_env] crash recovery timed out; alt=%.1fm", self.ob.chaser_alt)

            # 5. Hover.
            with self._msg_lock:
                self._msg.velocity.z = 0.0
                self._msg.header.stamp = rospy.Time(0)

        def _recover(self, timeout=15.0):
            """Establish a YOLO-detectable start before each episode (see reset()).
            GT-DRIVEN scaffolding: drives chaser to within 12 m of target (wide enough
            to be achievable on T4 orbit at 2 m/s — the old [5.5,7.5]m band settle was
            geometrically impossible on an orbiting target and caused every recovery to
            time out, leaving the drone at d=15m pointing the wrong way).
            Phase 2 yaw-aligns toward GT target and waits for YOLO confirmation (8s).
            Falls back to hover/yaw-scan if GT is missing."""
            # If drone crashed (altitude < 2m), re-arm and take off before proceeding.
            if self.ob.chaser_alt < 2.0:
                self._rearm_and_takeoff()
            # Warn if we entered _recover() at a dangerously low altitude (crash investigation).
            if self.ob.chaser_alt < 10.0:
                rospy.logwarn("[rl_env] _recover entered at LOW alt=%.2fm — policy descended into floor", self.ob.chaser_alt)
            lo, hi = self._rp['band_lo'], self._rp['band_hi']
            d_star = 0.5 * (lo + hi)
            SETTLE_DIST = 12.0   # accept "close enough" rather than in-band; YOLO detects ≤12m
            SETTLE_MIN  = 5.5    # but NEVER settle inside near-collision range — retreat if closer.
                                 # Without this, after a collision terminal (d≈2m) _recover saw
                                 # 2<12 → "settled" → exited → step 1 re-collided → 1-step loop that
                                 # deadlocked online training (685 collisions, ep_len≈5). Now a too-
                                 # close start falls through to the reposition block (mag<0 → retreat).
            r = rospy.Rate(20); t0 = rospy.Time.now(); settle = 0
            while not rospy.is_shutdown():
                el = (rospy.Time.now() - t0).to_sec()
                if el > timeout:
                    rospy.logwarn("[rl_env] _recover timed out (%.0fs); proceeding to yaw-align", timeout)
                    break
                td = self.ob.true_dist
                rel = self.ob.rel_w
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, dz = rel
                    rng = math.hypot(dx_w, dy_w)
                    yaw = self.ob.chaser_yaw
                    dbeta = math.atan2(dy_w, dx_w) - yaw
                    dbeta = math.atan2(math.sin(dbeta), math.cos(dbeta))
                    # wide settle: accept SETTLE_MIN ≤ d < SETTLE_DIST (T4 orbit can't hold
                    # [5.5,7.5]m). Too-close (d<SETTLE_MIN) is NOT framed → retreat below.
                    d_ref = td if not math.isnan(td) else rng
                    framed_gt = (SETTLE_MIN <= d_ref < SETTLE_DIST)
                else:
                    dbeta = 0.0; rng = float('nan'); framed_gt = False
                framed = framed_gt or (self.ob.valid and not math.isnan(td)
                                       and lo - 0.4 <= td <= hi + 0.4)
                if framed:
                    self._set_cmd(0, 0, 0, 0); settle += 1
                    if settle >= 3:          # ~0.15 s at distance (reduced: yaw no longer gating)
                        break
                    r.sleep(); continue
                settle = 0
                # --- GT reposition: LOCAL_NED via keepalive (correct fix) ---
                # Must use LOCAL_NED so vz is always world-z regardless of drone tilt.
                # Body-frame (BODY_NED) at 4 m/s forward causes ~15-20° bank → vz=0 in
                # body frame has a downward world component → chaser descends and crashes.
                # Approach: update _msg directly to LOCAL_NED so the keepalive thread
                # continuously republishes the approach command (no cancellation).
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, dz = rel
                    if rng > 1e-3:
                        mag = max(-4.0, min(4.0, 1.5 * (rng - d_star)))
                        vx_ned = mag * dx_w / rng   # velocity.x → world-x (dx_w); ENU probe-verified
                        vy_ned = mag * dy_w / rng   # velocity.y → world-y (dy_w)
                    else:
                        vx_ned = vy_ned = 0.0
                    vz_ned = max(-2.5, min(2.5, 0.8 * dz))   # ENU: +vz=UP; dz=ENU up → climb toward target
                    # Altitude floor in _recover(): identical to the one in step().
                    # Critical: without this, a chaser that DROPPED below target altitude keeps
                    # sinking (vz toward target < 0) for the full 15-s timeout, crashing.
                    _alt = self.ob.chaser_alt
                    if _alt < self._alt_floor:   # match step() ALT_FLOOR (configurable; baylands=14)
                        _climb = min(2.5, 1.5 * max(1.0, self._alt_floor - _alt))   # +vz = UP
                        if vz_ned < _climb:
                            vz_ned = _climb
                            rospy.logwarn_throttle(2.0,
                                "[rl_env] _recover alt-floor: z=%.2fm forcing vz_ned=%.2f",
                                _alt, vz_ned)
                    wz = max(-1.0, min(1.0, 1.5 * dbeta))   # ENU yaw_rate = CCW+; turn toward target
                    with self._msg_lock:
                        self._msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                        self._msg.velocity.x = float(vx_ned)
                        self._msg.velocity.y = float(vy_ned)
                        self._msg.velocity.z = float(vz_ned)
                        self._msg.yaw_rate = float(wz)
                        self._msg.header.stamp = rospy.Time(0)
                        self.pub.publish(self._msg)
                else:
                    self._set_cmd(0, 0, 0, 0.4 if el > 1.0 else 0.0)
                r.sleep()

            # --- Phase 2: yaw-align + YOLO wait (max 5s) ---
            yaw_t0 = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - yaw_t0).to_sec() < 5.0:
                if self.ob.valid:
                    rospy.loginfo("[rl_env] _recover: YOLO confirmed after yaw-align")
                    break
                rel = self.ob.rel_w
                if not any(math.isnan(v) for v in rel):
                    dx_w, dy_w, _ = rel
                    yaw = self.ob.chaser_yaw
                    dbeta = math.atan2(dy_w, dx_w) - yaw
                    dbeta = math.atan2(math.sin(dbeta), math.cos(dbeta))
                    wz = max(-1.0, min(1.0, 1.5 * dbeta))   # ENU yaw_rate = CCW+; turn toward target
                    self._set_cmd(0, 0, 0, wz)   # yaw only
                else:
                    self._set_cmd(0, 0, 0, 0.4)  # no GT → slow spin
                r.sleep()
            if not self.ob.valid:
                rospy.logwarn("[rl_env] _recover: YOLO not detecting after yaw-align; starting anyway")
            self._set_cmd(0, 0, 0, 0)   # LOCAL_NED zeros = world-frame hover


# --------------------------- policy runner (Step 2) ---------------------------
def _run_policy(ob, bc_path, run_secs):
    """CLOSED-LOOP: the BC policy flies the drone (replaces IBVS). Publishes body
    velocities at 20 Hz, a_prev = its OWN last action, frame-stack N=4 built here.
    This is the copycat test — a pure a_prev-echo would drift immediately.
    NB: no external safety filter (per design). A test-only hover-on-lost guard
    prevents a runaway if the target is lost > 3 s; it is NOT part of the controller."""
    import torch
    from collections import deque
    from rl_train_bc import BCPolicy
    ck = torch.load(bc_path, map_location='cpu', weights_only=False)
    net = BCPolicy(obs_dim=64); net.load_state_dict(ck['state_dict']); net.eval()
    caps = np.array(ck['caps'], dtype=np.float32)
    pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
    msg = PositionTarget(); msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
    msg.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                     PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                     PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                     PositionTarget.IGNORE_YAW)
    rospy.loginfo("[rl_env] POLICY mode — BC (%s) flying for %.0fs. No external filter.",
                  os.path.basename(bc_path), run_secs)
    r0 = rospy.Rate(20)
    while not rospy.is_shutdown() and not ob.valid:   # wait for first detection
        r0.sleep()
    stack = deque([ob.build()[0]] * 4, maxlen=4)
    a = np.zeros(4, dtype=np.float32)
    rate = rospy.Rate(20); t0 = rospy.Time.now(); n = 0; lost_since = None
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < run_secs:
        ob.a_prev = a                          # closed loop: a_prev = own last action
        obs, raw = ob.build()
        stack.append(obs)
        x = np.concatenate(stack).astype(np.float32)[None]      # (1,64) oldest..newest
        with torch.no_grad():
            a = net(torch.from_numpy(x))[0].numpy()
        # test-only safety (NOT the controller): hover if target lost > 3 s
        if ob.valid:
            lost_since = None
        elif lost_since is None:
            lost_since = rospy.Time.now()
        if lost_since is not None and (rospy.Time.now() - lost_since).to_sec() > 3.0:
            v = np.zeros(4, dtype=np.float32); tag = "LOST->HOVER"
        else:
            v = np.clip(a, -1.0, 1.0) * caps; tag = ""
        msg.header.stamp = rospy.Time(0)  # Time(0) → MAVROS uses FCU time
        msg.velocity.x, msg.velocity.y, msg.velocity.z = float(v[0]), float(v[1]), float(v[2])
        msg.yaw_rate = float(v[3]); pub.publish(msg)
        if n % 20 == 0:                        # ~1 Hz status
            print(f"t+{(rospy.Time.now()-t0).to_sec():4.0f}s  ex{obs[0]:+.3f} ey{obs[1]:+.3f}"
                  f"  d_hat{raw['d_hat']:5.2f} true{raw['true_dist']:5.2f}"
                  f"  v[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f},{v[3]:+.3f}] {tag}", flush=True)
        n += 1; rate.sleep()
    for _ in range(10):                        # leave a clean hover (avoid failsafe jerk)
        msg.header.stamp = rospy.Time(0)       # Time(0) → MAVROS uses FCU time
        msg.velocity.x = msg.velocity.y = msg.velocity.z = 0.0; msg.yaw_rate = 0.0
        pub.publish(msg); rate.sleep()
    rospy.signal_shutdown("policy run done")


# --------------------------- probe / record mains -----------------------------
def _run_probe(ob):
    tap = IbvsActionTap(ob)
    secs = float(rospy.get_param("~probe_secs", 12.0))   # self-terminate (clean flush)
    rospy.loginfo("[rl_env] PROBE mode — read-only; obs at 2 Hz for %.0f s.", secs)
    rate = rospy.Rate(2)
    hdr = " ".join(f"{n:>8s}" for n in OBS_NAMES)
    n = 0; t0 = rospy.Time.now()
    while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < secs:
        ob.a_prev = ob.latest_cmd            # display the live IBVS command in a_* cols
        obs, raw = ob.build()
        if n % 10 == 0:
            print("\n" + hdr + "   | d_hat  valid true_dist", flush=True)
        print(" ".join(f"{v:8.3f}" for v in obs) +
              f"   | {raw['d_hat']:5.2f}  {raw['valid']}     {raw['true_dist']:5.2f}", flush=True)
        n += 1
        rate.sleep()
    rospy.signal_shutdown("probe done")


def _run_record(ob):
    tap = IbvsActionTap(ob)
    out = os.path.expanduser(rospy.get_param("~out", "~/rl_demos/demo.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, 'w', newline='')
    wcsv = csv.writer(f)
    wcsv.writerow(['sim_time'] + OBS_NAMES +
                  ['act_vx','act_vy','act_vz','act_wz','valid','d_hat','true_dist'])
    rospy.loginfo("[rl_env] RECORD mode -> %s (20 Hz, read-only). Ctrl-C to stop.", out)
    rate = rospy.Rate(20)
    rows = 0
    prev = np.zeros(4, dtype=np.float32)     # a_(t-1): the PREVIOUS IBVS command
    while not rospy.is_shutdown():
        ob.a_prev = prev                     # obs carries a_(t-1); NOT the label -> no leak
        obs, raw = ob.build()
        label = ob.latest_cmd.copy()         # a_t: the CURRENT command = the decision to imitate
        wcsv.writerow([f"{rospy.Time.now().to_sec():.3f}"] +
                      [f"{v:.5f}" for v in obs] + [f"{v:.5f}" for v in label] +
                      [raw['valid'], f"{raw['d_hat']:.3f}", f"{raw['true_dist']:.3f}"])
        prev = label                         # advance a_(t-1)
        rows += 1
        if rows % 600 == 0:
            f.flush(); rospy.loginfo("[rl_env] %d rows", rows)
        rate.sleep()
    f.close()
    rospy.loginfo("[rl_env] saved %d rows -> %s", rows, out)


if __name__ == "__main__":
    rospy.init_node("rl_env", anonymous=True)
    mode = rospy.get_param("~mode", "probe")
    ob = ObsBuilder()
    if mode == "record":
        _run_record(ob)
    elif mode == "policy":
        _run_policy(ob,
                    os.path.expanduser(rospy.get_param("~bc", "~/fyp/rl/models/bc_policy_v2.pth")),
                    float(rospy.get_param("~run_secs", 30.0)))
    else:
        _run_probe(ob)
