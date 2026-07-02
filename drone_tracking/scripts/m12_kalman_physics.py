#!/usr/bin/env python3
# m12_kalman_physics.py — Phase C part 1 (offline, no flights).
# Physical justification of the Kalman process noise Q_vel against PX4 dynamics.
#
# The KF (kalman_filter_node.py) is a constant-velocity model on the image plane:
#   state x = [cx(px), cy(px), area(px^2), d/dt cx (px/s), d/dt cy (px/s), d/dt area]
#   predict: P = F P F^T + Q, with Q = diag[0.5,0.5,3, 6,6,3] added EVERY step.
# So Q_vel = 6 (px/s)^2 is literally the variance of the per-step (dt=0.05s)
# random-walk INCREMENT of the image-plane pixel velocity. We compare that to the
# largest per-step velocity increment a PX4 target can physically produce.
#
# Pinhole image velocity of a laterally-moving target at depth Z:
#   v_img [px/s]  = f * V_lat / Z
#   a_img [px/s^2]= f * A_lat / Z         (ignoring Z_dot cross terms)
# Per KF step the physical image-velocity increment is  d_v = a_img * dt.
import math
f = 277.0            # live camera focal length (px) — camera_info fx, not SDF
dt = 1.0/20.0        # KF predict step (s)
A_MAX = 3.0          # PX4 MPC_ACC_HOR (m/s^2) max commanded horizontal accel
Q_VEL = 6.0          # kalman_filter_node.py Q[3]=Q[4]
sigma_step = math.sqrt(Q_VEL)   # 1-sigma modeled per-step velocity increment (px/s)

print("="*76)
print(" M12 Phase C.1 — Kalman Q_vel vs PX4 dynamics (image-plane random walk)")
print("="*76)
print(f" f={f:.0f}px  dt={dt:.3f}s  MPC_ACC_HOR={A_MAX:.1f} m/s^2  Q_vel={Q_VEL:.0f} (px/s)^2")
print(f" modeled 1-sigma per-step velocity increment  sqrt(Q_vel) = {sigma_step:.3f} px/s")
print()
print(f" {'Z(m)':>5} {'a_img(px/s^2)':>13} {'d_v@Amax(px/s)':>15} {'d_v/sigma':>10} "
      f"{'Q_vel@1σ=Amax':>13} {'implied A@1σ(m/s2)':>18}")
for Z in (5, 8, 10, 15, 20, 30):
    a_img = f * A_MAX / Z                 # image accel at full PX4 accel
    d_v   = a_img * dt                    # per-step image-velocity increment at A_MAX
    ratio = d_v / sigma_step              # how many modeled sigmas a full-accel step is
    q_needed = d_v**2                     # Q_vel that would put A_MAX at exactly 1 sigma
    a_at_1sig = sigma_step * Z / (f*dt)   # lateral accel that the current 1σ corresponds to
    print(f" {Z:>5} {a_img:>13.1f} {d_v:>15.2f} {ratio:>10.2f} {q_needed:>13.1f} {a_at_1sig:>18.2f}")

# headline numbers at the 8 m nominal standoff
Z0=8.0
a_img0=f*A_MAX/Z0; d_v0=a_img0*dt; a1=sigma_step*Z0/(f*dt); nsig=d_v0/sigma_step
print()
print(" ── Slide statement (Z = 8 m nominal standoff) ─────────────────────────────")
print(f"  • At full PX4 lateral accel (3.0 m/s^2) the target's image velocity changes")
print(f"    by {d_v0:.2f} px/s per {dt*1000:.0f} ms step.")
print(f"  • Q_vel=6 models a 1σ per-step jerk of {sigma_step:.2f} px/s  ⇔  a lateral accel")
print(f"    of {a1:.2f} m/s^2  ({100*a1/A_MAX:.0f}% of MPC_ACC_HOR).")
print(f"  • So a FULL-authority (3.0 m/s^2) maneuver sits at {nsig:.2f}σ of the process")
print(f"    noise → the filter tracks a max-accel target within ~2σ WITHOUT the")
print(f"    innovation saturating, while still averaging out per-frame YOLO jitter.")
print(f"  • Q_vel scales with Z: a_img∝1/Z, so Q_vel=6 is MORE conservative far")
print(f"    (Z=30: {f*A_MAX/30*dt/sigma_step:.2f}σ) and TIGHTER near (Z=5: "
      f"{f*A_MAX/5*dt/sigma_step:.2f}σ) — 8 m is the design point.")
print(" ── Kalman state semantics (for the slide) ────────────────────────────────")
print("  x = [cx px, cy px, area px^2, d/dt cx px/s, d/dt cy px/s, d/dt area px^2/s]")
print("  x[2] = bbox AREA in px^2, clipped [0, 307200] = 640x480 (full frame).")
print("  R = diag[6,6,5] measurement var on [cx,cy,area]; Q_vel(cx,cy)=6, Q_pos=0.5.")
