# CLAUDE.md — FYP Context File
## AI-Based Drone-to-Drone Detection and Tracking
**Student:** Rawad Fakhredine | **Supervisor:** Dr. Ibrahim Sammour | **Program:** Masters in Robotics

---

## 1. Project Overview

Autonomous chaser drone detects, tracks, and follows a target drone using YOLOv8 + IBVS control. Evaluated in Gazebo/PX4 SITL simulation. Three configs: C1 (YOLO+IBVS), C2 (YOLO+KF+IBVS), C3 (YOLO+RL — in progress).

**C1/C2 FINAL BASELINE (sealed, committed `3d863be` + `699bedb`):**
- `alpha_dist_k=0.077` (calibrated: measured α·d²=0.077 vs GT; old 0.096 over-read 13%)
- `d_lpf=0.5` UNIFORM across all trajectories
- Standoff **[6,7] m UNIFORM** on ALL trajectories (no per-traj [9,11] anymore)
- `traj_track_kp=1.0` — target path-lock (fixes open-loop target drift; T1 4m→0.26m, T4 radius clean)
- **Option B** (`INCLINE_NO_HREVERSE=1`) for T6/T7 — forward + vertical zigzag, NO horizontal reverse
- **Vertical package** for T6/T7/T8: `KP_Z=6 / MAX_VZ=2.5 / MAX_ACCEL_VZ=6 / KI_Z=2.0 / INT_Z_MAX=1.3 / INT_Z_BLEED=0.70 / CHASER_ZDN=2.5`
- `SPAWN_YAW=96` for T2/T3

**C1/C2 RESULTS** (128 runs: T1–T8 × 8 seeds {42,43,45–50} × C1/C2, zone 1, 200 s):
- Custody: 100% T1–T6/T8; 98.9% T7 (both configs)
- HOLD: 94–98% except T3 cosmetic bimodality (T3 hold gate fixed to use pitch-comp ey_c)
- **THE C1-vs-C2 finding:** yaw-jerk C2 **1.6–3.3× smoother on ALL 8 trajectories**, every p<0.01 — the KF's contribution is smoothness, not tracking accuracy
- Safety: 3 sub-2m passes in 128 runs, all C1, all guarded corner passes; C2 worst 2.33m

**ADDITIVE RULE FOR RL:** Zero edits to `ibvs_controller_node.py`, `kalman_filter_node.py`, or `target_mover.py`. All RL code is new files only.

---

## 2. Environment & Stack

| Layer | Details |
|---|---|
| OS | WSL2 Ubuntu 20.04 |
| ROS | Noetic |
| Simulator | Gazebo Classic 11 |
| Autopilot | PX4 v1.13.3 SITL |
| Bridge | MAVROS |
| Python | 3.8 |
| RL framework | Stable-Baselines3 (SB3) v2.4.1 |
| Deep Learning | PyTorch |
| Detection | Ultralytics YOLOv8 (YOLOv8s, best_v41s.pt) |
| Kalman | hand-rolled 6-state KF (numpy) |
| Hardware | i5-13420H, 16GB RAM, RTX 4060 8GB |

**Key Paths:**
- Live scripts (run from here): `~/catkin_ws/src/drone_tracking/scripts/`
- Git repo (push from here ONLY): `~/Fyp_Drone_Detection_Tracking/drone_tracking/scripts/`
- PX4: `~/PX4-Autopilot/`
- YOLO model: `~/drone_detection/models/best.pt` (= best_v41s.pt / YOLOv8s)
- RL artifacts (outside repo): `~/fyp/rl/` — models, logs, datasets
- Results (raw): `~/fyp/Results/` → OneDrive `FYP/Results_raw`
- Results (curated reference): `~/fyp/Results_reference/` → OneDrive `FYP/Results`
- **Original/archived code: `~/fyp/original/`** — early .bak files, pre-RL baseline, nolockstep SDF files

**GitHub:** `rawad-fakhreddine/Fyp_Drone_Detection_Tracking` (branch: master)
**CRITICAL:** Always push from `~/Fyp_Drone_Detection_Tracking/` only. Never `git init` inside `catkin_ws/`.

---

## 3. Architecture

**C1:** `FPV Camera → YOLOv8 → IBVS → PX4`
**C2:** `FPV Camera → YOLOv8 → Kalman Filter → IBVS → PX4`
**C3 (RL, in progress):** `FPV Camera → YOLOv8 (frozen) → RL policy → body-velocity → PX4`

IBVS output: `/mavros/setpoint_raw/local` (PositionTarget, FRAME_BODY_NED: vx, vy, vz, yaw_rate).
State machine: TAKEOFF → SEARCH → APPROACH → HOLD ↔ (stale/lost) → SEARCH.

**Configs:**
| Config | Pipeline | Status |
|---|---|---|
| 1 | YOLO + IBVS | **Active — C1/C2 SEALED BASELINE** |
| 2 | YOLO + KF + IBVS | **Active — C1/C2 SEALED BASELINE** |
| 3 | YOLO + RL policy | **In progress** (supervisor directive 2026-08-11) |

**Ablation integrity:** C1 and C2 use IDENTICAL IBVS; the ONLY difference is the Kalman filter. C1 stays `detection_source=raw` with NO filtering.

---

## 4. RL Milestone — Config 3 Design (LOCKED)

**Scope:** RL replaces the CONTROL block only. YOLO perception stays frozen. C1/C2 are the baseline RL must beat on the same 8-traj / 8-seed matrix. SEARCH stays as-is (shared by all 3 configs — RL is scoped to the TRACKING regime).

**OBSERVATION** (16 scalars + frame-stack N=4):
`[ex, ey_c, d̂, ėx, ėy, ḋ, w, h, conf, t_since_det, pitch, roll, a_(t-1)]`
- `ey_c` = pitch-compensated vertical error (same math as IBVS :597)
- Frame-stack provides rate information AND temporal context; raise N if lag observed; LSTM fallback
- Dropout handling: FREEZE last obs + conf=0 flag

**Normalization table** (locked in RL_Milestone_Design.docx §7):
- ex/ey_c: ÷0.8 (frame half-width); d̂: ÷10; ėx/ėy: ÷0.5; ḋ: ÷3; w/h: ÷640/480; conf: as-is [0,1]; t_since_det: ÷3; pitch/roll: ÷30°; a_(t-1): ÷caps

**ACTION:** Continuous `[vx, vy, vz, wz]`, tanh → velocity caps `[8, 1.2, 2.5, 0.5]` m/s or rad/s.
No external safety filter. Safety is LEARNED via reward shaping.

**REWARD:**
- Gaussian centering: `exp(-(ex²+ey_c²)/(2σ²))` + alive bonus (kills suicidal-agent trap)
- Band penalty: penalize `|d̂ - d_star|` outside [6,7] m band
- Smoothness penalty: `‖a_t - a_{t-1}‖`
- `P_lost`: large penalty for sustained loss (largest term — closes suicide door)
- `P_safe`: penalty for d̂ < safety threshold
- GT legal for reward (training only); observation must not use GT

**EPISODES:**
- Option A (locked): fixed 30–40 s OR terminal collision (−50) OR sustained loss >3 s (−100)
- Timeout = truncation (not failure)

**ALGORITHM:** SAC primary (sample efficiency = binding constraint at ~1× RTF) + PPO baseline; TD3 optional. **Privileged critic** used (critic sees GT during training; actor sees only obs → no domain gap at test).

**BC WARM-START (done):**
- 158,727 tracking-regime pairs from 52 calibrated C1 runs (from existing flight_logger v2 logs)
- w/h/conf filled from α + aspect (teacher IBVS never uses them)
- BC v1 trained: held-out RMSE vx 0.066 / vy 0.024 / vz 0.053 m/s / wz 0.003 rad/s ≈ teacher noise floor
- 58 s on RTX 4060; model: `~/fyp/rl/models/bc_policy_v1.pth` (SB3 SAC actor dims for weight surgery)

---

## 5. RL Implementation Files

All in `~/catkin_ws/src/drone_tracking/scripts/` (and repo mirror). ADDITIVE — zero edits to C1/C2 code.

| File | Status | Purpose |
|---|---|---|
| `rl_env.py` | In progress | ObsBuilder from live ROS topics; pitch-comp ey; probe/record modes read-only; gym stub |
| `rl_bc_dataset.py` | Done | Extracts BC pairs from flight logs; fills w/h/conf from α+aspect |
| `rl_train_bc.py` | Done (BC v1 trained) | Behaviour cloning to warm-start SAC actor |
| `rl_train_sac.py` | Skeleton (blocked on env reset) | Online SAC training loop |
| `rl_eval_sac.py` | Done | Frozen policy evaluation harness (logs like a flight CSV) |
| `rl_test_episodes.py` | Done | Episode rollout tester |

**RL Training Setup:**
- World: `rl_empty.world` (flat, no trees, faster physics)
- SDF models: `iris_chaser_nolockstep/iris.sdf` + `target_iris_sitl_nolockstep/iris.sdf` (enable_lockstep=false)
- Training target: T4 orbital (contained moving target — one-way T2/T3 fly off-island → STUCK abort)
- Run command: `WORLD=rl_empty HEADLESS=1 VIEWER=0 NO_LOCKSTEP=1 SPEED_FACTOR=4 bash launch_stack.sh`

**4× Speedup (nolockstep) — Current Status:**
- Lockstep is always 1× RTF on WSL2 (PX4's HIL response ~4ms gates each step — can't speed up)
- Nolockstep target: `real_time_factor=4` in world + `PX4_SIM_SPEED_FACTOR=4` → 4 sim-s/wall-s
- **BLOCKING BUG (TIMESYNC):** Gazebo ROS API plugin forces `/use_sim_time=true` in its `Load()` regardless of launch file args. MAVROS uses sim_time (starts at 0); PX4 nolockstep uses wall_time (~1.787×10⁹s) → setpoints appear 56 years stale → rejected → OFFBOARD drops after ARM.
- **Fix (designed, not yet applied):** Split T1 launch: (1) `posix_sitl.launch` (Gazebo+PX4 only), (2) `wait_topic /clock`, (3) `rosparam set /use_sim_time false`, (4) launch MAVROS separately via `mavros px4.launch fcu_url:=udp://:14540@localhost:14557`
- World file (`rl_empty.world`) already set: `real_time_factor=4, real_time_update_rate=250, max_step_size=0.004` ✓
- PX4_SIM_SPEED_FACTOR only exported for NO_LOCKSTEP=1 (in lockstep it causes EKF fault → crash)

---

## 6. Benchmark Trajectories

| ID | Name | Parameters | Purpose |
|---|---|---|---|
| T1 | Static Hover | Stationary | Baseline HOLD stability |
| T2 | Slow Straight | 1.0 m/s, one-way, SPAWN_YAW=96 | Low-speed following |
| T3 | Fast Straight | 3.5 m/s, one-way, SPAWN_YAW=96 | High-speed following |
| T4 | Circular Orbit | R=8m, T=25s | Lateral tracking; **RL training target** |
| T5 | Lemniscate | a=8m, T=40s | Direction-change tracking |
| T6 | Inclined Medium | 15°, 2.0 m/s, Option B | 3D maneuvering |
| T7 | Inclined Hard | 35°, 3.0 m/s, Option B | 3D hard — validation gate |
| T8 | Up-Down Helix | R=8m, T=25s, continuous bounce | Altitude + lateral |
| T9 | Active Evasion | 7-axis Mamdani fuzzy | Stress test only — never accept criterion |

Seeds: {42, 43, 45–50}. Zone: 1. Duration: 200 s (C1/C2 matrix) / 150 s (deliverable).

---

## 7. Component Versions (Final C1/C2 Baseline)

| Component | File | Key params / notes |
|---|---|---|
| YOLO model | `best_v41s.pt` (= best.pt) | YOLOv8s; mAP50=0.991; ~1200 background negatives; v4n = rollback |
| Detection node | `yolo_detection_node.py` v3.3 | FP gate: w∈[3,300]px, w/h∈[0.8,6.0]; acquisition persistence 3 frames; publishes target_center + target_box + detector_status |
| Kalman filter | `kalman_filter_node.py` M9.8 | R=diag[6,6,5], Q=diag[0.5,0.5,3,6,6,3], PIXEL_JUMP=180px, MAX_REJ=4, damp=0.88 |
| IBVS controller | `ibvs_controller_node.py` | K_far=35, K_near=6, alpha_star=0.0067, ea_HOLD=0.010, max_vx=8, alpha_dist_k=0.077, d_lpf=0.5, KD_VX=1.5, INT_D_MAX=6, INT_BAND=2.5, INT_HOLD_ONLY=1, A_DEC=2.0, D_HOLD_MIN/MAX=6/7, MIN_DIST=2.5, PITCH_COMP=1.3, BAND_KP=0.4; emergency brake at alpha>0.033; 2-stage SEARCH |
| target_mover | `target_mover.py` v10.10 | traj_track_kp=1.0; Option B for T6/T7; continuous vz bounce for T8; one-way T2/T3; RISE_TO_Z=14m, Z_FLOOR=12m, Z_CEIL=24m |
| random_spawn | `random_spawn_target.py` v3.2 | Spawn 8–12 m in front, ALLOWED_ZONES {5,6,7,9}, zone 7 center (-45,-130), SPAWN_YAW=96 for T2/T3 |
| takeoff_both | `takeoff_both.py` v10.0 | TAKEOFF_ALT=14m; ARM_ATTEMPTS=40; disarms both on abort |
| flight_logger | `flight_logger.py` | 20 Hz; raw_det/flt_det=REAL/PRED/NONE; emerg flag; pitch/roll logged (needed for RL obs) |
| launch_stack | `launch_stack.sh` | `launch_stack.sh CONFIG TRAJ ZONE SEED [DURATION]`; VIEWER=0 for headless; HEADLESS=1 for no gzclient; NO_LOCKSTEP=1 for nolockstep mode; SPEED_FACTOR for nolockstep speedup |
| ms6_run_cells | `ms6_run_cells.sh` | Official C1/C2 matrix runner; per-trajectory env (standoff, vertical package, spawn yaw) |

**Target PX4 instance launch (DUAL SITL — per-instance working dir REQUIRED):**
```bash
PX4_BUILD=~/PX4-Autopilot/build/px4_sitl_default
mkdir -p "$PX4_BUILD/instance_1"
(cd "$PX4_BUILD/instance_1" && PX4_SIM_MODEL=iris \
   "$PX4_BUILD/bin/px4" -i 1 -d "$PX4_BUILD/etc" -w sitl_iris_1 -s etc/init.d-posix/rcS)
```
- DO NOT set `PX4_SIMULATOR=gazebo` (NO-OP in v1.13.3)
- DO NOT set `PX4_GZ_MODEL_NAME` (gz-sim/Garden only)
- `-i 1` → TCP port 4561; `interactive:=false` on T1 launch (no pxh stdin-EOF kill)

---

## 8. Coding Rules

**HARD RULES:**
- **ADDITIVE for RL:** Zero edits to `ibvs_controller_node.py`, `kalman_filter_node.py`, `target_mover.py`. RL = new files only.
- **UNIFORM CONTROLLER:** Final/comparison results use IDENTICAL controller across all trajectories. No per-traj parameter changes in reported numbers.
- **Push from `~/Fyp_Drone_Detection_Tracking/` only.** Never `git init` inside `catkin_ws/`.
- **Results stay OUTSIDE the git repo** (in `~/fyp/Results/`).

**STYLE:**
- Strategy before code; discuss rationale first
- Diagnostic-first debugging (isolated tests before fixes)
- Prefer complete file rewrites via `cat > file << 'PYEOF'` heredocs
- Bash mistake to watch: `source X && source Y rosrun` — must be `source X && source Y && rosrun`
- VIEWER=1 + LOSS_TIMEOUT=10 for diagnostic flights; HEADLESS=1 VIEWER=0 for RL training
- Keep DISPLAY set + gzclient alive (or HEADLESS=1) — FPV camera stops without Gazebo rendering
- ~15s between runs for cleanup
- det% is the CANARY — check before blaming the controller; restart WSL between marathon sessions

---

## 9. Key Learnings

**IBVS:**
- `ea_HOLD=0.010` is the #1 parameter (was 0.005 → 0% HOLD)
- `alpha_dist_k=0.077` calibrated from α·d² vs GT (0.096 over-read 13%)
- d_hat = √(k/α) — use ONLY on REAL detections (PRED frames steer, never teach distance)
- HOLD gate uses pitch-compensated ey_c (raw ey biased ~0.24 from cruise pitch → T3 0%-HOLD artifact)
- Integrator ceiling trap (F7): Ki·int_d_max·gain must exceed target cruise speed

**Kalman / Detection:**
- R=[6,6,5] final (R=[25,15,3000] TESTED AND REJECTED — worse on fast depth-change T7/T9)
- Q_vel=6.0 final (sweeps {2,4,6,8,10,12,14} — 6 wins; no live flights warranted)
- Detection failure on T6/T7 = in-frame small-target RAW miss (~91%), not gate (9%); control side exhausted
- Camera: 640×480 native, fx=277 live (not 307 from SDF); imgsz-1280 is upsampling only

**Dual SITL:**
- Per-instance working dir (`-w sitl_iris_1`) REQUIRED — without it lockstep destabilises
- mavparam writes persist across runs via PX4 eeprom → launch_stack explicitly restores CHASER_ZDN
- Lockstep RTF = always 1× on WSL2 regardless of world file `real_time_factor`
- `PX4_SIM_SPEED_FACTOR` in lockstep = FATAL (EKF fault → PX4 SIGABRT)

**RL-specific:**
- Online SAC training needs CONTAINED moving target — use T4 orbital (T2/T3 fly off-island → STUCK abort)
- Nolockstep TIMESYNC fix: split T1 launch, override use_sim_time=false after Gazebo starts, launch MAVROS separately
- `real_time_factor=4` in world + `PX4_SIM_SPEED_FACTOR=4` → 4× RTF in nolockstep
- Plugin constraint: `real_time_update_rate % 250 = 0` AND `1/real_time_update_rate = max_step_size` (250/0.004 ✓)

---

## 10. References

| Paper | Role |
|---|---|
| Sampedro et al., IROS 2018 | Founding DRL+IBVS for multirotors |
| Tuncer & Alpdemir, Software Impacts 2023 | Closest drone-to-drone PPO precedent |
| Fu / Wu et al., Drones MDPI 2023 | Fuzzy gain scheduling of IBVS — citation for C1/C2 adaptive gain |
| Jin et al., IEEE TIE 2022 | Policy-gradient visibility-preserving servo |
| Pereira, MSc Técnico Lisboa 2021 | IBVS vs PBVS on monocular drone — justifies IBVS choice |
| Luo et al., ICCV 2018 | A3C active tracking (discrete action — can't match speed) |
| Zhang et al., 2022 | SAC raw-image → 2D velocity |
| AgilePilot, 2025 | PPO+YOLO pose tracking |
| Caffyn et al., Neurocomputing 2024 | Benchmarks PPO/SAC/TD3 on quadcopter |
| He et al., IEEE TIE 2024 | Hierarchical RL + VS with smooth subgoals |
| Chaumette & Hutchinson, IEEE RA Mag 2006/2007 | Foundational IBVS math |

**Novelty:** YOLOv8 + Kalman + RL policy (Config 3) vs YOLOv8 + Kalman + IBVS (Config 2) on drone-to-drone tracking — RL replaces the control block entirely; no published work combines this exact stack on this problem.
