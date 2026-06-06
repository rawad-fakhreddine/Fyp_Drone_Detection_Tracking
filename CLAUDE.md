# CLAUDE.md — FYP Context File
## AI-Based Drone-to-Drone Detection and Tracking
**Student:** Rawad Fakhredine | **Supervisor:** Dr. Ibrahim Sammour | **Program:** Masters in Robotics

---

## 1. Project Summary

A fully autonomous **chaser drone** that detects, tracks, and follows a **target drone** using computer vision and AI, evaluated entirely in simulation with statistical validation results as the measure of success.

**One-sentence architecture:**
> FPV Camera → YOLOv8 (detection) → Kalman Filter (smoothing) → PPO agent (outer loop: outputs α*, λ) → IBVS controller (inner loop: pixel-error → velocity commands) → PX4/MAVROS → drone moves

**Architectural family:** Hierarchical "DRL-tunes-IBVS" paradigm. PPO is the strategic outer loop; IBVS is the tactical inner loop. Established by Sampedro et al. (IROS 2018), refined by Jin et al. (IEEE TIE 2022), Hu et al. (2022), and Wu et al. (Drones MDPI 2023).

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
| Detection | Ultralytics YOLOv8 |
| Kalman | filterpy |
| Hardware | i5-13420H, 16GB RAM, RTX 4060 8GB |

**Key Paths:**
- ROS package scripts: `~/catkin_ws/src/drone_tracking/scripts/`
- PX4 autopilot: `~/PX4-Autopilot/`
- Models (YOLO + PPO weights): `~/drone_detection/models/`
- Active YOLO model: `~/drone_detection/models/best.pt` (currently best_v4n.pt)
- PPO weights: `~/drone_detection/models/ppo_policy_weights_v5.pth`
- YOLO v4 raw capture frames: `~/drone_detection/capture_v4/raw/`
- Flight logs: `~/flight_log_*.csv`
- Ablation results: `~/results/` (per-run CSVs + summary.csv)

**GitHub:**
- Repo: `rawad-fakhreddine/Fyp_Drone_Detection_Tracking` (branch: master)
- Raw file access: `https://raw.githubusercontent.com/rawad-fakhreddine/Fyp_Drone_Detection_Tracking/master/drone_tracking/scripts/[filename]`
- **CRITICAL:** Always push from `~/Fyp_Drone_Detection_Tracking/` only. Never `git init` inside `catkin_ws/`.

---

## 3. Milestone History (M1–M8)

### M1 — Literature Review
- Surveyed perception, detection, dataset generation for drone-to-drone tracking.
- Concluded: Gazebo + PX4 SITL preferred over AirSim for ROS control-algorithm work; YOLOv8 with synthetic data is current standard.
- Key papers locked: Sampedro 2018 (DRL+IBVS founding), Tuncer & Alpdemir 2023 (closest PPO drone-to-drone precedent), Pereira 2021 (IBVS vs PBVS monocular justification).

### M2 — YOLO Training v1
- Dataset: 5 merged Roboflow datasets; `classes.txt` contamination fixed with `sed`.
- YOLOv8s fine-tuned on Colab: mAP50=0.979, Precision=0.994, Recall=0.940.
- Lesson: `labelImg classes.txt` contamination bug — verify/lock before labeling.

### M3 — Kalman Filter
- `kalman_filter_node.py`: subscribes to `/drone_tracking/target_center`, publishes `/drone_tracking/filtered_target`.
- Bridges YOLO dropouts up to 10 missed frames with prediction.
- Close-range collapse rejection: pixel jump > 100px AND alpha drops from > 3000 to < 900 in two frames → reject.

### M4 — Single PX4 SITL Setup
- Gazebo Classic 11 + PX4 v1.13.3 SITL + MAVROS on ROS Noetic.
- World: `baylands.world`. Drone model: `iris_fpv_cam`.
- OFFBOARD mode: must pre-stream setpoints before arming (WSL2 clock jitter causes auto-disarm).

### M5 — IBVS Controller (up to v6.15)
- Control law: `vc = -λ · Le⁺ · e` where `e = [ex, ey]` is pixel error, `Le` is interaction matrix.
- Why IBVS over PBVS: YOLO gives only pixel bounding boxes — no depth. IBVS is calibration-tolerant and keeps target in FOV by construction (Pereira 2021).
- Known failure modes addressed: fixed gain λ (solved by PPO), FOV loss at high gain (PPO modulates λ), close-range oscillation (α* clamping + Kalman rejection).

### M6 — PPO Agent (v4.5)
- Architecture: small MLP policy (SB3 PPO), action space: [α*, λ].
- Observation space: [ex, ey, α] normalized (divide α by 0.06).
- Reward: pure positive similarity reward (resolved collapse pattern).
- PPO collapse history: versions v1–v4 collapsed to constant output due to unbounded std, harsh penalties, missing lambda reward, observation scale imbalance. v4.5 fixed with normalized observations + pure positive similarity reward.
- Training: local WSL2 CPU (faster than GPU for small MLP — avoids transfer overhead).
- Export: PyTorch `.pth` state_dict only (never full SB3 `.zip` — breaks across Python versions).

### M7 — Dual PX4 SITL Architecture
- **M7.1:** Target drone spawned in Gazebo (visual only, no autopilot).
- **M7.2:** IBVS v6.15 finalized with reliable 50s+ HOLD phases.
- **M7.3:** Full dual PX4 SITL — chaser (instance 0, MAV_SYS_ID=1, `/mavros/*`) + target (instance 1, MAV_SYS_ID=2, `/target/mavros/*`). target_mover sends real MAVROS velocity commands.
- Dual-SITL gotchas: MAV_SYS_ID mismatch silently breaks target MAVROS; target spawn height < 0.5m causes physics glitch; world-frame velocity required (FRAME_BODY_NED for MAVROS).

### M8 — PPO v5 Retraining
- Retrained PPO to v5.2 with dual-SITL environment.
- **Result: near-constant output (α*≈0.011, λ≈0.50) — known limitation.**
- Root cause: domain gap between training distribution and live Gazebo. PPO sees valid states but outputs nearly uniform setpoints.
- Academic status: **valid ablation finding** — documents the gap between offline training and live environment. PPO theoretically adds value (learned gain schedule) but needs v6 retraining to demonstrate it empirically.

---

## 4. Milestone 9 — Full Detail (ACTIVE PHASE)

Milestone 9 is the **evaluation and optimization phase**. It covers: component tuning to achieve reliable HOLD, building the ablation study infrastructure, YOLO v4 dataset + training, and statistical validation.

### M9.1 — Evaluation Framework Design & target_mover v9.0
**What was done:**
- Locked the 9-trajectory benchmark plan (T1–T9) and 4-configuration ablation study.
- `target_mover.py` upgraded from v8.0 → v9.0: added `~trajectory` ROS parameter (1–9) dispatching to individual trajectory methods. T1–T8 are deterministic parametric; T9 preserves FuzzyEscaper class.
- Removed stray `.git` directory inside `scripts/` with `rm -rf`. Rule established: always push from `~/Fyp_Drone_Detection_Tracking/` only.
- CLAUDE.md fully rewritten to reflect milestone history, GitHub path, 9-trajectory plan, and measured YOLO fps.

**9 Benchmark Trajectories:**

| ID | Name | Parameters | Purpose |
|---|---|---|---|
| T1 | Static Hover | Stationary at 12m | Baseline HOLD stability |
| T2 | Slow Straight | 1.0 m/s | Low-speed following |
| T3 | Fast Straight | 3.5 m/s | High-speed following |
| T4 | Circular Orbit | R=8m, 1.5 m/s | Lateral tracking |
| T5 | Figure-8/Lemniscate | a=8m | Direction-change tracking |
| T6 | Inclined Medium | 15°, 2.0 m/s, random azimuth | 3D maneuvering |
| T7 | Inclined Hard | 35°, 3.0 m/s, random azimuth | 3D maneuvering (hard) |
| T8 | Up-Down Helix | R=8m, continuous ascent+descent | Altitude + lateral combined |
| T9 | Active Evasion (fuzzy) | Mamdani evasion model | Stress test only — NOT in A/B comparison |

**`target_mover.py` already supports `~trajectory` rosparam for T1–T9.**

---

### M9.2 — Fuzzy Evasion Model Development (target_mover v9.x)
**What was done:**
- Identified failure modes in v8.0 fuzzy model: circular orbiting behavior, altitude/heading conflicts, conservative speed outputs, boundary repulsion overriding escape logic.
- Began iterative redesign of the Mamdani fuzzy evasion system.

---

### M9.3 — Component Tuning Round 1 (IBVS v6.19–v6.20, target_mover v10.0–v10.2)
**What was done:**
- `random_spawn_target.py` v2.0: 8 GPS-surveyed zones covering full Baylands island. Confirmed good zones: C(-80,50) and H(30,-80). jitter=8m, z=0.5m.
- Ablation configuration plan locked (see Section 5).
- `flight_logger.py` and `analyze_flight_log.py` extended with `world_alt_err` column (M9.3).

**IBVS evolution this session:**

| Version | K_far | DEAD_ZONE | vel_smooth | Kp_z | pitch_comp | Key fix |
|---|---|---|---|---|---|---|
| v6.19 (start) | 30 | 0.001 | 0.35 | 1.8 | 0.8 | feedforward |
| v6.20 | 35 | 0.002 | 0.15 | 1.8 | 0.8 | directional SEARCH with last_cx/last_cy memory |

**target_mover evolution:**

| Version | Core change |
|---|---|
| v9.8 (start) | FOV-calibrated min speed — binary sprint/drift |
| v10.0 | Maneuver state machine + vy strafe |
| v10.1 | Speed floor 1.0, SPRINT increased |
| v10.2 | Always-escape rules, 7-axis Mamdani, dist-adaptive |

---

### M9.4 — Component Tuning Round 2 (MAJOR SESSION)
**What was done:**

#### Kalman M9.6 → M9.7 → M9.8
- M9.6: Q_pos fix (Q_pos 3.0 → 0.5) to reduce cx/cy jitter.
- M9.7: Q_vel 3.0 → 5.0 (more responsive to fast target motion).
- M9.8: Q_vel 5.0 → 6.0, PIXEL_JUMP_OUTLIER 120px → 180px, MAX_CONSECUTIVE_REJECTIONS 8 → 4.

**Final Kalman M9.8 parameters:**
```
R = [6, 6, 5]          # observation noise (cx, cy, alpha)
Q = [0.5, 0.5, 3, 6, 6, 3]  # process noise (pos, pos, alpha, vel_x, vel_y, vel_alpha)
PIXEL_JUMP_OUTLIER = 180px
MAX_CONSECUTIVE_REJECTIONS = 4
```

#### IBVS v6.21 → v6.23 → v6.24 → v6.25
- v6.21: Kd_a=150, Kp_z=2.5, pitch_comp=0.4, directional SEARCH with last_cx/cy.
- v6.22: max_vy/max_vz tuning.
- v6.23: max_vy=1.20, max_vz=1.5, max_wz=0.5, Kp_y=1.8, Kp_z=3.0.
- v6.24: **BREAKTHROUGH — ea HOLD threshold 0.005 → 0.010** (APPROACH→HOLD transition). This was the single most impactful fix — HOLD went from 0% to 98%.
- v6.25: SEARCH Stage 3 — fast 360° yaw rotation triggered after 5s (was 8s).

**⭐ Key insight: ea threshold is the most impactful IBVS parameter.** Relaxing from 0.005 to 0.010 restored HOLD across all simulations.

#### target_mover v10.3 → v10.4 → v10.5
- v10.3: EDGE→FAST rule, phi=0.40, vz increased. 7-axis Mamdani fuzzy evasion fully operational.
- v10.4: Longer maneuver intervals (all 4 phases extended by ~1.5–1.0s).
- v10.5: Speed universe [1,4] → [1,3.5], MFs rescaled, vx/vy clamps 3.5.

#### random_spawn_target v3.0
- 9 GPS-surveyed zones in Baylands, jitter=8m, z=0.5m.
- Zone parameter `str()` casting bug fixed.
- Zone whitelist for evaluation locked: {5, 6, 7, 9}.

#### YOLO v4 Dataset Pipeline (completed in M9.4)
- `auto_label_v4.py`: labeled 15,214 of 18,091 images at conf=0.35.
- `relabel_empties.py`: recovered 182 more at conf=0.20; 1,349 truly empty.
- `dedup_phash_v4.py`: pHash threshold=20 → removes 44.2% → ~10k images kept.
- Multi-box fix script: kept most-centered box only (95 affected files → 0 multi-box).
- `assemble_v4_v3mix.py`: v4 images (85/15 train/val split) + all 1,798 v3 train images; v3 files prefixed `v3_` to avoid collisions.
- `cleanup.sh`: kills full dual-PX4-SITL + Gazebo + MAVROS stack; confirms port release.
- Three obsolete scripts identified for deletion: `assemble_v4_training_set.py`, `dedup_v4.py`, `sort_v4_by_scenario.py`.

**Best M9.4 simulation metrics:**
| Metric | Value | Target | Status |
|---|---|---|---|
| HOLD% | 98% | ≥80% | ✅ Best run |
| Detection in HOLD | 100% | 100% | ✅ |
| Wrong-direction maneuvers | 0% | 0% | ✅ |
| Altitude error | +0.23m | <0.5m | ✅ |
| Mean separation in HOLD | 4.73m | 3–6m | ✅ |
| Target escaping | 54.8% | >50% | ✅ |
| Stale predictions | 0 | 0 | ✅ |
| OUTSIDE frames | 12% | <20% | ✅ |

**Note: HOLD% range across all M9.4 sims was 26–98% — inconsistency is the primary bottleneck.**

---

### M9.5 — Batch Runner Infrastructure + YOLO v4 Training
**What was done:**

#### Code Patches Applied (all verified on GitHub)
- `yolo_detection_node.py` (v3.1 + M9.5 patch): publishes `/drone_tracking/target_box` (geometry_msgs/Quaternion: cx, cy, w, h — real bbox dimensions, not square approximation).
- `yolo_debug_viewer.py` fix: was reconstructing boxes as perfect squares using `sqrt(area)`. Fixed to use real w, h from `/drone_tracking/target_box`.
- `ibvs_controller_node.py` (v6.25 + M9.5 patch): added `~detection_source` param. When `"raw"` → subscribes to `/drone_tracking/target_center` (raw YOLO) instead of `/drone_tracking/filtered_target` (Kalman). Confirmed `USE_PPO` is already a rosparam (not hardcoded).
- `target_mover.py` (v10.5 + M9.5 patch): added `~seed` param for `random.seed()` reproducibility.
- `random_spawn_target.py` (v3.0): zone whitelist {5,6,7,9} enforced; `~seed` param added.
- `extract_metrics.py`: new file — reads flight log CSV, appends one metrics row to `~/results/summary.csv`.

#### Batch Runner Status
| File | Status |
|---|---|
| `cleanup.sh` | ✅ Done + tested live |
| `launch_stack.sh` | ❌ Not yet created (formatting error during M9.5) |
| `run_config.sh` | ❌ Not yet created |
| `extract_metrics.py` | ✅ Done |
| `target_box` topic | ✅ Done |
| Zone whitelist + seed | ✅ Done |
| `USE_PPO` rosparam | ✅ Confirmed in IBVS |
| `detection_source` param | ✅ Done |

**Results save structure:**
```
~/results/
  summary.csv                              ← one row per run (all configs)
  Config1/zone{Z}_traj{T}_{TS}.csv        ← per-run flight logs
  Config2/zone{Z}_traj{T}_{TS}.csv
  Config3/zone{Z}_traj{T}_{TS}.csv
```

**Planned batch runner architecture (Option B, not yet coded):**
```
launch/
  tracking_base.launch   ← Gazebo + PX4 + MAVROS + both drones + takeoff + yolo + logger
  config1.launch         ← base + IBVS(use_ppo=false, detection_source=raw)
  config2.launch         ← base + kalman + IBVS(use_ppo=false, detection_source=kalman)
  config3.launch         ← base + kalman + ppo + IBVS(use_ppo=true, detection_source=kalman)
```
Each accepts args: `trajectory:=N zone:=N seed:=N duration:=N`.

#### YOLO v4 Training (Colab)
- Training notebook: `YOLO_Training_FYP.ipynb` on Google Drive (FYP_Rawad/).
- Dataset on Drive: `FYP_Rawad/v4_training_set.zip` (10,378 train + 1,513 val).
- Dataset `FILE_ID=1n-2ELvnEUFyTLQSOtQHz3qDwMdJwDAuk` (for `gdown`).
- All 3 models trained from COCO pretrained weights (fair comparison). epochs=80, patience=10, seed=42, cache=True.

| Model | Status | mAP50 | mAP50-95 | Precision | Recall | FPS (T4) | Train time |
|---|---|---|---|---|---|---|---|
| YOLOv8n | ✅ Done | 0.991 | 0.848 | 0.971 | 0.979 | 75.0 | 3.4h |
| YOLOv8s | 🔄 Interrupted at epoch 34 | — | — | — | — | — | — |
| YOLOv8m | ❌ Not started | — | — | — | — | — | — |

- `best_v4n.pt` deployed: `~/drone_detection/models/best.pt`.
- PPO weights: `~/drone_detection/models/ppo_policy_weights_v5.pth`.
- For v8s retry: run Colab cells 1, 2, 3, 4, 6, 6.5, 8, 9 (Cell 6.5 inserts v8n stats into `results_summary` so the JSON merges correctly for the comparison table).
- When running v8m later: upload `v4_results.json` (v8n + v8s stats) so Cell 4 loads prior results automatically.

---

## 5. Ablation Study (Locked Plan)

**Evaluation matrix:** T1–T8 (8 trajectories) × zones {5,6,7,9} (4 zones) × configs 1–3 (3 configs) = **96 base conditions × 5 repeats = 480 total runs**.

T9 (Active Evasion) = **stress test only**, excluded from A/B statistical comparison.

**4 Configurations:**
| Config | Pipeline | Key rosparam settings |
|---|---|---|
| 1 | YOLO + IBVS only | `USE_PPO:=false`, `detection_source:=raw` |
| 2 | YOLO + Kalman + IBVS | `USE_PPO:=false`, `detection_source:=kalman` |
| 3 | YOLO + Kalman + PPO + IBVS (full pipeline) | `USE_PPO:=true`, `detection_source:=kalman` |
| 4 (A+ extra) | End-to-end image→velocity (Config 4 architecture) | Not yet built |

**Config 4 architecture (planned, not yet built):**
- Pure end-to-end: replaces YOLO + Kalman + PPO + IBVS with a single network.
- Staged training: (1) encoder pretrain on v4 dataset → (2) behavioral cloning from Config 3 outputs → (3) SAC fine-tuning with curriculum.
- NOT a runtime module swap — it's a complete replacement.

**PPO v5.2 near-constant output is academically valid:** Documents the gap between offline training distribution and live simulation. Config 3 results with PPO v5.2 will show whether even this limited PPO adds value vs Config 2. PPO v6 retraining will improve this but is not required for initial ablation data collection.

**All M9.x simulations to date are tuning runs — NOT comparison data.**

---

## 6. Current Component Versions (End of M9.5)

| Component | Version | File | Key params / notes |
|---|---|---|---|
| YOLO model | v4n (deployed) | `~/drone_detection/models/best.pt` | mAP50=0.991, FPS=75 |
| Detection node | v3.1 + M9.5 patch | `yolo_detection_node.py` | 5-frame alpha median; publishes `/drone_tracking/target_center` (cx,cy,area) + `/drone_tracking/target_box` (cx,cy,w,h) |
| Kalman filter | M9.8 | `kalman_filter_node.py` | R=[6,6,5], Q=[0.5,0.5,3,6,6,3], PIXEL_JUMP=180px, MAX_CONSECUTIVE_REJECTIONS=4 |
| IBVS controller | v6.25 + M9.5 patch | `ibvs_controller_node.py` | K_far=35, dead_zone=0.002, smooth=0.15, Kd_a=150, Kp_z=2.5, pitch_comp=0.4, ea_hold=0.010, max_vx=3.5, SEARCH yaw-spin at 5s, `~detection_source` param, `~USE_PPO` rosparam |
| PPO agent | v5.2 | `ppo_agent_node.py` | Near-constant output (α*≈0.011, λ≈0.50). Weights: `ppo_policy_weights_v5.pth`. Retrain to v6 pending. |
| target_mover | v10.5 + M9.5 patch | `target_mover.py` | 7-axis Mamdani fuzzy evasion, speed universe [1,3.5], EDGE→FAST rule, `~trajectory` rosparam (T1–T9), `~seed` param |
| random_spawn | v3.0 | `random_spawn_target.py` | 9 zones, zone whitelist {5,6,7,9}, jitter=8m, z=0.5m, `~seed` param |
| takeoff_both | v9.9 | `takeoff_both.py` | TAKEOFF_ALT=12m |
| cleanup | done | `cleanup.sh` | Kills all ROS/PX4/Gazebo, confirms ports 11311, 4560, 4561 free |
| launch_stack | ❌ | `launch_stack.sh` | Not yet created |
| run_config | ❌ | `run_config.sh` | Not yet created |
| extract_metrics | done | `extract_metrics.py` | Appends one row to `~/results/summary.csv` |
| flight_logger | M9.3 | `flight_logger.py` | 10Hz CSV with world_alt_err column |
| analyze_flight | M9.3 | `analyze_flight_log.py` | 23+ sections |

---

## 7. Known Remaining Issues (Priority Order)

**P1 — HOLD% inconsistency (26–98% range):**
Root cause: SEARCH separation sometimes reaches 30–44m before re-acquisition; re-acquisition loss can be 40–52s. The directional SEARCH (v6.20 last_cx/cy memory) and yaw-spin (v6.25) help but don't always recover in time. This is the primary reliability bottleneck.

**P2 — Persistent altitude error +0.40m (best run: +0.23m):**
pitch_comp=0.4 helped but chaser still slightly above target. Consider world-frame altitude feedback as a future fix.

**P3 — PPO v5.2 near-constant output:**
Needs v6 retraining in current live environment. PPO v6 should target **far/recovery/maneuver regimes** (where the task actually lives), not just close/edge regimes.
- Add alpha-rate and phi to observation space.
- Reward function must penalize constant output.
- Train only after ablation infrastructure is complete and IBVS/target confirmed stable.

**P4 — launch_stack.sh and run_config.sh not created:**
Batch runner cannot run automated ablation without these. Creation failed during M9.5 due to formatting error. Must be built before 480-run evaluation.

**P5 — YOLOv8s training interrupted at epoch 34:**
Needs Colab retry (cells 1,2,3,4,6,6.5,8,9). Upload `v4_results.json` before starting to preserve v8n stats.

**P6 — YOLOv8m not started:**
Deferred — train in a separate fresh Colab session. Upload `v4_results.json` (v8n+v8s) so all 3 models appear in the comparison table.

---

## 8. Spawn Zones Reference

World: Baylands terrain. Safe coordinate range: X=[-343,343], Y=[-269,269]. Model pose z=-1.3 (Gazebo offset).

**Zone whitelist for evaluation:** {5, 6, 7, 9} (Zone 6 needs yaw-away from tree line at spawn).

**Confirmed reliable zones:** C(-80,50) and H(30,-80).

---

## 9. Architecture Diagram

```
FPV Camera (iris_fpv_cam, forward-facing)
       ↓
YOLOv8 detection node (v3.1)
  → /drone_tracking/target_center  (cx, cy, area)
  → /drone_tracking/target_box     (cx, cy, w, h)
       ↓
Kalman filter node (M9.8)
  → /drone_tracking/filtered_target (smoothed cx, cy, area + velocity estimates)
       ↓
PPO agent node (v5.2)
  inputs: [ex, ey, α] normalized
  outputs: [α*, λ]  (currently near-constant)
       ↓
IBVS controller node (v6.25)
  e = [ex - x*, ey - y*]
  vz proportional to (α - α*)
  vc = -λ · Le⁺ · e
  → /mavros/setpoint_velocity/cmd_vel  (world frame)
       ↓
PX4/MAVROS (attitude stabilization)
```

**State machine (IBVS):** APPROACH → HOLD ↔ SEARCH → APPROACH

---

## 10. Key Learnings & Failure Modes

### IBVS
- **ea threshold is the #1 parameter.** ea < 0.010 for HOLD transition (was 0.005 — caused 0% HOLD).
- SEARCH separation > 30m + 40s re-acquisition gap is the primary HOLD% inconsistency driver.
- `setpoint_velocity/cmd_vel` interprets in world frame. Use `setpoint_raw/local` with `FRAME_BODY_NED` for body-frame commands.

### Kalman
- Q_pos=0.5 (not 3.0) — higher Q_pos causes cx/cy jitter.
- PIXEL_JUMP=180px avoids rejecting legitimate fast target motion.
- MAX_CONSECUTIVE_REJECTIONS=4 (not 8) — faster fallback to predictions.
- Close-range collapse rule: pixel jump > PIXEL_JUMP AND alpha drops from > 3000 to < 900 → reject detection.

### PPO
- Training alpha range MUST match real Gazebo values (0.001–0.04).
- Observation normalization (divide alpha by 0.06) MUST be consistent between training and deployment.
- CPU faster than GPU for small MLP (avoid transfer overhead).
- Never use full SB3 `.zip` export — use PyTorch `.pth` state_dict only.
- Near-constant output from v5.2 = domain gap, not architecture failure.

### YOLO / Detection
- pHash dedup threshold=20 removes ~44% of v4 frames while preserving diversity.
- Close-range instability: alpha collapses when target fills frame → Kalman poisoning → cascade failure. Fixed with collapse rejection rule.
- `classes.txt` must remain `chmod 644` in labelImg workflows.
- Confidence threshold 0.35 for auto-labeling.

### Dual SITL
- MAV_SYS_ID mismatch silently breaks target MAVROS.
- Target spawn height must be ≥ 0.5m (physics glitch below).
- Never publish zero-velocity during WAITING phase (triggers premature OFFBOARD mode).
- Each drone reports local_position=(0,0,0) at own spawn point — use `/gazebo/model_states` for world-frame inter-drone distance.

### Git / Bash
- Always push from `~/Fyp_Drone_Detection_Tracking/`. Never `git init` inside `catkin_ws`.
- Recurring bash mistake: `source X && source Y rosrun` (space before `rosrun`). Must be `source X && source Y && rosrun`.
- Prefer complete file rewrites via `cat > file << 'PYEOF'` heredoc commands.
- Verify each component in isolation before full pipeline launch.

---

## 11. Immediate Next Steps (Priority Order)

1. **Create `launch_stack.sh` and `run_config.sh`** — blocking the entire 480-run ablation batch.
2. **Test v10.5 + Kalman M9.8** — verify HOLD% > 80% stable across 3–5 flights before running ablation.
3. **Run ablation Config 1** (YOLO + IBVS only): `USE_PPO=false`, `detection_source=raw`.
4. **Retry YOLOv8s training** on Colab (cells 1,2,3,4,6,6.5,8,9). Upload `v4_results.json` first.
5. **PPO v6 retraining** — after ablation infra complete and IBVS/target confirmed stable.
6. **Statistical validation** (5 flights × 9 trajectories) — after HOLD% confirmed consistently > 80%.

---

## 12. Coding Preferences (Rawad's Style)

- **Strategy before code:** Always discuss logic and rationale before implementation begins.
- **Diagnostic-first debugging:** Isolated tests before proposing fixes.
- **Complete file rewrites** via `cat > file << 'PYEOF'` heredoc commands (not `cp`).
- **sed/Python patches** for targeted modifications to existing files.
- **GitHub workflow:** Rawad pushes; Claude fetches via raw URL, modifies, returns sed commands or full files.
- **Back up before destructive operations** using timestamped naming.
- **Token-efficient responses** — avoid exhaustive explanations when iterating on code.
- **Private vs. supervisor-facing:** Rawad distinguishes what Dr. Sammour needs to see (slides) from design rationale (private).
- **Watch for bash mistake:** `source X && source Y rosrun` → space before `rosrun` instead of `&&`.

---

## 13. Key References

| Paper | Role in FYP |
|---|---|
| Sampedro et al., IROS 2018 | Founding paper of DRL+IBVS for multirotors. Cite when defending hierarchical architecture. |
| Tuncer & Alpdemir, Software Impacts 2023 | Closest published drone-to-drone PPO precedent. Cite when asked "has anyone done this before?" |
| Wu/Fu et al., Drones MDPI 2023 | DRL tunes IBVS gain to prevent FOV loss. Structurally closest to PPO+IBVS. |
| Jin et al., IEEE TIE 2022 | Policy-gradient method learns visibility-preserving servo policies. Most structurally similar. |
| Pereira, MSc Técnico Lisboa 2021 | IBVS vs PBVS on monocular Parrot AR Drone 2.0 — justifies monocular IBVS choice. |
| Abdessameud & Janabi-Sharifi, Automatica 2015 | Lyapunov-based stability proof for IBVS on VTOL UAV. |
| Chaumette & Hutchinson, IEEE RA Mag 2006/2007 | Foundational IBVS tutorial. Core reference for IBVS math. |
| Lin et al., Actuators 2022 | PPO is most stable continuous-action RL for quadrotor. |
| Caffyn et al., Neurocomputing 2024 | Benchmarks 8 RL algorithms on quadcopter — PPO vs SAC vs TD3. |
| Yi et al., IEEE T-IV 2025 | Safe RL + visual servoing for quadrotor tracking unknown targets. Most recent direct analogue. |
| He et al., IEEE TIE 2024 | Hierarchical RL + VS with smooth subgoals — exact paradigm. |

**Novelty statement:**
> "This project builds on the drone-to-drone PPO tracking approach of Tuncer & Alpdemir (2023) by replacing end-to-end velocity regression with a hierarchical controller in which PPO supplies setpoints to an IBVS inner loop, following the DRL-tunes-IBVS paradigm of Sampedro (2018), Jin (2022), and Wu (2023). To the best of our knowledge, no published work combines YOLOv8 + Kalman + PPO + IBVS + PX4 on the drone-to-drone problem."
