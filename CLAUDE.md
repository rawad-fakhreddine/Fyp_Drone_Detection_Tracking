# CLAUDE.md — FYP Context File
## AI-Based Drone-to-Drone Detection and Tracking
**Student:** Rawad Fakhredine | **Supervisor:** Dr. Ibrahim Sammour | **Program:** Masters in Robotics

> **Changelog**
> - **2026-06-30 — Vertical-channel Kp_z + max_vz probe (M10.3): FLAT → SEVENTH/EIGHTH levers ruled out, the ey-axis T6/T7 failure is CONFIRMED detection-limited (not vertical-tracking-limited):** the ey axis owns the T6/T7 losses (AUC 0.98–0.99) and the vertical gain `Kp_z` was the one untested gain on it — swept live on T6+T7/Zone-1, Config 2, seeds {42,43,45}, 200 s, LOSS_TIMEOUT=60, latch OFF, banner verified every run. **Pre-flight inspection:** `Kp_z=3.0/Ki_z=0.04/Kd_z=0.5` and `max_vz=1.5` were hardcoded; `vz=-gain·(Kp_z·ey+Ki_z·∫ey+Kd_z·dey)` then clipped to ±max_vz → both rosparam-ified (`~Kp_z`/`~max_vz`, defaults = current → plain run byte-for-byte baseline), banner extended to log them. **Two pre-checks off the existing baselines (read-only):** **(1) Saturation NOT binding** — pooled HOLD `|cmd_vz|` mean 0.41 / p90 0.72 / p99 1.49, **frac at cap 0.93 %** (T6 **0 %**; T7 only a 0.3–7 % p99 tail) → `Kp_z` has headroom, gain sweep VALID, cell D deferred. **(2) ey near-centre at dropout** — across 125 raw `REAL→NONE` onsets the target sits at mean `|ey|`=**0.19** of the 0.8 frame-edge clip, slope **+0.005/s** (flat), only **7 %** edge-ward ramps → the box vanishes while still well-centred vertically = the pure-detection signature (a centering *lag* would ramp ey to the edge first); forecasts outcome (b). **Sweep A(Kp_z 3.0)/B(4.5,×1.5)/C(6.0,×2.0) × T6/T7 × 3 seeds = 18 runs, all clean (stale 0/fail 0/banner-bad 0):** **T6 FLAT** (HOLD 38.9/38.4/39.8, no seed-consistent gain, vzSat ~0 %) → outcome (b). **T7 does NOT clear the win gate** — the headline +8 pts (19.0→27.8) is **not seed-consistent** (two of three seeds flat-to-*down* at the non-saturating cell B; the lift sits **inside the measured ~8.9 pt T7 noise floor**), and cell C **saturates 11.4 %** (cap binding *induced* by the ×2 gain, baseline was 2.2 %). **Oscillation guard never trips** — ey zero-crossing ≤0.125/s everywhere and *falls* as Kp_z rises (0.125→0.096 on T6) → no ringing regime reached even at ×2, closing the "push harder" door (same shape as the Kp_wz/Kp_y result). **Cell D follow-on (literal outcome-(c) branch, max_vz×1.5=2.25, Kp_z back to baseline, T7×3):** raising the cap **fully relieves saturation** (0 % at 2.25) but HOLD still **not seed-consistent** (19.0→25.6, per-seed Δ [s42 +10.4, s43 +13.7, **s45 −4.1**], Δmean +6.7 inside the noise floor), eyZC 0.110≈baseline. **VERDICT: outcome (b) — the lever is NEITHER Kp_z NOR max_vz; the vertical channel is CLOSED.** The ey-axis ownership of T6/T7 is real but the failing quantity is **mid-range detector recall, not vertical tracking authority** (empirically confirms the near-centre ey-drift pre-check). Aborts stayed 2–3/3 at every cell including D. **No gain change to the file** (no cell cleared the win+oscillation gate). The hard regime (T6/T7-class) is now ruled out across R, Q_vel, standoff/α\*, damping, SEARCH-heading/τ, Kp_wz/Kp_y, **Kp_z, and max_vz** — the control side is exhausted; the residual is the detection-resolution lever (RUNG-2 native camera res / v4.2 retrain, per the 2026-06-22 seals). **Commit = rosparam plumbing only, a verified no-op** (`~Kp_z` default 3.0 / `~max_vz` default 1.5; `KP_Z`/`MAX_VZ` launch knobs; banner extended). Tools (catkin, untracked): `vz_precheck.py`, `vz_sweep.sh`, `vz_sweep_analyze.py`, `vz_cellD.sh`; deliverable `~/fyp/Results/diagnostics/vz_sweep/{M10.3_vz_analysis.txt, sweep.log.summary, cellD.log.summary}`.
> - **2026-06-22 — RESOLUTION lever MEASURED, not predicted (M10.3 Part A/B/C): the "free imgsz-1280 win / recovers most" forecast in the bullet below is REFUTED — the camera is 640×480 NATIVE so imgsz-1280 is UPSAMPLING, and the real lever is native CAMERA resolution (ladder RUNG 2):** **Part A (read-only, gates everything):** `fpv_cam.sdf` camera sensor = **640×480** (hfov 1.2217 rad/70° in the SDF, but the live `camera_info` publishes fx=277.19 ≈98° — known discrepancy, resolution is unambiguous); `yolo_detection_node.py` passes the full-native frame to YOLO with **no `imgsz` arg → ultralytics default 640**, and **no pre-resize**. So YOLO already consumes the full native sensor — **imgsz-1280 cannot recover real pixels (there are none beyond 640×480); it interpolates.** This flips the prior "free win" framing. **Part B (one capture run, gains untouched, banner Kp_wz=0.90/Kp_y=1.80 verified):** new untracked tools `clean_capture_record.py` + `clean_capture.sh` flew T7/Zone-1/Config-2/seed-42 and saved **2129 UN-annotated in-FOV frames** (pristine pixels — fixes the burned-GT-marker blocker that made the offline recovery count impossible before) + per-frame metadata. Composition re-confirms the split on fresh data: **460 in-frame raw-miss (det_state=LOST) vs 9 gate-withheld (CONFIRMING) ≈ 98 %/2 % raw** (even more raw-dominant than the pooled 91/9); raw-misses sit at **median 29.4 m / 5.9 px**, live-detections at 8.8 m / 15.9 px. **Part C (offline re-inference, `recover_imgsz_test.py`, same model best_v41s.pt, live gate+conf 0.55 mirrored):** recovery of the 460 raw-miss frames — **imgsz640 (=live control) 36 % · imgsz1280 42 % (+6 pts) · 2×2 tiled 46 % (+10 pts)**. The genuine resolution increment over the live-equivalent control is only **+6 pts (1280) / +10 pts (tiling)** — NOT "most." The still-missed population stays at **median ~4.5 px (>30 m) at every setting**; the sub-5 px / >40 m band is **0 % recall** and unrecoverable by any resolution trick because the 640×480 sensor never captured the detail. Fresh recall-vs-px curve (this capture): 100 % ≥11 px (≤11 m), 85 % @7–9 px (~17 m), 49 % @5–7 px (~23 m), **0 % <5 px (>~40 m)**. **Part D — ladder rung = RUNG 2 (raise NATIVE camera resolution), NOT rung 1 (free imgsz):** the +6/+10-pt interpolation/tiling gain is the upsampling ceiling on the current sensor. Tiling's +10 pts is a **lower bound** on what true native-2× (1280×960 SDF) buys, because tiling has no new optics while a native-2× capture doubles real apparent px → maps the 17–23 m shoulder (5–9 px→10–18 px) from ~50–85 % toward ~95 % recall. **Residual after native-2× = the >40 m / <5 px tail (0 % recall now, →<10 px even at 2×): the slice that needs a v4.2 retrain (higher-res training + small/far-target augmentation), or is accepted as beyond useful tracking range.** **Number for Dr. Sammour: imgsz-alone (zero sim change) buys only +6–10 pts → not worth it as the primary fix; the real fix is native sensor resolution (SDF width/height), which a v4.2 retrain only complements at the far tail.** This CORRECTS finding (6) of the bullet below (resolution is neither "free" nor "recovers most"); findings (1)–(5) stand. Tools (catkin, untracked): `clean_capture_record.py`, `clean_capture.sh`, `recover_imgsz_test.py`; deliverable `~/fyp/Results/diagnostics/cap_T7/{frames/*.png, clean_capture_meta.csv}`. No code/gain/trajectory changes; only this CLAUDE.md seal.
> - **2026-06-22 — FOV-loss mechanism RE-DIAGNOSED (M10.3): the hard-trajectory loss is TRIGGERED by an in-frame small-target DETECTOR-MISS, not FOV-departure — reopens DETECTION as the one untested lever:** a headless GT-overlay recorder (`gt_overlay_record.py`, mp4 + per-frame CSV) + loss-onset classifier (`gt_loss_analyze.py`) run on T3/T6/T7 at **clean Zone 1** (Config 2, seed 42, banner Kp_wz=0.90/Kp_y=1.80, latch OFF; all 3 aborted on the 60 s SEARCH watchdog), plus pooled Zone-1 gt+flight CSVs (seeds {42,43,45}). **Intrinsics bug caught+fixed:** the recorder first fell back to SDF fx=457; forced onto the live `camera_info` **fx=277.19** (the value the canonical gt CSVs use). The shared projector has a clean horizontal axis (u-resid 0.5 px) but a **−42.6 px vertical bias** (≈9° pitch-model error → vertical-edge calls approximate, horizontal solid). **Six offline findings:** **(1) Loss ONSET = in-frame detector-miss** — every first onset has the GT INSIDE the frame near centre at **mid-range 14–17 m, box steady (~10 px), no shrink, no canopy on the LOS** (Zone-1 clearance 69.5 m, every loss <40 m → occlusion stays ruled out, trees ruled out by eye in the mp4s). **(2) Detection envelope (pooled, in-FOV frames):** recall ≥97 % out to **15 m / ≥14 px**; soft shoulder 15–25 m (72–83 %, 7–11 px); **crosses 50 % at ~28 m / ~5–6 px; ~0 % beyond 40 m.** **(3) Gate-vs-raw — CORRECTED:** the in-frame miss is **~91 % raw-YOLO (no candidate at all: det_state=LOST, det_w=0), only ~9 % persistence-gate** (candidate withheld) — *the earlier "~55 % gate / ~44 % raw" estimate used a wrong det_state mapping and is REVISED to 9/91*; the gate is a low-payoff lever, raw model recall is the problem. **(4) Closure NOT speed-limited:** during SEARCH cmd_vx is pinned at **max 4.5 m/s, faster than every target (T3 3.5 / T6 2.0 / T7 3.0)**; APPROACH median 1.13 (headroom unused because unneeded) → **K_far/max_vx is a dead lever**. **(5) SEARCH geometry:** **81 %** of pooled SEARCH frames the target is OUT of frame (already diverged), 19 % in-frame-undetected → recovery-failure is FOV-divergence (the heading-latch τ sweep already proved this dead). **(6) Resolution cost is FREE:** imgsz 1280 = **47 fps**, 2×2 tiling = **27 fps** (RTX 4060) — both above the ~20 fps pipeline budget, so higher-res inference costs nothing operationally; px-doubling maps the 7–14 px miss-shoulder into the ≥14 px = 99 % band → **predicted to recover most 15–30 m raw-misses** (exact recovery count needs a CLEAN-frame re-inference — the recorded mp4s have the GT marker burned onto the target, so a valid count is impossible offline; only the fps cost was measured). **Decision/framing: the "static tuning DONE / six-lever structural ceiling" stands for the CONTROL side, but all six levers (R, Q_vel, standoff/α*, damping, SEARCH-heading/τ, Kp_wz/Kp_y) were controller-side — the DETECTOR was never a lever. The ceiling decomposes into a detection-TRIGGER (mid-range small-target raw recall — tunable via resolution) + an FOV-divergence RECOVERY-failure (the structural part). Highest-payoff next step = a detection-resolution probe (imgsz 1280 / raise native camera res) BEFORE any expensive v4.2 retrain.** Reconciles with — does not overturn — the 2026-06-15 z7 "99.6 % OUT_OF_FOV" finding: that counted the whole dropout population (tail-dominated by the divergent SEARCH = the *consequence*) on a close-range (2.3 m) overshoot run; this is **onset-based** and a **mid-range** loss. Tools (catkin, untracked): `gt_overlay_record.py`, `gt_loss_analyze.py`, `occlusion_record.sh`; deliverables `~/fyp/Results/diagnostics/watch_T{3,6,7}.mp4` + `watch_T{3,6,7}_frames.csv`. No code/gain/trajectory changes; only this CLAUDE.md seal.
> - **2026-06-20 — Kp_wz × Kp_y angular-tracking gain sweep (M10.3 step 1): FLAT → SIXTH lever ruled out, static tuning DONE:** the one controller param that could have mattered if T7 were *heading*-limited — the yaw-rate + lateral centering gains — was swept live on T7/Zone-5, seeds {42,43,45}, Config 2, 200 s, LOSS_TIMEOUT=60. Coarse OFAT direction-finder: **A** base (Kp_wz 0.9, Kp_y 1.8) · **B** wz↑ (1.5, 1.8, ×1.67) · **C** both↑ (1.5, 2.7). A runner banner-compare bug (`0.9`≠node's `%.2f` `0.90`) re-ran each cell 3× — **harmless (gains correct every run, CSV copied pre-check) and turned into a fixed-gain noise estimate** (bug since fixed). **Three thesis-grade specifics:** **(1) Noise floor as a number — at IDENTICAL gain+seed, T7/Zone-5 HOLD% spreads ~8.9 pts run-to-run** (base s42: **28.1 / 26.0 / 34.8**); every B/C paired Δ (B +2.4/+0.3/+0.7, C +5.0/−0.1/−1.7; cell-means A 18.0, B 19.2, C 19.0) sits *inside* that band → **statistically flat**. This number pre-empts "did you run enough reps?", justifies reporting **T7 as a stress appendix with variance, not a point estimate**, and sets the **minimum rep count for hard (T7-class) trajectories** in the final matrix. **(2) Structural claim scoped to REGIME, not global — the fast-inclined maneuvering regime (T7-class) sits at a structural FOV/closure ceiling, while the tracker holds ~90 % HOLD on T1/T2/T4/T5/T8.** All **27** runs aborted to the watchdog (ended in SEARCH, spans 102–182 s) **regardless of gain, base included** → the failure is **qualitative (target leaves FOV faster than any centering gain can track), not a tunable margin**. Frame as a **measured ceiling on aggressive 3D evasion — a found-and-proved limit, never a tracker defect**. **(3) The oscillation result closes the "push harder" door — B/C yaw zero-crossing came in BELOW base (0.102 / 0.080 vs 0.175 Hz):** the gains had real headroom, reached **no ringing regime even at ×1.67/×1.5**, and still moved nothing → forecloses "you just didn't push the gains hard enough" (the pre-stated over-gain disqualifier was 3-seed-mean ≥ 0.21 Hz = baseline mean+2σ; never approached). **Decision: adopt NO cell; Kp_wz=0.9 / Kp_y=1.8 unchanged in the file.** Saturation incidental (wz cap grazed 2–3 % on one seed in B/C, zero benefit → not the lever, outcome (c) ruled out). **Committed `ea75620` = rosparam plumbing only, a verified no-op** (`~Kp_wz`/`~Kp_y` defaults 0.9/1.8 → plain run byte-for-byte baseline; `KP_WZ`/`KP_Y` launch knobs; `extract_metrics` END-col `hold_yaw_zerocross_hz`). T7 is now a **CHARACTERIZED structural FOV/closure ceiling across SIX independent levers** (R, Q_vel, standoff/α\*, damping, SEARCH-heading/τ, **Kp_wz/Kp_y**). **Static tuning is DONE.** Next (M10.3 item 2) = adaptive-IBVS mechanism — **form OPEN, NOT defaulted to fuzzy; a Rawad+supervisor strategy decision before any code.** Tools (catkin, untracked): `kp_sweep.sh`, `kp_sweep_analyze.py`.
> - **2026-06-20 — SEARCH velocity-latch (IBVS v6.30): T7 unrecoverable-SEARCH root cause found + characterized as an FOV/closure limit; latch shipped default-OFF as an opt-in evasion aid:** **Root cause** — the "velocity-predicted SEARCH" (v6.22+) always flew BLIND: the Kalman zeroes its velocity state at `max_dropout`=30 frames=**1.5 s** of dropout, but IBVS only enters SEARCH at `detection_timeout`=**3.0 s** after loss and reads `_kf_vx` THEN → it was always 0, so `_search_base_cwz`=0 and `_pred_cx` froze (Stage-2 = blind ±30° sweep around heading 0). This (not damping/R/Q/standoff) is why a lost maneuvering target diverged. **Fix (v6.30, light single-knob):** latch the loss-instant `v0=(_kf_vx,_kf_vy)` from the last tracking frame (while still valid) and scale it by one exponential decay weight **`w(t)=exp(-t/τ)`**, t=time since loss; one knob `~search_tau` (default 1.0 s), one clamp (`SEARCH_STEP_CLAMP_PX`=12 px/tick). SEARCH-ONLY, gated by `~search_latch`; flag-off ≡ byte-for-byte v6.28 (svx=live `_kf_vx`=0 by 1.5 s); HOLD/APPROACH untouched; Config-1 (no `kalman_velocity`) → v0=0 → graceful blind search. **τ sweep {1,2,3,4} (T7+T9 × seeds {42,43,45}, Config 2 Zone 5, paired vs v6.28) — NO win-both value:** **T7 never lifts at ANY τ** (ΔHOLD −1.1…−2.5, Δrecov −22…−39 everywhere) → **confirms T7-Zone5 is FOV/closure-limited, NOT heading-limited** (agrees with the standoff-sweep + FOV-loss findings; a better SEARCH heading cannot fix a problem that isn't about heading). **T9 evasion improves ONLY at τ=1.0** (Δrecov **+52.8**, ΔOOF **−676**, aborts removed on 2/3 seeds); at τ≥2 the stale-heading regression re-opens (Δrecov −14…−39). **Mechanism: the latch helps reactive evasion most when it commits LEAST** (the fuzzy target reverses mid-gap → a long-trusted stale heading is poison). **Decision: flag default-OFF** (`~search_latch`=False in node + `SEARCH_LATCH:-false` in launch_stack) → a plain run = v6.28 baseline, **ablation matrix + T7 gate unaffected**; the time-bound latch is preserved as an **opt-in evasion aid** (`SEARCH_LATCH=true`, τ=1.0 best when enabled). **The T7 maneuvering-target failure is now a CHARACTERIZED structural FOV/closure limit, not a tunable deficiency** (ruled out across R, Q_vel, standoff/α\*, damping, and SEARCH-heading/τ — all five levers). Tools (catkin, untracked): `search_latch_ab_v630.sh`, `search_latch_analyze_v630.py`, `tau_sweep.sh`, `tau_sweep_analyze.py`, `takeoff_probe.sh`. Aside: **seed 44 is a takeoff DUD in Zone 5** (failed the EKF/arming gate 6×, upstream of IBVS) → clean paired seed set for Zone 5 = **{42,43,45}**.
> - **2026-06-17 — R=[25,15,3000] TESTED AND REJECTED; R stays [6,6,5] (M9.8):** the M9.9 R re-characterization was reverted in code (catkin + repo) back to `R=diag[6,6,5]`; the M9.9 banner/changelog were removed. R=[25,15,3000] was characterized on T1/T4 (static/slow, apparent size ~constant) where heavy alpha smoothing is free; on T7/T9 (fast depth change) the fixed large R_alpha lags true apparent size → worse estimate. Confirmed by HOLD gate + offline same-stream replay (8/10 streams worse on alpha). **Lesson: R_alpha cannot be one fixed value across static+maneuvering regimes.** The f=277 camera-focal comment fixes in `yolo_detection_node.py` are correct and KEPT (unrelated to R). Next knob to evaluate = Q_vel (process noise), the regime-relevant lever for maneuvering targets.
> - **2026-06-11 — stress trio flown: guard VERIFIED on T4; two new fixes (FP gate v3.3 + max_vx 4.5) NOT yet flown:** **Trio results (z7, seed 42, 300 s, Config 2):** **T4 — HOLD 89.4%, det 99.6%, sep 6.48 m, min_sep 1.59 m, 1 emergency-brake engagement → v6.26 guard VERIFIED** (no 0.4 m-class event). **T3 — HOLD 9.8%, sep grew past 20 m, aborted 57 s: chaser `max_vx` 3.5 == target speed 3.5 → ZERO closure capability (structural).** **T7 — HOLD 11.3%, det 35%, aborted 157 s: during SEARCH YOLO latched a tree-line false positive ("DRONE 27×8") → IBVS chased the phantom, phase left SEARCH (watchdog reset), recovery failed.** Fixes (staged, fly the trio again): **detection node v3.3** — plausibility filter (box w ∈ [3,300] px from the iris's true size 0.52–0.77 m tip-to-tip + f=307; w/h ∈ [0.8,6.0]) + **acquisition persistence** (no accepted det >1 s → LOST; re-acquire needs ≥3 consecutive frames <80 px apart; starts LOST; gated frames publish the NaN no-detection point; params `~persist_frames`/`~persist_max_jump`/`~lost_after`; `/drone_tracking/detector_status` for the viewer's [GATED] tag). *Ablation integrity: the gate decides WHICH detections exist; accepted coordinates pass untouched — identical in raw/kalman.* **IBVS v6.27** — `max_vx` 3.5→**4.5** (`~max_vx`; ~30% speed advantage; PX4 `MPC_XY_VEL_MAX`=12 default, no clip; brake authority rises to −4.5 too; `max_vx_retreat` unchanged pending re-run emerg data). **ATTRIBUTION:** both land before the re-run — T3 improvement attributes to max_vx, T7 recovery to the FP gate (orthogonal mechanisms). Also: viewer **v1.2** (rolling 300-frame rate + conf + [GATED]), random_spawn **v3.2** (zone 7 center → (−45,−130), ≥42 m off the y=−180 edge; spawn yaw → east ±40°, PRE-BASELINE then LOCKED), summary consolidation (old schema → `summary_archive_preEmerg.csv`; `summary_v2.csv` → `summary.csv`), `plot_run.py` QA plots. **Future card (play ONLY if the runtime gate proves insufficient — a model change resets R characterization): "YOLO v4.1 negative-image retrain"** — add the 1,349 empty frames as background negatives + an FP-rate eval cell (false detections per 100 background frames) to the Colab notebook.
> - **2026-06-11 — ALPHA_EMERGENCY guard (IBVS v6.26) + watchdog protocol; NOT yet flown:** collision forensics on the T2 z5 CSV nailed the P1 root cause: **the brake branch was working but clamped** — `max_vx_retreat=0.50` m/s pinned `cmd_vx` at −0.50 for the entire 6 s drift from 2.1 m → 0.40 m while the 1 m/s target closed (alpha peak 0.353). Kalman largely exonerated (3 collapse-rejection rows; PRED bridging correct). **IBVS v6.26**: emergency brake guard ABOVE the control law — engage at `~alpha_emergency` **0.033** (3.6× the healthy-HOLD alpha ceiling 0.0091 measured across T1/T2; ~2× the HOLD band ceiling 0.0167; 10× below the collision peak; crossed 5.2 s before closest approach in the recorded collision), release below `~alpha_emergency_exit` 0.7×=0.0231 (hysteresis); while engaged `vx=−max_vx` bypassing vel_smooth AND the retreat clamp, vy/vz/wz keep tracking; target lost while engaged → release to normal SEARCH (never brake blind); engaged state on `/drone_tracking/emergency_brake` → **flight_logger `emerg` column** (flag, NOT a phase). *Caveat (recorded honestly): a pure-alpha trigger cannot give ≥1 s margin at a hypothetical 3.5 m/s closure (alpha=0.033 ↔ ≈1.4 m; 1 s at 3.5 m/s needs trigger at ≈4.9 m where alpha≈alpha_star — inside the HOLD band). Real recorded closures near HOLD were ~0.3–0.5 m/s; the guard's −3.5 matches the target's max 3.5, so once engaged separation cannot keep shrinking.* **extract_metrics** +4 END-appended columns: `emergency_brake_pct`, `emergency_brake_count`, `aborted`, `mission_duration_s` (logged sim-time span) → new schema rows divert to `summary_v2.csv` (B-series mechanism). **Watchdog protocol**: first-acquisition grace (SEARCH counts toward abort only after first APPROACH/HOLD); **tuning runs LOSS_TIMEOUT=10, official `--matrix` batches export 60** (10 s censors the recovery metric + biases against Config 1; a 60 s abort is a recorded outcome); mission end prints "Xs wall / Ys sim" (RTF-drop visibility). Validation = Rawad's stress trio T3/T4/T7.
> - **2026-06-11 — flight session 2: spawn 8–12 m LOCKED; 10 s loss watchdog; NEW P1 = chaser-target collision:** sequencing gates + `takeoff_both` v10.0 **VERIFIED in flight (3/3 clean launches**, EKF gate released at sim ~37 s each time, zero PX4 deaths; watchdog **abort path NOT yet exercised**). Old 3–6 m spawn caused instant FOV loss + 59 s SEARCH the moment the mover started (smoke run, det 36%) → `random_spawn_target.py` **v3.1**: spawn separation **8–12 m — LOCKED PROTOCOL PARAMETER** (see §13; changing it = mandatory baseline reset). `launch_stack.sh` gained a **loss watchdog**: IBVS phase = SEARCH for `LOSS_TIMEOUT` (default **10 s**) consecutive seconds → abort run, still save CSV + analysis (tagged "RUN ABORTED") + metrics row, batch continues; new env knobs `START_DIST`, `LOSS_TIMEOUT`. Tuning results (NOT official baselines): **T1 z7 static 300 s — HOLD 88.5%, det 99.6%, sep 3.98 m, alt err −0.02 m**; **T2 z5 slow (stopped at 180 s) — HOLD 97.7% of mission, det ~99%, sep 4.81 m, alt err +0.18 m**. Findings: altitude bias is **target-motion-dependent** (−0.02 m static → +0.18 m at 1 m/s); HOLD separation **drifts with target speed** (pursuit lag — λ-scheduler territory); **v8s live in-pipeline fps ≈ 19–20** (same as v3 — record for the n-vs-s table); **RTF = 1.00** → 300 s sim ≈ 5 min wall → full matrix ≈ 16 h. **NEW P1 — chaser-target collision** observed in flight (no safety floor in IBVS; see §12): next task = **ALPHA_EMERGENCY brake guard → stress trio T3/T4/T7** → resume pipeline (§10).
> - **2026-06-11 — sequencing fix (wall-vs-SIM time race; "weird chaser" crash):** a 300 s run crashed on the pad: chaser `ARM → FAIL` ×10 at **sim 8 s** (EKF not re-aligned after the T2 zone teleport — a ~200 m jump at sim 0–1 during initial alignment), takeoff_both gave up after 10 attempts and **abandoned the already-armed target** with no setpoint stream → it tipped over (the "acting weird" screenshot). Root issue: stages were gated by **wall-clock sleeps**, but PX4 readiness is a **SIM-time** property and sim time freezes during the lockstep window (T2 target spawn → T5 attach) — so some runs armed at sim 24 s+ (fine) and others at sim 8 s (crash). Fixes: `launch_stack.sh` gained **readiness gates** — `wait_fcu` (MAVROS `connected: True`) after T1 (before the teleport — topic existence fires ~30 s early, mid world-load) and after T6 (heartbeats imply instance 1 attached + rcS done = freeze over), plus **`wait_sim_time 25` before T10** (EKF settle floor). `takeoff_both.py` → **v10.0**: ARM_ATTEMPTS 10→40 (~20 s sim, outlasts EKF alignment) + **disarm-on-abort** (never leave one drone armed). Static checks only — NOT yet smoke-tested (Rawad runs Gazebo next session).
> - **2026-06-11 — infrastructure VERIFIED (post pxh-EOF fix):** smoke test PASSED (Config 2, T4, zone 7, seed 42, 120 s: HOLD 77.9%, det 99.87%, takeoff_ready OK in 1 s, zero deaths). `run_config.sh` batch validated **2/2** (seeds 42/43); same-seed repeat near-identical (HOLD 77.86 vs 78.06, n=1554 both) → paired-comparison basis confirmed. **YOLOv8s live fps measured: ~19–20 on CUDA** (closes the step-2 fps item; n-vs-s table still pending v8n live number). New `sync_results.sh` mirrors `~/fyp/Results` → `C:\Users\fakhe\OneDrive\Desktop\FYP\Results` (auto-runs at end of every `run_config.sh` batch; results stay on ext4 during runs — /mnt/c is slow and OneDrive locks files). `yolo_debug_viewer.py` → **v1.1** (stale-box cleared after 0.5 s + NO DETECTION banner; was: frozen box forever + inflated rate) and now auto-launched by `launch_stack.sh` as stage **TV** (`VIEWER=0` to disable for unattended batches). **Standard run duration = 300 s** (the built-in default; the 120 s runs were smoke-test overrides only). Parallel sims ruled out (single Gazebo world, fixed ports 4560/4561, one GPU) — sequential batches via `run_config.sh` are the design.
> - **2026-06-10 — pxh-EOF fix (REAL cause of chaser-PX4 death):** the recurring "chaser PX4 dies → Gazebo closes / T10 takeoff_ready TIMEOUT" was the chaser's **interactive pxh shell reading EOF on stdin**. `launch_stack.sh` backgrounds roslaunch (`&` → stdin=/dev/null); `posix_sitl.launch` defaults `interactive:=true`, so pxh starts the instant rcS completes (= when baylands finishes loading, ~30 s in) and exits px4 cleanly (`pxh> Exiting NOW.`); `sitl` is `required="true"` → roslaunch kills master+Gazebo → all later nodes die (`Failed to initialize time` / `Connection refused`). Death lands "at T5/T10" only because that's when world load finishes — both the B2 theory AND the T5 working-dir theory misattributed it (the T5 multi-instance form is still correct and kept; verified standalone it blocks properly on TCP 4561). **Fix: `interactive:=false` on the T1 roslaunch** (px4 runs with `-d`, no pxh). Target PX4's "rcS return value: 2" in that run = SIGINT(2) from cleanup, not a startup failure.
> - **2026-06-10 — T5 dual-SITL fix:** target PX4 (instance 1) now launches via PX4's own multi-instance Classic form (per-instance working dir + `-w sitl_iris_1` + rootfs `<build>/etc`), fixing the **chaser-PX4 death** that closed the Gazebo window at T5/T10. The earlier **B2 "fix" was wrong**: `PX4_SIMULATOR=gazebo` is a NO-OP in v1.13.3 (never read by `rcS`; starts no 2nd gzserver). True regression = missing working-dir isolation destabilising lockstep on the shared Gazebo. M7.3 "verified command" block corrected; B2 row marked SUPERSEDED.
> - **2026-06-10 — M9.6 step 2:** YOLOv8s deployed (`best.pt` = best_v4s.pt, 11.14M params, 28.6 GFLOPs; v4n kept as `best_v4n.pt` rollback). Detection node → **v3.2** (`~device`/`~conf` rosparams + rolling-fps log; detection logic unchanged). CUDA available (RTX 4060, torch 2.4.1+cu121). Results restructured to **`~/fyp/Results/Config{1-4}/`** (`launch_stack`/`run_config`/`extract_metrics` repointed; old `~/results` never materialized → nothing to archive).
> - **2026-06-10 — M9.6:** fuzzy adaptive IBVS locked (Mamdani λ-scheduling, no RL), YOLOv8m dropped (n-vs-s only), PPO/Config 3/Config 4 parked pending supervisor. Bugs B1–B10 **FIXED in Phase B**; target_mover bumped to **v10.6** (B1 + SPRINT MF rescaled to fit universe [1.0,3.5]); Kalman strings unified to M9.8. CLAUDE.md corrected against master (code = source of truth).

---

## 1. Project Summary

A fully autonomous **chaser drone** that detects, tracks, and follows a **target drone** using computer vision and AI, evaluated entirely in simulation with statistical validation results as the measure of success.

**One-sentence architecture (deployed Configs 1–2):**
> FPV Camera → YOLOv8 (detection) → [Kalman Filter — Config 2 only] → IBVS controller (pixel-error → **body-frame velocity setpoint** on `/mavros/setpoint_raw/local`) → PX4/MAVROS → drone moves

**Planned full pipeline (Config 3, PARKED):** insert PPO outer loop (α*, λ) between Kalman and IBVS.

**Architectural family:** Hierarchical "DRL-tunes-IBVS" paradigm. PPO is the strategic outer loop; IBVS is the tactical inner loop. Established by Sampedro et al. (IROS 2018), refined by Jin et al. (IEEE TIE 2022), Hu et al. (2022), and Wu/Fu et al. (Drones MDPI 2023). **M9.6 decision:** the outer-loop λ adaptation is implemented as **Mamdani fuzzy gain scheduling** (no RL) for the deployed ablation; PPO is parked.

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
| RL framework | Stable-Baselines3 (SB3) v2.4.1 (PPO parked) |
| Deep Learning | PyTorch |
| Detection | Ultralytics YOLOv8 |
| Kalman | hand-rolled 6-state KF (numpy, not filterpy) |
| Hardware | i5-13420H, 16GB RAM, RTX 4060 8GB |

**Key Paths:**
- ROS package scripts (live, run from here): `~/catkin_ws/src/drone_tracking/scripts/`
- Git repo (push from here ONLY): `~/Fyp_Drone_Detection_Tracking/drone_tracking/scripts/`
- PX4 autopilot: `~/PX4-Autopilot/`
- Models (YOLO + PPO weights): `~/drone_detection/models/`
- Active YOLO model: `~/drone_detection/models/best.pt` (**= best_v4s.pt / YOLOv8s, deployed M9.6 step 2**; rollback kept as `best_v4n.pt`; path hardcoded in `yolo_detection_node.py`, device selectable via `~device` rosparam)
- PPO weights: `~/drone_detection/models/ppo_policy_weights_v5.pth` (parked)
- Flight log (latest run): `~/flight_log_latest.csv`
- Ablation results: `~/fyp/Results/` (M9.6 step 2 layout: `summary.csv` + `Config{1,2,3,4}/traj{T}_zone{Z}_{TS}.csv` + matching `.analysis.txt`). Old `~/results` never materialized (no pre-fix runs) → nothing archived. Results stay **OUTSIDE** the git repo.

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
- Control law: pixel error `e = [ex, ey]` → body-frame velocity setpoint (PD/PID per axis, not the textbook `-λ·Le⁺·e` form — see §Control Chain).
- Why IBVS over PBVS: YOLO gives only pixel bounding boxes — no depth. IBVS is calibration-tolerant and keeps target in FOV by construction (Pereira 2021).
- Known failure modes addressed: fixed gain λ (planned: fuzzy scheduler), FOV loss at high gain, close-range oscillation (α* clamping + Kalman rejection).

### M6 — PPO Agent (v4.5) — **PARKED at M9.6**
- Architecture: small MLP policy (SB3 PPO), action space: [α*, λ].
- Observation space: [ex, ey, α] normalized (divide α by 0.06).
- Reward: pure positive similarity reward.
- Export: PyTorch `.pth` state_dict only (never full SB3 `.zip`).
- **M9.6 status: PARKED** pending supervisor decision on Config 3 redesign. The deployed λ adaptation is fuzzy, not RL.

### M7 — Dual PX4 SITL Architecture
- **M7.1:** Target drone spawned in Gazebo.
- **M7.2:** IBVS v6.15 finalized with reliable 50s+ HOLD phases.
- **M7.3:** Full dual PX4 SITL — chaser (instance 0, MAV_SYS_ID=1, `/mavros/*`) + target (instance 1, MAV_SYS_ID=2, `/target/mavros/*`).
- **Verified target PX4 launch command (Gazebo Classic, instance 1 attaching to the EXISTING Gazebo from T1 — CORRECTED 2026-06-10 "T5 fix"):**
  ```bash
  # From PX4's own Tools/gazebo_sitl_multiple_run.sh (~L37). The per-instance
  # working dir (-w) is REQUIRED — without it lockstep on the single shared
  # Gazebo destabilises and the CHASER PX4 dies (taking Gazebo down with it).
  PX4_BUILD=~/PX4-Autopilot/build/px4_sitl_default
  mkdir -p "$PX4_BUILD/instance_1"
  (cd "$PX4_BUILD/instance_1" && PX4_SIM_MODEL=iris \
     "$PX4_BUILD/bin/px4" -i 1 -d "$PX4_BUILD/etc" -w sitl_iris_1 -s etc/init.d-posix/rcS)
  ```
  - **Do NOT** set `PX4_SIMULATOR=gazebo` — it is a **NO-OP** in v1.13.3 (referenced nowhere in `rcS`/`px4-rc.simulator`; does **not** start a 2nd gzserver). **Do NOT** set `PX4_GZ_MODEL_NAME` (gz-sim/Garden, ignored in Classic). `-i 1` → simulator TCP port **4561**, matching the target SDF `mavlink_tcp_port=4561`. `PX4_SIM_MODEL=iris` → airframe **`10016_iris`** (the only iris airframe in this build; legacy `4001` no longer exists). The target **model** is spawned separately by **T2** (`random_spawn_target.py` → `gazebo_ros spawn_model`), so T5 launches ONLY the firmware.
- Dual-SITL gotchas: MAV_SYS_ID mismatch silently breaks target MAVROS; target spawn height < 0.5m causes physics glitch; world-frame velocity required for the target (FRAME_LOCAL_NED for `/target/mavros/setpoint_raw/local`).

### M8 — PPO v5 Retraining — **PARKED at M9.6**
- Retrained PPO to v5.2 with dual-SITL environment. Result: near-constant output (α*≈0.011, λ≈0.50).
- Root cause: domain gap between offline training distribution and live Gazebo.
- **M9.6 status: PARKED.** Documented as a valid ablation finding; v6 retraining not pursued in current plan.

---

## 4. Milestone 9 — Evaluation & Optimization Phase

### M9.1 — Evaluation Framework Design & target_mover v9.0
- Locked the 9-trajectory benchmark (T1–T9) and the configuration ablation.
- `target_mover.py` gained `~trajectory` rosparam (1–9). T1–T8 deterministic parametric; T9 = FuzzyEscaper.
- Removed stray `.git` inside `scripts/`. Rule: always push from `~/Fyp_Drone_Detection_Tracking/` only.

**9 Benchmark Trajectories:**

| ID | Name | Parameters | Purpose |
|---|---|---|---|
| T1 | Static Hover | Stationary | Baseline HOLD stability |
| T2 | Slow Straight | 1.0 m/s | Low-speed following |
| T3 | Fast Straight | 3.5 m/s | High-speed following |
| T4 | Circular Orbit | R=8m, T=25s | Lateral tracking |
| T5 | Lemniscate | a=8m, T=40s | Direction-change tracking |
| T6 | Inclined Medium | 15°, 2.0 m/s, random azimuth | 3D maneuvering |
| T7 | Inclined Hard | 35°, 3.0 m/s, random azimuth | **3D hard — deterministic VALIDATION GATE** |
| T8 | Up-Down Helix | R=8m, T=25s, continuous ascent+descent | Altitude + lateral combined |
| T9 | Active Evasion (fuzzy) | 7-axis Mamdani evasion | **Stress test only — reactive, never an accept criterion** |

### M9.2 — Fuzzy Evasion Model Development (target_mover v9.x)
- Iterative redesign of the Mamdani fuzzy evasion (target_mover) addressing circular orbiting, altitude/heading conflicts, conservative speeds, boundary-repulsion override.

### M9.3 — Component Tuning Round 1
- `random_spawn_target.py` v2.0→v3.0: GPS-surveyed zones; ablation whitelist {5,6,7,9}.
- `flight_logger.py` + `analyze_flight_log.py` gained `world_alt_err` column.
- IBVS v6.19→v6.20 (K_far 30→35, directional SEARCH memory); target_mover v9.8→v10.2.

### M9.4 — Component Tuning Round 2 (major session)
- **Kalman M9.6→M9.7→M9.8:** Q_pos→0.5, Q_vel→6.0, PIXEL_JUMP→180, MAX_REJ→4.
- **IBVS v6.21→v6.25:** v6.24 **breakthrough — ea HOLD threshold 0.005→0.010** (HOLD 0%→98%). v6.22 rebuilt SEARCH as the **2-stage velocity-predicted** design that is still deployed.
  `<!-- VERIFY: old CLAUDE.md claimed "v6.25 = Stage 3 / 360° at 5s". NOT in code. Deployed SEARCH is 2-stage (Stage 1 0–3s velocity-predicted, Stage 2 3s+ ±30° sweep). The file is titled v6.25 but its changelog stops at v6.22. -->`
- **target_mover v10.3→v10.5:** 7-axis Mamdani evasion, EDGE→FAST, speed/MF rescaling.
- **YOLO v4 dataset pipeline:** auto-label → relabel empties → pHash dedup (~44% removed) → v3-mix assembly.
- Best run hit HOLD 98%, but HOLD% ranged 26–98% across runs — **inconsistency is the primary bottleneck** (P1 at the time; now P2 — see §12).

### M9.5 — Batch Runner Infrastructure + YOLO v4 Training
- `yolo_detection_node.py` publishes `/drone_tracking/target_box` (real cx,cy,w,h).
- `ibvs_controller_node.py` gained `~detection_source` (raw|kalman); `~use_ppo` confirmed rosparam.
- `target_mover.py` + `random_spawn_target.py` gained `~seed`.
- `extract_metrics.py` created (appends one row to `~/results/summary.csv`).
- `cleanup.sh` done + tested. **`launch_stack.sh` / `run_config.sh` were created (exist on master) but untested and buggy — see Known Bugs.**
- YOLO v4 training (Colab, from COCO weights, epochs=80, seed=42): **v8n done** (mAP50=0.991, mAP50-95=0.848, P=0.971, R=0.979, 75 FPS on T4). v8s interrupted at epoch 34. v8m not started.

### M9.6 — Strategy Lock + Infrastructure Bug Fixes (current session)
**Locked decisions (the spine of the remaining work):**
- **Adaptive IBVS = Mamdani fuzzy gain scheduling** (reuse FuzzyEscaper pattern from `target_mover.py`; cite Fu et al., Drones MDPI 2023). Formula-based scheduling = documented fallback. Replaces the fixed `0.70` λ-fallback. See §Adaptive IBVS.
- **Deployed YOLO going forward = YOLOv8s** (supervisor decision). v8m DROPPED — comparison is n-vs-s only. v8s must be deployed *before* any noise (R) measurement (R is model-dependent).
- **PPO, Config 3 redesign, Config 4 = PARKED** pending supervisor.
- **Universal validation gate:** 2×T7 (seeds {42,43}) pass/fail gate + 2×T9 (seeds {42,43}) stress test, 10 Hz logging. Seeds are fixed and reused forever → paired comparison (identical target path per seed on deterministic trajectories). T7 is deterministic (differences attributable to the change); T9 is reactive (robustness evidence only, never the accept criterion).
- **Bugs B1–B10 identified and fixed in Phase B** (see Known Bugs section).

---

## 5. Ablation Study (revised at M9.6)

**Primary evaluation matrix (deployed):** T1–T8 × zones {5,6,7,9} × **Configs 1–2**, ≥3 seeded repeats, paired significance test. T9 reported separately as stress test.

**Configurations:**
| Config | Pipeline | Key rosparams | Status |
|---|---|---|---|
| 1 | YOLO + IBVS (fuzzy-λ) | `use_ppo:=false`, `detection_source:=raw` | **Active** |
| 2 | YOLO + Kalman + IBVS (fuzzy-λ) | `use_ppo:=false`, `detection_source:=kalman` | **Active** |
| 3 | YOLO + Kalman + PPO + IBVS | `use_ppo:=true`, `detection_source:=kalman` | **PARKED** (PPO pending supervisor) |
| 4 | End-to-end image→velocity | — | **PARKED** |

**Ablation integrity (verbatim, critical):** Config 1 and Config 2 use the **IDENTICAL** fuzzy-scheduled IBVS; the **ONLY** difference is the Kalman filter. Config 1 stays `detection_source=raw` with **NO filtering added**. `lambda_gain` comes from the scheduler in **both** configs. (Phase 1 of the scheduler uses alpha only — identical in raw/kalman modes. Phase 2's alpha-rate input is the controller-internal Kd_a feedforward signal, also identical in both modes — so adding it does not break Config-1 integrity.)

**All M9.x simulations to date are tuning runs — NOT comparison data.**

---

## 6. Control Chain — What Each Block Actually Controls

```
IBVS → /mavros/setpoint_raw/local (PositionTarget, FRAME_BODY_NED:
        vx, vy, vz, yaw_rate — body-frame VELOCITY setpoint)
     → MAVROS → PX4 SITL cascaded controllers
        (velocity → attitude → rate → motor mixing)
     → 4 motor PWM → Gazebo Iris physics
```

- **PX4 handles ALL low-level flight control** — velocity→attitude→rate→mixing, thrust limits, stabilization. **This is NOT our code.**
- **IBVS maps visual pixel error → desired body-frame velocity setpoint.** It does NOT command motors and does NOT model aerodynamics.
- Therefore **"adaptive IBVS" = adapting how pixel error maps to velocity setpoints (the gains), never touching motor-level control.**
- The "IBVS" here is a per-axis PD/PID controller in pixel/alpha space (not the literal `vc = -λ·Le⁺·e` interaction-matrix form). The `λ`/`gain` term is a single forward-aggression multiplier — that is the knob the fuzzy scheduler tunes.
- **target_mover** commands the target via `/target/mavros/setpoint_raw/local` (FRAME_LOCAL_NED velocities + yaw_rate).

---

## 7. Adaptive IBVS — Fuzzy Gain Scheduling Plan (No RL)

**Decision (locked): Mamdani fuzzy gain scheduling**, reusing the FuzzyEscaper pattern from `target_mover.py` (triangular/trapezoidal MFs, max-min inference, centroid defuzzification). Citation: **Fu et al., Drones MDPI 2023** (fuzzy gain scheduling of IBVS — classical replacement for PPO's λ). Formula-based scheduling = documented fallback if fuzzy tuning stalls.

**What it replaces:** the fixed `0.70` λ-fallback. In `compute_velocities()`:
```python
lam_gain = (.4 + .6*self.lam) if self.ppo_is_active() else .70   # 0.70 = the constant the scheduler replaces
gain = gain_scale * lam_gain
```
`gain` multiplies **both** the K_far approach branch (`vx_p = K_far·√(-ea-dead)·gain`) **and** the K_near brake branch (`vx = -K_near·√(ea-dead)·gain`), so a single multiplier modulates forward aggression symmetrically — exactly where PPO's λ acted.

### Phase 1 — schedule `lambda_gain ∈ [0.4, 1.0]` from alpha ONLY
- K_far=35, K_near=6, all Y/Z/yaw PIDs stay fixed. One fuzzy output, one input.
- Draft rules: alpha VERY_SMALL → HIGH gain; SMALL → MED_HIGH; MEDIUM → MEDIUM; LARGE → MED_LOW; VERY_LARGE → LOW.
- MF breakpoints anchored to: `alpha_star = 0.0067`, `ea_HOLD = ±0.010`, `alpha_min_valid = 0.0005`, working range `0.001–0.04`.
- **Config-integrity safe:** alpha is identical in raw and kalman detection modes.

### Phase 2 — only if Phase 1 plateaus: add controller-internal alpha-rate
- Input = the `dea`/Kd_a feedforward signal IBVS already computes (identical in both detection modes).
- Rules: close + fast-closing → LOW; far + receding → HIGH; mid + stable → MEDIUM; very close → MINIMUM.

### Phase 3 — if still needed: schedule K_far / K_near directly.

### Safety / observability (all phases)
- **Slew-rate limit** on gain (≤ 0.02 per 20 Hz cycle).
- **Hard clamp** to universe bounds.
- New `sched_gain` column in `flight_logger` → gain-vs-distance plot is the report evidence of adaptation.

**Ablation integrity (verbatim):** Config 1 and Config 2 use the IDENTICAL fuzzy-scheduled IBVS; the ONLY difference is the Kalman filter. Config 1 stays `detection_source=raw` with NO filtering added. `lambda_gain` comes from the scheduler in both configs.

---

## 8. Kalman & SEARCH Parameter Optimization Plan

**Current Kalman M9.8 (confirmed in code):** `R=diag[6,6,5]`, `Q=diag[0.5,0.5,3,6,6,3]` (so Q_vel cx/cy = **6.0**), `PIXEL_JUMP_OUTLIER=180px`, `MAX_CONSECUTIVE_REJECTIONS=4`, `velocity_damping=0.88`, publishes `/drone_tracking/filtered_target` + `/drone_tracking/kalman_velocity`.
`<!-- RESOLVED 2026-06-10 (B10 extension): the stale "Q_vel cx/cy 6.0→3.0" docstring claim was DELETED and the title / changelog / startup log unified to "M9.8" (strings only). Live Q_vel stays 6.0 (correct, unchanged). -->`

### R tuning (AFTER v8s deployment only — R is model-dependent)
Plan a reusable `measure_yolo_noise.py` (do not write yet):
- Subscribe to `/drone_tracking/target_center` + `/gazebo/model_states`.
- Project ground-truth target into camera frame using chaser pose + `iris_fpv_cam` **EXTRINSICS read from the SDF** (not assumed identity) + intrinsics (f≈307 px, cx=320, cy=240). Nearest-timestamp matching.
- Measure during **MOTION** (T4 circle, 60–120 s) — hover-only σ is optimistic; optionally also a hover run for a hover-vs-moving comparison table.
- Output σ_cx / σ_cy / σ_alpha (optionally distance-binned), recommended R vs current [6,6,5]; data → `~/results/yolo_noise_characterization.csv`.
- **Build ONE reusable ground-truth-projection utility** shared with the Q-sweep analyzer. Re-run after ANY YOLO change.

### Q tuning (target-behavior-dependent; carries across YOLO models)
- Q_pos stays 0.3–1.0; **Q_vel is the critical knob.** Sweep `Q_vel ∈ {2,4,6,8,10}` (current = 6).
- Per value, run the gate protocol; metrics: mean |cx_filt − cx_gt| in HOLD, derivative-signal std, SEARCH recovery time, HOLD% stability.

### velocity_damping + SEARCH consistency
- Sweep `damp ∈ {0.82, 0.85, 0.88, 0.91, 0.94}`.
- **Consistency check:** useful-velocity lifetime `T_useful = −1/(20·ln(damp))`. At damp=0.88 → ≈0.39 s, but SEARCH Stage 1 lasts 3.0 s → ~2.6 s of Stage 1 extrapolates on stale velocity.
- Options: raise damp, shorten Stage 1, or (preferred long-term) replace the hard stage cutoff with `trust_factor = damp^(dropout_frames)` blending.

### PIXEL_JUMP (after v8s deployment)
- Measure max legitimate frame-to-frame centroid jump during fast T7/T9; set `PIXEL_JUMP = 1.5 × max`.

### Universal protocol (every change above)
2×T7 {42,43} gate + 2×T9 {42,43} stress; 10 Hz logging; **commit only if no regression on BOTH T7 flights.**

---

## 9. Known Bugs (B1–B10)

Identified 2026-06-10 against master; **all FIXED in Phase B (2026-06-10)** unless noted. File + location given.

| ID | File / location | Bug | Fix | Status |
|---|---|---|---|---|
| **B1** | `target_mover.py` `_compute_fuzzy_velocity` (~L474) | `if phi>0.40: au=(phi-.60)/.40` fires at phi>0.40 but formula assumes phi≥0.60 → for phi∈(0.40,0.60) `au<0`, so `f_vz=ed*1.3*au` moves target TOWARD chaser altitude (anti-escape). Also OVERWRITES the 9 fuzzy altitude rules whenever phi>0.40. | Threshold to 0.60 (or clamp au∈[0,1]); deliberate blend-vs-overwrite. | **FIXED** |
| **B2** | `launch_stack.sh` T5 (~L88) | Target PX4 uses `PX4_GZ_MODEL_NAME=` (gz-sim / Gazebo Garden syntax) — wrong for PX4 v1.13.3 + Gazebo Classic; instance likely never binds to the spawned model. | ~~Use `PX4_SIM_MODEL=iris PX4_SIMULATOR=gazebo ./bin/px4 -i 1 -s etc/init.d-posix/rcS`~~ — **this "fix" was itself broken** (ran instance 1 with no working-dir isolation → destabilised lockstep → killed the chaser PX4 → Gazebo closed; `PX4_SIMULATOR` is a no-op, not a 2nd-gzserver trigger). | **SUPERSEDED → see "T5 fix" (2026-06-10): per-instance `-w` working dir, PX4's multi-instance form. See §M7.3.** |
| **B3** | `extract_metrics.py` (~L46) | `detected = r.get('raw_det','') in ('1','1.0','True')` — but logger writes `"REAL"/"NONE"` → detection_rate always 0. | `== 'REAL'`; add filtered detection rate. | **FIXED** |
| **B4** | `launch_stack.sh` `wait_topic()` (~L48,L120) | Checks topic EXISTENCE (`rostopic list \| grep`), not content. `/drone_tracking/takeoff_ready` exists (latched advert) long before takeoff completes → mission starts mid-climb. | `wait_bool_true()` waiting for `data: True` (takeoff_both publishes `Bool(data=True)`); keep `wait_topic` for `/mavros/state`, `/target/mavros/state`. | **FIXED** |
| **B5** | `launch_stack.sh` T11 (after T10) + `flight_logger.py` | Logger launched LAST (after target_mover) → misses TAKEOFF + early SEARCH. Logging rate only 4 Hz. | Launch logger BEFORE takeoff_both; raise 4 Hz → 10 Hz. | **FIXED** |
| **B6** | `target_mover.py` `_gazebo_states_cb` (~L273) | Bare `except: pass` swallows model-name mismatches → fuzzy distance silently computed from zeros. | Warn-once if `_got_world_pos` still False ~5 s into MOVING (don't change matching logic). | **FIXED** |
| **B7** | `run_config.sh` (~L43) | "random" zone/traj uses unseeded `$RANDOM` → batches not reproducible; no deterministic matrix mode. | Add `--matrix` (iterate T1–8 × zones {5,6,7,9} = 32 runs, seed=SEED_START+index); `--matrix --repeats N` offsets seeds by 100/repeat. | **FIXED** |
| **B8** | `cleanup.sh` | No log hygiene → 480 runs flood `~/.ros/log` + `/tmp/T*.log`. | Append `rosclean purge -y`; delete `/tmp/T*_zone*` older than 7 days. | **FIXED** |
| **B9** | `extract_metrics.py` | Missing frozen metrics: filtered detection rate, recovery-time-after-loss, wrong-direction%. | Add `flt_detection_rate`, `pred_rate`, `mean_recovery_time_s`, `wrong_direction_pct` (reuse `analyze_flight_log.py` definition: APPROACH+HOLD+REAL rows where `(ea<-0.002 and cmd_vx<-0.05) or (ea>0.002 and cmd_vx>0.05)`). Append new columns at end (back-compat). | **FIXED** |
| **B10** | `ibvs_controller_node.py` L119; `target_mover.py` strings/docstring; `kalman_filter_node.py` (extension) | Stale version strings: IBVS startup log "v6.22" (file v6.25); target_mover logs/banner "v10.3" (file v10.5); comment "clamp 4.0" but code clamps 3.5; Kalman titled "M9.8" but log/changelog say "M9.6" with a stale "Q_vel 6.0→3.0" claim. | IBVS log→"v6.25"; target_mover strings→**v10.6** + docstring universe [1.0,3.5]; **SPRINT MF rescaled to trap(3.00,3.35,3.50,3.50)** to fit the sampled universe (finishes the v10.5 rescale — deliberate behaviour change, T9 baseline resets); Kalman strings unified to "M9.8" + stale Q_vel claim deleted (no numeric change). | **FIXED** |

`<!-- RESOLVED 2026-06-10: code wins — universe _SP = [1.0,3.5], output min(...,3.5). The v10.5 rescale was incomplete (only the universe was shrunk; SPRINT MF was left at trap(3.20,3.60,4.00,4.00) with its plateau outside [1.0,3.5] → top escape speed silently capped at 0.75 membership). v10.6 finishes it: SPRINT=trap(3.00,3.35,3.50,3.50); docstring/comments say [1.0,3.5]. Deliberate behaviour change → T9 baseline resets. -->`

---

## 10. Execution Order (locked)

> **Current position (2026-06-11, post-trio):** stress trio flown — **guard VERIFIED on T4 (HOLD 89.4%, 1 engagement, min_sep 1.59 m)**; T3 failed on the max_vx speed ceiling, T7 failed on a tree-line phantom capture during SEARCH. Fixes staged (NOT committed): **detection FP gate v3.3 + IBVS v6.27 max_vx=4.5** (+ viewer v1.2, zone 7 geometry v3.2, summary consolidation). **NEXT: Rawad re-flies the trio (LOSS_TIMEOUT=60 all three) → commit (carries the trio-validated v6.26 too) → resume step 3 onward** (noise char → Q_vel → damping → scheduler). Attribution: T3 improvement → max_vx; T7 recovery → FP gate (orthogonal).

1. **Fix B1–B10** (Phase B) → smoke-test `launch_stack` (Config 2, T4, zone 7, seed 42, 120 s).
2. **YOLOv8s — DONE.** ✅ Trained + deployed as `best.pt` (= best_v4s.pt, 11.14M params, CUDA/RTX 4060); `~device`/`~conf` rosparams + rolling-fps log added (node v3.2); results restructured to `~/fyp/Results/`; live fps ≈ 19–20 measured; **n-vs-s table recorded (§11): v8s mAP50=0.990, mAP50-95=0.848, P=0.973, R=0.977 — statistically identical to v8n, pipeline-bound at runtime.**
3. **Noise characterization** → update R. *(Unblocked once step-2 smoke test passes — R is model-dependent, now that v8s is deployed. PIXEL_JUMP re-check also pending per §8.)*
4. **Q_vel sweep.**
5. **Damping + SEARCH consistency + PIXEL_JUMP re-check.**
6. **Fuzzy λ scheduler Phase 1** → gate → Phase 2 only if needed.
7. **Freeze parameters** → full matrix: T1–T8 × zones {5,6,7,9} × Configs 1–2, ≥3 seeded repeats, paired significance test; T9 reported separately.
8. **Report writing.**

---

## 11. Current Component Versions (End of M9.6)

| Component | Version | File | Key params / notes |
|---|---|---|---|
| YOLO model | **v4.1 (best_v41s.pt)** (deployed; flight-validation pending) | `~/drone_detection/models/best_v41s.pt` (node default now points here; v4s preserved as `best_v4s_backup.pt`) | YOLOv8s. **v4.1 trained with ~1200 background negatives to reduce phantom-lock FPs. Metrics: mAP50=0.991, mAP50-95=0.862, P=0.976, R=0.979. FP/100 on holdout: 1.4 (vs 0.0 v4s). Accepted: core metrics within tolerance, phantom-lock validation pending T7 flight test.** Prior v4s: 11.14M params, 28.6 GFLOPs, 130 layers, CUDA (RTX 4060, torch 2.4.1+cu121). v4n kept as `best_v4n.pt` rollback. **n-vs-s table recorded (see below §11) — closes the step-2 item.** v8m DROPPED. |
| Detection node | **v3.3** (FP gate; flight-validation pending) | `yolo_detection_node.py` | 5-frame alpha median; publishes `/drone_tracking/target_center` (cx,cy,area) + `/drone_tracking/target_box` (cx,cy,w,h) + **`/drone_tracking/detector_status`** (String "STATE,conf,w,h"). `~device`/`~conf` rosparams; rolling-fps log (now + per-100-frame rejection counts). **v3.3 gate (T7 phantom-chase fix):** plausibility filter — box w ∈ [`~min_box_px` 3, `~max_box_px` 300] px (anchored to iris 0.52–0.77 m tip-to-tip, f=307: w_px=307·W/Z), w/h ∈ [`~aspect_min` 0.8, `~aspect_max` 6.0]; best PLAUSIBLE box wins (confidence order). **Acquisition persistence:** no accepted det > `~lost_after` (1.0 s) → LOST; re-acquisition needs `~persist_frames` (3) consecutive frames < `~persist_max_jump` (80 px) apart; node starts LOST; gated frames publish the NaN no-detection point. **Ablation-integrity argument: the gate decides WHICH detections exist; it does not filter coordinates of accepted ones — raw and kalman configs see identical behavior.** Model path still hardcoded. |
| Kalman filter | M9.8 (file title) / logs "M9.6" | `kalman_filter_node.py` | R=diag[6,6,5], Q=diag[0.5,0.5,3,**6,6**,3], PIXEL_JUMP=180px, MAX_REJ=4, damp=0.88. Publishes `/drone_tracking/filtered_target` + `/drone_tracking/kalman_velocity`. |
| IBVS controller | **v6.30** (SEARCH latch default-OFF; emergency guard VERIFIED on T4) | `ibvs_controller_node.py` | Output `/mavros/setpoint_raw/local` (FRAME_BODY_NED vx,vy,vz,wz). K_far=35, K_near=6, Kd_a=150, dead=0.002, smooth=0.15, Kp_y=1.8, **Kp_z=3.0**, pitch_comp=0.4, **alpha_star=0.0067**, **ea_HOLD=0.010**, HOLD=(\|ex\|<.12 ∧ \|ey\|<.12 ∧ \|ea\|<.010), **max_vx=4.5 (`~max_vx`; v6.27 — T3 proved 3.5 = zero closure on a 3.5 m/s target; PX4 MPC_XY_VEL_MAX=12 won't clip)**, max_vx_retreat=0.5. **λ-fallback=0.70** (→ fuzzy scheduler). **2-stage SEARCH** (Stage1 0–3s velocity-predicted 2.5 m/s; Stage2 3s+ 4.5 m/s ±30° sweep). **v6.30 SEARCH velocity-latch — DEFAULT-OFF opt-in evasion aid:** latch loss-instant `v0` × exponential decay `w(t)=exp(-t/τ)` for the SEARCH heading; `~search_latch` (default **False** → flag-off ≡ byte-for-byte v6.28), `~search_tau` (default 1.0 s), per-step clamp 12 px. SEARCH-only; HOLD/APPROACH untouched; Config-1 unaffected. τ sweep proved no win-both value — T7 is FOV/closure-limited (heading can't fix), T9 evasion recovery +52.8 at τ=1.0 only. **v6.26 emergency brake guard (VERIFIED in flight, T4 2026-06-11: 1 engagement, min_sep 1.59 m):** engage alpha>`~alpha_emergency` (0.033), release <`~alpha_emergency_exit` (0.0231); engaged → vx=−max_vx (now −4.5; bypasses smooth + retreat clamp), publishes `/drone_tracking/emergency_brake`. rosparams `~use_ppo`, `~detection_source`, `~alpha_emergency`, `~alpha_emergency_exit`, `~max_vx`, `~search_latch`, `~search_tau`, **`~Kp_wz` (default 0.9)**, **`~Kp_y` (default 1.8)**. **M10.3 (`ea75620`): `~Kp_wz`/`~Kp_y` rosparam-overridable (defaults = baseline) for the angular-tracking gain sweep — swept ×1.67/×1.5, FLAT (sixth lever ruled out, see changelog); gains UNCHANGED in file, Kd_wz/Kd_y stay fixed.** **M10.3 vertical probe (2026-06-30): `~Kp_z` (default 3.0) + `~max_vz` (default 1.5) rosparam-overridable (`KP_Z`/`MAX_VZ` launch knobs), banner logs Kp_z/max_vz. Swept Kp_z ×1.5/×2.0 + max_vz ×1.5 on T6/T7 — FLAT (seventh/eighth levers ruled out; ey failure = detection-limited); Kp_z=3.0/max_vz=1.5 UNCHANGED in file, Ki_z/Kd_z fixed.** |
| PPO agent | v5.2 | `ppo_agent_node.py` | **PARKED.** Near-constant output. Weights `ppo_policy_weights_v5.pth`. |
| target_mover | **v10.6** (was v10.5; +B1, +B10) | `target_mover.py` | 7-axis Mamdani fuzzy evasion. Speed defuzz universe **[1.0,3.5]** (`_SP`), output `min(...,3.5)` → speed ∈ [1.0,3.5]. Output `/target/mavros/setpoint_raw/local` (FRAME_LOCAL_NED). `~trajectory` (T1–T9), `~seed`. |
| random_spawn | **v3.2** (2026-06-11) | `random_spawn_target.py` | 9 zones, ALLOWED_ZONES {5,6,7,9}, jitter, z=0.5m, `~seed`, `~zone`, `~dist`. **Spawn separation 8–12 m — LOCKED protocol parameter (was 3–6 m; see §13).** **v3.2: zone 7 center (−45,−170)→(−45,−130) + CLEAR_VIEW_YAW (zone 7 faces east π/2±40°, same single yaw draw remapped → zones 5/6/9 unchanged per seed) — PRE-BASELINE then LOCKED (§13).** |
| takeoff_both | **v10.0 (VERIFIED in flight 2026-06-11)** | `takeoff_both.py` | **TAKEOFF_ALT=14.0m.** ARM_ATTEMPTS=40 (~20 s sim — outlasts post-teleport EKF alignment); disarms BOTH drones if arming aborts. Publishes latched `/drone_tracking/takeoff_ready` + `/drone_tracking/target_takeoff_ready` (Bool data=True at completion). |
| cleanup | done (+B8) | `cleanup.sh` | Kills all ROS/PX4/Gazebo; frees ports; rosclean purge + old /tmp log delete. |
| launch_stack | +B2→**T5fix**,B4,B5,**pxh-EOF fix**,+TV,**+seq gates (VERIFIED)**,**+loss watchdog** | `launch_stack.sh` | One full run: `launch_stack.sh CONFIG TRAJ ZONE SEED [DURATION]` (DURATION default **300 s**). T1 roslaunch passes **`interactive:=false`** (px4 `-d`, no pxh — pxh stdin-EOF was the real chaser-killer; see changelog). T5 = PX4 multi-instance form (`-w sitl_iris_1`, rootfs `<build>/etc`); see §M7.3. Stage **TV** auto-starts the debug viewer (`VIEWER=0` to skip). **Readiness gates:** `wait_fcu` after T1 + T6, `wait_sim_time 25` before T10 — **verified in flight 2026-06-11 (3/3 clean).** **Loss watchdog:** phase SEARCH for `LOSS_TIMEOUT` s (default **10** = tuning; `--matrix` exports **60** — protocol rule, see changelog) → abort run, save CSV/analysis/metrics ("RUN ABORTED" tag, `aborted=1`), batch continues (abort path not yet exercised). **First-acquisition grace:** SEARCH counts only after the first APPROACH/HOLD. Mission end prints "Xs wall / Ys sim". Env knobs: `VIEWER`, `START_DIST`, `LOSS_TIMEOUT`. |
| run_config | **VERIFIED 2026-06-11** (+B7, +auto-sync, +matrix LOSS_TIMEOUT) | `run_config.sh` | Batch loop; `--matrix` deterministic grid (**exports LOSS_TIMEOUT=60** — official runs measure recovery; aborts are recorded outcomes); calls `sync_results.sh` at batch end. Validated 2/2 (seeds 42/43). |
| sync_results | new 2026-06-11 | `sync_results.sh` | Mirrors `~/fyp/Results` → `/mnt/c/Users/fakhe/OneDrive/Desktop/FYP/Results` (rsync; manual or batch-end). |
| yolo_debug_viewer | **v1.2** | `yolo_debug_viewer.py` | Live bbox overlay window. **v1.2: detection rate = ROLLING last-300-frames window (was cumulative); shows box conf + w×h; [GATED] tag while detector v3.3 is LOST/CONFIRMING (candidate shown but withheld from the controller).** Stale box cleared after 0.5 s (+NO DETECTION banner). Auto-launched as stage TV; press Q or cleanup kills it. |
| extract_metrics | done (+B3,B9; M9.6 step 2 path; **+step 3 cols**) | `extract_metrics.py` | Appends one row to `~/fyp/Results/summary.csv`. **2026-06-11 consolidation: old-schema file archived as `summary_archive_preEmerg.csv` (4 pre-guard tuning rows); `summary_v2.csv` promoted to `summary.csv` (append verified, no divert).** Header-mismatch safety divert (→ `summary_v2.csv`) still in place. END-appended: `emergency_brake_pct`, `emergency_brake_count`, `aborted` (from `--aborted`), `mission_duration_s` (logged sim span), **`hold_yaw_zerocross_hz` (M10.3 `ea75620` — cmd_wz sign-changes/s over HOLD; the over-gain/dropped-damping-ratio detector; local summary.csv migrated +1 trailing col to avoid divert).** |
| flight_logger | M9.3 (+B5 → 10 Hz; **+`emerg` col**) | `flight_logger.py` | Output `~/flight_log_latest.csv`; `raw_det`/`flt_det` = REAL/PRED/NONE; `true_dist_3d`, `world_alt_err`; **`emerg`** ('1'/'0', END-appended) from `/drone_tracking/emergency_brake` — a flag, NOT a phase (watchdog/HOLD% parse the phase string unchanged). |
| analyze_flight | M9.3 | `analyze_flight_log.py` | 23+ sections incl. wrong-direction events. |
| plot_run | new 2026-06-11 | `plot_run.py` | 4-panel QA PNG per flight CSV (XY shape / altitude+phase shading / separation+min / alpha vs v6.26 guard thresholds, log). `--batch DIR` renders every `*.csv` (skips up-to-date PNGs) + one-line summary per run. XY panel = per-drone LOCAL frames (shape QA only — world XY never logged; separation truth = panel 3). Pre-`emerg` CSVs handled gracefully. |

### YOLOv8 n-vs-s comparison (report table, recorded 2026-06-12)

Both trained on the v4 dataset (Colab, from COCO weights, epochs=80, seed=42, identical data/augmentation).

| Model | Params | mAP50 | mAP50-95 | Precision | Recall | Colab T4 fps | Live in-pipeline fps (RTX 4060) |
|---|---|---|---|---|---|---|---|
| YOLOv8n (`best_v4n.pt`, rollback) | 3.2M | 0.991 | 0.848 | 0.971 | 0.979 | 75 | ~19–20 |
| **YOLOv8s (`best.pt`, deployed)** | 11.1M | 0.990 | 0.848 | 0.973 | 0.977 | — (not recorded) | ~19–20 |

**Takeaway:** accuracy metrics are statistically indistinguishable on this dataset; live fps is identical (~19–20) because the pipeline (ROS image transport + per-frame Python overhead), not the model, is the bottleneck on the RTX 4060 — so v8s's extra capacity is free at runtime. v8s deployed per supervisor decision; v8m dropped.

### Iris target true dimensions (from `iris.sdf` — gate-bound anchor, recorded 2026-06-12)

Body box **0.47 × 0.47 × 0.11 m**; rotors at (±0.13, ±0.20/0.22) m with prop radius **0.128 m** → tip-to-tip span **0.696 m** (lateral Y), **0.516 m** (longitudinal X), **0.767 m** (diagonal). Expected YOLO box width w_px = 307·W/Z (f=307 px, W≈0.70 m representative): Z=5 m → 43 px, 10 m → 21 px, 15 m → 14 px, 20 m → 11 px, 30 m → 7 px, 40 m → 5 px, 60 m → 3.6 px. Detection-gate bounds derived: `~min_box_px`=3 (0.52 m narrow aspect at 60 m → 2.7 px), `~max_box_px`=300 (0.77 m diagonal at 0.8 m → 295 px).

---

## 12. Known Remaining Issues (Priority Order)

*(Renumbered 2026-06-11 when the collision issue was opened: old P1–P6 → P2–P7.)*

**P1 — Chaser-target collision: GUARD VERIFIED IN FLIGHT (T4 trio run 2026-06-11: 1 engagement, min_sep 1.59 m — no 0.4 m-class event).** Note v6.27 raises engaged brake authority to −4.5 (vx=−max_vx). Originally observed in flight (T2 z5 seed42: 0.40 m closest approach at sim 144, alpha peak 0.353). **Forensics (2026-06-11) corrected the root-cause ranking:** the dominant cause is the **`max_vx_retreat=0.50` m/s brake clamp** — `cmd_vx` was pinned at −0.50 for the entire 6 s drift from 2.1 m → 0.40 m, so any target closing faster than 0.5 m/s out-runs the brake (K_near=6 vs K_far=35 asymmetry is real but secondary: the K_near branch had already saturated the clamp). **Kalman diagnostic answered: NOT a main contributor** — only 3 close-range collapse-rejection rows near the peak; PRED bridging tracked correctly; IBVS was braking on good data, it simply had no authority. **Fix deployed:** `ALPHA_EMERGENCY=0.033` (`~alpha_emergency`; 3.6× the healthy-HOLD ceiling 0.0091 from T1/T2, ~2× the HOLD band ceiling 0.0167, 10× below the collision peak; crossed 5.2 s before closest approach in the recorded collision), exit hysteresis at 0.7× = 0.0231 (`~alpha_emergency_exit`); engaged → `vx=−max_vx` bypassing vel_smooth and the retreat clamp, vy/vz/wz keep tracking; lost-while-engaged → release to SEARCH (never brake blind); `emerg` flag column in flight_logger + `emergency_brake_pct`/`emergency_brake_count` in summary. Identical in all configs (ablation-safe). Known limit: pure-alpha trigger ⇒ <1 s margin at a hypothetical full 3.5 m/s closure (see changelog); recorded real closures were 0.3–0.5 m/s. The fuzzy λ scheduler (step 6) remains the systematic fix; the guard is the safety envelope beneath it.

**P2 — HOLD% inconsistency (26–98%)** *(was P1)*: SEARCH separation sometimes reaches 30–44m before re-acquisition; recovery loss 40–52s. The 2-stage velocity-predicted SEARCH helps but doesn't always recover in time. Primary reliability bottleneck. *(2026-06-11: the 8–12 m spawn fix removed the immediate post-takeoff loss component. Trio decomposed the rest into two named causes, both fixed pending re-run: **T3 = max_vx speed ceiling** (3.5 vs target 3.5 → no closure; v6.27 = 4.5) and **T7 = tree-line false-positive capture during SEARCH** (phantom pulled the phase out of SEARCH, resetting the watchdog and corrupting recovery; detector gate v3.3). Re-run quantifies what inconsistency remains.)*

**P3 — Altitude error** *(was P2)*: historical +0.40m (best +0.23m). **2026-06-11 finding: target-motion-dependent** — measured −0.02 m on a static target (T1) vs +0.18 m at 1 m/s (T2); expect worse at higher speeds. pitch_comp=0.4 helps; consider world-frame altitude feedback.

**P4 — Infrastructure bugs B1–B10** *(was P3)*: fixed in Phase B; **sequencing gates + takeoff_both v10.0 + batch chaining VERIFIED in flight 2026-06-11 (3/3 clean launches).** Watchdog abort path not yet exercised in flight.

**P5 — YOLOv8s deployment** *(was P4)*: **DONE (M9.6 step 2)** — v8s deployed as `best.pt` (11.14M params, CUDA/RTX 4060), node v3.2 with `~device`/`~conf` + fps log, results restructured to `~/fyp/Results/`. **Live in-pipeline fps measured ≈ 19–20 (2026-06-11). n-vs-s table recorded (§11, 2026-06-12) — CLOSED.** Unblocks R characterization (step 3).

**P6 — YOLOv8m** *(was P5)*: **DROPPED** (n-vs-s comparison only).

**P7 — PPO v5.2 / Config 3** *(was P6)*: **PARKED** pending supervisor decision on the RL redesign. The deployed λ adaptation is fuzzy, not RL.

---

## 13. Spawn Zones Reference

World: Baylands terrain. Safe coordinate range: X=[-343,343], Y=[-269,269]. Model pose z offset handled in `random_spawn_target.py`.

**LOCKED PROTOCOL PARAMETER (2026-06-11): target spawn distance = 8–12 m in front of the chaser** (`random_spawn_target.py` v3.1; was 3–6 m — the old value caused instant target loss + 59 s SEARCH on the smoke run). This defines the initial conditions for **ALL** future runs; `~dist` / `START_DIST` overrides are for diagnostics only. **Changing this later = mandatory baseline reset.**

**Zone 7 geometry adjusted PRE-BASELINE then LOCKED (v3.2, 2026-06-11):** center (−45,−170) → **(−45,−130)** (≥42 m from the y=−180 land edge at full jitter; was as little as 2 m) and **spawn yaw constrained east (π/2 ± 40°, `CLEAR_VIEW_YAW`)** so the initial 8–12 m view has open ground/sky behind the target instead of the western tree line (T7 phantom-FP trigger area). The yaw draw count is unchanged, so zones 5/6/9 spawns stay identical per seed. No baseline data existed at the old geometry → no reset owed; from here this is part of the locked protocol.

**Zone whitelist for evaluation:** {5, 6, 7, 9} (enforced as `ALLOWED_ZONES` in code; Zone 6 needs yaw-away from tree line at spawn).
**Confirmed reliable zones (historical):** C(-80,50) and H(30,-80).

---

## 14. Architecture Diagram (deployed Config 2)

```
FPV Camera (iris_fpv_cam, forward-facing)
       ↓
YOLOv8 detection node (v3.1)
  → /drone_tracking/target_center  (cx, cy, area)
  → /drone_tracking/target_box     (cx, cy, w, h)
       ↓
Kalman filter node (M9.8)   [Config 2 only; Config 1 = raw]
  → /drone_tracking/filtered_target (smoothed cx, cy, area; z<0 = prediction)
  → /drone_tracking/kalman_velocity (vx, vy, valpha — SEARCH direction)
       ↓
[PPO agent node (v5.2)  — PARKED, Config 3 only]
       ↓
IBVS controller node (v6.25)
  ex = (cx-cx0)/cx0 - x* ;  ey (+ pitch comp) ;  ea = alpha - alpha_star
  vx from K_far/K_near·√(|ea|)·gain  (+ Kd_a feedforward) ;  vy,vz,wz = PID
  gain = gain_scale · λ_fallback(0.70)   ← fuzzy scheduler will replace 0.70
  → /mavros/setpoint_raw/local  (PositionTarget, FRAME_BODY_NED: vx,vy,vz,yaw_rate)
       ↓
PX4 / MAVROS (velocity → attitude → rate → motor mixing → Gazebo physics)
```

**State machine (IBVS):** TAKEOFF → SEARCH → APPROACH → HOLD ↔ (stale/lost) → SEARCH.

---

## 15. Key Learnings & Failure Modes

### IBVS
- **ea threshold is the #1 parameter.** ea < 0.010 for HOLD transition (was 0.005 → 0% HOLD).
- SEARCH separation > 30m + 40s re-acquisition gap is the primary HOLD% inconsistency driver.
- **Output is `/mavros/setpoint_raw/local` with FRAME_BODY_NED (body-frame velocity).** *Historical M7 lesson:* `setpoint_velocity/cmd_vel` interprets in WORLD frame — that earlier topic was abandoned in favour of `setpoint_raw/local` for body-frame control. (Do not reintroduce the cmd_vel/world-frame claim into the live architecture.)

### Kalman
- Q_pos=0.5 (not 3.0) — higher Q_pos causes cx/cy jitter.
- PIXEL_JUMP=180px avoids rejecting legitimate fast target motion.
- MAX_CONSECUTIVE_REJECTIONS=4 — faster fallback to predictions.
- Close-range collapse rule: pixel jump > PIXEL_JUMP AND alpha drops from > 3000 to < 900 → reject.
- velocity_damping=0.88 decays velocity during dropout (see SEARCH consistency note in §8).

### PPO (parked)
- Training alpha range MUST match real Gazebo (0.001–0.04); normalization (÷0.06) consistent train/deploy.
- CPU faster than GPU for small MLP. Never export full SB3 `.zip` — `.pth` state_dict only.
- Near-constant v5.2 output = domain gap, not architecture failure.

### YOLO / Detection
- pHash dedup threshold=20 removes ~44% of v4 frames while preserving diversity.
- Close-range instability: alpha collapses when target fills frame → Kalman poisoning → fixed with collapse rejection rule.
- Confidence threshold 0.35 for auto-labeling; `classes.txt` must stay `chmod 644`.

### Dual SITL
- MAV_SYS_ID mismatch silently breaks target MAVROS.
- Target spawn height ≥ 0.5m (physics glitch below).
- Never publish zero-velocity during WAITING (triggers premature OFFBOARD).
- Each drone reports local_position=(0,0,0) at its own spawn — use `/gazebo/model_states` for world-frame inter-drone distance.

### Git / Bash
- Always push from `~/Fyp_Drone_Detection_Tracking/`. Never `git init` inside `catkin_ws`.
- Recurring bash mistake: `source X && source Y rosrun` (space before `rosrun`). Must be `source X && source Y && rosrun`.
- Prefer complete file rewrites via `cat > file << 'PYEOF'` heredocs.

---

## 16. target_mover Behaviors (for report accuracy)

- **T1–T8 are velocity-commanded** (open-loop velocity setpoints integrated by PX4) → slight ground-path drift vs the ideal geometric path is expected and acceptable; commands are identical per seed.
- **Boundary repulsion is centered on the target's LOCAL spawn origin** (`pos_x/pos_y` from `/target/mavros/local_position/pose`), which confines each run near its zone — intended.
- **All randomness uses Python `random`, seeded via `~seed`** → T1–T8 fully reproducible per seed; **T9** is seed-deterministic in its random draws but closed-loop reactive (depends on live chaser pose), so it is a stress test, never an accept criterion.
- Phase flow: `WAITING → RISING → SETTLING → MOVING`. RISE_TO_Z=14m, Z_FLOOR=12m, Z_CEIL=24m.

---

## 17. Coding Preferences (Rawad's Style)

- **Strategy before code:** discuss logic and rationale before implementation.
- **Diagnostic-first debugging:** isolated tests before fixes.
- **Complete file rewrites** via `cat > file << 'PYEOF'` heredocs (not `cp`).
- **sed/Python patches** for targeted edits.
- **GitHub workflow:** Rawad pushes; Claude fetches via raw URL, modifies, returns sed commands or full files.
- **Back up before destructive ops** with timestamped naming.
- **Token-efficient responses** when iterating on code.
- **Private vs supervisor-facing:** distinguish slides (Dr. Sammour) from design rationale (private).
- **Watch the bash mistake:** `source X && source Y rosrun` → must be `&& rosrun`.

---

## 18. Key References

| Paper | Role in FYP |
|---|---|
| Sampedro et al., IROS 2018 | Founding paper of DRL+IBVS for multirotors. Defends hierarchical architecture. |
| Tuncer & Alpdemir, Software Impacts 2023 | Closest published drone-to-drone PPO precedent. |
| **Fu / Wu et al., Drones MDPI 2023** | **DRL tunes IBVS gain to prevent FOV loss. Citation for fuzzy gain scheduling of IBVS — the classical replacement for PPO's λ adopted at M9.6.** |
| Jin et al., IEEE TIE 2022 | Policy-gradient visibility-preserving servo policies. Structurally similar. |
| Pereira, MSc Técnico Lisboa 2021 | IBVS vs PBVS on monocular AR Drone — justifies monocular IBVS choice. |
| Abdessameud & Janabi-Sharifi, Automatica 2015 | Lyapunov stability proof for IBVS on VTOL UAV. |
| Chaumette & Hutchinson, IEEE RA Mag 2006/2007 | Foundational IBVS tutorial — core IBVS math. |
| Lin et al., Actuators 2022 | PPO most stable continuous-action RL for quadrotor. |
| Caffyn et al., Neurocomputing 2024 | Benchmarks 8 RL algorithms on quadcopter (PPO vs SAC vs TD3). |
| Yi et al., IEEE T-IV 2025 | Safe RL + visual servoing for quadrotor tracking unknown targets. |
| He et al., IEEE TIE 2024 | Hierarchical RL + VS with smooth subgoals — exact paradigm. |

**Novelty statement:**
> "This project builds on the drone-to-drone PPO tracking approach of Tuncer & Alpdemir (2023) by replacing end-to-end velocity regression with a hierarchical controller in which a higher-level policy supplies setpoints/gains to an IBVS inner loop, following the DRL-tunes-IBVS paradigm of Sampedro (2018), Jin (2022), and Wu/Fu (2023). At M9.6 the gain-adaptation layer is realized as Mamdani fuzzy gain scheduling (a classical, interpretable alternative to PPO). To the best of our knowledge, no published work combines YOLOv8 + Kalman + adaptive (fuzzy/PPO) gain + IBVS + PX4 on the drone-to-drone problem."
