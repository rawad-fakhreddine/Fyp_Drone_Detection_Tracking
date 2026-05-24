#!/usr/bin/env python3
"""
analyze_flight_log.py — M9.3
=============================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

Focused on IBVS, Fuzzy Logic, and YOLO Detection.

Sections:
  YOLO & DETECTION
    1.  Detection breakdown by phase
    2.  Alpha stats (raw vs filtered)
    3.  Frame-to-frame alpha/centroid jitter
    4.  Close-range events (alpha > 0.04)
    5.  Long PRED runs (Kalman bridging)
    6.  Stale-prediction events

  IBVS CONTROL
    7.  Velocity commands per phase
    8.  vx response to alpha-error (binned)
    9.  Wrong-direction events
    10. Backward-when-far events

  GEOMETRY
    11. True 3D separation (Gazebo world frame)
    12. Altitude tracking — world frame (no spawn bias)

  PPO → IBVS SETPOINT
    13. PPO alpha_star stats
    14. IBVS tracking of PPO setpoint (alpha vs alpha*)
    15. PPO response to target motion

  FUZZY LOGIC (TARGET EVASION)
    16. Escape quality (separation + closing rate + trajectory)
    17. OU heading perturbation summary
    18. Fuzzy output distribution (speed / omega / vz)
    19. Escape effectiveness (phi vs response + IBVS alpha trend)

  SUMMARY
    20. Tracking quality summary

Removed vs M8.2: Sections 11 (TAKEOFF tracking), 12 (target pos verbose),
                 17 (PPO lambda stats — merged into summary).
Fixed vs M8.2:   Sec 11 (was 13) uses true_dist_3d; Sec 12 (was 14) uses
                 world_alt_err; Sec 20 (was 15) adds tracking-time HOLD%.

Backward-compatible: missing columns are skipped gracefully.

Usage:
  python3 analyze_flight_log.py
  python3 analyze_flight_log.py --file path.csv
  python3 analyze_flight_log.py --window 100 200
"""

import math, sys, os, argparse
import numpy as np

ALPHA_STAR    = 0.0067
BWF_FAR_ALPHA = 0.003


def load_csv(path):
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def to_float(val, default=float('nan')):
    try:    return float(val)
    except: return default


def analyze(rows, label="full"):
    n = len(rows)
    if n == 0:
        print("  (no rows)"); return

    duration = to_float(rows[-1]['sim_time']) - to_float(rows[0]['sim_time'])
    rate     = n / duration if duration > 0 else 0
    print("=== Flight log: %s ===" % label)
    print("Rows: %d    Duration: %.1f s    Rate: %.1f Hz" % (n, duration, rate))

    # ── Column availability ────────────────────────────────────────────
    has_target    = 'target_px'      in rows[0]
    has_ppo       = 'ppo_alpha_star' in rows[0]
    has_fuzzy     = 'fuzzy_d'        in rows[0]
    has_true_dist = 'true_dist_3d'   in rows[0]
    has_world_alt = 'world_alt_err'  in rows[0]   # M9.3

    # ── Parse core arrays ─────────────────────────────────────────────
    phases    = [r['phase'] for r in rows]
    raw_alpha = np.array([to_float(r['raw_alpha']) for r in rows])
    flt_alpha = np.array([to_float(r['flt_alpha']) for r in rows])
    raw_det   = [r['raw_det'] for r in rows]
    flt_det   = [r['flt_det'] for r in rows]
    cmd_vx    = np.array([to_float(r['cmd_vx'])    for r in rows])
    cmd_vy    = np.array([to_float(r['cmd_vy'])    for r in rows])
    cmd_vz    = np.array([to_float(r['cmd_vz'])    for r in rows])
    cmd_frame = np.array([to_float(r.get('cmd_frame', 0)) for r in rows])
    raw_cx    = np.array([to_float(r['raw_cx'])    for r in rows])
    flt_cx    = np.array([to_float(r['flt_cx'])    for r in rows])
    ea        = np.array([to_float(r['ea'])         for r in rows])

    if has_target:
        target_px     = np.array([to_float(r['target_px'])     for r in rows])
        target_py     = np.array([to_float(r['target_py'])     for r in rows])
        target_pz     = np.array([to_float(r['target_pz'])     for r in rows])
        target_cmd_vx = np.array([to_float(r['target_cmd_vx']) for r in rows])
        target_cmd_vy = np.array([to_float(r['target_cmd_vy']) for r in rows])
        target_cmd_vz = np.array([to_float(r['target_cmd_vz']) for r in rows])

    if has_true_dist:
        true_dist_3d = np.array([to_float(r['true_dist_3d']) for r in rows])

    if has_world_alt:
        world_alt_err = np.array([to_float(r['world_alt_err']) for r in rows])

    if has_ppo:
        ppo_alpha_star = np.array([to_float(r['ppo_alpha_star']) for r in rows])
        ppo_lambda     = np.array([to_float(r['ppo_lambda'])     for r in rows])

    if has_fuzzy:
        fuzzy_d       = np.array([to_float(r['fuzzy_d'])           for r in rows])
        fuzzy_ddot    = np.array([to_float(r['fuzzy_ddot'])        for r in rows])
        fuzzy_phi     = np.array([to_float(r['fuzzy_phi'])         for r in rows])
        fuzzy_perturb = np.array([to_float(r['fuzzy_perturb_deg']) for r in rows])
        fuzzy_speed   = np.array([to_float(r['fuzzy_speed'])       for r in rows])
        fuzzy_omega   = np.array([to_float(r['fuzzy_omega'])       for r in rows])
        fuzzy_vz      = np.array([to_float(r['fuzzy_vz'])          for r in rows])

    unique_phases = sorted(set(phases))

    def pstats(name, arr, n_vals=None):
        """Print compact percentile stats line."""
        v = arr[~np.isnan(arr)]
        if len(v) == 0: return
        nn = n_vals if n_vals is not None else len(v)
        print("  %-16s n=%4d  mean=%+.5f  std=%.5f  "
              "p5=%+.5f  p50=%+.5f  p95=%+.5f" %
              (name, nn, np.mean(v), np.std(v),
               np.percentile(v, 5), np.median(v), np.percentile(v, 95)))

    # ════════════════════════════════════════════════════════════════════
    # 1. Detection breakdown
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 1. Detection breakdown by phase ===")
    print("  %-15s %5s %9s %9s  %9s %9s %9s" %
          ("phase", "n", "raw=REAL", "raw=NONE", "flt=REAL", "flt=PRED", "flt=NONE"))
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph]
        nn  = len(idx)
        rr  = sum(1 for i in idx if raw_det[i] == "REAL")
        rn  = sum(1 for i in idx if raw_det[i] == "NONE")
        fr  = sum(1 for i in idx if flt_det[i] == "REAL")
        fp  = sum(1 for i in idx if flt_det[i] == "PRED")
        fn  = sum(1 for i in idx if flt_det[i] == "NONE")
        print("  %-15s %5d %8.1f%% %8.1f%%  %8.1f%% %8.1f%% %8.1f%%" %
              (ph, nn,
               100*rr/nn if nn else 0, 100*rn/nn if nn else 0,
               100*fr/nn if nn else 0, 100*fp/nn if nn else 0,
               100*fn/nn if nn else 0))

    # ════════════════════════════════════════════════════════════════════
    # 2. Alpha stats per phase (REAL frames only)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 2. Alpha stats per phase (REAL frames only) ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph and raw_det[i] == "REAL"]
        if not idx: continue
        print("-- %s --" % ph)
        for name, arr in [("raw_alpha", raw_alpha), ("flt_alpha", flt_alpha)]:
            pstats(name, arr[idx])
        deltas = np.abs(flt_alpha[idx] - raw_alpha[idx])
        pstats("|flt - raw|", deltas[~np.isnan(deltas)])

    # ════════════════════════════════════════════════════════════════════
    # 3. Frame-to-frame jitter (consecutive REAL frames)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 3. Frame-to-frame jitter (consecutive REAL frames) ===")
    for ph in ["APPROACH", "HOLD"]:
        idx = [i for i, p in enumerate(phases) if p == ph and raw_det[i] == "REAL"]
        if len(idx) < 2: continue
        consec = [(idx[j], idx[j+1]) for j in range(len(idx)-1) if idx[j+1] == idx[j]+1]
        if not consec: continue
        print("-- %s --" % ph)
        d_ra = np.array([abs(raw_alpha[b] - raw_alpha[a]) for a, b in consec])
        d_fa = np.array([abs(flt_alpha[b] - flt_alpha[a]) for a, b in consec])
        d_rc = np.array([abs(raw_cx[b]    - raw_cx[a])    for a, b in consec])
        d_fc = np.array([abs(flt_cx[b]    - flt_cx[a])    for a, b in consec])
        for name, arr in [("|Δraw_alpha|", d_ra), ("|Δflt_alpha|", d_fa),
                          ("|Δraw_cx|",    d_rc), ("|Δflt_cx|",    d_fc)]:
            pstats(name, arr)

    # ════════════════════════════════════════════════════════════════════
    # 4. Close-range events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 4. Close-range events (flt_alpha > 0.04) ===")
    close_idx = [i for i in range(n) if flt_alpha[i] > 0.04]
    print("  Frames: %d (%.1f%% of total)" % (len(close_idx), 100*len(close_idx)/n if n else 0))

    # ════════════════════════════════════════════════════════════════════
    # 5. Long PRED runs (Kalman bridging)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 5. Long PRED runs (≥ 1.0 s) ===")
    pred_runs = []; cur_start = None
    for i in range(n):
        if flt_det[i] == "PRED":
            if cur_start is None: cur_start = i
        else:
            if cur_start is not None:
                dur_s = to_float(rows[i]['sim_time']) - to_float(rows[cur_start]['sim_time'])
                if dur_s >= 1.0: pred_runs.append((cur_start, i-1, dur_s))
                cur_start = None
    if not pred_runs: print("  (none)")
    for s, e, d in pred_runs:
        print("  rows %d-%d  %.1fs  phase=%s" % (s, e, d, phases[s]))

    # ════════════════════════════════════════════════════════════════════
    # 6. Stale-prediction events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 6. Stale-prediction events (PRED with stale alpha) ===")
    last_real_alpha = None; stale_count = 0
    for i in range(n):
        if raw_det[i] == "REAL":
            last_real_alpha = raw_alpha[i]
        elif flt_det[i] == "PRED" and last_real_alpha is not None:
            if last_real_alpha > 0 and abs(flt_alpha[i] - last_real_alpha) / last_real_alpha > 0.3:
                stale_count += 1
    print("  Found %d stale-prediction frame(s)" % stale_count)

    # ════════════════════════════════════════════════════════════════════
    # 7. Velocity commands per phase
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 7. Velocity commands per phase ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph]
        if not idx: continue
        vx = cmd_vx[idx]; vx = vx[~np.isnan(vx)]
        if not len(vx): continue
        print("-- %s --" % ph)
        pstats("cmd_vx", vx)
        if ph in ("APPROACH", "HOLD"):
            tot = len(vx) or 1
            print("  fwd=%.1f%%  bwd=%.1f%%  zero=%.1f%%" %
                  (100*np.sum(vx> 0.05)/tot,
                   100*np.sum(vx<-0.05)/tot,
                   100*np.sum(np.abs(vx)<=0.05)/tot))

    # ════════════════════════════════════════════════════════════════════
    # 8. vx response to alpha-error (APPROACH+HOLD, REAL)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 8. vx response to alpha-error (APPROACH+HOLD, REAL) ===")
    ah_real = [i for i in range(n) if phases[i] in ("APPROACH","HOLD")
               and flt_det[i]=="REAL" and cmd_frame[i]==8]
    bins = [
        ("err_a < -0.010 (very far)", lambda e: e < -0.010),
        ("-0.010 ≤ err_a < -0.005",  lambda e: -0.010 <= e < -0.005),
        ("-0.005 ≤ err_a < -0.002",  lambda e: -0.005 <= e < -0.002),
        ("-0.002 ≤ err_a ≤ +0.002",  lambda e: -0.002 <= e <= 0.002),
        ("+0.002 < err_a ≤ +0.005",  lambda e:  0.002 < e <= 0.005),
        ("+0.005 < err_a ≤ +0.010",  lambda e:  0.005 < e <= 0.010),
        ("err_a > +0.010 (close)",    lambda e:  e > 0.010),
    ]
    print("  %-35s %5s %10s %10s %7s" % ("bin", "n", "mean_vx", "p50_vx", "wrong%"))
    for lbl, cond in bins:
        idx = [i for i in ah_real if cond(ea[i])]
        if not idx: print("  %-35s %5d %10s" % (lbl, 0, "no data")); continue
        vx = cmd_vx[idx]
        wrong = sum(1 for i in idx if (ea[i]<-0.002 and cmd_vx[i]<-0.05) or
                                       (ea[i]> 0.002 and cmd_vx[i]> 0.05))
        print("  %-35s %5d %10.4f %10.4f %6.1f%%" %
              (lbl, len(idx), np.mean(vx), np.median(vx), 100*wrong/len(idx)))

    # ════════════════════════════════════════════════════════════════════
    # 9. Wrong-direction events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 9. Wrong-direction events (APPROACH+HOLD, REAL) ===")
    active = [i for i in range(n) if phases[i] in ("APPROACH","HOLD")
              and flt_det[i]=="REAL" and cmd_frame[i]==8]
    wrong  = [i for i in active if (ea[i]<-0.002 and cmd_vx[i]<-0.05) or
                                    (ea[i]> 0.002 and cmd_vx[i]> 0.05)]
    print("  %d / %d frames (%.1f%% of active REAL)" %
          (len(wrong), len(active), 100*len(wrong)/len(active) if active else 0))

    # ════════════════════════════════════════════════════════════════════
    # 10. Backward-when-far events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 10. Backward-when-far events (alpha<%.3f, vx<-0.05, ≥3 frames) ===" % BWF_FAR_ALPHA)
    bwf = []; run_start = None
    for i in range(n):
        if flt_alpha[i] < BWF_FAR_ALPHA and cmd_vx[i] < -0.05:
            if run_start is None: run_start = i
        else:
            if run_start is not None and (i - run_start) >= 3:
                bwf.append((run_start, i-1, i-run_start))
            run_start = None
    print("  Found %d event(s)%s" % (len(bwf), ":"))
    for s, e, l in bwf[:5]:
        print("    rows %d-%d (%d frames) phase=%s alpha=%.5f vx=%.3f" %
              (s, e, l, phases[s], flt_alpha[s], cmd_vx[s]))

    # ════════════════════════════════════════════════════════════════════
    # 11. True 3D separation (Gazebo world frame)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 11. True 3D separation per phase (Gazebo world frame) ===")
    if not has_true_dist:
        print("  (no true_dist_3d column — run with M8.2+ logger)")
    else:
        for ph in unique_phases:
            idx = [i for i, p in enumerate(phases) if p == ph and not np.isnan(true_dist_3d[i])]
            if not idx: continue
            d = true_dist_3d[idx]
            print("  %-12s n=%4d  mean=%.2fm  std=%.2f  p5=%.2f  p50=%.2f  p95=%.2f  max=%.2fm" %
                  (ph, len(d), np.mean(d), np.std(d),
                   np.percentile(d,5), np.median(d), np.percentile(d,95), np.max(d)))

    # ════════════════════════════════════════════════════════════════════
    # 12. Altitude tracking — world frame (no spawn bias)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 12. Altitude tracking — world frame (HOLD phase) ===")
    hold_idx = [i for i in range(n) if phases[i] == "HOLD"]
    if has_world_alt:
        wae = world_alt_err[hold_idx]; wae = wae[~np.isnan(wae)]
        if len(wae) > 0:
            print("  world_alt_err (chaser_wz - target_wz):")
            print("  mean=%+.2fm  std=%.2f  p5=%+.2f  p50=%+.2f  p95=%+.2f  "
                  "min=%+.2f  max=%+.2fm" %
                  (np.mean(wae), np.std(wae),
                   np.percentile(wae,5), np.median(wae), np.percentile(wae,95),
                   np.min(wae), np.max(wae)))
            ok = np.sum(np.abs(wae) < 1.0)
            print("  Within ±1m: %d / %d frames (%.1f%%)" %
                  (ok, len(wae), 100*ok/len(wae)))
        else:
            print("  (no world_alt_err data in HOLD)")
    elif has_target:
        # Fallback: MAVROS cross-frame (has ~0.5m spawn bias — note this)
        pos_z = np.array([to_float(r['pos_z']) for r in rows])
        alt_err_mavros = pos_z[hold_idx] - target_pz[hold_idx]
        alt_err_mavros = alt_err_mavros[~np.isnan(alt_err_mavros)]
        if len(alt_err_mavros) > 0:
            print("  alt_err MAVROS (pos_z - target_pz, ~0.5m spawn bias):")
            print("  mean=%+.2fm  std=%.2f  p5=%+.2f  p50=%+.2f  p95=%+.2f" %
                  (np.mean(alt_err_mavros), np.std(alt_err_mavros),
                   np.percentile(alt_err_mavros,5), np.median(alt_err_mavros),
                   np.percentile(alt_err_mavros,95)))
            print("  ⚠ Upgrade to M9.3 logger for world_alt_err (no spawn bias)")
    else:
        print("  (no altitude data available)")

    # ════════════════════════════════════════════════════════════════════
    # 13. PPO alpha_star stats
    # ════════════════════════════════════════════════════════════════════
    if not has_ppo:
        print("\n=== 13-15. (skipped — no PPO columns) ===")
    else:
        valid_ppo = ~np.isnan(ppo_alpha_star)

        print("\n=== 13. PPO alpha_star stats per phase ===")
        print("  Expected: far → high α* (approach), hold → ~0.007, close → low α* (retreat)")
        for ph in unique_phases:
            idx = [i for i, p in enumerate(phases) if p == ph and valid_ppo[i]]
            if not idx: continue
            vals = ppo_alpha_star[idx]; vals = vals[~np.isnan(vals)]
            if not len(vals): continue
            print("  %-12s n=%4d  mean=%.5f  std=%.5f  p5=%.5f  p50=%.5f  p95=%.5f" %
                  (ph, len(vals), np.mean(vals), np.std(vals),
                   np.percentile(vals,5), np.median(vals), np.percentile(vals,95)))

        # ════════════════════════════════════════════════════════════════
        # 14. IBVS tracking of PPO setpoint
        # ════════════════════════════════════════════════════════════════
        print("\n=== 14. IBVS tracking of PPO setpoint (APPROACH+HOLD, REAL) ===")
        ah_ppo = [i for i in range(n) if phases[i] in ("APPROACH","HOLD")
                  and flt_det[i]=="REAL" and valid_ppo[i]]
        if len(ah_ppo) > 5:
            err_sp = (flt_alpha[ah_ppo] - ppo_alpha_star[ah_ppo])
            err_sp = err_sp[~np.isnan(err_sp)]
            if len(err_sp):
                print("  err (flt_alpha - alpha*): mean=%+.5f  std=%.5f  "
                      "p5=%+.5f  p50=%+.5f  p95=%+.5f" %
                      (np.mean(err_sp), np.std(err_sp),
                       np.percentile(err_sp,5), np.median(err_sp),
                       np.percentile(err_sp,95)))
                alpha_bins = [
                    ("alpha<0.004 (far)",    lambda a: a < 0.004),
                    ("0.004-0.008 (mid)",    lambda a: 0.004 <= a < 0.008),
                    ("0.008-0.012 (hold)",   lambda a: 0.008 <= a < 0.012),
                    ("alpha>0.012 (close)",  lambda a: a > 0.012),
                ]
                for lbl, cond in alpha_bins:
                    sub = [i for i in ah_ppo if cond(flt_alpha[i]) and not np.isnan(ppo_alpha_star[i])]
                    if len(sub) < 3: continue
                    print("    %-22s n=%4d  mean_α=%.5f  mean_α*=%.5f  mean_λ=%.3f" %
                          (lbl, len(sub), np.mean(flt_alpha[sub]),
                           np.mean(ppo_alpha_star[sub]), np.mean(ppo_lambda[sub])))
        else:
            print("  (insufficient data)")

        # ════════════════════════════════════════════════════════════════
        # 15. PPO response to target motion
        # ════════════════════════════════════════════════════════════════
        print("\n=== 15. PPO response to target motion (HOLD) ===")
        print("  Expected: faster target → higher alpha* and lambda")
        if has_target:
            hpt = [i for i in range(n) if phases[i]=="HOLD" and valid_ppo[i]
                   and not np.isnan(target_cmd_vx[i]) and not np.isnan(target_cmd_vy[i])]
            if len(hpt) > 10:
                t_spd = np.sqrt(target_cmd_vx[hpt]**2 + target_cmd_vy[hpt]**2)
                spd_bins = [
                    ("static (<0.1 m/s)",     lambda s: s < 0.1),
                    ("slow (0.1-0.5 m/s)",    lambda s: 0.1 <= s < 0.5),
                    ("fast (>0.5 m/s)",        lambda s: s >= 0.5),
                ]
                print("  %-22s %5s %12s %10s" % ("target speed", "n", "mean_α*", "mean_λ"))
                for lbl, cond in spd_bins:
                    mask = np.array([cond(s) for s in t_spd])
                    si   = np.array(hpt)[mask]
                    if len(si) < 3: print("  %-22s %5d %12s" % (lbl, len(si), "no data")); continue
                    print("  %-22s %5d %12.5f %10.3f" %
                          (lbl, len(si), np.mean(ppo_alpha_star[si]), np.mean(ppo_lambda[si])))
            else:
                print("  (insufficient data)")
        else:
            print("  (no target columns)")

        # PPO summary inline
        hold_ppo = [i for i in range(n) if phases[i]=="HOLD" and valid_ppo[i]]
        if hold_ppo:
            ma = np.mean(ppo_alpha_star[hold_ppo]); sa = np.std(ppo_alpha_star[hold_ppo])
            ml = np.mean(ppo_lambda[hold_ppo]);     sl = np.std(ppo_lambda[hold_ppo])
            print("\n  PPO HOLD summary: α*=%.5f±%.5f  λ=%.3f±%.3f" % (ma, sa, ml, sl))
            print("  α* variance: %s" % ("✓ OK (std=%.5f)" % sa if sa >= 0.0005
                                          else "⚠ low (PPO may be constant)"))
            print("  α* range:    %s" % ("✓ in [0.003, 0.020]" if 0.003 <= ma <= 0.020
                                          else "⚠ outside expected range"))

    # ════════════════════════════════════════════════════════════════════
    # 16. Escape quality — Fuzzy
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 16. Escape quality — Fuzzy target motion (HOLD) ===")
    if not has_fuzzy:
        print("  (no fuzzy columns)")
    else:
        hold_f = [i for i in range(n) if phases[i]=="HOLD" and not np.isnan(fuzzy_d[i])]
        if len(hold_f) < 5:
            print("  (insufficient HOLD frames with fuzzy data)")
        else:
            fd = fuzzy_d[hold_f]; fv = fuzzy_ddot[hold_f]
            print("  Separation (fuzzy_d):")
            print("    n=%d  mean=%.2fm  std=%.2f  p5=%.2f  p50=%.2f  p95=%.2f  max=%.2fm" %
                  (len(fd), np.mean(fd), np.std(fd), np.percentile(fd,5),
                   np.median(fd), np.percentile(fd,95), np.max(fd)))
            pct_c = 100*np.sum(fv >  0.05)/len(fv)
            pct_s = 100*np.sum(np.abs(fv)<=0.05)/len(fv)
            pct_r = 100*np.sum(fv < -0.05)/len(fv)
            print("  Closing rate: approaching=%.1f%%  stable=%.1f%%  receding=%.1f%%  "
                  "mean=%+.3f m/s" % (pct_c, pct_s, pct_r, np.mean(fv)))

            if has_target:
                tpx = target_px[hold_f]; tpy = target_py[hold_f]; tpz = target_pz[hold_f]
                vm  = ~np.isnan(tpx)
                if np.sum(vm) > 5:
                    xr = np.max(tpx[vm])-np.min(tpx[vm])
                    yr = np.max(tpy[vm])-np.min(tpy[vm])
                    zr = np.max(tpz[vm])-np.min(tpz[vm])
                    net = math.sqrt((tpx[vm][-1]-tpx[vm][0])**2 + (tpy[vm][-1]-tpy[vm][0])**2)
                    print("  Trajectory coverage: X=%.1fm  Y=%.1fm  Z=%.1fm  net XY=%.1fm" %
                          (xr, yr, zr, net))

    # ════════════════════════════════════════════════════════════════════
    # 17. OU perturbation summary
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 17. OU heading perturbation — HOLD ===")
    if not has_fuzzy:
        print("  (no fuzzy columns)")
    else:
        hold_p = [i for i in range(n) if phases[i]=="HOLD" and not np.isnan(fuzzy_perturb[i])]
        if len(hold_p) < 5:
            print("  (insufficient data)")
        else:
            fp = fuzzy_perturb[hold_p]
            pct_lim = 100*np.sum(np.abs(fp) >= 89.0)/len(fp)
            print("  n=%d  mean=%.1f°  std=%.1f°  |mean|=%.1f°  max=%.1f°" %
                  (len(fp), np.mean(fp), np.std(fp), np.mean(np.abs(fp)), np.max(np.abs(fp))))
            print("  At ±90° limit: %.1f%% %s" %
                  (pct_lim, "⚠ high clamping" if pct_lim > 15 else "✓ OK"))
            # Correlation with distance
            fd_p = fuzzy_d[hold_p]; valid = ~(np.isnan(fp)|np.isnan(fd_p))
            if np.sum(valid) > 10:
                r = np.corrcoef(np.abs(fp[valid]), fd_p[valid])[0, 1]
                print("  |perturb| vs distance: r=%.3f %s" %
                      (r, "✓ grows with d" if r > 0.2 else "⚠ weak distance-dependence"))

    # ════════════════════════════════════════════════════════════════════
    # 18. Fuzzy output distribution (HOLD)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 18. Fuzzy output distribution (HOLD) ===")
    if not has_fuzzy:
        print("  (no fuzzy columns)")
    else:
        hold_s = [i for i in range(n) if phases[i]=="HOLD" and not np.isnan(fuzzy_speed[i])]
        if hold_s:
            sp = fuzzy_speed[hold_s]; om = fuzzy_omega[hold_s]; vz_ = fuzzy_vz[hold_s]
            print("  speed=%.3f±%.3f m/s   omega=%.3f±%.3f rad/s   vz=%.3f±%.3f m/s" %
                  (np.mean(sp), np.std(sp), np.mean(om), np.std(om), np.mean(vz_), np.std(vz_)))
            print("  Speed regimes:  SLOW<0.5=%.1f%%  0.5-0.75=%.1f%%  0.75-1.0=%.1f%%  SPRINT>1.0=%.1f%%" %
                  (100*np.sum(sp<0.50)/len(sp), 100*np.sum((sp>=0.50)&(sp<0.75))/len(sp),
                   100*np.sum((sp>=0.75)&(sp<1.00))/len(sp), 100*np.sum(sp>=1.00)/len(sp)))
            print("  Vz regimes:     FAST_DIVE<-0.3=%.1f%%  HOLD±0.08=%.1f%%  FAST_CLIMB>0.3=%.1f%%" %
                  (100*np.sum(vz_<-0.30)/len(vz_), 100*np.sum(np.abs(vz_)<=0.08)/len(vz_),
                   100*np.sum(vz_>0.30)/len(vz_)))

    # ════════════════════════════════════════════════════════════════════
    # 19. Escape effectiveness — phi vs response + IBVS alpha trend
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 19. Escape effectiveness (HOLD) ===")
    if not has_fuzzy:
        print("  (no fuzzy columns)")
    else:
        hold_ef = [i for i in range(n) if phases[i]=="HOLD"
                   and not np.isnan(fuzzy_phi[i]) and not np.isnan(fuzzy_speed[i])]
        if len(hold_ef) < 10:
            print("  (insufficient data)")
        else:
            phi  = fuzzy_phi[hold_ef]; spd = fuzzy_speed[hold_ef]
            omg  = fuzzy_omega[hold_ef]; ddt = fuzzy_ddot[hold_ef]
            print("  FOV exposure (phi) → fuzzy response:")
            print("  %-22s %5s %10s %10s %10s" % ("phi bin","n","mean_spd","mean_ω","mean_ḋ"))
            for lbl, cond in [
                ("OUTSIDE  (phi<0.2)", lambda p: p <  0.2),
                ("EDGE     (0.2-0.5)", lambda p: 0.2<=p<0.5),
                ("PARTIAL  (0.5-0.8)", lambda p: 0.5<=p<0.8),
                ("CENTERED (phi>0.8)", lambda p: p >= 0.8),
            ]:
                m = np.array([cond(p) for p in phi])
                if np.sum(m) < 3: continue
                print("  %-22s %5d %10.3f %10.3f %+10.3f" %
                      (lbl, np.sum(m), np.mean(spd[m]), np.mean(omg[m]), np.mean(ddt[m])))

            if has_ppo:
                hold_ep = [i for i in hold_ef if not np.isnan(flt_alpha[i])]
                if len(hold_ep) > 10:
                    ddt_ = fuzzy_ddot[hold_ep]; fa_ = flt_alpha[hold_ep]
                    print("\n  IBVS alpha vs target escape direction:")
                    for lbl, mask in [
                        ("Chaser CLOSING  (ḋ>+0.05)", ddt_> 0.05),
                        ("Chaser RECEDING (ḋ<-0.05)", ddt_<-0.05),
                    ]:
                        if np.sum(mask) > 3:
                            print("    %-30s n=%d  mean_alpha=%.5f" %
                                  (lbl, np.sum(mask), np.mean(fa_[mask])))

            n_hf = len(hold_ef)
            pct_esc = 100*np.sum(fuzzy_ddot[hold_ef]<-0.05)/n_hf
            pct_clo = 100*np.sum(fuzzy_ddot[hold_ef]> 0.05)/n_hf
            print("\n  Escape verdict: %.1f%% receding  %.1f%% closing  → %s" %
                  (pct_esc, pct_clo,
                   "✓ target NET-ESCAPING" if pct_esc > pct_clo else "⚠ chaser NET-CLOSING"))

    # ════════════════════════════════════════════════════════════════════
    # 20. Tracking quality summary
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 20. Tracking quality summary ===")
    hold_rows   = [i for i in range(n) if phases[i]=="HOLD"]
    track_rows  = [i for i in range(n) if phases[i] in ("APPROACH","HOLD","SEARCH")]
    search_rows = [i for i in range(n) if phases[i]=="SEARCH"]

    hold_time   = len(hold_rows)   / rate if rate > 0 else 0
    track_time  = len(track_rows)  / rate if rate > 0 else 0
    search_time = len(search_rows) / rate if rate > 0 else 0

    det_real = sum(1 for i in hold_rows if raw_det[i]=="REAL")
    det_pct  = 100*det_real/len(hold_rows) if hold_rows else 0
    mean_alpha_hold = np.nanmean(flt_alpha[hold_rows]) if hold_rows else float('nan')

    print("  Flight duration:        %.1f s" % duration)
    print("  Tracking time:          %.1f s (APPROACH+HOLD+SEARCH)" % track_time)
    print("  HOLD time:              %.1f s (%.0f%% total | %.0f%% tracking)" %
          (hold_time,
           100*hold_time/duration     if duration    > 0 else 0,
           100*hold_time/track_time   if track_time  > 0 else 0))
    print("  SEARCH time:            %.1f s (%.0f%% total)" %
          (search_time, 100*search_time/duration if duration > 0 else 0))
    print("  Detection rate (HOLD):  %.1f%%" % det_pct)
    print("  Mean alpha (HOLD):      %.5f  (target=%.5f)" % (mean_alpha_hold, ALPHA_STAR))

    if has_true_dist and hold_rows:
        d = true_dist_3d[hold_rows]; d = d[~np.isnan(d)]
        if len(d):
            print("  Mean separation (HOLD): %.2fm  (p5=%.1f  p95=%.1f)" %
                  (np.mean(d), np.percentile(d,5), np.percentile(d,95)))
            print("  Separation <2m:  %d frames (%.1f%%)" %
                  (np.sum(d<2), 100*np.sum(d<2)/len(d)))
            print("  Separation >8m:  %d frames (%.1f%%)" %
                  (np.sum(d>8), 100*np.sum(d>8)/len(d)))

    if has_fuzzy and hold_rows:
        fd_h = fuzzy_d[[i for i in hold_rows if not np.isnan(fuzzy_d[i])]]
        if len(fd_h):
            pct_esc = 100*np.sum(fuzzy_ddot[[i for i in hold_rows if not np.isnan(fuzzy_ddot[i])]]<-0.05) / \
                      len([i for i in hold_rows if not np.isnan(fuzzy_ddot[i])]) if \
                      len([i for i in hold_rows if not np.isnan(fuzzy_ddot[i])]) else 0
            print("  Target escaping (HOLD): %.1f%%" % pct_esc)

    if has_world_alt and hold_rows:
        wae = world_alt_err[hold_rows]; wae = wae[~np.isnan(wae)]
        if len(wae):
            print("  Alt error world (HOLD): mean=%+.2fm  std=%.2fm" %
                  (np.mean(wae), np.std(wae)))

    if has_ppo and hold_rows:
        hp = [i for i in hold_rows if not np.isnan(ppo_alpha_star[i])]
        if hp:
            print("  PPO α* (HOLD):          %.5f±%.5f   λ=%.3f±%.3f" %
                  (np.mean(ppo_alpha_star[hp]), np.std(ppo_alpha_star[hp]),
                   np.mean(ppo_lambda[hp]), np.std(ppo_lambda[hp])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=os.path.expanduser('~/flight_log_latest.csv'))
    parser.add_argument('--window', nargs=2, type=int, default=None,
                        help='Row range: --window START END')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("ERROR: %s not found" % args.file); sys.exit(1)

    rows  = load_csv(args.file)
    label = os.path.basename(args.file)
    if args.window:
        s, e  = args.window
        rows  = rows[s:e]
        label += " [rows %d-%d]" % (s, e)

    analyze(rows, label)


if __name__ == '__main__':
    main()