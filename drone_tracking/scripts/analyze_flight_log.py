#!/usr/bin/env python3
"""
analyze_flight_log.py — M7.3 dual-PX4 edition
===============================================

All original 11 sections preserved from M7.2.
New sections 12-15 for dual-PX4 target data:

  12. Target position & velocity stats per phase
  13. 3D distance (chaser-to-target) stats per phase
  14. Chaser-vs-target altitude tracking
  15. Tracking quality summary (one-line per-phase report)

Backward-compatible: if target columns are missing (old CSV), sections 12-15 are skipped.

Usage:
  python3 analyze_flight_log.py                  # latest log
  python3 analyze_flight_log.py --window 100 200 # rows 100-200 only
  python3 analyze_flight_log.py --file path.csv  # specific file
"""

import sys, os, argparse
import numpy as np

ALPHA_STAR = 0.0067
BWF_FAR_ALPHA = 0.003


def load_csv(path):
    import csv
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_float(val, default=float('nan')):
    try:
        v = float(val)
        return v
    except (ValueError, TypeError):
        return default


def analyze(rows, label="full"):
    n = len(rows)
    if n == 0:
        print("  (no rows)")
        return

    duration = to_float(rows[-1]['sim_time']) - to_float(rows[0]['sim_time'])
    rate = n / duration if duration > 0 else 0
    print("=== Flight log: %s ===" % label)
    print("Rows: %d    Duration: %.1f s    Effective rate: %.1f Hz" % (n, duration, rate))

    # ── Check for target columns (M7.3) ───────────────────────────────
    has_target = 'target_px' in rows[0] and 'dist_3d' in rows[0]

    # ── Parse arrays ──────────────────────────────────────────────────
    phases = [r['phase'] for r in rows]
    raw_alpha = np.array([to_float(r['raw_alpha']) for r in rows])
    flt_alpha = np.array([to_float(r['flt_alpha']) for r in rows])
    raw_det = [r['raw_det'] for r in rows]
    flt_det = [r['flt_det'] for r in rows]
    cmd_vx = np.array([to_float(r['cmd_vx']) for r in rows])
    cmd_vy = np.array([to_float(r['cmd_vy']) for r in rows])
    cmd_vz = np.array([to_float(r['cmd_vz']) for r in rows])
    cmd_frame = np.array([to_float(r.get('cmd_frame', 0)) for r in rows])
    raw_cx = np.array([to_float(r['raw_cx']) for r in rows])
    flt_cx = np.array([to_float(r['flt_cx']) for r in rows])
    pos_x = np.array([to_float(r['pos_x']) for r in rows])
    pos_y = np.array([to_float(r['pos_y']) for r in rows])
    pos_z = np.array([to_float(r['pos_z']) for r in rows])
    ea = np.array([to_float(r['ea']) for r in rows])

    if has_target:
        target_px = np.array([to_float(r['target_px']) for r in rows])
        target_py = np.array([to_float(r['target_py']) for r in rows])
        target_pz = np.array([to_float(r['target_pz']) for r in rows])
        target_cmd_vx = np.array([to_float(r['target_cmd_vx']) for r in rows])
        target_cmd_vy = np.array([to_float(r['target_cmd_vy']) for r in rows])
        target_cmd_vz = np.array([to_float(r['target_cmd_vz']) for r in rows])
        dist_3d = np.array([to_float(r['dist_3d']) for r in rows])

    unique_phases = sorted(set(phases))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 1: Detection breakdown by phase
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 1. Detection breakdown by phase ===")
    print("  %-15s %5s %9s %9s  %9s %9s %9s" %
          ("phase", "n", "raw=REAL", "raw=NONE", "flt=REAL", "flt=PRED", "flt=NONE"))
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph]
        nn = len(idx)
        rr = sum(1 for i in idx if raw_det[i] == "REAL")
        rn = sum(1 for i in idx if raw_det[i] == "NONE")
        fr = sum(1 for i in idx if flt_det[i] == "REAL")
        fp = sum(1 for i in idx if flt_det[i] == "PRED")
        fn = sum(1 for i in idx if flt_det[i] == "NONE")
        print("  %-15s %5d %8.1f%% %8.1f%%  %8.1f%% %8.1f%% %8.1f%%" %
              (ph, nn,
               100*rr/nn if nn else 0, 100*rn/nn if nn else 0,
               100*fr/nn if nn else 0, 100*fp/nn if nn else 0, 100*fn/nn if nn else 0))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 2: Alpha stats per phase
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 2. Alpha stats per phase (raw=REAL frames only) ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph and raw_det[i] == "REAL"]
        if not idx:
            continue
        print("-- %s --" % ph)
        for name, arr in [("raw_alpha", raw_alpha), ("flt_alpha", flt_alpha)]:
            vals = arr[idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            print("  %-15s n=%4d  mean=%+.5f  std=%.5f  min=%+.5f  p5=%+.5f  "
                  "p50=%+.5f  p95=%+.5f  max=%+.5f" %
                  (name, len(vals), np.mean(vals), np.std(vals), np.min(vals),
                   np.percentile(vals, 5), np.median(vals),
                   np.percentile(vals, 95), np.max(vals)))
        # filter-raw delta
        deltas = np.abs(flt_alpha[idx] - raw_alpha[idx])
        deltas = deltas[~np.isnan(deltas)]
        if len(deltas):
            print("  %-15s n=%4d  mean=%+.5f  std=%.5f  min=%+.5f  p5=%+.5f  "
                  "p50=%+.5f  p95=%+.5f  max=%+.5f" %
                  ("|flt - raw|", len(deltas), np.mean(deltas), np.std(deltas),
                   np.min(deltas), np.percentile(deltas, 5), np.median(deltas),
                   np.percentile(deltas, 95), np.max(deltas)))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3: Frame-to-frame jumps
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 3. Frame-to-frame jumps (consecutive REAL frames) ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph and raw_det[i] == "REAL"]
        if len(idx) < 2:
            continue
        print("-- %s --" % ph)
        consec = [(idx[j], idx[j+1]) for j in range(len(idx)-1) if idx[j+1] == idx[j]+1]
        if not consec:
            continue
        d_ra = np.array([abs(raw_alpha[b] - raw_alpha[a]) for a, b in consec])
        d_fa = np.array([abs(flt_alpha[b] - flt_alpha[a]) for a, b in consec])
        d_rc = np.array([abs(raw_cx[b] - raw_cx[a]) for a, b in consec])
        d_fc = np.array([abs(flt_cx[b] - flt_cx[a]) for a, b in consec])
        for name, arr in [
            ("|d raw_alpha|", d_ra), ("|d flt_alpha|", d_fa),
            ("|d raw_cx|", d_rc), ("|d flt_cx|", d_fc)
        ]:
            print("  %-15s n=%4d  mean=%+.5f  std=%.5f  min=%+.5f  p5=%+.5f  "
                  "p50=%+.5f  p95=%+.5f  max=%+.5f" %
                  (name, len(arr), np.mean(arr), np.std(arr), np.min(arr),
                   np.percentile(arr, 5), np.median(arr),
                   np.percentile(arr, 95), np.max(arr)))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 4: Close-range events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 4. Close-range events (flt_alpha > 0.04) ===")
    close_idx = [i for i in range(n) if flt_alpha[i] > 0.04]
    print("  Frames: %d (%.1f%% of total)" % (len(close_idx), 100*len(close_idx)/n if n else 0))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 5: Long PRED runs
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 5. Long PRED runs (\u2265 1.0 s) ===")
    pred_runs = []
    cur_start = None
    for i in range(n):
        if flt_det[i] == "PRED":
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                dur_s = (to_float(rows[i]['sim_time']) - to_float(rows[cur_start]['sim_time']))
                if dur_s >= 1.0:
                    pred_runs.append((cur_start, i-1, dur_s))
                cur_start = None
    if not pred_runs:
        print("  (none)")
    for start, end, dur in pred_runs:
        print("  rows %d-%d  %.1fs  phase=%s" % (start, end, dur, phases[start]))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 6: Velocity command stats per phase
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 6. Velocity command stats per phase ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph]
        if not idx:
            continue
        print("-- %s --" % ph)
        vx_vals = cmd_vx[idx]
        vx_vals = vx_vals[~np.isnan(vx_vals)]
        if len(vx_vals):
            print("  %-15s n=%4d  mean=%+.3f  std=%.3f  min=%+.3f  p5=%+.3f  "
                  "p50=%+.3f  p95=%+.3f  max=%+.3f" %
                  ("cmd_vx", len(vx_vals), np.mean(vx_vals), np.std(vx_vals),
                   np.min(vx_vals), np.percentile(vx_vals, 5), np.median(vx_vals),
                   np.percentile(vx_vals, 95), np.max(vx_vals)))
        if ph in ("APPROACH", "HOLD"):
            fwd = sum(1 for v in vx_vals if v > 0.05)
            bwd = sum(1 for v in vx_vals if v < -0.05)
            nz = sum(1 for v in vx_vals if abs(v) <= 0.05)
            total = len(vx_vals) if len(vx_vals) else 1
            print("  vx direction:  forward=%.1f%%  backward=%.1f%%  near-zero=%.1f%%" %
                  (100*fwd/total, 100*bwd/total, 100*nz/total))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 7: vx response to alpha-error
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 7. vx response to alpha-error (APPROACH+HOLD, REAL only) ===")
    ah_real = [i for i in range(n) if phases[i] in ("APPROACH", "HOLD")
               and flt_det[i] == "REAL" and cmd_frame[i] == 8]
    bins = [
        ("err_a < -0.010 (very far)", lambda e: e < -0.010),
        ("-0.010 \u2264 err_a < -0.005", lambda e: -0.010 <= e < -0.005),
        ("-0.005 \u2264 err_a < -0.002", lambda e: -0.005 <= e < -0.002),
        ("-0.002 \u2264 err_a \u2264 +0.002", lambda e: -0.002 <= e <= 0.002),
        ("+0.002 < err_a \u2264 +0.005", lambda e: 0.002 < e <= 0.005),
        ("+0.005 < err_a \u2264 +0.010", lambda e: 0.005 < e <= 0.010),
        ("err_a > +0.010 (very close)", lambda e: e > 0.010),
    ]
    print("  %-37s %5s %10s %10s %7s" % ("bin", "n", "mean_vx", "p50_vx", "wrong%"))
    for label, cond in bins:
        idx = [i for i in ah_real if cond(ea[i])]
        if not idx:
            print("  %-37s %5d %10s" % (label, 0, "no data"))
            continue
        vx = cmd_vx[idx]
        wrong = sum(1 for i in idx if (ea[i] < -0.002 and cmd_vx[i] < -0.05) or
                    (ea[i] > 0.002 and cmd_vx[i] > 0.05))
        print("  %-37s %5d %10.4f %10.4f %6.1f%%" %
              (label, len(idx), np.mean(vx), np.median(vx), 100*wrong/len(idx)))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 8: Wrong-direction events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 8. Wrong-direction events (active REAL frames only) ===")
    active_real = [i for i in range(n) if phases[i] in ("APPROACH", "HOLD")
                   and flt_det[i] == "REAL" and cmd_frame[i] == 8]
    wrong = [i for i in active_real if
             (ea[i] < -0.002 and cmd_vx[i] < -0.05) or
             (ea[i] > 0.002 and cmd_vx[i] > 0.05)]
    print("  %d frames (%.1f%% of active REAL)" %
          (len(wrong), 100*len(wrong)/len(active_real) if active_real else 0))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 9: Backward-when-far events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 9. Backward-when-far events ===")
    print("    flt_alpha < %.3f (target far)" % BWF_FAR_ALPHA)
    print("    cmd_vx    < -0.05 (commanding backward)")
    print("    minimum 3 consecutive frames")
    bwf_events = []
    run_start = None
    for i in range(n):
        if flt_alpha[i] < BWF_FAR_ALPHA and cmd_vx[i] < -0.05:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= 3:
                bwf_events.append((run_start, i-1, i - run_start))
            run_start = None
    print("\n  Found %d event(s):" % len(bwf_events))
    for start, end, length in bwf_events[:5]:
        print("    rows %d-%d (%d frames) phase=%s alpha=%.5f vx=%.3f" %
              (start, end, length, phases[start], flt_alpha[start], cmd_vx[start]))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 10: Stale-prediction events
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 10. Stale-prediction events (PRED with stale alpha) ===")
    last_real_alpha = None
    stale_count = 0
    for i in range(n):
        if raw_det[i] == "REAL":
            last_real_alpha = raw_alpha[i]
        elif flt_det[i] == "PRED" and last_real_alpha is not None:
            if last_real_alpha > 0 and abs(flt_alpha[i] - last_real_alpha) / last_real_alpha > 0.3:
                stale_count += 1
    print("  Found %d stale-prediction frame(s)" % stale_count)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 11: TAKEOFF tracking
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 11. TAKEOFF tracking (commanded vs. actual world velocity) ===")
    tk_idx = [i for i in range(n) if phases[i] == "TAKEOFF" and cmd_frame[i] == 1]
    if len(tk_idx) < 3:
        print("  (insufficient TAKEOFF frames with cmd_frame=1)")
    else:
        print("    For TAKEOFF frames with cmd_frame=1 (FRAME_LOCAL_NED).")
        dt_arr = []
        for j in range(1, len(tk_idx)):
            i0, i1 = tk_idx[j-1], tk_idx[j]
            t0 = to_float(rows[i0]['sim_time'])
            t1 = to_float(rows[i1]['sim_time'])
            dt_val = t1 - t0
            if dt_val > 0.01:
                dt_arr.append((i1, dt_val))
        if dt_arr:
            for axis, cmd_arr, pos_arr in [
                ("cmd_vx", cmd_vx, pos_x), ("cmd_vy", cmd_vy, pos_y), ("cmd_vz", cmd_vz, pos_z)
            ]:
                cmd_vals, actual_vals = [], []
                for idx_val, dt_val in dt_arr:
                    cmd_vals.append(cmd_arr[idx_val])
                    prev_idx = tk_idx[tk_idx.index(idx_val)-1] if idx_val in tk_idx else idx_val-1
                    actual_vals.append((pos_arr[idx_val] - pos_arr[prev_idx]) / dt_val)
                cmd_m = np.mean(cmd_vals)
                act_m = np.mean(actual_vals)
                err_m = cmd_m - act_m
                flag = "\u26A0 POOR TRACKING" if abs(err_m) > 0.02 else "\u2713 OK"
                print("  %s:  n=%3d  mean_cmd=%+.3f  mean_actual=%+.3f  mean_err=%+.3f  %s" %
                      (axis, len(cmd_vals), cmd_m, act_m, err_m, flag))
            # Origin drift
            tk_x = pos_x[tk_idx]
            tk_y = pos_y[tk_idx]
            drifts = np.sqrt((tk_x - tk_x[0])**2 + (tk_y - tk_y[0])**2)
            print("  Origin drift: max=%.3fm  final=(%.3f, %.3f)m" %
                  (np.max(drifts), tk_x[-1] - tk_x[0], tk_y[-1] - tk_y[0]))

    # ════════════════════════════════════════════════════════════════════
    # M7.3 SECTIONS — only if target columns exist
    # ════════════════════════════════════════════════════════════════════
    if not has_target:
        print("\n=== 12-15. (skipped — no target columns in CSV) ===")
        return

    # Check if we actually have target data (not all NaN)
    valid_target = ~np.isnan(target_px)
    if np.sum(valid_target) < 5:
        print("\n=== 12-15. (skipped — fewer than 5 rows with target data) ===")
        return

    # ════════════════════════════════════════════════════════════════════
    # SECTION 12: Target position & velocity stats
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 12. Target drone position & velocity stats per phase ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph and valid_target[i]]
        if not idx:
            continue
        print("-- %s (n=%d) --" % (ph, len(idx)))
        tpx, tpy, tpz = target_px[idx], target_py[idx], target_pz[idx]
        print("  target_pos     x: [%.1f, %.1f]  y: [%.1f, %.1f]  z: [%.1f, %.1f]" %
              (np.min(tpx), np.max(tpx), np.min(tpy), np.max(tpy),
               np.min(tpz), np.max(tpz)))

        tvx = target_cmd_vx[idx]
        tvy = target_cmd_vy[idx]
        tvz = target_cmd_vz[idx]
        # Only use rows where both vx and vy are valid
        valid_v = ~np.isnan(tvx) & ~np.isnan(tvy)
        if np.sum(valid_v) > 0:
            h_speed = np.sqrt(tvx[valid_v]**2 + tvy[valid_v]**2)
            print("  target_h_speed mean=%.2f  p50=%.2f  max=%.2f" %
                  (np.mean(h_speed), np.median(h_speed), np.max(h_speed)))
            tvz_valid = tvz[~np.isnan(tvz)]
            if len(tvz_valid):
                print("  target_vz      mean=%+.3f  std=%.3f  min=%+.3f  max=%+.3f" %
                      (np.mean(tvz_valid), np.std(tvz_valid),
                       np.min(tvz_valid), np.max(tvz_valid)))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 13: 3D distance stats
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 13. 3D chaser-to-target distance per phase ===")
    for ph in unique_phases:
        idx = [i for i, p in enumerate(phases) if p == ph and not np.isnan(dist_3d[i])]
        if not idx:
            continue
        d = dist_3d[idx]
        print("  %-12s n=%4d  mean=%.2fm  std=%.2f  min=%.2f  p5=%.2f  "
              "p50=%.2f  p95=%.2f  max=%.2fm" %
              (ph, len(d), np.mean(d), np.std(d), np.min(d),
               np.percentile(d, 5), np.median(d),
               np.percentile(d, 95), np.max(d)))

    # ════════════════════════════════════════════════════════════════════
    # SECTION 14: Altitude tracking (chaser vs target)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 14. Chaser-vs-target altitude tracking (HOLD phase) ===")
    hold_idx = [i for i in range(n) if phases[i] == "HOLD" and valid_target[i]]
    if len(hold_idx) > 5:
        alt_err = pos_z[hold_idx] - target_pz[hold_idx]
        print("  alt_err (chaser - target):  mean=%+.2fm  std=%.2f  "
              "min=%+.2f  p5=%+.2f  p50=%+.2f  p95=%+.2f  max=%+.2fm" %
              (np.mean(alt_err), np.std(alt_err), np.min(alt_err),
               np.percentile(alt_err, 5), np.median(alt_err),
               np.percentile(alt_err, 95), np.max(alt_err)))
        # XY distance only (how well chaser follows horizontally)
        xy_dist = np.sqrt((pos_x[hold_idx] - target_px[hold_idx])**2 +
                          (pos_y[hold_idx] - target_py[hold_idx])**2)
        print("  xy_dist:                    mean=%.2fm  std=%.2f  "
              "p50=%.2f  p95=%.2f  max=%.2fm" %
              (np.mean(xy_dist), np.std(xy_dist), np.median(xy_dist),
               np.percentile(xy_dist, 95), np.max(xy_dist)))
    else:
        print("  (insufficient HOLD frames with target data)")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 15: Tracking quality summary
    # ════════════════════════════════════════════════════════════════════
    print("\n=== 15. Tracking quality summary ===")
    total_time = duration
    hold_rows = [i for i in range(n) if phases[i] == "HOLD"]
    hold_time = len(hold_rows) / rate if rate > 0 else 0
    search_rows = [i for i in range(n) if phases[i] == "SEARCH"]
    search_time = len(search_rows) / rate if rate > 0 else 0
    det_real = sum(1 for i in hold_rows if raw_det[i] == "REAL")
    det_pct = 100 * det_real / len(hold_rows) if hold_rows else 0

    hold_dist_idx = [i for i in hold_rows if not np.isnan(dist_3d[i])]
    mean_dist = np.mean(dist_3d[hold_dist_idx]) if hold_dist_idx else float('nan')

    hold_alpha_idx = [i for i in hold_rows if raw_det[i] == "REAL"]
    mean_alpha = np.mean(flt_alpha[hold_alpha_idx]) if hold_alpha_idx else float('nan')

    print("  Flight duration:          %.1f s" % total_time)
    print("  HOLD time:                %.1f s (%.0f%%)" %
          (hold_time, 100*hold_time/total_time if total_time > 0 else 0))
    print("  SEARCH time:              %.1f s (%.0f%%)" %
          (search_time, 100*search_time/total_time if total_time > 0 else 0))
    print("  Detection rate (HOLD):    %.1f%%" % det_pct)
    print("  Mean 3D distance (HOLD):  %.2f m" % mean_dist)
    print("  Mean alpha (HOLD):        %.5f (target=%.5f)" % (mean_alpha, ALPHA_STAR))
    if hold_dist_idx:
        d = dist_3d[hold_dist_idx]
        close = sum(1 for v in d if v < 2.0)
        far = sum(1 for v in d if v > 8.0)
        print("  Distance < 2m:            %d frames (%.1f%%)" %
              (close, 100*close/len(d)))
        print("  Distance > 8m:            %d frames (%.1f%%)" %
              (far, 100*far/len(d)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=os.path.expanduser('~/flight_log_latest.csv'))
    parser.add_argument('--window', nargs=2, type=int, default=None,
                        help='Row range: --window START END')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("ERROR: %s not found" % args.file)
        sys.exit(1)

    rows = load_csv(args.file)
    label = os.path.basename(args.file)

    if args.window:
        s, e = args.window
        rows = rows[s:e]
        label += " [rows %d-%d]" % (s, e)

    analyze(rows, label)


if __name__ == '__main__':
    main()