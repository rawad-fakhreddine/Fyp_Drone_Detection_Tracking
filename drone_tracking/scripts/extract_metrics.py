#!/usr/bin/env python3
"""
extract_metrics.py — Extract key metrics from a flight CSV log for the summary.
Appends one row to ~/fyp/Results/summary.csv.

M9.6 (B3, B9):
  * B3: detection_rate now uses raw_det == 'REAL' (flight_logger writes the
    strings REAL/NONE/PRED, not 1/0 — so the old in('1','1.0','True') test was
    always 0). Adds flt_detection_rate (flt_det == 'REAL') and pred_rate
    (flt_det == 'PRED').
  * B9: adds mean_recovery_time_s (mean duration of contiguous SEARCH episodes)
    and wrong_direction_pct (same definition as analyze_flight_log.py §9).
  New columns are appended at the END of the summary schema for backward
  compatibility. On a header mismatch the row is diverted to summary_v2.csv
  rather than corrupting existing rows.

M9.6 step 3 (IBVS v6.26 guard + watchdog protocol), appended at END like B9:
  * emergency_brake_pct   — % of mission rows (SEARCH/APPROACH/HOLD) with
                            emerg=1 ('' if the CSV predates the column)
  * emergency_brake_count — number of distinct engagements (0->1 transitions)
  * aborted               — '1' if the run was cut by the loss watchdog
                            (passed by launch_stack via --aborted)
  * mission_duration_s    — actual logged sim-time span (last - first
                            sim_time row); makes HOLD% comparable across
                            aborted/full runs of different lengths

Usage:
  python3 extract_metrics.py --csv PATH --config N --zone Z --traj T --seed S --duration D [--aborted 0|1]
"""
import os, sys, csv, argparse, datetime
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True)
    ap.add_argument('--config',   required=True, type=int)
    ap.add_argument('--zone',     required=True)
    ap.add_argument('--traj',     required=True, type=int)
    ap.add_argument('--seed',     required=True, type=int)
    ap.add_argument('--duration', required=True, type=int)
    ap.add_argument('--aborted',  type=int, default=0)   # 1 = cut by loss watchdog
    ap.add_argument('--summary',  default=os.path.expanduser("~/fyp/Results/summary.csv"))
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[extract_metrics] WARNING: empty CSV"); sys.exit(0)

    def pct(filter_fn):
        n = sum(1 for r in rows if filter_fn(r))
        return 100.0 * n / len(rows)
    def fmean(col, filter_fn=lambda r: True):
        vals = []
        for r in rows:
            if not filter_fn(r): continue
            try:
                v = float(r.get(col, ''))
                if not np.isnan(v): vals.append(v)
            except (ValueError, TypeError): pass
        return float(np.mean(vals)) if vals else float('nan')
    def fnum(r, col):
        try:    return float(r.get(col, ''))
        except (ValueError, TypeError): return None

    phase        = lambda P: lambda r: r.get('phase','') == P
    is_hold      = phase('HOLD')
    # B3: flight_logger writes REAL / NONE / PRED strings
    detected     = lambda r: r.get('raw_det','') == 'REAL'
    flt_detected = lambda r: r.get('flt_det','') == 'REAL'
    flt_predict  = lambda r: r.get('flt_det','') == 'PRED'

    # ── B9: wrong-direction % (replicates analyze_flight_log.py §9) ──────────
    #   active = APPROACH/HOLD rows with flt_det==REAL and cmd_frame==8 (BODY_NED)
    #   wrong  = (ea < -0.002 and cmd_vx < -0.05) or (ea > 0.002 and cmd_vx > 0.05)
    def _is_body_frame(r):
        v = fnum(r, 'cmd_frame')
        return v is not None and int(v) == 8
    active = [r for r in rows
              if r.get('phase','') in ('APPROACH', 'HOLD')
              and r.get('flt_det','') == 'REAL' and _is_body_frame(r)]
    def _wrong(r):
        ea = fnum(r, 'ea'); vx = fnum(r, 'cmd_vx')
        if ea is None or vx is None: return False
        return (ea < -0.002 and vx < -0.05) or (ea > 0.002 and vx > 0.05)
    n_wrong = sum(1 for r in active if _wrong(r))
    wrong_direction_pct = (100.0 * n_wrong / len(active)) if active else float('nan')

    # ── B9: mean recovery time — contiguous SEARCH episodes ─────────────────
    #   episode duration = last_t - first_t; an episode still open at EOF is
    #   ignored (we don't know when it would have ended).
    episodes = []
    cur_start = cur_last = None
    for r in rows:
        if r.get('phase','') == 'SEARCH':
            t = fnum(r, 'sim_time')
            if t is None: continue
            if cur_start is None: cur_start = t
            cur_last = t
        else:
            if cur_start is not None and cur_last is not None:
                episodes.append(cur_last - cur_start)   # closed episode
            cur_start = cur_last = None
    # cur_start still set here => last episode open at EOF => ignored
    mean_recovery_time_s = float(np.mean(episodes)) if episodes else float('nan')

    # ── M9.6 step 3: v6.26 emergency brake stats (mission rows only) ─────────
    has_emerg = 'emerg' in rows[0]
    mission = [r for r in rows if r.get('phase','') in ('SEARCH','APPROACH','HOLD')]
    if has_emerg and mission:
        n_emerg = sum(1 for r in mission if r.get('emerg','') == '1')
        emergency_brake_pct = round(100.0 * n_emerg / len(mission), 2)
        # distinct engagements = 0->1 transitions over the full row sequence
        emergency_brake_count = 0
        prev = '0'
        for r in rows:
            cur = r.get('emerg', '0')
            if cur == '1' and prev != '1': emergency_brake_count += 1
            prev = cur
    else:
        emergency_brake_pct = ''      # CSV predates the emerg column
        emergency_brake_count = ''

    # ── M9.6 step 3: actual logged sim-time span (abort visibility) ─────────
    times = [t for t in (fnum(r, 'sim_time') for r in rows) if t is not None]
    mission_duration_s = round(max(times) - min(times), 1) if times else ''

    # ── M10.3: HOLD-phase yaw zero-crossing rate (oscillation guard) ─────────
    #   Sign changes of the published cmd_wz across consecutive HOLD samples,
    #   per second of HOLD sim-time. Over-gain (dropped damping ratio with fixed
    #   Kd_wz) shows up as yaw ringing -> elevated crossing rate -> a disqualifier
    #   for the Kp_wz/Kp_y sweep even if HOLD% rises (M10.3 sweep accept gate).
    hold_rows = [r for r in rows if r.get('phase','') == 'HOLD']
    def _sgn(v): return 1 if v > 1e-4 else (-1 if v < -1e-4 else 0)
    xings = 0; prev_s = None; ht0 = ht1 = None
    for r in hold_rows:
        wz = fnum(r, 'cmd_wz'); t = fnum(r, 'sim_time')
        if wz is None or t is None: continue
        if ht0 is None: ht0 = t
        ht1 = t
        s = _sgn(wz)
        if prev_s is not None and s != 0 and prev_s != 0 and s != prev_s: xings += 1
        if s != 0: prev_s = s
    hold_span = (ht1 - ht0) if (ht0 is not None and ht1 is not None and ht1 > ht0) else 0.0
    hold_yaw_zerocross_hz = round(xings / hold_span, 3) if hold_span > 5.0 else ''

    def _opt(x, nd):
        return round(x, nd) if not np.isnan(x) else ''

    metrics = {
        'timestamp':            datetime.datetime.now().isoformat(timespec='seconds'),
        'config':               args.config,
        'zone':                 args.zone,
        'trajectory':           args.traj,
        'seed':                 args.seed,
        'duration_s':           args.duration,
        'n_samples':            len(rows),
        'takeoff_pct':          round(pct(phase('TAKEOFF')),  2),
        'search_pct':           round(pct(phase('SEARCH')),   2),
        'approach_pct':         round(pct(phase('APPROACH')), 2),
        'hold_pct':             round(pct(is_hold),           2),
        'detection_rate':       round(pct(detected),          2),
        'hold_mean_sep':        round(fmean('true_dist_3d', is_hold),  3),
        'hold_mean_alt_err':    round(fmean('world_alt_err', is_hold), 3),
        'csv_file':             os.path.basename(args.csv),
        # ── M9.6 new columns (appended at END for backward compatibility) ──
        'flt_detection_rate':   round(pct(flt_detected), 2),
        'pred_rate':            round(pct(flt_predict),  2),
        'mean_recovery_time_s': _opt(mean_recovery_time_s, 2),
        'wrong_direction_pct':  _opt(wrong_direction_pct,  2),
        # ── M9.6 step 3 columns (appended at END for backward compatibility) ──
        'emergency_brake_pct':   emergency_brake_pct,
        'emergency_brake_count': emergency_brake_count,
        'aborted':               args.aborted,
        'mission_duration_s':    mission_duration_s,
        # ── M10.3 column (appended at END for backward compatibility) ──
        'hold_yaw_zerocross_hz': hold_yaw_zerocross_hz,
    }

    fieldnames = list(metrics.keys())

    # ── Write, preserving backward compatibility ────────────────────────────
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    target = args.summary
    if os.path.exists(args.summary):
        with open(args.summary, newline='') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and existing_header != fieldnames:
            # Header mismatch — do NOT corrupt old rows. Divert to summary_v2.csv.
            target = os.path.join(os.path.dirname(args.summary), 'summary_v2.csv')
            print("[extract_metrics] WARNING: '%s' header differs from the new "
                  "M9.6 schema.\n   Writing this run to '%s' instead "
                  "(old file left untouched)." % (args.summary, target))

    write_header = not os.path.exists(target)
    with open(target, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow(metrics)

    print("\n=== Run metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<22s} {v}")
    print(f"\n✓ Appended to {target}")

if __name__ == '__main__':
    main()
