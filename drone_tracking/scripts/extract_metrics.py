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

M12 Phase D-prep (C1-vs-C2 measurement hardening), appended at END like above:
  hold_pct_mission, cmd_var_{vx,vy,vz,wz}, rms_jerk_{vx,vy,vz,wz},
  deriv_noise_{dex,dey,dea}, time_to_first_loss_s, n_dropouts_bridged,
  reacq_error_px, actuation_effort. Exact definitions are in-line at the
  computation block below. jerk/derivative dt is taken from consecutive
  sim_time deltas (rate-robust); deriv_noise needs the M12 logger columns and
  is '' on pre-M12 CSVs. Adding these columns changes the header, so runs will
  divert to summary_v2.csv unless a fresh --summary is given (recommended for
  the Phase D campaign: --summary ~/fyp/Results/summary_m12.csv).

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

    # ══ M12 Phase D-prep: C1-vs-C2 discriminating metric family ══════════════
    # Definitions (all rate-robust: dt taken from consecutive sim_time deltas):
    #   hold_pct_mission     — 100*HOLD / (SEARCH+APPROACH+HOLD)  [post-takeoff denom]
    #   cmd_var_{vx,vy,vz,wz}— variance of the commanded velocity, APPROACH+HOLD rows
    #   rms_jerk_{...}       — RMS of jerk = 2nd difference of cmd velocity / dt^2,
    #                          over contiguous APPROACH+HOLD triplets. NOTE the cmd
    #                          is already vel-smoothed in the controller, so this is
    #                          the jerk of the SMOOTHED command (what the actuator sees).
    #   deriv_noise_{dex,dey,dea} — std of the controller's error derivative during
    #                          CONTINUOUS-detection APPROACH+HOLD segments (ctrl_state==1),
    #                          dropping the first sample of each segment to exclude the
    #                          re-acquisition spike -> measures steady-state signal noise.
    #                          M12-only (needs dex/ctrl_state); '' on older CSVs.
    #   time_to_first_loss_s — sim_time from the first REAL YOLO detection to the first
    #                          subsequent frame where it drops (raw_det REAL->NONE).
    #   n_dropouts_bridged   — count of YOLO dropouts (<1.5 s) that recovered WITHOUT
    #                          the phase entering SEARCH (C2: Kalman-PRED bridged;
    #                          C1: frozen hold recovered before the 3 s timeout).
    #   reacq_error_px       — mean px distance between the pre-dropout centroid and the
    #                          re-acquisition centroid over those short dropouts
    #                          (config source: raw_cx/cy in C1, flt_cx/cy in C2).
    #   actuation_effort     — sum over mission rows & 4 axes of |v[i]-v[i-1]|*dt.
    has_m12 = 'ctrl_state' in rows[0]
    AH = ('APPROACH', 'HOLD')

    mission_rows = [r for r in rows if r.get('phase','') in ('SEARCH','APPROACH','HOLD')]
    hold_pct_mission = (round(100.0 * sum(1 for r in mission_rows if r.get('phase','')=='HOLD')
                              / len(mission_rows), 2) if mission_rows else '')

    def _segments(pred):
        segs=[]; cur=[]
        for r in rows:
            if pred(r): cur.append(r)
            elif cur:   segs.append(cur); cur=[]
        if cur: segs.append(cur)
        return segs

    ah_segs = _segments(lambda r: r.get('phase','') in AH)

    def _cmd_var(col):
        vals=[]
        for seg in ah_segs:
            for r in seg:
                v=fnum(r,col)
                if v is not None and not np.isnan(v): vals.append(v)
        return round(float(np.var(vals)),6) if len(vals)>1 else ''

    def _rms_jerk(col):
        js=[]
        for seg in ah_segs:
            for i in range(2,len(seg)):
                t2,t1,t0=(fnum(seg[i],'sim_time'),fnum(seg[i-1],'sim_time'),fnum(seg[i-2],'sim_time'))
                v2,v1,v0=(fnum(seg[i],col),fnum(seg[i-1],col),fnum(seg[i-2],col))
                if None in (t2,t1,t0,v2,v1,v0): continue
                d1,d0=t2-t1,t1-t0
                if d1<=0 or d0<=0 or d1>0.5 or d0>0.5: continue
                dt=0.5*(d1+d0)
                js.append((v2-2*v1+v0)/(dt*dt))
        return round(float(np.sqrt(np.mean(np.square(js)))),4) if js else ''

    def _deriv_noise(col):
        if not has_m12: return ''
        vals=[]
        for seg in _segments(lambda r: r.get('phase','') in AH and fnum(r,'ctrl_state')==1.0):
            if len(seg)<=2: continue
            for r in seg[1:]:                       # drop the re-acquisition spike
                v=fnum(r,col)
                if v is not None and not np.isnan(v): vals.append(v)
        return round(float(np.std(vals)),5) if len(vals)>1 else ''

    # time_to_first_loss_s
    time_to_first_loss_s=''; seen_real=False; t_first=None
    for r in rows:
        det=r.get('raw_det',''); t=fnum(r,'sim_time')
        if t is None: continue
        if det=='REAL' and not seen_real: seen_real=True; t_first=t
        elif seen_real and det!='REAL': time_to_first_loss_s=round(t-t_first,2); break

    # n_dropouts_bridged + reacq_error_px  (config-appropriate centroid source)
    src_cx = 'raw_cx' if args.config==1 else 'flt_cx'
    src_cy = 'raw_cy' if args.config==1 else 'flt_cy'
    n_bridged=0; reacq_errs=[]
    in_drop=False; onset_t=None; pre_row=None; had_search=False; prev_real=None
    for r in rows:
        det=r.get('raw_det',''); t=fnum(r,'sim_time'); ph=r.get('phase','')
        if det=='REAL':
            if in_drop and onset_t is not None and t is not None and (t-onset_t)<1.5 and not had_search:
                n_bridged+=1
                pcx,pcy=fnum(pre_row,src_cx),fnum(pre_row,src_cy)
                rcx,rcy=fnum(r,src_cx),fnum(r,src_cy)
                if None not in (pcx,pcy,rcx,rcy):
                    reacq_errs.append(((rcx-pcx)**2+(rcy-pcy)**2)**0.5)
            in_drop=False; onset_t=None; had_search=False; prev_real=r
        else:
            if not in_drop and prev_real is not None:
                in_drop=True; onset_t=t; pre_row=prev_real; had_search=(ph=='SEARCH')
            elif in_drop and ph=='SEARCH': had_search=True
    n_dropouts_bridged=n_bridged
    reacq_error_px=round(float(np.mean(reacq_errs)),2) if reacq_errs else ''

    # actuation_effort
    eff=0.0; got_eff=False; prev=None
    for r in rows:
        if r.get('phase','') not in ('SEARCH','APPROACH','HOLD'): prev=None; continue
        t=fnum(r,'sim_time'); vs=[fnum(r,c) for c in ('cmd_vx','cmd_vy','cmd_vz','cmd_wz')]
        if t is None or any(v is None for v in vs): prev=None; continue
        if prev is not None:
            dt=t-prev[0]
            if 0<dt<1.0:
                eff+=sum(abs(vs[a]-prev[1][a]) for a in range(4))*dt; got_eff=True
        prev=(t,vs)
    actuation_effort=round(eff,4) if got_eff else ''

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
        # ── M12 Phase D-prep: C1-vs-C2 discriminating family (END-appended) ──
        'hold_pct_mission':      hold_pct_mission,
        'cmd_var_vx':  _cmd_var('cmd_vx'), 'cmd_var_vy':  _cmd_var('cmd_vy'),
        'cmd_var_vz':  _cmd_var('cmd_vz'), 'cmd_var_wz':  _cmd_var('cmd_wz'),
        'rms_jerk_vx': _rms_jerk('cmd_vx'),'rms_jerk_vy': _rms_jerk('cmd_vy'),
        'rms_jerk_vz': _rms_jerk('cmd_vz'),'rms_jerk_wz': _rms_jerk('cmd_wz'),
        'deriv_noise_dex': _deriv_noise('dex'), 'deriv_noise_dey': _deriv_noise('dey'),
        'deriv_noise_dea': _deriv_noise('dea'),
        'time_to_first_loss_s':  time_to_first_loss_s,
        'n_dropouts_bridged':    n_dropouts_bridged,
        'reacq_error_px':        reacq_error_px,
        'actuation_effort':      actuation_effort,
    }

    fieldnames = list(metrics.keys())

    # ── Write, preserving backward compatibility ────────────────────────────
    # Never append into a file whose header differs from this schema (that would
    # misalign columns). Walk candidates [given summary, summary_v2, summary_m12]
    # and pick the first that either does not exist or already matches fieldnames.
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    def _hdr(path):
        if not os.path.exists(path): return None
        with open(path, newline='') as f: return next(csv.reader(f), [])
    d = os.path.dirname(args.summary)
    candidates = [args.summary,
                  os.path.join(d, 'summary_v2.csv'),
                  os.path.join(d, 'summary_m12.csv')]
    target = None
    for c in candidates:
        h = _hdr(c)
        if h is None or h == fieldnames: target = c; break
    if target is None:                     # all exist with different schemas
        target = os.path.join(d, 'summary_m12.csv')
    if target != args.summary:
        print("[extract_metrics] NOTE: '%s' header differs from this schema; "
              "writing to '%s' (existing files untouched)." % (args.summary, target))

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
