#!/usr/bin/env python3
"""score.py CSV [CSV...] — standard scorecard for the T3/zone1 vx campaign."""
import csv, sys, math, statistics as st
import numpy as np

for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path)))
    mission = [r for r in rows if r['phase'] in ('SEARCH', 'APPROACH', 'HOLD')]
    if not mission:
        print(path, "no mission rows"); continue
    t0 = float(mission[0]['sim_time'])
    hold = [r for r in mission if r['phase'] == 'HOLD']
    trk = [r for r in mission if r['phase'] in ('APPROACH', 'HOLD')]
    fov = [r for r in mission if r.get('in_fov', '') in ('0', '1')]
    det = [r for r in mission if r['raw_det'] == 'REAL']
    win = [r for r in trk if 5 <= float(r['sim_time']) - t0 <= 30 and r['true_dist_3d']]
    err = [abs(float(r['true_dist_3d']) - 8.0) for r in win]
    dm = [float(r['true_dist_3d']) for r in win]
    vx = [float(r['cmd_vx']) for r in trk if r['raw_det'] == 'REAL']
    dv = [abs(b - a) for a, b in zip(vx, vx[1:])]
    dall = [float(r['true_dist_3d']) for r in mission if r['true_dist_3d']]
    emg = sum(1 for r in mission if r.get('emerg') == '1')
    name = path.split('/')[-2] + '/' + path.split('/')[-1][-23:-4]
    print("%s\n  HOLD %.1f%%  det %.1f%%  FOV %.1f%%  | |d-8| %.2f  dmean %.2f  "
          "| vxJp90 %.2f  closest %.2f  emerg %.1fs  dur %.0fs"
          % (name, 100 * len(hold) / len(mission), 100 * len(det) / len(mission),
             100 * sum(1 for r in fov if r['in_fov'] == '1') / max(len(fov), 1),
             st.mean(err) if err else float('nan'), st.mean(dm) if dm else float('nan'),
             sorted(dv)[int(.9 * len(dv))] if dv else float('nan'),
             min(dall) if dall else float('nan'), emg / 20.0,
             float(mission[-1]['sim_time']) - t0))
    # health check 1: target altitude flatness (z-hold verification)
    tz = [float(r['target_wz']) for r in mission if r.get('target_wz')]
    if tz:
        print("  [traj] target z: min %.2f max %.2f (span %.2f m — z-hold %s)"
              % (min(tz), max(tz), max(tz) - min(tz),
                 "OK" if max(tz) - min(tz) < 0.8 else "DRIFTING"))
    # health check 2: world-frame straightness (ground truth)
    w = [(float(r['target_wx']), float(r['target_wy'])) for r in mission
         if r.get('target_wx') and r.get('target_wy')]
    if len(w) > 100:
        a = np.array(w); mv = a[np.hypot(a[:, 0] - a[0, 0], a[:, 1] - a[0, 1]) > 0.5]
        if len(mv) > 50:
            c = mv.mean(0); u, s, vt = np.linalg.svd(mv - c)
            lat = (mv - c) @ vt[1]
            print("  [traj] world-frame lateral dev: rms %.2f max %.2f m (%s)"
                  % (lat.std(), abs(lat).max(),
                     "straight" if lat.std() < 1.5 else "CHECK PATH"))
