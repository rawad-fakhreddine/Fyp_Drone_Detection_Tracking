#!/usr/bin/env python3
"""collision.py CSV... — flag chaser-target and tree/terrain strikes from
the logged Gazebo deltas (gz_dx/dy/dz, chaser/target world positions)."""
import csv, sys, math

for path in sys.argv[1:]:
    rows = [r for r in csv.DictReader(open(path))
            if r['phase'] in ('SEARCH', 'APPROACH', 'HOLD')]
    if len(rows) < 30:
        print(path.split('/')[-1], "too few rows"); continue
    name = path.split('/')[-2] + '/' + path.split('/')[-1][-12:-4]
    t0 = float(rows[0]['sim_time'])
    rows_by_t = {round(float(r['sim_time']) - t0, 1): r.get('emerg', '0') for r in rows}

    # 1) target collision: true 3D separation below a body-contact threshold
    tgt = [(float(r['sim_time']) - t0, float(r['true_dist_3d']))
           for r in rows if r['true_dist_3d']]
    hits = [(t, d) for t, d in tgt if d < 1.5]
    mind = min((d for _, d in tgt), default=float('nan'))

    # 2) tree/terrain strike: chaser commands motion but world pos is frozen,
    #    or a sharp altitude loss into canopy height.
    cw = [(float(r['sim_time']) - t0, float(r['chaser_wx']), float(r['chaser_wy']),
           float(r['chaser_wz']),
           math.hypot(float(r['cmd_vx']), float(r['cmd_vy'])))
          for r in rows if r.get('chaser_wx')]
    stalls = []
    strike_z = None
    if len(cw) > 12:
        peak = cw[0][3]
        for i in range(10, len(cw)):
            dt = cw[i][0] - cw[i - 10][0]
            if dt <= 0:
                continue
            disp = math.hypot(cw[i][1] - cw[i - 10][1], cw[i][2] - cw[i - 10][2])
            # real obstacle only: commanding motion, not moving, NOT braking,
            # and past the initial-acquisition transient (>12 s)
            braking = rows_by_t.get(round(cw[i][0], 1), '0') == '1'
            if (cw[i][4] > 1.5 and disp < 0.15 and not braking
                    and cw[i][0] > 12):
                stalls.append(cw[i][0])
            peak = max(peak, cw[i][3])
            if cw[i][3] < peak - 4 and cw[i][3] < 10 and strike_z is None:
                strike_z = (cw[i][0], cw[i][3], peak)

    print("%s" % name)
    print("  TARGET: min sep %.2f m | %d frames <1.5 m%s"
          % (mind, len(hits),
             ("  FIRST HIT t=%.0fs d=%.2f" % (hits[0][0], hits[0][1])) if hits else ""))
    zs = [c[3] for c in cw]
    print("  TREE/GND: %d motion-stall frames%s | chaser_z %.1f-%.1f m"
          % (len(stalls),
             ("  FIRST STALL t=%.0fs" % stalls[0]) if stalls else "",
             (min(zs) if zs else float('nan')), (max(zs) if zs else float('nan'))))
    if strike_z:
        print("  !! ALTITUDE-LOSS STRIKE: z=%.1f (from peak %.1f) at t=%.0fs"
              % (strike_z[1], strike_z[2], strike_z[0]))
