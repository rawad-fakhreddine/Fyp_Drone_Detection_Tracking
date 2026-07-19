#!/usr/bin/env python3
"""traj_verify.py CSV TRAJ — fit the target's GT path against the trajectory
equation (spec vs measured). Uses flight_logger's target_wx/wy/wz columns.
Specs (M9.1): T1 hover | T2 1.0 m/s straight | T3 3.5 straight | T4 circle
R=8 T=25 | T5 lemniscate a=8 (y=a/2) T=40 | T6 15 deg 2.0 | T7 35 deg 3.0 |
T8 helix R=8 T=25 vz 0.15 bounce."""
import csv, math, sys, statistics as st

CSV, TRAJ = sys.argv[1], int(sys.argv[2])
rows = [r for r in csv.DictReader(open(CSV))
        if r.get('target_wx') not in (None, '', 'nan')]
tx = [float(r['target_wx']) for r in rows]
ty = [float(r['target_wy']) for r in rows]
tz = [float(r['target_wz']) for r in rows]
# MOVING segment = after the target reaches cruise altitude (world ~12.5+)
i0 = next((i for i, z in enumerate(tz) if z > 12.5), 0) + 50
# DT measured from the CSV clock (logger is 20 Hz, NOT the nominal 10)
ts = [float(r['sim_time']) for r in rows]
DT = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
tx, ty, tz = tx[i0:], ty[i0:], tz[i0:]
n = len(tx)
print("traj T%d | %d GT samples (%.0f s of MOVING)" % (TRAJ, n, n * DT))

def speed_h():
    v = [math.hypot(tx[i+10]-tx[i], ty[i+10]-ty[i])/(10*DT) for i in range(0, n-10, 10)]
    return st.mean(v), st.pstdev(v)

if TRAJ == 1:
    print("T1 hover: x std %.2f  y std %.2f  z std %.2f (want ~0)"
          % (st.pstdev(tx), st.pstdev(ty), st.pstdev(tz)))
elif TRAJ in (2, 3):
    vh, vs = speed_h()
    # straightness: lateral RMS from the best-fit line through the path
    mx, my = st.mean(tx), st.mean(ty)
    sxx = sum((x-mx)**2 for x in tx); sxy = sum((x-mx)*(y-my) for x, y in zip(tx, ty))
    th = 0.5 * math.atan2(2*sxy, sxx - sum((y-my)**2 for y in ty))
    lat = [-(x-mx)*math.sin(th)+(y-my)*math.cos(th) for x, y in zip(tx, ty)]
    print("T%d straight: speed %.2f±%.2f (spec %.1f) | lateral RMS %.2f m | z std %.2f"
          % (TRAJ, vh, vs, 1.0 if TRAJ == 2 else 3.5,
             math.sqrt(st.mean([l*l for l in lat])), st.pstdev(tz)))
elif TRAJ in (4, 8):
    # LS circle fit (Kasa): radius + center, period from unwrapped angle
    mx, my = st.mean(tx), st.mean(ty)
    r = [math.hypot(x-mx, y-my) for x, y in zip(tx, ty)]
    R = st.mean(r)
    ang = [math.atan2(y-my, x-mx) for x, y in zip(tx, ty)]
    unw = [ang[0]]
    for a in ang[1:]:
        d = a - unw[-1]
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        unw.append(unw[-1] + d)
    T = 2*math.pi*(n-1)*DT/abs(unw[-1]-unw[0]) if abs(unw[-1]-unw[0]) > 0.5 else float('inf')
    print("T%d circle: R %.2f±%.2f (spec 8) | period %.1f s (spec 25) | speed %.2f (spec %.2f)"
          % (TRAJ, R, st.pstdev(r), T, speed_h()[0], 2*math.pi*8/25))
    if TRAJ == 8:
        vzs = [(tz[i+10]-tz[i])/(10*DT) for i in range(0, n-10, 10)]
        mag = st.mean([abs(v) for v in vzs])
        flips = sum(1 for a, b in zip(vzs, vzs[1:])
                    if a*b < 0 and abs(a) > 0.05 and abs(b) > 0.05)
        print("   helix vz: |vz| mean %.2f m/s (spec 0.15) | z range %.1f-%.1f | vz flips %d"
              % (mag, min(tz), max(tz), flips))
elif TRAJ == 5:
    # Gerono: x=A sin(wt), y=(A/2) sin(2wt) about the center
    mx, my = st.mean(tx), st.mean(ty)
    # principal axis (long axis of the 8)
    sxx = sum((x-mx)**2 for x in tx); syy = sum((y-my)**2 for y in ty)
    sxy = sum((x-mx)*(y-my) for x, y in zip(tx, ty))
    th = 0.5*math.atan2(2*sxy, sxx-syy)
    u = [ (x-mx)*math.cos(th)+(y-my)*math.sin(th) for x, y in zip(tx, ty)]
    w = [-(x-mx)*math.sin(th)+(y-my)*math.cos(th) for x, y in zip(tx, ty)]
    A = (max(u)-min(u))/2.0; B = (max(w)-min(w))/2.0
    # period: zero-crossings of u (2 per period)
    zc = sum(1 for a, b in zip(u, u[1:]) if a*b < 0)
    T = 2.0*(n-1)*DT/zc if zc else float('inf')
    print("T5 lemniscate: long semi-axis %.2f (spec 8) | short %.2f (spec 4) | period %.1f s (spec 40)"
          % (A, B, T))
elif TRAJ in (6, 7):
    spd, slp = (2.0, 15.0) if TRAJ == 6 else (3.0, 35.0)
    sl = []
    for i in range(0, n-10, 10):
        vh = math.hypot(tx[i+10]-tx[i], ty[i+10]-ty[i])/(10*DT)
        vz = abs(tz[i+10]-tz[i])/(10*DT)
        if vh > 0.3: sl.append(math.degrees(math.atan2(vz, vh)))
    x0, y0 = tx[0], ty[0]
    disp = [math.hypot(x-x0, y-y0) for x, y in zip(tx, ty)]
    v3 = [math.sqrt((tx[i+10]-tx[i])**2+(ty[i+10]-ty[i])**2+(tz[i+10]-tz[i])**2)/(10*DT)
          for i in range(0, n-10, 10)]
    print("T%d incline: slope %.1f deg (spec %.0f) | 3D speed %.2f±%.2f (spec %.1f) | "
          "max horiz disp %.1f (bound 62) | z %.1f-%.1f"
          % (TRAJ, st.median(sl), slp, st.mean(v3), st.pstdev(v3), spd,
             max(disp), min(tz), max(tz)))
