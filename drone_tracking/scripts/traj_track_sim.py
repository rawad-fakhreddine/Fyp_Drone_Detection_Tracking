#!/usr/bin/env python3
"""traj_track_sim.py — OFFLINE closed-loop test of the trajectory-tracking fix.

Simulates the target as a velocity plant (PX4 does not achieve a commanded
velocity instantly) and integrates position for all 8 trajectories under two
control laws:
  OPEN   : v_cmd = formula velocity            (today's behaviour)
  CLOSED : v_cmd = formula + Kp*(ideal - actual)   (the proposed fix)
where `ideal` is the perfect (plant-free) integral of the same formula.

Goal: show CLOSED makes the actual path track the exact formula regardless of
the plant, on all 8 trajectories. This proves the fix before any Gazebo flight.

Plant is tunable so we can check it against the measured real deviations
(T5 real semi-axes ~9.4/5.2). --plant first|second , --tau , --wn , --zeta .
"""
import math, sys
import numpy as np

DT = 0.02          # 50 Hz, matches target_mover
DUR = 200.0
N = int(DUR / DT)
Z_FLOOR, Z_CEIL = 12.0, 24.0
ANCHOR = np.array([0.0, 0.0, 14.0])

# ---- plant options -------------------------------------------------------
PLANT = 'first'
TAU = 0.35          # first-order velocity lag (s)
WN, ZETA = 6.0, 0.6  # second-order velocity loop


def plant_step(state, v_cmd):
    """Advance the per-axis velocity plant one DT toward v_cmd. Returns v_act."""
    if PLANT == 'first':
        v = state['v']
        v += (v_cmd - v) * DT / TAU
        state['v'] = v
        return v
    else:  # second order: v tracks v_cmd with wn, zeta
        v, a = state['v'], state['a']
        a += DT * (WN * WN * (v_cmd - v) - 2 * ZETA * WN * a)
        v += DT * a
        state['v'], state['a'] = v, a
        return v


def new_plant():
    return [{'v': 0.0, 'a': 0.0} for _ in range(3)]


# ---- trajectory formula velocities (world frame) -------------------------
class Law:
    """Reimplements target_mover T1-T8 velocity, with the bounce/shuttle state
    driven by whichever position (ideal or actual) is passed in."""
    def __init__(self, t):
        self.t = t
        self.sdir = 1.0
        self.vzdir = 1.0

    def vel(self, e, pos):
        t = self.t
        if t == 1:
            return 0.0, 0.0, 0.0
        if t in (2, 3):
            spd = 1.0 if t == 2 else 3.5
            return spd, 0.0, 0.0            # az = 0 (world +x)
        if t == 4:
            R, w = 8.0, 2 * math.pi / 25.0
            return -R * w * math.sin(w * e), R * w * math.cos(w * e), 0.0
        if t == 5:
            A, w = 8.0, 2 * math.pi / 40.0
            return A * w * math.cos(w * e), A * w * math.cos(2 * w * e), 0.0
        if t == 8:
            R, w = 8.0, 2 * math.pi / 25.0
            if self.vzdir > 0 and pos[2] >= Z_CEIL - 1:
                self.vzdir = -1.0
            elif self.vzdir < 0 and pos[2] <= Z_FLOOR + 1:
                self.vzdir = 1.0
            return -R * w * math.sin(w * e), R * w * math.cos(w * e), 0.15 * self.vzdir
        if t in (6, 7):
            spd, sl = (2.0, 15.0) if t == 6 else (3.0, 35.0)
            slr = math.radians(sl)
            lat, vb = spd * math.cos(slr), spd * math.sin(slr)
            if self.vzdir > 0 and pos[2] >= Z_CEIL - 1:
                self.vzdir = -1.0
            elif self.vzdir < 0 and pos[2] <= Z_FLOOR + 1:
                self.vzdir = 1.0
            along = pos[0] - ANCHOR[0]
            if abs(along) > 60.0 and self.sdir == 1.0:
                self.sdir = -1.0
            elif abs(along) < 2.0 and self.sdir == -1.0:
                self.sdir = 1.0
            return lat * self.sdir, 0.0, vb * self.vzdir
        return 0.0, 0.0, 0.0


def integrate_ideal(t):
    """Perfect (plant-free) integral of the formula = the exact target path."""
    law = Law(t)
    pos = ANCHOR.copy()
    P = np.empty((N, 3))
    for i in range(N):
        v = np.array(law.vel(i * DT, pos))
        pos = pos + v * DT
        P[i] = pos
    return P


def run(t, ideal, kp):
    """Simulate with the plant; kp=0 => OPEN, kp>0 => CLOSED. Returns path."""
    law = Law(t)
    pl = new_plant()
    pos = ANCHOR.copy()
    P = np.empty((N, 3))
    for i in range(N):
        ff = np.array(law.vel(i * DT, pos))
        cmd = ff + kp * (ideal[i] - pos) if kp > 0 else ff
        v = np.array([plant_step(pl[k], cmd[k]) for k in range(3)])
        pos = pos + v * DT
        P[i] = pos
    return P


def geom(t, P):
    """One-line geometry metric for trajectory t from path P (skip transient)."""
    Q = P[500:]                              # drop first 10 s
    x, y, z = Q[:, 0] - Q[:, 0].mean(), Q[:, 1] - Q[:, 1].mean(), Q[:, 2]
    if t == 1:
        r = np.sqrt(x**2 + y**2 + (z - z.mean())**2)
        return 'drift %.2f m (spec 0)' % r.mean()
    if t in (2, 3):
        lat = y
        return 'lateral RMS %.2f m (spec 0)' % math.sqrt((lat**2).mean())
    if t in (4, 8):
        r = np.hypot(x, y)
        return 'R %.2f+/-%.2f (spec 8)' % (r.mean(), r.std())
    if t == 5:
        return 'semi-axes %.2f/%.2f (spec 8/4)' % (
            (x.max() - x.min()) / 2, (y.max() - y.min()) / 2)
    if t in (6, 7):
        sl = 15.0 if t == 6 else 35.0
        dx, dz = np.diff(Q[:, 0]), np.diff(Q[:, 2])
        m = np.abs(dx) > 1e-6
        return 'slope %.1f (spec %.0f)' % (
            np.degrees(np.arctan2(np.abs(dz[m]), np.abs(dx[m]))).mean(), sl)
    return ''


def track_err(ideal, P):
    """Mean 3-D distance between actual and exact-formula path (after transient)."""
    d = np.linalg.norm(P[500:] - ideal[500:], axis=1)
    return d.mean(), d.max()


def main():
    global PLANT, TAU, WN, ZETA, KP
    kp = 2.0
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--plant': PLANT = args[i + 1]
        elif a == '--tau': TAU = float(args[i + 1])
        elif a == '--wn': WN = float(args[i + 1])
        elif a == '--zeta': ZETA = float(args[i + 1])
        elif a == '--kp': kp = float(args[i + 1])
    print('plant=%s tau=%.2f wn=%.1f zeta=%.2f kp=%.1f\n' % (PLANT, TAU, WN, ZETA, kp))
    print('%-3s %-26s %-26s %-10s %-10s' %
          ('T', 'OPEN-LOOP (today)', 'CLOSED-LOOP (fix)', 'err_open', 'err_fix'))
    for t in range(1, 9):
        ideal = integrate_ideal(t)
        Po = run(t, ideal, 0.0)
        Pc = run(t, ideal, kp)
        eo = track_err(ideal, Po); ec = track_err(ideal, Pc)
        print('T%-2d %-26s %-26s %6.2f/%-4.2f %6.2f/%-4.2f' %
              (t, geom(t, Po), geom(t, Pc), eo[0], eo[1], ec[0], ec[1]))
    print('\nerr = mean/max 3-D distance (m) from the exact formula path')


if __name__ == '__main__':
    main()
