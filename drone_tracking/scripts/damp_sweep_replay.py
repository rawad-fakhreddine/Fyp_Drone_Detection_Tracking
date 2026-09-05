#!/usr/bin/env python3
"""
damp_sweep_replay.py — confound-free velocity_damping comparison (offline).
Same KF machinery as q_sweep_replay.py, R held [6,6,5] and Q baseline; only
velocity_damping is swept over {0.88,0.92,0.95,0.97,0.99}. Same raw-YOLO stream
through every damping -> only damping changes.

Damping decays the velocity state ONLY on dropout/outlier frames (matches the
node), and is HARD-ZEROED at max_dropout=30 frames (1.5s). So its effect is
confined to the FIRST 30 frames of each gap. We therefore score dropout-
prediction error (kf vs ground-truth proj) split by:
  - gap position:  EARLY (<=30 frames into the gap, damping live)
                   LATE  (>30 frames, velocity already zeroed -> damping-blind)
  - target motion: STRAIGHT vs TURNING (GT heading change since gap start >30deg)
                   -> exposes the stale-velocity overshoot failure mode.

Usage: damp_sweep_replay.py CSV [CSV ...]
"""
import sys, csv, math
import numpy as np

DAMPS = [0.88, 0.92, 0.95, 0.97, 0.99]
R_KEPT = [6.0, 6.0, 5.0]
Q_BASE = [0.5, 0.5, 3.0, 6.0, 6.0, 3.0]
MAXDROP = 30
TURN_DEG = 30.0


class KF:
    OUTLIER_PREV_ALPHA = 3000.0
    OUTLIER_NEW_ALPHA = 900.0
    PIXEL_JUMP = 180.0
    MAX_REJ = 4

    def __init__(self, damp):
        self.x = np.zeros(6); self.P = np.eye(6) * 500.0
        dt = 1.0 / 20.0
        self.F = np.eye(6); self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        self.H = np.zeros((3, 6)); self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.R = np.diag(R_KEPT); self.Q = np.diag(Q_BASE)
        self.damp = damp
        self.init = False; self.drop = 0; self.rej = 0; self.maxdrop = MAXDROP

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def _outlier(self, mx, my, mz):
        prev = self.x[2]; jump = math.hypot(mx - self.x[0], my - self.x[1])
        if prev > self.OUTLIER_PREV_ALPHA and mz < self.OUTLIER_NEW_ALPHA:
            return True
        if jump > self.PIXEL_JUMP and self.rej < self.MAX_REJ:
            return True
        return False

    def step(self, mx, my, mz):
        is_drop = (mx != mx) or (my != my) or (mz != mz)
        if not is_drop:
            if not self.init:
                self.x[0:3] = [mx, my, mz]; self.x[3:6] = 0.0
                self.init = True; self.drop = 0; self.rej = 0
                self.predict(); self.update(np.array([mx, my, mz]))
            else:
                if self._outlier(mx, my, mz) and self.rej >= self.MAX_REJ:
                    self.x[2] = mz; self.x[5] = 0.0; self.P[2, 2] = 500.0; self.P[5, 5] = 500.0
                    self.rej = 0; self.drop = 0
                    self.predict(); self.update(np.array([mx, my, mz])); is_drop = False
                elif self._outlier(mx, my, mz):
                    self.rej += 1; self.drop += 1
                    self.x[3] *= self.damp; self.x[4] *= self.damp; self.x[5] *= self.damp
                    if self.drop >= self.maxdrop: self.x[3:6] = 0.0
                    self.predict(); is_drop = True
                else:
                    self.drop = 0; self.rej = 0
                    self.predict(); self.update(np.array([mx, my, mz]))
        else:
            self.drop += 1
            if not self.init:
                return None
            self.x[3] *= self.damp; self.x[4] *= self.damp; self.x[5] *= self.damp
            if self.drop >= self.maxdrop: self.x[3:6] = 0.0
            self.predict()
        self.x[0] = min(max(self.x[0], 0.0), 640.0)
        self.x[1] = min(max(self.x[1], 0.0), 480.0)
        self.x[2] = min(max(self.x[2], 0.0), 307200.0)
        return (self.x[0], self.x[1], self.drop, is_drop)


def fnum(s):
    try:
        v = float(s); return v if v == v else None
    except (ValueError, TypeError):
        return None


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float('nan')


def gt_headings(rows):
    """GT heading (rad) per row from smoothed proj_u/proj_v deltas; None if unknown."""
    pu = [fnum(r.get('proj_u')) for r in rows]
    pv = [fnum(r.get('proj_v')) for r in rows]
    head = [None] * len(rows)
    W = 3
    for i in range(len(rows)):
        a, b = i - W, i
        if a < 0 or pu[a] is None or pu[b] is None or pv[a] is None or pv[b] is None:
            continue
        dx, dy = pu[b] - pu[a], pv[b] - pv[a]
        if dx * dx + dy * dy < 1.0:
            head[i] = head[i - 1] if i > 0 else None     # too small to be reliable
        else:
            head[i] = math.atan2(dy, dx)
    return head


def angdiff(a, b):
    d = abs(a - b) % (2 * math.pi)
    return d if d <= math.pi else 2 * math.pi - d


def replay(rows, damp, head):
    kf = KF(damp)
    # err buckets: dict key (gappos, turn) -> list of per-axis errors (cx,cy combined as hypot)
    buckets = {('EARLY', 'STRAIGHT'): [], ('EARLY', 'TURNING'): [],
               ('LATE', 'STRAIGHT'): [], ('LATE', 'TURNING'): []}
    all_drop = []
    head0 = None          # GT heading captured at the start of the current gap
    for i, r in enumerate(rows):
        mx, my, mz = fnum(r.get('raw_cx')), fnum(r.get('raw_cy')), fnum(r.get('raw_alpha'))
        if r.get('raw_det') != 'REAL' or None in (mx, my, mz):
            mx = my = mz = float('nan')
        prev_drop_active = (kf.drop > 0)
        out = kf.step(mx, my, mz)
        if out is None:
            continue
        kfx, kfy, gappos, is_drop = out
        if is_drop:
            if not prev_drop_active:                 # gap just started: capture heading
                head0 = head[i - 1] if i > 0 else None
            if r.get('in_fov') == '1':
                pu, pv = fnum(r.get('proj_u')), fnum(r.get('proj_v'))
                if pu is not None and pv is not None:
                    e = math.hypot(kfx - pu, kfy - pv)
                    all_drop.append(e)
                    gp = 'EARLY' if gappos <= MAXDROP else 'LATE'
                    turn = 'STRAIGHT'
                    h = head[i]
                    if head0 is not None and h is not None and angdiff(h, head0) > math.radians(TURN_DEG):
                        turn = 'TURNING'
                    buckets[(gp, turn)].append(e)
    return all_drop, buckets


def main():
    streams = sys.argv[1:]
    hdr = "%-16s | %-5s | %8s | %8s %8s | %8s %8s | %5s" % (
        "stream", "damp", "drpAll", "earlyStr", "earlyTrn", "lateStr", "lateTrn", "nDrop")
    print(hdr); print("-" * len(hdr))
    agg = {d: {'all': [], ('EARLY', 'STRAIGHT'): [], ('EARLY', 'TURNING'): [],
               ('LATE', 'STRAIGHT'): [], ('LATE', 'TURNING'): []} for d in DAMPS}
    for path in streams:
        rows = list(csv.DictReader(open(path)))
        name = path.split('/')[-1].replace('.csv', '')
        head = gt_headings(rows)
        for d in DAMPS:
            alld, b = replay(rows, d, head)
            print("%-16s | %-5.2f | %8.2f | %8.2f %8.2f | %8.2f %8.2f | %5d" % (
                name, d, rms(alld),
                rms(b[('EARLY', 'STRAIGHT')]), rms(b[('EARLY', 'TURNING')]),
                rms(b[('LATE', 'STRAIGHT')]), rms(b[('LATE', 'TURNING')]), len(alld)))
            agg[d]['all'] += alld
            for k in b:
                agg[d][k] += b[k]
        print("-" * len(hdr))

    print("\n=== POOLED across all streams (RMS px, kf vs GT proj, dropout frames) ===")
    print("%-5s | %8s | %8s %8s | %8s %8s | %6s %6s" % (
        "damp", "drpAll", "earlyStr", "earlyTrn", "lateStr", "lateTrn", "nEarly", "nLate"))
    for d in DAMPS:
        a = agg[d]
        ne = len(a[('EARLY', 'STRAIGHT')]) + len(a[('EARLY', 'TURNING')])
        nl = len(a[('LATE', 'STRAIGHT')]) + len(a[('LATE', 'TURNING')])
        print("%-5.2f | %8.2f | %8.2f %8.2f | %8.2f %8.2f | %6d %6d" % (
            d, rms(a['all']),
            rms(a[('EARLY', 'STRAIGHT')]), rms(a[('EARLY', 'TURNING')]),
            rms(a[('LATE', 'STRAIGHT')]), rms(a[('LATE', 'TURNING')]), ne, nl))
    print("\nEARLY = <=%d dropout frames (damping live); LATE = >%d (velocity hard-zeroed)." % (MAXDROP, MAXDROP))
    print("STRAIGHT/TURNING = GT heading change since gap start <=/> %.0f deg. current damp=0.88." % TURN_DEG)


if __name__ == '__main__':
    main()
