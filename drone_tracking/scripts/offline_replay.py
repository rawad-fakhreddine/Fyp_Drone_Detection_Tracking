#!/usr/bin/env python3
"""
offline_replay.py — confound-free R comparison.
Replays a recorded raw-YOLO measurement stream (from an estimation_logger CSV)
through a faithful copy of kalman_filter_node's filter with old-R vs new-R, and
scores each filtered estimate against the SAME ground-truth on IDENTICAL frames.
This removes the closed-loop path divergence that confounds live cross-R runs.

The KF copy mirrors kalman_filter_node.py exactly: x0/P0, F(dt=1/20), H, Q,
velocity_damping=0.88, alpha-collapse + pixel-jump outlier rules, MAX_REJ=4,
dropout prediction, state clip. Only R differs between the two passes.

Usage: offline_replay.py CSV [CSV ...]
"""
import sys, csv, math
import numpy as np


class KF:
    OUTLIER_PREV_ALPHA = 3000.0
    OUTLIER_NEW_ALPHA = 900.0
    PIXEL_JUMP = 180.0
    MAX_REJ = 4

    def __init__(self, R_diag):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 500.0
        dt = 1.0 / 20.0
        self.F = np.eye(6); self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        self.H = np.zeros((3, 6)); self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.R = np.diag(R_diag)
        self.Q = np.diag([0.5, 0.5, 3.0, 6.0, 6.0, 3.0])
        self.damp = 0.88
        self.init = False; self.drop = 0; self.rej = 0; self.maxdrop = 30

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
        prev = self.x[2]
        jump = math.hypot(mx - self.x[0], my - self.x[1])
        if prev > self.OUTLIER_PREV_ALPHA and mz < self.OUTLIER_NEW_ALPHA:
            return True
        if jump > self.PIXEL_JUMP and self.rej < self.MAX_REJ:
            return True
        return False

    def step(self, mx, my, mz):
        """Mirror kalman callback. Returns (cx, cy, alpha, is_dropout) or None."""
        is_drop = (mx != mx) or (my != my) or (mz != mz)   # nan check
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
        return (self.x[0], self.x[1], self.x[2], is_drop)


def fnum(s):
    try:
        v = float(s)
        return v if v == v else None
    except (ValueError, TypeError):
        return None


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float('nan')


def std(xs):
    n = len(xs)
    if n < 2: return float('nan')
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def replay(rows, R_diag):
    kf = KF(R_diag)
    e = {'cx': [], 'a': []}; d = {'cx': [], 'a': []}
    for r in rows:
        mx, my, mz = fnum(r['raw_cx']), fnum(r['raw_cy']), fnum(r['raw_alpha'])
        if r.get('raw_det') != 'REAL' or None in (mx, my, mz):
            mx = my = mz = float('nan')           # dropout frame
        out = kf.step(mx, my, mz)
        if out is None:
            continue
        if r.get('in_fov') != '1':
            continue
        pu, pv, pa = fnum(r['proj_u']), fnum(r['proj_v']), fnum(r['proj_alpha'])
        if None in (pu, pa):
            continue
        ecx = out[0] - pu; ea = out[2] - pa
        e['cx'].append(ecx); e['a'].append(ea)
        if out[3]:
            d['cx'].append(ecx); d['a'].append(ea)
    return e, d


def raw_baseline(rows):
    rc, ra = [], []
    for r in rows:
        if r.get('in_fov') != '1' or r.get('raw_det') != 'REAL':
            continue
        mx, ma = fnum(r['raw_cx']), fnum(r['raw_alpha'])
        pu, pa = fnum(r['proj_u']), fnum(r['proj_alpha'])
        if None in (mx, ma, pu, pa):
            continue
        rc.append(mx - pu); ra.append(ma - pa)
    return rc, ra


def main():
    print("%-16s | %7s | %8s %8s | %8s %8s || %9s %9s | %10s %10s" % (
        "stream", "rawErrcx", "kfErrcx_o", "kfErrcx_n",
        "drpcx_o", "drpcx_n", "rawErra", "kfErra_o", "kfErra_n", "drpa_o/n"))
    print("-" * 130)
    for path in sys.argv[1:]:
        rows = list(csv.DictReader(open(path)))
        rc, ra = raw_baseline(rows)
        eo, do = replay(rows, [6.0, 6.0, 5.0])
        en, dn = replay(rows, [25.0, 15.0, 3000.0])
        name = path.split('/')[-1].replace('.csv', '')
        print("%-16s | %7.1f | %8.1f %8.1f | %8.1f %8.1f || %9.1f %9.1f %9.1f | %5.1f/%-5.1f" % (
            name, rms(rc),
            rms(eo['cx']), rms(en['cx']), rms(do['cx']), rms(dn['cx']),
            rms(ra), rms(eo['a']), rms(en['a']), rms(do['a']), rms(dn['a'])))
    print("\n(All errors RMS vs ground truth, in-FOV frames; same raw stream through")
    print(" both R. _o=old-R[6,6,5]  _n=new-R[25,15,3000]  drp=dropout/prediction frames.)")


if __name__ == '__main__':
    main()
