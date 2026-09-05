#!/usr/bin/env python3
"""
q_sweep_replay.py — confound-free Q_vel comparison (offline).
Same machinery as offline_replay.py, but R is HELD at the kept value [6,6,5]
and Q_vel (the cx/cy velocity process-noise diagonal entries) is swept over
{6,8,10,12,14}. Same raw-YOLO stream through every Q -> only Q changes.

For each (stream, Q_vel) vs ground-truth (proj_*) on in-FOV frames:
  - kfErr cx / cy  : RMS estimate error, all in-FOV frames
  - drpErr cx / cy : RMS estimate error on DROPOUT (prediction) frames only
  - velStd vx / vy : std of the filtered velocity estimate (jitter proxy;
                     too-high Q -> jittery velocity)
Q_vel for alpha (index 5) is left at the committed 3.0; only cx/cy vel noise
(indices 3,4) is swept, matching "Q_vel" in the kalman node.

Usage: q_sweep_replay.py CSV [CSV ...]
"""
import sys, csv, math
import numpy as np

Q_VELS = [6, 8, 10, 12, 14]
R_KEPT = [6.0, 6.0, 5.0]


class KF:
    OUTLIER_PREV_ALPHA = 3000.0
    OUTLIER_NEW_ALPHA = 900.0
    PIXEL_JUMP = 180.0
    MAX_REJ = 4

    def __init__(self, q_vel):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 500.0
        dt = 1.0 / 20.0
        self.F = np.eye(6); self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        self.H = np.zeros((3, 6)); self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.R = np.diag(R_KEPT)
        # committed Q with cx/cy velocity entries swept:
        self.Q = np.diag([0.5, 0.5, 3.0, float(q_vel), float(q_vel), 3.0])
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
        return (self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], is_drop)


def fnum(s):
    try:
        v = float(s); return v if v == v else None
    except (ValueError, TypeError):
        return None


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float('nan')


def std(xs):
    n = len(xs)
    if n < 2: return float('nan')
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def replay(rows, q_vel):
    kf = KF(q_vel)
    e = {'cx': [], 'cy': []}; d = {'cx': [], 'cy': []}; v = {'x': [], 'y': []}
    for r in rows:
        mx, my, mz = fnum(r['raw_cx']), fnum(r['raw_cy']), fnum(r['raw_alpha'])
        if r.get('raw_det') != 'REAL' or None in (mx, my, mz):
            mx = my = mz = float('nan')
        out = kf.step(mx, my, mz)
        if out is None:
            continue
        v['x'].append(out[3]); v['y'].append(out[4])
        if r.get('in_fov') != '1':
            continue
        pu, pv, pa = fnum(r['proj_u']), fnum(r['proj_v']), fnum(r['proj_alpha'])
        if None in (pu, pv):
            continue
        ecx = out[0] - pu; ecy = out[1] - pv
        e['cx'].append(ecx); e['cy'].append(ecy)
        if out[5]:
            d['cx'].append(ecx); d['cy'].append(ecy)
    return e, d, v


def main():
    streams = sys.argv[1:]
    # per-stream block
    hdr = "%-18s | %-5s | %7s %7s | %8s %8s | %7s %7s | %5s" % (
        "stream", "Qvel", "kfErrcx", "kfErrcy", "drpErrcx", "drpErrcy",
        "velStdx", "velStdy", "nDrop")
    print(hdr); print("-" * len(hdr))
    agg = {q: {'cx': [], 'cy': [], 'dcx': [], 'dcy': []} for q in Q_VELS}
    for path in streams:
        rows = list(csv.DictReader(open(path)))
        name = path.split('/')[-1].replace('.csv', '')
        for q in Q_VELS:
            e, d, v = replay(rows, q)
            print("%-18s | %-5d | %7.2f %7.2f | %8.2f %8.2f | %7.3f %7.3f | %5d" % (
                name, q, rms(e['cx']), rms(e['cy']), rms(d['cx']), rms(d['cy']),
                std(v['x']), std(v['y']), len(d['cx'])))
            agg[q]['cx'] += e['cx']; agg[q]['cy'] += e['cy']
            agg[q]['dcx'] += d['cx']; agg[q]['dcy'] += d['cy']
        print("-" * len(hdr))

    print("\n=== POOLED across all streams ===")
    print("%-5s | %7s %7s | %8s %8s" % ("Qvel", "kfErrcx", "kfErrcy", "drpErrcx", "drpErrcy"))
    for q in Q_VELS:
        a = agg[q]
        print("%-5d | %7.2f %7.2f | %8.2f %8.2f" % (
            q, rms(a['cx']), rms(a['cy']), rms(a['dcx']), rms(a['dcy'])))
    print("\n(RMS px vs ground truth, in-FOV. R held [6,6,5]. Qvel = cx/cy velocity")
    print(" process-noise diagonal; alpha-vel Q left at 3.0. velStd = filtered")
    print(" velocity std (px/frame) = jitter proxy. current committed Qvel = 6.)")


if __name__ == '__main__':
    main()
