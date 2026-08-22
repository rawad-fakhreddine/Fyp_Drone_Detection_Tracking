#!/usr/bin/env python3
"""
rl_bc_dataset.py — RL Milestone (Config 3) · BC data: flight CSVs -> training pairs.

Converts the existing 20 Hz flight logs (the IBVS teacher flying the calibrated
baseline) into (observation, action) pairs for the behaviour-cloning warm-start.
OFFLINE tool — no ROS needed; reads ~/fyp/Results/Config1/*.csv, writes one .npz.

Observation reconstruction (MUST match rl_env.py ObsBuilder.build() — same
constants, same clipping; see FYP/RL/observation/Observation_Design.docx §7):
    ex      = ex_ctrl        (pitch-compensated, exactly as the controller saw it)
    ey      = ey_ctrl
    d_hat   = sqrt(0.077 / raw_alpha)          (recomputed with the CALIBRATED k,
                                                regardless of the k the run flew)
    dex,dey = logged controller rates (deriv_lpf 0.6 — same filter rl_env uses)
    dd      = EMA(0.6) finite difference of d_hat (not logged; reconstructed)
    w, h    = filled from alpha + iris aspect prior r=2.0  (w=sqrt(A*r), h=sqrt(A/r))
              — the TEACHER never uses w/h/conf, so these dims carry no BC signal;
              they only need plausible runtime statistics.
    conf    = 1.0 on REAL frames, 0.0 otherwise (proxy)
    t_nodet = seconds since last REAL row (clip 1.0)
    pitch/roll = pitch_deg/roll_deg -> radians   (sign convention validated by the
                 probe flight; BC is robust to a global sign either way)
    a_prev  = previous row's cmd_* / caps [8.0, 1.2, 2.5, 0.5]
Action label = this row's cmd_* / caps, clipped to [-1,1].

Row filter: phase in {APPROACH, HOLD} only (the tracking regime the RL is scoped
to — no TAKEOFF/SEARCH/DISARMED rows). Files are the train/val split unit.

Frame stacking: emits X (N,16) single-frame plus Xs (N,64) = 4 consecutive frames
concatenated OLDEST-FIRST (matches SB3 VecFrameStack ordering for vector obs).
Stacks never cross a file boundary or a time gap > 0.15 s.

Usage:
  python3 rl_bc_dataset.py --since 2026-08-06 --out ~/fyp/rl/datasets/bc_v1.npz
"""
import os, csv, glob, math, argparse, re
import numpy as np

# ---- constants: MUST match rl_env.py -----------------------------------------
K_CAL      = 0.077
CAPS       = np.array([8.0, 1.2, 2.5, 0.5], dtype=np.float32)
RATE_LPF   = 0.6
ASPECT_R   = 3.3     # measured live box aspect (w~43px / h~13px at 6-7 m); was 2.0
CONF_PROXY = 0.85    # live YOLO conf ~0.8 on REAL frames; was a 1.0 proxy
AREA_NORM  = 640.0 * 480.0
STACK_N    = 4
OBS_NAMES  = ["ex","ey","d_hat","dex","dey","dd","w","h","conf","t_nodet",
              "pitch","roll","a_vx","a_vy","a_vz","a_wz"]

def normalize(ex, ey, d, dex, dey, dd, w, h, conf, t_nd, pitch, roll, a_prev):
    """Identical clipping/scaling to rl_env.ObsBuilder.build()."""
    return np.array([
        np.clip(ex, -1.5, 1.5),
        np.clip(ey, -1.5, 1.5),
        np.clip(d / 20.0, 0.0, 2.0),
        np.clip(dex / 2.0, -1.0, 1.0),
        np.clip(dey / 2.0, -1.0, 1.0),
        np.clip(dd  / 5.0, -1.0, 1.0),
        np.clip(w / 100.0, 0.0, 3.0),
        np.clip(h / 100.0, 0.0, 3.0),
        np.clip(conf, 0.0, 1.0),
        min(t_nd, 1.0),
        np.clip(pitch / 0.5, -1.0, 1.0),
        np.clip(roll  / 0.5, -1.0, 1.0),
        a_prev[0], a_prev[1], a_prev[2], a_prev[3],
    ], dtype=np.float32)

def f(row, key, default=float('nan')):
    try:
        v = float(row[key]);  return v if not math.isnan(v) else default
    except (KeyError, ValueError, TypeError):
        return default

def convert_file(path):
    """-> (X (n,16), Y (n,4), t (n,), d_true (n,)) for tracking-regime rows of one run.
    d_true = Gazebo ground-truth 3-D distance (for reward recompute in the RL prefill;
    the BC trainer ignores it)."""
    X, Y, T, D = [], [], [], []
    d_prev = None; dd = 0.0; t_prev = None
    t_last_real = None; a_prev = np.zeros(4, dtype=np.float32)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t = f(row, 'sim_time')
            if math.isnan(t): continue
            det_real = (row.get('raw_det') == 'REAL')
            if det_real: t_last_real = t
            # a_prev must advance over ALL rows (it is the previous command,
            # whatever phase it was issued in)
            cmd = np.array([f(row,'cmd_vx',0), f(row,'cmd_vy',0),
                            f(row,'cmd_vz',0), f(row,'cmd_wz',0)], dtype=np.float32)
            cmd_n = np.clip(cmd / CAPS, -1.0, 1.0)
            phase = row.get('phase', '')
            alpha = f(row, 'raw_alpha')
            if phase in ('APPROACH', 'HOLD') and det_real and alpha > 1e-9:
                d = min(math.sqrt(K_CAL / alpha), 40.0)
                if d_prev is not None and t_prev is not None and 0 < t - t_prev < 0.15:
                    dd = RATE_LPF*dd + (1-RATE_LPF)*(d - d_prev)/(t - t_prev)
                else:
                    dd = 0.0
                d_prev, t_prev = d, t
                area_px = alpha * AREA_NORM
                w = math.sqrt(area_px * ASPECT_R); h = math.sqrt(area_px / ASPECT_R)
                t_nd = 0.0 if t_last_real is None else min(t - t_last_real, 1.0)
                obs = normalize(
                    f(row,'ex_ctrl',0), f(row,'ey_ctrl',0), d,
                    f(row,'dex',0), f(row,'dey',0), dd,
                    w, h, CONF_PROXY, t_nd,
                    math.radians(f(row,'pitch_deg',0)), math.radians(f(row,'roll_deg',0)),
                    a_prev)
                X.append(obs); Y.append(cmd_n); T.append(t)
                D.append(f(row, 'true_dist_3d'))     # GT distance for RL reward recompute
            else:
                d_prev = None; dd = 0.0        # rate chain breaks outside tracking
            a_prev = cmd_n
    return (np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32),
            np.array(T), np.array(D, dtype=np.float32))

def stack(X, T):
    """(n,16),(n,) -> (m, 16*STACK_N) oldest-first, only over contiguous time."""
    if len(X) < STACK_N: return np.zeros((0, 16*STACK_N), dtype=np.float32), np.zeros(0, dtype=np.int64)
    idx = []
    for i in range(STACK_N-1, len(X)):
        if T[i] - T[i-STACK_N+1] < 0.15*STACK_N:
            idx.append(i)
    idx = np.array(idx, dtype=np.int64)
    Xs = np.concatenate([X[idx-(STACK_N-1-j)] for j in range(STACK_N)], axis=1)
    return Xs, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser('~/fyp/Results/Config1'))
    ap.add_argument('--since', default='2026-08-06')
    ap.add_argument('--until', default='2026-12-31',
                    help='exclude CSVs dated AFTER this (keeps RL-contaminated runs out; '
                         'e.g. --until 2026-08-16 drops the post-handoff thrash CSVs)')
    ap.add_argument('--out', default=os.path.expanduser('~/fyp/rl/datasets/bc_v1.npz'))
    ap.add_argument('--val-files', type=int, default=3, help='newest N files held out for validation')
    ap.add_argument('--zero-aprev', action='store_true',
                    help='zero the 4 a_prev columns (12:16) so the clone learns to IGNORE its '
                         'own last action -> removes the closed-loop runaway channel (SAC v2). '
                         'Pair with rl_env ~zero_aprev:=true at deploy.')
    a = ap.parse_args()

    files = []
    for p in sorted(glob.glob(os.path.join(a.src, 'traj*_zone*_*.csv'))):
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_', p)
        if m and a.since <= m.group(1) <= a.until: files.append(p)
    if not files:
        print("no files matched"); return
    print(f"{len(files)} files since {a.since}")

    val_set = set(sorted(files, key=os.path.getmtime)[-a.val_files:])
    tr, va = {'X':[], 'Y':[], 'Xs':[], 'Ys':[]}, {'X':[], 'Y':[], 'Xs':[], 'Ys':[]}
    kept = 0
    for p in files:
        X, Y, T, _D = convert_file(p)          # _D (GT dist) used by the RL prefill, not BC
        if len(X) < STACK_N: continue
        if a.zero_aprev and len(X):
            X[:, 12:16] = 0.0                  # a_vx,a_vy,a_vz,a_wz -> ignored by the clone
        Xs, idx = stack(X, T)
        dst = va if p in val_set else tr
        dst['X'].append(X); dst['Y'].append(Y)
        dst['Xs'].append(Xs); dst['Ys'].append(Y[idx])
        kept += len(X)
        print(f"  {os.path.basename(p):48s} rows={len(X):6d} stacked={len(Xs):6d}"
              f"{'  [VAL]' if p in val_set else ''}")
    out = {}
    for name, d in (('train', tr), ('val', va)):
        for k in d:
            out[f'{name}_{k}'] = np.concatenate(d[k]) if d[k] else np.zeros((0,))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, obs_names=np.array(OBS_NAMES), caps=CAPS,
                        k_cal=K_CAL, stack_n=STACK_N, **out)
    print(f"\nsaved {a.out}")
    print(f"train: {out['train_Xs'].shape[0]} stacked pairs · val: {out['val_Xs'].shape[0]}"
          f" · total raw rows kept: {kept}")

if __name__ == '__main__':
    main()
