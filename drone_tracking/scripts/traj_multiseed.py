#!/usr/bin/env python3
"""traj_multiseed.py [TRAJ...] — overlay the REAL target path from ALL matrix
seeds against the THEORETICAL formula, per (hard) trajectory.

Each seed's path is centred on its own centroid and rotated so its principal
horizontal axis lines up, so the SHAPES overlay and the spread is visible. This
answers "is the real-vs-theoretical deviation consistent across many runs?" using
the runs we already have (8 seeds x 2 configs in the sealed matrix; target path
is identical for C1/C2 on a seed, so we read C1).

Output: one PNG per trajectory (3D overlay + best 2D projection) + a metric
table, under ~/fyp/Results_reference/10_Trajectory_Verify/multiseed/.
Usage: traj_multiseed.py [5 6 7]   (default: the hard set T5,T6,T7)
"""
import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

MDIR = os.path.expanduser('~/fyp/Results_reference/09_Official_Matrix')
OUT = os.path.expanduser('~/fyp/Results_reference/10_Trajectory_Verify/multiseed')
os.makedirs(OUT, exist_ok=True)
SEEDS = ['42', '43', '45', '46', '47', '48', '49', '50']
TRAJS = [int(a) for a in sys.argv[1:]] or [5, 6, 7]
NAMES = {4: 'Circular orbit R=8', 5: 'Lemniscate a=8 T=40',
         6: 'Inclined 15 deg 2.0 m/s', 7: 'Inclined 35 deg 3.0 m/s',
         8: 'Up-down helix R=8'}
CMAP = plt.get_cmap('viridis')


def load(t, s):
    path = os.path.join(MDIR, 'T%d_C1_s%s.csv' % (t, s))
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path))
            if r.get('target_wx') not in (None, '', 'nan')]
    tx = np.array([float(r['target_wx']) for r in rows])
    ty = np.array([float(r['target_wy']) for r in rows])
    tz = np.array([float(r['target_wz']) for r in rows])
    i0 = next((i for i, z in enumerate(tz) if z > 12.5), 0) + 50
    return tx[i0:], ty[i0:], tz[i0:]


def center_align(tx, ty, tz):
    """Centre on centroid; rotate so the XY principal axis is +X; centre Z."""
    cx, cy, cz = tx.mean(), ty.mean(), tz.mean()
    x, y, z = tx - cx, ty - cy, tz - cz
    U, _, _ = np.linalg.svd(np.vstack([x, y]), full_matrices=False)
    ang = math.atan2(U[0, 1], U[0, 0])
    xr = x * math.cos(ang) + y * math.sin(ang)
    yr = -x * math.sin(ang) + y * math.cos(ang)
    return xr, yr, z


def metric(t, tx, ty, tz):
    xr, yr, zr = center_align(tx, ty, tz)
    if t in (4, 8):
        r = np.hypot(xr, yr)
        return 'R %.2f+/-%.2f (spec 8)' % (r.mean(), r.std())
    if t == 5:
        return 'semi-axes %.1f/%.1f (spec 8/4)' % (
            (xr.max() - xr.min()) / 2, (yr.max() - yr.min()) / 2)
    if t in (6, 7):
        sl = 15.0 if t == 6 else 35.0
        dx = np.diff(xr); dz = np.diff(zr)
        m = np.abs(dx) > 1e-6
        slope = np.degrees(np.arctan2(np.abs(dz[m]), np.abs(dx[m])))
        return 'slope %.1f (spec %.0f) | z-span %.1f m' % (
            np.median(slope), sl, zr.max() - zr.min())
    return ''


def theo(t):
    """Theoretical curve in the centred+aligned frame (x', y', z')."""
    if t in (4, 8):
        a = np.linspace(0, 2 * np.pi, 300)
        return 8 * np.cos(a), 8 * np.sin(a), np.zeros_like(a)
    if t == 5:
        s = np.linspace(0, 2 * np.pi, 400)
        return 8 * np.sin(s), 4 * np.sin(2 * s), np.zeros_like(s)
    if t in (6, 7):
        # Inclined SHUTTLE (target_mover._ti): the horizontal along-track sweep
        # (reverses at INCLINE_MAX_DIST=60 m) and the vertical bounce (reverses
        # at the Z bounds 13..23 m) run INDEPENDENTLY, both at |v|=spd. Each
        # zig/zag leg therefore has slope vb/lat = tan(angle). Simulate it and
        # centre to match the real (centre_align centres on the centroid).
        spd = 2.0 if t == 6 else 3.0
        sl = math.radians(15 if t == 6 else 35)
        lat = spd * math.cos(sl); vb = spd * math.sin(sl)
        dt = 0.05; n = int(200.0 / dt)
        a = 0.0; z = 18.0; ad = 1.0; zd = 1.0
        A = np.empty(n); Z = np.empty(n)
        for i in range(n):
            a += lat * ad * dt
            if a >= 60.0: ad = -1.0
            elif a <= 0.0: ad = 1.0
            z += vb * zd * dt
            if z >= 23.0: zd = -1.0
            elif z <= 13.0: zd = 1.0
            A[i] = a; Z[i] = z
        A -= A.mean(); Z -= Z.mean()          # centre like the real data
        return A, np.zeros_like(A), Z         # straight azimuth -> y'=0
    return None


def plot_traj(t):
    seeds = [(s, load(t, s)) for s in SEEDS]
    seeds = [(s, d) for s, d in seeds if d is not None]
    fig = plt.figure(figsize=(15, 6.5))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    rows = []
    for i, (s, (tx, ty, tz)) in enumerate(seeds):
        xr, yr, zr = center_align(tx, ty, tz)
        col = CMAP(i / max(len(seeds) - 1, 1))
        ax3.plot(xr, yr, zr, color=col, lw=0.8, alpha=0.75)
        if t in (6, 7):                      # side view: along-track vs Z
            ax2.plot(xr, zr, color=col, lw=0.8, alpha=0.75, label='s%s' % s)
        else:                                # top-down XY
            ax2.plot(xr, yr, color=col, lw=0.8, alpha=0.75, label='s%s' % s)
        rows.append((s, metric(t, tx, ty, tz)))
    th = theo(t)
    if th is not None:
        ax3.plot(th[0], th[1], th[2], 'r--', lw=2.0, alpha=0.9, label='Theoretical')
        if t in (6, 7):                      # side view = along-track (x') vs Z (z')
            ax2.plot(th[0], th[2], 'r--', lw=2.0, alpha=0.9, label='Theoretical')
        else:                                # top-down = x' vs y'
            ax2.plot(th[0], th[1], 'r--', lw=2.4, label='Theoretical')
    ax3.set_title('T%d - %s : %d seeds (Real), Theoretical dashed'
                  % (t, NAMES[t], len(seeds)), fontsize=10)
    ax3.set_xlabel('X\' (m)'); ax3.set_ylabel('Y\' (m)'); ax3.set_zlabel('Z\' (m)')
    ax3.view_init(elev=22, azim=-58)
    ax2.set_aspect('equal', 'datalim')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc='best')
    if t in (6, 7):
        ax2.set_xlabel('along-track (m)'); ax2.set_ylabel('Z (m, centred)')
        ax2.set_title('Side view: climb/descend')
    else:
        ax2.set_xlabel('X\' (m)'); ax2.set_ylabel('Y\' (m)')
        ax2.set_title('Top-down: shape spread')
    fig.tight_layout()
    p = os.path.join(OUT, 'T%d_multiseed.png' % t)
    fig.savefig(p, dpi=125); plt.close(fig)
    print('\nT%d %s' % (t, NAMES[t]))
    for s, m in rows:
        print('   s%s  %s' % (s, m))
    print('   -> %s' % p)


if __name__ == '__main__':
    for t in TRAJS:
        plot_traj(t)
