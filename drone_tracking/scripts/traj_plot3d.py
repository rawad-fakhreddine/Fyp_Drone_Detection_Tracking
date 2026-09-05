#!/usr/bin/env python3
"""traj_plot3d.py — 3D actual-vs-ideal target-trajectory plots + error.

Reads the flight_logger target_wx/wy/wz (world XYZ) columns from the official
matrix CSVs, trims to the MOVING segment, and overlays the SPEC formula path
(M9.1) so the deviation is visible and quantified. T1-T8 are velocity-commanded
(PX4 integrates them open-loop), so drift from the ideal is expected — this
tool measures how much.

Output: per-trajectory PNG + a 2x4 montage, under OUTDIR.
Usage:  traj_plot3d.py [SEED=42] [MATRIX_DIR] [OUTDIR]
"""
import csv, math, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DEF_MDIR = os.path.expanduser('~/fyp/Results_reference/09_Official_Matrix')
DEF_OUT = os.path.expanduser('~/fyp/Results_reference/10_Trajectory_Verify')

NAMES = {1: 'Static hover', 2: 'Slow straight 1.0 m/s', 3: 'Fast straight 3.5 m/s',
         4: 'Circular orbit R=8 T=25', 5: 'Lemniscate a=8 T=40',
         6: 'Inclined 15 deg 2.0 m/s', 7: 'Inclined 35 deg 3.0 m/s',
         8: 'Up-down helix R=8', 10: 'Random waypoints',
         11: 'Random waypoints + speed'}
ACT = '#1f77b4'   # flown path
IDE = '#d62728'   # ideal formula
GRN = '#2ca02c'   # start


def load(path):
    rows = [r for r in csv.DictReader(open(path))
            if r.get('target_wx') not in (None, '', 'nan')]
    tx = np.array([float(r['target_wx']) for r in rows])
    ty = np.array([float(r['target_wy']) for r in rows])
    tz = np.array([float(r['target_wz']) for r in rows])
    ts = np.array([float(r['sim_time']) for r in rows])
    i0 = next((i for i, z in enumerate(tz) if z > 12.5), 0) + 50  # MOVING seg
    return tx[i0:], ty[i0:], tz[i0:], ts[i0:]


def _prindir(tx, ty, cx, cy):
    # P is 2xN; the 2D principal direction is the first LEFT singular vector
    # (column of U), NOT a right singular vector (which is N-dimensional).
    P = np.vstack([tx - cx, ty - cy])
    U, _, _ = np.linalg.svd(P, full_matrices=False)
    return U[:, 0]  # unit principal direction in XY


def ideal(t, tx, ty, tz, ts):
    """Return (ix, iy, iz, label, errtext). Single point => scatter marker."""
    dt = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
    cx, cy, cz = tx.mean(), ty.mean(), tz.mean()

    if t == 1:
        rad = np.sqrt((tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2)
        return (np.array([cx]), np.array([cy]), np.array([cz]),
                'Theoretical: fixed point',
                'drift mean %.2f m, max %.2f m (spec ~0)' % (rad.mean(), rad.max()))

    if t in (2, 3):
        spd = 1.0 if t == 2 else 3.5
        d = _prindir(tx, ty, cx, cy)
        proj = (tx - cx) * d[0] + (ty - cy) * d[1]
        s = np.linspace(proj.min(), proj.max(), 50)
        ix, iy, iz = cx + s * d[0], cy + s * d[1], np.full(50, cz)
        lat = -(tx - cx) * d[1] + (ty - cy) * d[0]
        vh = np.hypot(np.diff(tx), np.diff(ty)) / dt
        return (ix, iy, iz, 'Theoretical: straight %.1f m/s' % spd,
                'speed %.2f m/s (spec %.1f) | lateral RMS %.2f m | z std %.2f m'
                % (vh.mean(), spd, math.sqrt((lat**2).mean()), tz.std()))

    if t == 4:
        R, th = 8.0, np.linspace(0, 2 * np.pi, 200)
        ix, iy, iz = cx + R * np.cos(th), cy + R * np.sin(th), np.full(200, cz)
        r = np.hypot(tx - cx, ty - cy)
        return (ix, iy, iz, 'Theoretical: circle R=8',
                'radius %.2f+/-%.2f m (spec 8)' % (r.mean(), r.std()))

    if t == 8:                              # helix = ideal R=8 circle + the real up/down
        ang = np.arctan2(ty - cy, tx - cx)
        ix, iy, iz = cx + 8 * np.cos(ang), cy + 8 * np.sin(ang), tz.copy()
        r = np.hypot(tx - cx, ty - cy)
        return (ix, iy, iz, 'Theoretical: helix R=8 (up+down)',
                'radius %.2f m (spec 8) | z %.1f-%.1f m'
                % (r.mean(), tz.min(), tz.max()))

    if t == 5:
        d = _prindir(tx, ty, cx, cy)
        ang = math.atan2(d[1], d[0])
        s = np.linspace(0, 2 * np.pi, 400)
        uu, ww = 8 * np.sin(s), 4 * np.sin(2 * s)
        ix = cx + uu * math.cos(ang) - ww * math.sin(ang)
        iy = cy + uu * math.sin(ang) + ww * math.cos(ang)
        iz = np.full(400, cz)
        up = (tx - cx) * math.cos(ang) + (ty - cy) * math.sin(ang)
        wp = -(tx - cx) * math.sin(ang) + (ty - cy) * math.cos(ang)
        return (ix, iy, iz, 'Theoretical: figure-8 a=8,4',
                'semi-axes %.1f/%.1f m (spec 8/4)'
                % ((up.max() - up.min()) / 2, (wp.max() - wp.min()) / 2))

    if t in (6, 7):
        spd, sl = (2.0, 15.0) if t == 6 else (3.0, 35.0)
        d = _prindir(tx, ty, cx, cy)
        s = (tx - cx) * d[0] + (ty - cy) * d[1]        # along-track position
        # ideal = straight inclined shuttle: real along-track + real altitude
        # (both are well tracked), so the intended bounce shows on the reference
        # instead of appearing to oscillate around a flat line.
        ix, iy, iz = cx + s * d[0], cy + s * d[1], tz.copy()
        vh = np.hypot(np.diff(tx), np.diff(ty)) / dt
        vz = np.abs(np.diff(tz)) / dt
        m = vh > 0.3
        slope = np.degrees(np.arctan2(vz[m], vh[m]))
        return (ix, iy, iz, 'Theoretical: incline %.0f deg (climb/descend)' % sl,
                'slope %.1f deg (spec %.0f) | z %.1f-%.1f m'
                % (np.median(slope), sl, tz.min(), tz.max()))
    if t in (10, 11):                      # random waypoints: no formula overlay
        return (np.array([tx[0]]), np.array([ty[0]]), np.array([tz[0]]),
                'start (random-waypoint path)',
                'seeded random path | x[%.0f,%.0f] y[%.0f,%.0f] z[%.0f,%.0f] m'
                % (tx.min(), tx.max(), ty.min(), ty.max(), tz.min(), tz.max()))
    return np.array([cx]), np.array([cy]), np.array([cz]), '', ''


def _set_scale(ax, xs, ys, zs, min_half=1.0):
    # Enforce a minimum half-span so near-perfect (cm-scale) paths are not
    # auto-zoomed until their tiny residual fills the panel and looks messy.
    spans = []
    for setlim, a in ((ax.set_xlim, xs), (ax.set_ylim, ys), (ax.set_zlim, zs)):
        c = 0.5 * (a.min() + a.max())
        half = max(0.5 * (a.max() - a.min()), min_half)
        setlim(c - half, c + half); spans.append(2 * half)
    m = max(spans)
    ax.set_box_aspect((spans[0] / m, spans[1] / m, max(spans[2] / m, 0.12)))


def draw(ax, t, tx, ty, tz, ix, iy, iz, ilab, err, title=True):
    ax.plot(tx, ty, tz, color=ACT, lw=1.1, label='Real (simulated)')
    if len(ix) > 1:
        ax.plot(ix, iy, iz, color=IDE, lw=1.8, ls='--', label=ilab)
    else:
        ax.scatter(ix, iy, iz, color=IDE, s=70, marker='*', label=ilab)
    ax.scatter([tx[0]], [ty[0]], [tz[0]], color=GRN, s=35, marker='o', label='start')
    ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
    ax.set_zlabel('Z (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-58)
    _set_scale(ax, np.concatenate([tx, np.ravel(ix)]),
               np.concatenate([ty, np.ravel(iy)]),
               np.concatenate([tz, np.ravel(iz)]))
    if title:
        ax.set_title('T%d - %s\n%s' % (t, NAMES[t], err), fontsize=9)
    ax.legend(loc='upper left', fontsize=7)


def plot_single(csv_path, t, out):
    tx, ty, tz, ts = load(csv_path)
    ix, iy, iz, ilab, err = ideal(t, tx, ty, tz, ts)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    draw(ax, t, tx, ty, tz, ix, iy, iz, ilab, err)
    fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print('T%d  %-24s  %s\n   -> %s' % (t, NAMES[t], err, out))


def main():
    args = sys.argv[1:]
    # single-CSV mode: traj_plot3d.py <CSV> <TRAJ> [OUTPNG]
    if args and os.path.isfile(args[0]):
        csv_path, t = args[0], int(args[1])
        out = args[2] if len(args) > 2 else os.path.expanduser(
            '~/fyp/Results_reference/10_Trajectory_Verify/T%d_traj3d_fix.png' % t)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_single(csv_path, t, out)
        return
    # batch mode: traj_plot3d.py [SEED] [MATRIX_DIR] [OUTDIR]
    seed = args[0] if args else '42'
    mdir = args[1] if len(args) > 1 else DEF_MDIR
    out = args[2] if len(args) > 2 else DEF_OUT
    os.makedirs(out, exist_ok=True)
    def cpath(tt):
        return os.path.join(mdir, 'T%d_C1_s%s.csv' % (tt, seed))
    for t in range(1, 9):
        tx, ty, tz, ts = load(cpath(t))
        ix, iy, iz, ilab, err = ideal(t, tx, ty, tz, ts)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        draw(ax, t, tx, ty, tz, ix, iy, iz, ilab, err)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'T%d_traj3d_s%s.png' % (t, seed)), dpi=130)
        plt.close(fig)
        print('T%d  %-24s  %s' % (t, NAMES[t], err))
    fig = plt.figure(figsize=(20, 10))
    for t in range(1, 9):
        tx, ty, tz, ts = load(cpath(t))
        ix, iy, iz, ilab, err = ideal(t, tx, ty, tz, ts)
        ax = fig.add_subplot(2, 4, t, projection='3d')
        draw(ax, t, tx, ty, tz, ix, iy, iz, ilab, err)
    fig.suptitle('Target trajectory: Real simulated (blue) vs Theoretical formula (red dashed) - seed %s'
                 % seed, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out, 'ALL_traj3d_s%s.png' % seed), dpi=110)
    plt.close(fig)
    print('\nmontage -> %s' % os.path.join(out, 'ALL_traj3d_s%s.png' % seed))


if __name__ == '__main__':
    main()
