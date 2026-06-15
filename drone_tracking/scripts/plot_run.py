#!/usr/bin/env python3
"""
plot_run.py — trajectory + run QA plots from flight_logger CSVs (M9.6 step 3).

Usage:
  python3 plot_run.py CSV [--out PNG]      one run  -> one 4-panel PNG
  python3 plot_run.py --batch DIR          all DIR/*.csv (skips PNGs newer
                                           than their CSV); one-line summary
                                           per run.

Panels:
  1. Top-down XY     — target + chaser paths, equal aspect, spawns marked.
                       NOTE: the logger records each drone in its OWN MAVROS
                       local frame (origin = that drone's spawn; both frames
                       ENU-aligned). World XY is never logged, so this panel
                       is exact for SHAPE QA but the spawn offset between the
                       two paths is not represented — true separation is
                       panel 3 (true_dist_3d, Gazebo world frame).
  2. Altitude vs t   — chaser pos_z + target target_pz; shaded where
                       phase != HOLD; red marks where emerg=1.
  3. Separation vs t — true_dist_3d; refs 3 m / 6 m; min annotated; emerg marks.
  4. alpha vs t      — log scale; refs alpha_star=0.0067, HOLD ceiling 0.0167,
                       ALPHA_EMERGENCY=0.033, exit 0.0231.

Older CSVs without the emerg column are handled (overlay skipped, no crash).

Expected trajectory shapes (QA reference, target path in panel 1):
  T1 point | T2/T3 straight line (bounce at 60 m) | T4 circle R=8 |
  T5 figure-8 (Gerono, a=8) | T6 inclined 15 deg line | T7 inclined 35 deg
  line | T8 helix R=8 with slow climb/descend (vz 0.15) | T9 irregular
  evasion. Trajectories are velocity-commanded, so mild drift of the shape
  over time is EXPECTED and not a bug.
"""
import os, re, csv, glob, argparse

import matplotlib
matplotlib.use('Agg')            # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

ALPHA_STAR      = 0.0067
ALPHA_HOLD_CEIL = 0.0167         # alpha_star + ea_HOLD band
ALPHA_EMERG     = 0.033          # IBVS v6.26 ~alpha_emergency
ALPHA_EMERG_EXIT= 0.0231         # 0.7 x


def load_csv(path):
    """Return dict of numpy arrays (floats; '' -> nan) + string columns."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {}
    keys = rows[0].keys()
    str_cols = ('phase', 'raw_det', 'flt_det', 'armed', 'emerg')
    for k in keys:
        if k in str_cols:
            cols[k] = np.array([r.get(k, '') or '' for r in rows])
        else:
            vals = []
            for r in rows:
                v = r.get(k, '')
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(np.nan)
            cols[k] = np.array(vals)
    cols['_n'] = len(rows)
    return cols


def parse_traj(path):
    m = re.search(r'traj(\d+)_zone(\d+)', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def run_was_aborted(csv_path):
    """'1' if the sibling .analysis.txt carries the RUN ABORTED tag, '?' if
    no analysis file exists (the CSV alone does not record the abort)."""
    ana = csv_path[:-4] + '.analysis.txt' if csv_path.endswith('.csv') else None
    if not ana or not os.path.exists(ana):
        return '?'
    with open(ana, errors='replace') as f:
        return '1' if 'ABORTED' in f.read() else '0'


def shade_non_hold(ax, t, phase):
    """Shaded bands wherever phase != HOLD (SEARCH/APPROACH visible at a glance)."""
    bad = phase != 'HOLD'
    if not bad.any():
        return
    # contiguous runs of bad
    edges = np.flatnonzero(np.diff(bad.astype(int)))
    starts = [0] if bad[0] else []
    starts += [e + 1 for e in edges if bad[e + 1]]
    ends = [e for e in edges if bad[e]]
    if bad[-1]:
        ends.append(len(bad) - 1)
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color='orange', alpha=0.15, lw=0)


def emerg_mask(d):
    if 'emerg' not in d:
        return None
    m = d['emerg'] == '1'
    return m if m.any() else None


def mark_emerg(ax, t, y, mask):
    if mask is not None:
        ax.plot(t[mask], y[mask], 'r.', ms=4, label='emerg=1', zorder=5)


def plot_run(csv_path, out_path=None):
    d = load_csv(csv_path)
    if d is None:
        print(f"[plot_run] SKIP empty CSV: {csv_path}")
        return None
    t = d['sim_time']
    traj, zone = parse_traj(csv_path)
    em = emerg_mask(d)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()
    title = os.path.basename(csv_path)
    if traj:
        title += f"   (T{traj}, zone {zone})"
    fig.suptitle(title, fontsize=12)

    # ── 1. Top-down XY (per-drone local frames, ENU-aligned) ────────────
    ok_t = ~np.isnan(d['target_px']) & ~np.isnan(d['target_py'])
    ok_c = ~np.isnan(d['pos_x']) & ~np.isnan(d['pos_y'])
    if ok_c.any():
        ax1.plot(d['pos_x'][ok_c], d['pos_y'][ok_c], color='grey', lw=0.8,
                 alpha=0.7, label='chaser (own local frame)')
        ax1.plot(d['pos_x'][ok_c][0], d['pos_y'][ok_c][0], 's', color='grey',
                 ms=8, label='chaser spawn')
    if ok_t.any():
        sc = ax1.scatter(d['target_px'][ok_t], d['target_py'][ok_t],
                         c=t[ok_t], cmap='viridis', s=3, label='target (own local frame)')
        fig.colorbar(sc, ax=ax1, label='sim time (s)')
        ax1.plot(d['target_px'][ok_t][0], d['target_py'][ok_t][0], '^',
                 color='green', ms=9, label='target spawn')
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.set_title('Top-down XY — shape QA only (frames share orientation, NOT origin)',
                  fontsize=9)
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(alpha=0.3)

    # ── 2. Altitude vs time ─────────────────────────────────────────────
    ax2.plot(t, d['pos_z'], color='grey', lw=0.9, label='chaser z (local)')
    ax2.plot(t, d['target_pz'], color='green', lw=0.9, label='target z (local)')
    shade_non_hold(ax2, t, d['phase'])
    mark_emerg(ax2, t, d['pos_z'], em)
    ax2.set_xlabel('sim time (s)'); ax2.set_ylabel('altitude (m)')
    ax2.set_title('Altitude — shaded = phase != HOLD', fontsize=9)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # ── 3. Separation vs time ───────────────────────────────────────────
    sep = d['true_dist_3d']
    ok = ~np.isnan(sep)
    ax3.plot(t[ok], sep[ok], color='steelblue', lw=0.9, label='true_dist_3d')
    ax3.axhline(3.0, color='green', ls='--', lw=0.8, label='HOLD band 3-6 m')
    ax3.axhline(6.0, color='green', ls='--', lw=0.8)
    mark_emerg(ax3, t, sep, em)
    min_sep = np.nan
    if ok.any():
        i = np.nanargmin(sep)
        min_sep = sep[i]
        ax3.annotate(f'min {min_sep:.2f} m @ {t[i]:.1f}s', xy=(t[i], min_sep),
                     xytext=(t[i], min_sep + 2),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     color='red', fontsize=8)
    ax3.set_xlabel('sim time (s)'); ax3.set_ylabel('separation (m)')
    ax3.set_title('True 3D separation (Gazebo world frame)', fontsize=9)
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # ── 4. alpha vs time (log) ──────────────────────────────────────────
    alpha = d['flt_alpha']
    valid = (d['flt_det'] != 'NONE') & ~np.isnan(alpha) & (alpha > 0)
    ax4.plot(t[valid], alpha[valid], '.', color='steelblue', ms=2, label='flt_alpha')
    for v, c, lab in ((ALPHA_STAR, 'green', 'alpha* 0.0067'),
                      (ALPHA_HOLD_CEIL, 'olive', 'HOLD ceil 0.0167'),
                      (ALPHA_EMERG_EXIT, 'orange', 'emerg exit 0.0231'),
                      (ALPHA_EMERG, 'red', 'EMERGENCY 0.033')):
        ax4.axhline(v, color=c, ls='--', lw=0.9, label=lab)
    if em is not None:
        mark_emerg(ax4, t, alpha, em & valid)
    ax4.set_yscale('log')
    ax4.set_xlabel('sim time (s)'); ax4.set_ylabel('alpha (log)')
    ax4.set_title('alpha vs IBVS v6.26 guard thresholds', fontsize=9)
    ax4.legend(fontsize=7, loc='best'); ax4.grid(alpha=0.3, which='both')

    out = out_path or (csv_path[:-4] + '.png')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110)
    plt.close(fig)

    # one-line summary stats
    hold_pct = 100.0 * np.count_nonzero(d['phase'] == 'HOLD') / d['_n']
    n_emerg = 0
    if em is not None:
        e = (d['emerg'] == '1').astype(int)
        n_emerg = int(np.count_nonzero(np.diff(e) == 1) + (1 if e[0] else 0))
    return dict(traj=traj, zone=zone, hold_pct=hold_pct, min_sep=min_sep,
                emerg_count=(n_emerg if 'emerg' in d else '-'),
                aborted=run_was_aborted(csv_path), out=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('csv', nargs='?', help='one flight CSV')
    ap.add_argument('--out', help='output PNG (single-run mode)')
    ap.add_argument('--batch', metavar='DIR', help='render every DIR/*.csv')
    args = ap.parse_args()

    if args.batch:
        csvs = sorted(glob.glob(os.path.join(os.path.expanduser(args.batch), '*.csv')))
        if not csvs:
            print(f"[plot_run] no CSVs in {args.batch}"); return
        for c in csvs:
            png = c[:-4] + '.png'
            if os.path.exists(png) and os.path.getmtime(png) > os.path.getmtime(c):
                print(f"  skip (PNG up to date): {os.path.basename(c)}")
                continue
            s = plot_run(c)
            if s:
                ms = f"{s['min_sep']:.2f}m" if not np.isnan(s['min_sep']) else 'n/a'
                print(f"  T{s['traj']} z{s['zone']}  HOLD {s['hold_pct']:5.1f}%  "
                      f"min_sep {ms}  emerg {s['emerg_count']}  "
                      f"aborted {s['aborted']}  -> {os.path.basename(s['out'])}")
    elif args.csv:
        s = plot_run(os.path.expanduser(args.csv), args.out)
        if s:
            print(f"✓ {s['out']}")
    else:
        ap.error('give a CSV or --batch DIR')


if __name__ == '__main__':
    main()
