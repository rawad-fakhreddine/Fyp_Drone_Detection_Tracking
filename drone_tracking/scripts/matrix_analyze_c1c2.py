#!/usr/bin/env python3
"""matrix_analyze_c1c2.py — M11 Config 1 vs Config 2 paired analysis.
=============================================================================
Read-only. Reads ~/fyp/Results/summary.csv, isolates the frozen matrix
(zone 1, configs {1,2}, seeds {42,43,45}), and reports per (config,traj)
mean +/- SD over seeds for the FROZEN metrics, plus a paired Config-1-vs-2
test (Wilcoxon signed-rank, paired-t fallback) within each trajectory class.

Panels (frozen):
  control-ceiling   : T2, T4, T5, T8   (bounded/periodic)
  detection-limited : T3, T6, T7       (hard)
  appendix          : T9               (stress — reported, never an A/B verdict)

Frozen metrics (no post-hoc additions):
  hold_pct, hold_mean_sep, detection_rate, mean_recovery_time_s,
  wrong_direction_pct, hold_mean_alt_err  + abort count / mission_duration_s.
NOTE: summary 'detection_rate' is the WHOLE-MISSION raw-det rate (the only
detection column logged), labelled accordingly; it is not HOLD-restricted.

Usage:  matrix_analyze_c1c2.py [--summary ~/fyp/Results/summary.csv]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy import stats

CONTROL = [2, 4, 5, 8]
DETECT  = [3, 6, 7]
APPENDIX = [9]
SEEDS = [42, 43, 45]
ZONE = 1
# (column, label, lower_is_better)
METRICS = [
    ('hold_pct',             'HOLD %',                 False),
    ('hold_mean_sep',        'sep (HOLD) m',           None),
    ('detection_rate',       'det% (mission)',         False),
    ('mean_recovery_time_s', 'recovery s',             True),
    ('wrong_direction_pct',  'wrong-dir %',            True),
    ('hold_mean_alt_err',    'alt err m',              None),
    ('mission_duration_s',   'mission s',              None),
]


def ms(arr):
    a = np.array([x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if a.size == 0:
        return (np.nan, np.nan, 0)
    return (a.mean(), a.std(ddof=1) if a.size > 1 else 0.0, a.size)


def paired_test(c1, c2):
    """c1,c2: equal-length lists paired elementwise; drop NaN pairs."""
    pairs = [(a, b) for a, b in zip(c1, c2)
             if not (np.isnan(a) or np.isnan(b))]
    if len(pairs) < 3:
        return (np.nan, len(pairs), 'n<3', np.nan)
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    diff = b - a                      # Config2 - Config1
    med = float(np.median(diff))
    if np.allclose(diff, 0):
        return (1.0, len(pairs), 'wilcoxon(all-equal)', 0.0)
    try:
        w, p = stats.wilcoxon(a, b)   # paired
        test = 'wilcoxon'
    except Exception:
        t, p = stats.ttest_rel(a, b)
        test = 'paired-t'
    return (float(p), len(pairs), test, med)


def load(summary):
    df = pd.read_csv(summary)
    for c in ('config', 'trajectory', 'seed', 'zone'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[(df.zone == ZONE) & (df.config.isin([1, 2])) & (df.seed.isin(SEEDS))]
    # keep the LATEST row per (config,traj,seed) — re-runs supersede
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['config', 'trajectory', 'seed'], keep='last')
    for col, _, _ in METRICS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['aborted'] = pd.to_numeric(df.get('aborted', 0), errors='coerce').fillna(0)
    return df


def cell(df, cfg, traj, col):
    """ordered-by-seed values for one (config,traj); NaN where a seed missing."""
    out = []
    for s in SEEDS:
        r = df[(df.config == cfg) & (df.trajectory == traj) & (df.seed == s)]
        out.append(float(r[col].iloc[0]) if len(r) else np.nan)
    return out


def panel(df, name, trajs):
    print("\n" + "=" * 78)
    print(" PANEL: %s   (T%s)" % (name, ", T".join(map(str, trajs))))
    print("=" * 78)
    # coverage
    for traj in trajs:
        for cfg in (1, 2):
            got = [s for s in SEEDS if len(df[(df.config == cfg) &
                   (df.trajectory == traj) & (df.seed == s)])]
            if len(got) < len(SEEDS):
                print("  ! coverage T%d C%d: only seeds %s of %s"
                      % (traj, cfg, got, SEEDS))
    # per-metric table + paired verdict pooled across the panel
    for col, label, lower in METRICS:
        print("\n  %-16s   %-22s %-22s   paired C1vsC2" % (label, "Config 1 (m±SD)", "Config 2 (m±SD)"))
        pc1, pc2 = [], []
        for traj in trajs:
            v1 = cell(df, 1, traj, col); v2 = cell(df, 2, traj, col)
            pc1 += v1; pc2 += v2
            m1, s1, n1 = ms(v1); m2, s2, n2 = ms(v2)
            print("    T%-3d           %8.2f ± %-6.2f (n%d)  %8.2f ± %-6.2f (n%d)"
                  % (traj, m1, s1, n1, m2, s2, n2))
        p, npair, test, med = paired_test(pc1, pc2)
        if np.isnan(p):
            verdict = "insufficient pairs (%s)" % test
        else:
            sig = "SIGNIFICANT" if p < 0.05 else "n.s. (indistinguishable)"
            dirn = ("C2<C1" if med < 0 else "C2>C1" if med > 0 else "no diff")
            better = ""
            if lower is not None and p < 0.05 and med != 0:
                helps = (med < 0) == bool(lower)
                better = " -> Kalman %s" % ("HELPS" if helps else "HURTS")
            verdict = "%s  p=%.3f  %s  Δmed(C2-C1)=%+.2f%s  [%s,n=%d]" % (
                sig, p, dirn, med, better, test, npair)
        print("    => %s" % verdict)


def abort_summary(df, trajs, name):
    print("\n  -- aborts (%s) --" % name)
    for cfg in (1, 2):
        for traj in trajs:
            r = df[(df.config == cfg) & (df.trajectory == traj)]
            nab = int(r['aborted'].sum())
            if nab:
                durs = r[r['aborted'] == 1]['mission_duration_s']
                md = durs.mean() if len(durs) else float('nan')
                print("    C%d T%d: %d/%d aborted (mean dur %.0fs)"
                      % (cfg, traj, nab, len(r), md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default=os.path.expanduser('~/fyp/Results/summary.csv'))
    args = ap.parse_args()
    df = load(args.summary)
    if df.empty:
        print("no matching rows (zone %d, cfg{1,2}, seeds %s)" % (ZONE, SEEDS)); sys.exit(1)
    print("#" * 78)
    print("# M11 ABLATION — Config 1 (YOLO+IBVS, raw) vs Config 2 (YOLO+Kalman+IBVS)")
    print("# Config 3/4 OUT OF SCOPE (PPO parked, C4 unbuilt). Zone %d, seeds %s." % (ZONE, SEEDS))
    print("# Paired Wilcoxon (paired-t fallback); p<0.05 = significant.")
    print("# rows used: %d" % len(df))
    print("#" * 78)
    panel(df, "control-ceiling", CONTROL); abort_summary(df, CONTROL, "control")
    panel(df, "detection-limited", DETECT); abort_summary(df, DETECT, "detection")
    panel(df, "APPENDIX (stress, not an A/B verdict)", APPENDIX)
    abort_summary(df, APPENDIX, "appendix")
    print("\n" + "#" * 78)
    print("# READOUT: the diagnostics predict Kalman's effect surfaces in the")
    print("# recovery/stability band (recovery s, sep SD), not clean HOLD pass/fail.")
    print("# Trajectories flagged 'n.s.' above are a finding: Kalman adds nothing there.")
    print("#" * 78)


if __name__ == '__main__':
    main()
