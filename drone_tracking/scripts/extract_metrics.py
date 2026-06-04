#!/usr/bin/env python3
"""
extract_metrics.py — Extract key metrics from a flight CSV log for the summary.
Appends one row to ~/results/summary.csv.

Usage:
  python3 extract_metrics.py --csv PATH --config N --zone Z --traj T --seed S --duration D
"""
import os, sys, csv, argparse, datetime
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',      required=True)
    ap.add_argument('--config',   required=True, type=int)
    ap.add_argument('--zone',     required=True)
    ap.add_argument('--traj',     required=True, type=int)
    ap.add_argument('--seed',     required=True, type=int)
    ap.add_argument('--duration', required=True, type=int)
    ap.add_argument('--summary',  default=os.path.expanduser("~/results/summary.csv"))
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[extract_metrics] WARNING: empty CSV"); sys.exit(0)

    def pct(filter_fn):
        n = sum(1 for r in rows if filter_fn(r))
        return 100.0 * n / len(rows)
    def fmean(col, filter_fn=lambda r: True):
        vals = []
        for r in rows:
            if not filter_fn(r): continue
            try:
                v = float(r.get(col, ''))
                if not np.isnan(v): vals.append(v)
            except (ValueError, TypeError): pass
        return float(np.mean(vals)) if vals else float('nan')

    phase     = lambda P: lambda r: r.get('phase','') == P
    is_hold   = phase('HOLD')
    detected  = lambda r: r.get('raw_det','') in ('1', '1.0', 'True')

    metrics = {
        'timestamp':        datetime.datetime.now().isoformat(timespec='seconds'),
        'config':           args.config,
        'zone':             args.zone,
        'trajectory':       args.traj,
        'seed':             args.seed,
        'duration_s':       args.duration,
        'n_samples':        len(rows),
        'takeoff_pct':      round(pct(phase('TAKEOFF')),  2),
        'search_pct':       round(pct(phase('SEARCH')),   2),
        'approach_pct':     round(pct(phase('APPROACH')), 2),
        'hold_pct':         round(pct(is_hold),           2),
        'detection_rate':   round(pct(detected),          2),
        'hold_mean_sep':    round(fmean('true_dist_3d', is_hold),  3),
        'hold_mean_alt_err':round(fmean('world_alt_err', is_hold), 3),
        'csv_file':         os.path.basename(args.csv),
    }

    # Append to summary.csv (create with header if absent)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    write_header = not os.path.exists(args.summary)
    with open(args.summary, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header: w.writeheader()
        w.writerow(metrics)

    print("\n=== Run metrics ===")
    for k, v in metrics.items():
        print(f"  {k:<20s} {v}")
    print(f"\n✓ Appended to {args.summary}")

if __name__ == '__main__':
    main()