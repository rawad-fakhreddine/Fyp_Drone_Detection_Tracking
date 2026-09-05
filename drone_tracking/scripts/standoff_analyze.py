#!/usr/bin/env python3
"""standoff_analyze.py — summarize the IBVS standoff sweep.
Reads the per-run flight + gt_projection CSVs in ~/fyp/Results/diagnostics/standoff
and prints the per-run metrics + per-standoff averages over seeds.

Per run:
  flight CSV  -> HOLD%, det-in-HOLD%, closest approach, mean/min sep in HOLD
  gt CSV      -> OUT_OF_FOV frames (yolo_det==NONE & in_fov==0), in-FOV%, det%
  launch log  -> aborted flag
"""
import os, csv, math, glob, statistics as st

SWEEP = os.path.expanduser('~/fyp/Results/diagnostics/standoff')
LABELS = [('current', '~3.0m'), ('s5', '5m'), ('s6', '6m'), ('s7', '7m')]
SEEDS = [42, 43]
MISSION = ('SEARCH', 'APPROACH', 'HOLD')


def fnum(s):
    try:
        v = float(s); return v if v == v else None
    except (ValueError, TypeError):
        return None


def flight_metrics(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    hold = [r for r in rows if r.get('phase') == 'HOLD']
    mission = [r for r in rows if r.get('phase') in MISSION]
    if not mission:
        return None
    holdpct = 100.0 * len(hold) / len(mission)
    det_in_hold = (100.0 * sum(1 for r in hold if r.get('raw_det') == 'REAL') / len(hold)) if hold else float('nan')
    # closest approach: min true 3D separation while actively tracking
    track = [fnum(r.get('true_dist_3d')) for r in rows if r.get('phase') in ('APPROACH', 'HOLD')]
    track = [d for d in track if d is not None and d > 0.05]
    closest = min(track) if track else float('nan')
    sep_hold = [fnum(r.get('true_dist_3d')) for r in hold]
    sep_hold = [d for d in sep_hold if d is not None and d > 0.05]
    mean_sep = st.mean(sep_hold) if sep_hold else float('nan')
    min_sep = min(sep_hold) if sep_hold else float('nan')
    return dict(holdpct=holdpct, det_in_hold=det_in_hold, closest=closest,
                mean_sep=mean_sep, min_sep=min_sep, n_mission=len(mission))


def gt_metrics(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    n = len(rows)
    none = [r for r in rows if r.get('yolo_det') == 'NONE']
    real = sum(1 for r in rows if r.get('yolo_det') == 'REAL')
    infov = sum(1 for r in rows if r.get('in_fov') == '1')
    out_of_fov = sum(1 for r in none if r.get('in_fov') == '0')
    too_far = 0
    for r in none:
        if r.get('in_fov') == '1':
            pw = fnum(r.get('proj_w_px'))
            if pw is not None and pw < 8.0:
                too_far += 1
    return dict(n=n, none=len(none), real=real, out_of_fov=out_of_fov,
                too_far=too_far, infov_pct=100.0*infov/n, det_pct=100.0*real/n)


def aborted_flag(label, seed):
    lg = '/tmp/standoff_launch_T7_s%d_%s.log' % (seed, label)
    if os.path.exists(lg):
        return 'Y' if 'ABORTED' in open(lg, errors='ignore').read() else 'N'
    return '?'


def main():
    print('\nPER-RUN:')
    hdr = '%-9s %-4s | %6s %8s %8s | %7s %7s %7s | %7s %7s %6s %5s' % (
        'standoff', 'seed', 'HOLD%', 'detHOLD', 'abort',
        'OUT_FOV', 'inFOV%', 'det%', 'closest', 'meanSep', 'minSep', 'nMis')
    print(hdr); print('-' * len(hdr))
    agg = {lab: {k: [] for k in ('holdpct', 'det_in_hold', 'out_of_fov', 'infov_pct',
                                 'det_pct', 'closest', 'mean_sep', 'min_sep')}
           for lab, _ in LABELS}
    aborts = {lab: 0 for lab, _ in LABELS}
    for lab, dist in LABELS:
        for seed in SEEDS:
            fm = flight_metrics(os.path.join(SWEEP, 'T7_s%d_%s_flight.csv' % (seed, lab)))
            gm = gt_metrics(os.path.join(SWEEP, 'T7_s%d_%s_gt.csv' % (seed, lab)))
            ab = aborted_flag(lab, seed)
            if ab == 'Y':
                aborts[lab] += 1
            if fm is None or gm is None:
                print('%-9s %-4d | %6s (missing flight=%s gt=%s)' % (
                    lab, seed, '--', fm is not None, gm is not None))
                continue
            print('%-9s %-4d | %6.1f %8.1f %8s | %7d %7.1f %7.1f | %7.2f %7.2f %6.2f %5d' % (
                lab, seed, fm['holdpct'], fm['det_in_hold'], ab,
                gm['out_of_fov'], gm['infov_pct'], gm['det_pct'],
                fm['closest'], fm['mean_sep'], fm['min_sep'], fm['n_mission']))
            agg[lab]['holdpct'].append(fm['holdpct'])
            agg[lab]['det_in_hold'].append(fm['det_in_hold'])
            agg[lab]['out_of_fov'].append(gm['out_of_fov'])
            agg[lab]['infov_pct'].append(gm['infov_pct'])
            agg[lab]['det_pct'].append(gm['det_pct'])
            agg[lab]['closest'].append(fm['closest'])
            agg[lab]['mean_sep'].append(fm['mean_sep'])
            agg[lab]['min_sep'].append(fm['min_sep'])

    def m(xs):
        xs = [x for x in xs if x == x]
        return st.mean(xs) if xs else float('nan')

    print('\nAVERAGED OVER SEEDS:')
    hdr2 = '%-9s %-6s | %6s %8s | %8s %7s %7s | %7s %7s %7s | %6s' % (
        'standoff', 'dist', 'HOLD%', 'detHOLD', 'OUT_FOV', 'inFOV%', 'det%',
        'closest', 'meanSep', 'minSep', 'abort')
    print(hdr2); print('-' * len(hdr2))
    for lab, dist in LABELS:
        a = agg[lab]
        print('%-9s %-6s | %6.1f %8.1f | %8.0f %7.1f %7.1f | %7.2f %7.2f %7.2f | %2d/%d' % (
            lab, dist, m(a['holdpct']), m(a['det_in_hold']), m(a['out_of_fov']),
            m(a['infov_pct']), m(a['det_pct']), m(a['closest']), m(a['mean_sep']),
            m(a['min_sep']), aborts[lab], len(SEEDS)))
    print('\nOUT_FOV = YOLO=NONE frames with target projected outside the image (FOV loss).')
    print('closest/sep = true 3D separation (m). detHOLD = raw YOLO detection rate within HOLD phase.')


if __name__ == '__main__':
    main()
