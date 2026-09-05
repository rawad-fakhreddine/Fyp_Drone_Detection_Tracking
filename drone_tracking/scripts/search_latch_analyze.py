#!/usr/bin/env python3
"""search_latch_analyze.py — A/B summary for the v6.29 SEARCH velocity latch.
Reads per-run flight + gt CSVs in ~/fyp/Results/diagnostics/search_latch and
reports base (blind SEARCH) vs latch (loss-instant v0) per trajectory,
averaged over seeds. Key metric = recovery rate from detection loss.
"""
import os, csv, math, statistics as st

AB = os.path.expanduser('~/fyp/Results/diagnostics/search_latch')
MISSION = ('SEARCH', 'APPROACH', 'HOLD')
TRACK = ('APPROACH', 'HOLD')


def fnum(s):
    try:
        v = float(s); return v if v == v else None
    except (ValueError, TypeError):
        return None


def flight_metrics(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    mission = [r for r in rows if r.get('phase') in MISSION]
    if not mission:
        return None
    n = len(mission)
    hold = sum(1 for r in mission if r.get('phase') == 'HOLD')
    det = sum(1 for r in mission if r.get('raw_det') == 'REAL')
    # sim span of mission
    ts = [fnum(r.get('sim_time')) for r in mission]
    ts = [t for t in ts if t is not None]
    dur = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    # loss episodes after first acquisition
    phases = [r.get('phase') for r in mission]
    times = [fnum(r.get('sim_time')) for r in mission]
    first_acq = next((i for i, p in enumerate(phases) if p in TRACK), None)
    n_loss = n_rec = 0
    rec_times = []
    if first_acq is not None:
        i = first_acq
        while i < len(phases):
            if phases[i] == 'SEARCH':
                j = i
                while j < len(phases) and phases[j] == 'SEARCH':
                    j += 1
                n_loss += 1
                if j < len(phases):              # episode ended by re-acquisition
                    n_rec += 1
                    t0 = times[i]; t1 = times[j]
                    if t0 is not None and t1 is not None:
                        rec_times.append(t1 - t0)
                i = j
            else:
                i += 1
    rec_rate = (100.0 * n_rec / n_loss) if n_loss else float('nan')
    mean_rec = st.mean(rec_times) if rec_times else float('nan')
    closest = [fnum(r.get('true_dist_3d')) for r in rows if r.get('phase') in TRACK]
    closest = [d for d in closest if d is not None and d > 0.05]
    return dict(holdpct=100.0*hold/n, detpct=100.0*det/n, dur=dur,
                n_loss=n_loss, n_rec=n_rec, rec_rate=rec_rate, mean_rec=mean_rec,
                closest=min(closest) if closest else float('nan'))


def gt_out_of_fov(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    none = [r for r in rows if r.get('yolo_det') == 'NONE']
    oof = sum(1 for r in none if r.get('in_fov') == '0')
    return oof


def aborted(tag):
    lg = '/tmp/ablatch_launch_%s.log' % tag
    if os.path.exists(lg):
        return 1 if 'ABORTED' in open(lg, errors='ignore').read() else 0
    return None


def collect(traj, seeds, variant):
    acc = dict(holdpct=[], detpct=[], dur=[], rec_rate=[], mean_rec=[],
               oof=[], aborted=[], n_loss=[], closest=[])
    for s in seeds:
        tag = 'T%d_s%d_%s' % (traj, s, variant)
        fm = flight_metrics(os.path.join(AB, tag + '_flight.csv'))
        oof = gt_out_of_fov(os.path.join(AB, tag + '_gt.csv'))
        ab = aborted(tag)
        if fm is None:
            continue
        acc['holdpct'].append(fm['holdpct']); acc['detpct'].append(fm['detpct'])
        acc['dur'].append(fm['dur']); acc['rec_rate'].append(fm['rec_rate'])
        acc['mean_rec'].append(fm['mean_rec']); acc['n_loss'].append(fm['n_loss'])
        acc['closest'].append(fm['closest'])
        if oof is not None: acc['oof'].append(oof)
        if ab is not None: acc['aborted'].append(ab)
    return acc


def m(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float('nan')


def block(traj, seeds):
    b = collect(traj, seeds, 'base')
    l = collect(traj, seeds, 'latch')
    print('\n=== T%d  (seeds %s)  base[blind] vs latch[v0]  averaged ===' %
          (traj, ','.join(map(str, seeds))))
    print('%-22s %12s %12s %10s' % ('metric', 'base', 'latch', 'delta'))
    print('-' * 58)
    def row(name, key, fmt='%.1f', better='up'):
        bv, lv = m(b[key]), m(l[key])
        d = lv - bv
        arrow = ''
        if d == d:
            good = (d > 0) if better == 'up' else (d < 0)
            arrow = '  ✓' if (good and abs(d) > 1e-9) else ('  ✗' if abs(d) > 1e-9 else '')
        print(('%-22s ' + fmt + ' ' + fmt + ' ' + fmt + '%s') %
              (name, bv, lv, d, arrow))
    row('HOLD%', 'holdpct', '%12.1f', 'up')
    row('detection%', 'detpct', '%12.1f', 'up')
    row('recovery_rate%', 'rec_rate', '%12.1f', 'up')
    row('mean_recovery_s', 'mean_rec', '%12.1f', 'down')
    row('loss_episodes', 'n_loss', '%12.1f', 'down')
    row('OUT_OF_FOV', 'oof', '%12.1f', 'down')
    row('aborted(of %d)' % len(seeds), 'aborted', '%12.1f', 'down')
    row('mission_dur_s', 'dur', '%12.1f', 'up')
    row('closest_m', 'closest', '%12.2f', 'up')


def main():
    block(7, [42, 43, 44])
    block(9, [42, 43, 44])
    block(4, [42])
    print('\n✓ = latch better, ✗ = worse. recovery_rate = %% of loss episodes that re-acquired.')
    print('VERDICT RULE: accept if recovery/HOLD improves on T7/T9 with no T4 regression.')


if __name__ == '__main__':
    main()
