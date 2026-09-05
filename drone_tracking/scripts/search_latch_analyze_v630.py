#!/usr/bin/env python3
"""search_latch_analyze_v630.py — PER-SEED paired A/B table for the v6.30
time-bound SEARCH heading latch. base (flag-off == v6.28 blind SEARCH) vs
new (flag-on == v6.30 decaying latch). Reads ~/fyp/Results/diagnostics/
search_latch_v630/{tag}_flight.csv + {tag}_gt.csv. Prints every seed's
base->new delta (the bimodality hides in averages) plus the per-traj mean,
then evaluates the pre-stated accept criterion.
"""
import os, csv, sys, statistics as st

AB = os.path.expanduser('~/fyp/Results/diagnostics/search_latch_v630')
MISSION = ('SEARCH', 'APPROACH', 'HOLD')
TRACK = ('APPROACH', 'HOLD')
SEEDS = [int(s) for s in (sys.argv[1:] or ['42', '43', '44'])]


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
    ts = [fnum(r.get('sim_time')) for r in mission]; ts = [t for t in ts if t is not None]
    dur = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    phases = [r.get('phase') for r in mission]
    times = [fnum(r.get('sim_time')) for r in mission]
    first_acq = next((i for i, p in enumerate(phases) if p in TRACK), None)
    n_loss = n_rec = 0; rec_times = []
    if first_acq is not None:
        i = first_acq
        while i < len(phases):
            if phases[i] == 'SEARCH':
                j = i
                while j < len(phases) and phases[j] == 'SEARCH':
                    j += 1
                n_loss += 1
                if j < len(phases):
                    n_rec += 1
                    t0, t1 = times[i], times[j]
                    if t0 is not None and t1 is not None:
                        rec_times.append(t1 - t0)
                i = j
            else:
                i += 1
    rec_rate = (100.0 * n_rec / n_loss) if n_loss else float('nan')
    mean_rec = st.mean(rec_times) if rec_times else float('nan')
    closest = [fnum(r.get('true_dist_3d')) for r in rows if r.get('phase') in TRACK]
    closest = [d for d in closest if d is not None and d > 0.05]
    return dict(holdpct=100.0 * hold / n, detpct=100.0 * det / n, dur=dur,
                n_loss=n_loss, n_rec=n_rec, rec_rate=rec_rate, mean_rec=mean_rec,
                closest=min(closest) if closest else float('nan'))


def gt_oof(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    none = [r for r in rows if r.get('yolo_det') == 'NONE']
    return sum(1 for r in none if r.get('in_fov') == '0') if rows else None


def aborted(tag):
    lg = '/tmp/ab630_launch_%s.log' % tag
    return (1 if 'ABORTED' in open(lg, errors='ignore').read() else 0) if os.path.exists(lg) else None


def get(traj, seed, variant):
    tag = 'T%d_s%d_%s' % (traj, seed, variant)
    fm = flight_metrics(os.path.join(AB, tag + '_flight.csv'))
    if fm is None:
        return None
    fm['oof'] = gt_oof(os.path.join(AB, tag + '_gt.csv'))
    fm['aborted'] = aborted(tag)
    return fm


def fmt(v, f='%.1f'):
    return ('n/a' if v is None or v != v else f % v)


def per_seed_block(traj, seeds):
    print('\n' + '=' * 78)
    print('T%d  PER-SEED  base[v6.28 blind] -> new[v6.30 time-bound]   (Config2 Z5)' % traj)
    print('=' * 78)
    hdr = '%-5s %-6s %8s %8s %8s %9s %7s %6s %7s' % (
        'seed', 'var', 'HOLD%', 'det%', 'recov%', 'mrec_s', 'OOF', 'abort', 'dur_s')
    print(hdr); print('-' * len(hdr))
    deltas = {k: [] for k in ('holdpct', 'detpct', 'rec_rate', 'oof', 'aborted')}
    for s in seeds:
        b = get(traj, s, 'base'); l = get(traj, s, 'latch')
        for var, m_ in (('base', b), ('latch', l)):
            if m_ is None:
                print('%-5d %-6s %8s %8s %8s %9s %7s %6s %7s' % (s, var, '-', '-', '-', '-', '-', '-', '-'))
            else:
                print('%-5d %-6s %8s %8s %8s %9s %7s %6s %7s' % (
                    s, var, fmt(m_['holdpct']), fmt(m_['detpct']), fmt(m_['rec_rate']),
                    fmt(m_['mean_rec']), fmt(m_['oof'], '%.0f'), fmt(m_['aborted'], '%.0f'),
                    fmt(m_['dur'], '%.0f')))
        if b and l:
            d = '   Δ(new-base): HOLD%%=%+.1f det%%=%+.1f recov%%=%s OOF=%s abort=%s' % (
                l['holdpct'] - b['holdpct'], l['detpct'] - b['detpct'],
                ('%+.1f' % (l['rec_rate'] - b['rec_rate'])) if (l['rec_rate'] == l['rec_rate'] and b['rec_rate'] == b['rec_rate']) else 'n/a',
                ('%+.0f' % (l['oof'] - b['oof'])) if (l['oof'] is not None and b['oof'] is not None) else 'n/a',
                ('%+.0f' % (l['aborted'] - b['aborted'])) if (l['aborted'] is not None and b['aborted'] is not None) else 'n/a')
            print(d)
            for k in deltas:
                bv = b.get(k); lv = l.get(k)
                if bv is not None and lv is not None and bv == bv and lv == lv:
                    deltas[k].append(lv - bv)
        print('-' * len(hdr))
    means = {k: (st.mean(v) if v else float('nan')) for k, v in deltas.items()}
    print('  MEAN Δ  HOLD%%=%+.1f  det%%=%+.1f  recov%%=%+.1f  OOF=%+.1f  abort=%+.1f  (n=%d paired)' % (
        means['holdpct'], means['detpct'], means['rec_rate'], means['oof'], means['aborted'],
        len(deltas['holdpct'])))
    return means


def main():
    print('SEEDS:', SEEDS)
    t7 = per_seed_block(7, SEEDS)
    t9 = per_seed_block(9, SEEDS)
    t4 = per_seed_block(4, [42])
    print('\n' + '#' * 78)
    print('ACCEPT CRITERION (pre-stated):')
    c7 = t7['rec_rate'] >= -1e-6 and t7['holdpct'] >= -1e-6 if (t7['rec_rate'] == t7['rec_rate']) else (t7['holdpct'] >= -1e-6)
    c9 = (t9['rec_rate'] >= -1e-6) if (t9['rec_rate'] == t9['rec_rate']) else True
    c4 = abs(t4['holdpct']) <= 3.0 and abs(t4['detpct']) <= 3.0
    print('  T7 wins-or-ties (recov%% & HOLD%% not lower)     : %s  (ΔHOLD%%=%+.1f Δrecov%%=%+.1f)' % (
        'PASS' if c7 else 'FAIL', t7['holdpct'], t7['rec_rate']))
    print('  T9 no recovery-rate regression on average       : %s  (Δrecov%%=%+.1f ΔHOLD%%=%+.1f)' % (
        'PASS' if c9 else 'FAIL', t9['rec_rate'], t9['holdpct']))
    print('  T4 no harm (|ΔHOLD%%|<=3 & |Δdet%%|<=3)           : %s  (ΔHOLD%%=%+.1f Δdet%%=%+.1f)' % (
        'PASS' if c4 else 'FAIL', t4['holdpct'], t4['detpct']))
    print('  VERDICT: %s' % ('RECOMMEND COMMIT' if (c7 and c9 and c4) else
                             'DO NOT COMMIT — report honestly (default-OFF or document limit)'))
    print('#' * 78)


if __name__ == '__main__':
    main()
