#!/usr/bin/env python3
"""tau_sweep_analyze.py — tau-vs-trajectory curve for the v6.30 light SEARCH
heading latch. Baseline (v6.28) and tau=1.0 latch come from the search_latch_v630
batch; tau in {2,3,4} from the search_tau sweep. Reports per-seed then mean Δ
(latch[tau] - baseline), paired per seed, for T7 and T9. Finds the tau that
maximizes T7 recovery while keeping T9 Δrecov >= 0.
"""
import os, csv, statistics as st

V630 = os.path.expanduser('~/fyp/Results/diagnostics/search_latch_v630')
TAUD = os.path.expanduser('~/fyp/Results/diagnostics/search_tau')
SEEDS = [42, 43, 45]
TAUS = [1.0, 2.0, 3.0, 4.0]
MISSION = ('SEARCH', 'APPROACH', 'HOLD'); TRACK = ('APPROACH', 'HOLD')


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
    phases = [r.get('phase') for r in mission]
    times = [fnum(r.get('sim_time')) for r in mission]
    first = next((i for i, p in enumerate(phases) if p in TRACK), None)
    nl = nr = 0; rt = []
    if first is not None:
        i = first
        while i < len(phases):
            if phases[i] == 'SEARCH':
                j = i
                while j < len(phases) and phases[j] == 'SEARCH':
                    j += 1
                nl += 1
                if j < len(phases):
                    nr += 1
                    if times[i] is not None and times[j] is not None:
                        rt.append(times[j] - times[i])
                i = j
            else:
                i += 1
    return dict(holdpct=100.0 * hold / n, detpct=100.0 * det / n,
                rec_rate=(100.0 * nr / nl) if nl else float('nan'),
                n_loss=nl)


def gt_oof(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    none = [r for r in rows if r.get('yolo_det') == 'NONE']
    return sum(1 for r in none if r.get('in_fov') == '0') if rows else None


def base_path(traj, seed, kind):   # kind: flight|gt
    return os.path.join(V630, 'T%d_s%d_base_%s.csv' % (traj, seed, kind))


def latch_path(tau, traj, seed, kind):
    if abs(tau - 1.0) < 1e-9:
        return os.path.join(V630, 'T%d_s%d_latch_%s.csv' % (traj, seed, kind))
    tdir = 'tau%d' % int(tau) if float(tau).is_integer() else 'tau%s' % tau
    return os.path.join(TAUD, tdir, 'T%d_s%d_latch_%s.csv' % (traj, seed, kind))


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.mean(xs) if xs else float('nan')


def curve_for(traj):
    rows = []
    for tau in TAUS:
        dH = dR = dO = []
        dH, dR, dO = [], [], []
        seeddet = []
        for s in SEEDS:
            b = flight_metrics(base_path(traj, s, 'flight'))
            l = flight_metrics(latch_path(tau, traj, s, 'flight'))
            bo = gt_oof(base_path(traj, s, 'gt'))
            lo = gt_oof(latch_path(tau, traj, s, 'gt'))
            if b and l:
                dH.append(l['holdpct'] - b['holdpct'])
                if b['rec_rate'] == b['rec_rate'] and l['rec_rate'] == l['rec_rate']:
                    dR.append(l['rec_rate'] - b['rec_rate'])
                if bo is not None and lo is not None:
                    dO.append(lo - bo)
                seeddet.append((s, b, l, bo, lo))
            else:
                seeddet.append((s, b, l, bo, lo))
        rows.append((tau, mean(dH), mean(dR), mean(dO), seeddet))
    return rows


def fmt(v, f='%+.1f'):
    return ('n/a' if v is None or v != v else f % v)


def main():
    print('SEEDS %s   (Δ = latch[tau] - baseline v6.28, paired per seed)\n' % SEEDS)
    t7 = curve_for(7)
    t9 = curve_for(9)
    by_tau = {}
    print('=' * 72)
    print('  tau | T7 ΔHOLD  T7 Δrecov |  T9 ΔHOLD  T9 Δrecov   T9 ΔOOF')
    print('-' * 72)
    for (tau, h7, r7, o7, _), (_, h9, r9, o9, _) in zip(t7, t9):
        by_tau[tau] = dict(h7=h7, r7=r7, h9=h9, r9=r9, o9=o9)
        print('  %3.0f | %8s  %9s | %8s  %9s  %8s' % (
            tau, fmt(h7), fmt(r7), fmt(h9), fmt(r9), fmt(o9, '%+.0f')))
    print('=' * 72)

    # per-seed detail
    for traj, rows in (('T7', t7), ('T9', t9)):
        print('\n--- %s per-seed HOLD%% / recov%% (base -> tau1 / tau2 / tau3 / tau4) ---' % traj)
        for s in SEEDS:
            cells = []
            b = flight_metrics(base_path(int(traj[1]), s, 'flight'))
            bh = ('%.0f' % b['holdpct']) if b else '-'
            br = ('%.0f' % b['rec_rate']) if (b and b['rec_rate'] == b['rec_rate']) else 'nan'
            for tau in TAUS:
                l = flight_metrics(latch_path(tau, int(traj[1]), s, 'flight'))
                lh = ('%.0f' % l['holdpct']) if l else '-'
                lr = ('%.0f' % l['rec_rate']) if (l and l['rec_rate'] == l['rec_rate']) else 'nan'
                cells.append('%s/%s' % (lh, lr))
            print('  s%d  base %s/%s  ->  %s' % (s, bh, br, '  '.join(cells)))

    # decision
    print('\n' + '#' * 72)
    print('FIND: max T7 Δrecov subject to T9 Δrecov >= 0  (no evasion regression)')
    ok = [(tau, d) for tau, d in by_tau.items()
          if (d['r9'] == d['r9'] and d['r9'] >= -1e-6) and (d['h7'] == d['h7'])]
    # require T7 to actually improve (ΔHOLD>0 or Δrecov>0)
    winners = [(tau, d) for tau, d in ok if (d['h7'] > 1e-6 or (d['r7'] == d['r7'] and d['r7'] > 1e-6))]
    if winners:
        best = max(winners, key=lambda kd: (kd[1]['r7'] if kd[1]['r7'] == kd[1]['r7'] else -1e9, kd[1]['h7']))
        tau, d = best
        print('  WIN-BOTH tau = %.0f : T7 ΔHOLD %+.1f Δrecov %+.1f ; T9 ΔHOLD %+.1f Δrecov %+.1f ΔOOF %+.0f'
              % (tau, d['h7'], d['r7'], d['h9'], d['r9'], d['o9']))
        print('  -> COMMIT CANDIDATE tau=%.0f (default-on)' % tau)
    else:
        print('  NO win-both tau: T7 only lifts where T9 Δrecov < 0.')
        print('  -> fall back to flag default-OFF (evasion aid); report tradeoff curve.')
    print('#' * 72)


if __name__ == '__main__':
    main()
