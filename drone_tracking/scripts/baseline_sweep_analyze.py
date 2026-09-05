#!/usr/bin/env python3
"""
baseline_sweep_analyze.py — M10.3 baseline diagnostic sweep -> weakness map.
READ-ONLY. Reads ~/fyp/Results/diagnostics/baseline_sweep/T<t>/T<t>_s<seed>_{flight,gt}.csv.

Per run:  HOLD%, abort, mean+closest sep, recovery time, vx/vy/wz saturation,
          gt dropout classification (OUT_OF_FOV / TOO_FAR / OCCLUSION_OR_MISS).
Per traj: mean +/- spread over 3 seeds, pass/fail, dominant failure mode,
          ex/ey/ea owner via 2 s pre-break AUC.
NO control change proposed — descriptive map only. T9 reported separately.
"""
import os, csv, glob, math, bisect, statistics as st

ROOT = os.path.expanduser("~/fyp/Results/diagnostics/baseline_sweep")
SEEDS = [42, 43, 45]
TRAJS = [2, 3, 4, 5, 6, 7, 8, 9]
MAX_VX, MAX_VY, MAX_WZ = 4.5, 1.20, 0.50
SAT = 0.98
MIN_DET_W = 8.0
DUR = 200.0


def num(r, c):
    try:
        v = float(r.get(c, ''))
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None


def load(f):
    with open(f) as fh:
        return list(csv.DictReader(fh))


def classify_gt(gt_rows):
    """Fractions of detection-LOSS frames (yolo_det=='NONE') by cause."""
    cats = {'OUT_OF_FOV': 0, 'TOO_FAR': 0, 'OCCLUSION_OR_MISS': 0}
    loss = 0
    for r in gt_rows:
        if r.get('yolo_det') != 'NONE':
            continue
        loss += 1
        in_fov = r.get('in_fov') == '1'
        pw = num(r, 'proj_w_px')
        if not in_fov:
            cats['OUT_OF_FOV'] += 1
        elif pw is not None and pw < MIN_DET_W:
            cats['TOO_FAR'] += 1
        else:
            cats['OCCLUSION_OR_MISS'] += 1
    if loss == 0:
        return {k: 0.0 for k in cats}, 0
    return {k: 100.0 * v / loss for k, v in cats.items()}, loss


def analyze_flight(rows):
    mission = [r for r in rows if r.get('phase') in ('SEARCH', 'APPROACH', 'HOLD')]
    hold = [r for r in rows if r.get('phase') == 'HOLD']
    track = [r for r in rows if r.get('phase') in ('APPROACH', 'HOLD')]
    holdpct = 100.0 * len(hold) / len(mission) if mission else 0.0
    times = [num(r, 'sim_time') for r in rows if num(r, 'sim_time') is not None]
    span = (max(times) - min(times)) if times else 0.0
    last = rows[-1].get('phase') if rows else ''
    aborted = 1 if (last == 'SEARCH' and span < 190.0) else 0
    seps = [num(r, 'true_dist_3d') for r in mission if num(r, 'true_dist_3d') is not None]
    mean_sep = st.mean(seps) if seps else float('nan')
    min_sep = min(seps) if seps else float('nan')
    # saturation over tracking rows
    def satfrac(col, cap):
        v = [abs(num(r, col)) for r in track if num(r, col) is not None]
        return 100.0 * sum(1 for x in v if x >= SAT * cap) / len(v) if v else 0.0
    sat_vx, sat_vy, sat_wz = satfrac('cmd_vx', MAX_VX), satfrac('cmd_vy', MAX_VY), satfrac('cmd_wz', MAX_WZ)
    # recovery: closed SEARCH episodes after first APPROACH/HOLD
    seen = False; eps = []; cs = cl = None
    for r in rows:
        ph = r.get('phase')
        if ph in ('APPROACH', 'HOLD'): seen = True
        if ph == 'SEARCH' and seen:
            t = num(r, 'sim_time')
            if t is None: continue
            if cs is None: cs = t
            cl = t
        else:
            if cs is not None and cl is not None: eps.append(cl - cs)
            cs = cl = None
    rec = st.mean(eps) if eps else float('nan')
    return dict(holdpct=holdpct, span=span, aborted=aborted, mean_sep=mean_sep,
                min_sep=min_sep, sat_vx=sat_vx, sat_vy=sat_vy, sat_wz=sat_wz,
                rec=rec, seps=seps)


def auc_prebreak(rows, win_s=2.0):
    """AUC for |ex|,|ey|,|ea| separating fine vs degrading (2 s pre-break)."""
    t = [num(r, 'sim_time') for r in rows]
    onset_t = []
    for i in range(1, len(rows)):
        ph0, ph1 = rows[i - 1].get('phase'), rows[i].get('phase')
        d0, d1 = rows[i - 1].get('raw_det'), rows[i].get('raw_det')
        if ph0 == 'HOLD' and ph1 in ('APPROACH', 'SEARCH'):
            if t[i] is not None: onset_t.append(t[i])
        elif ph1 in ('APPROACH', 'HOLD') and d0 == 'REAL' and d1 in ('PRED', 'NONE'):
            if t[i] is not None: onset_t.append(t[i])
    fine = {'ex': [], 'ey': [], 'ea': []}
    degr = {'ex': [], 'ey': [], 'ea': []}
    for i, r in enumerate(rows):
        ti = t[i]
        if ti is None: continue
        ex, ey, ea = num(r, 'ex'), num(r, 'ey'), num(r, 'ea')
        if ex is None or ey is None or ea is None: continue
        if r.get('phase') == 'HOLD' and r.get('raw_det') == 'REAL' and abs(ex) < 0.10 and abs(ey) < 0.10:
            fine['ex'].append(abs(ex)); fine['ey'].append(abs(ey)); fine['ea'].append(abs(ea))
        elif r.get('phase') in ('APPROACH', 'HOLD') and any(0 <= (ot - ti) <= win_s for ot in onset_t):
            degr['ex'].append(abs(ex)); degr['ey'].append(abs(ey)); degr['ea'].append(abs(ea))
    out = {}
    for k in ('ex', 'ey', 'ea'):
        fv, dv = sorted(fine[k]), degr[k]
        if not fv or not dv:
            out[k] = None; continue
        n = len(fv); tot = 0
        for x in dv:
            lo = bisect.bisect_left(fv, x); hi = bisect.bisect_right(fv, x)
            tot += lo + (hi - lo) * 0.5
        out[k] = tot / (len(dv) * n)
    return out, sum(len(v) for v in fine.values()) // 3, sum(len(v) for v in degr.values()) // 3


def main():
    print("\n" + "=" * 96)
    print("M10.3 BASELINE DIAGNOSTIC SWEEP — Config 2 (deployed, gains untouched), Zone 5, seeds", SEEDS)
    print("=" * 96)
    pertraj = {}
    for tr in TRAJS:
        rundata = []
        for s in SEEDS:
            fp = os.path.join(ROOT, f"T{tr}", f"T{tr}_s{s}_flight.csv")
            gp = os.path.join(ROOT, f"T{tr}", f"T{tr}_s{s}_gt.csv")
            if not os.path.exists(fp):
                rundata.append((s, None, None, None)); continue
            fl = analyze_flight(load(fp))
            gt = (classify_gt(load(gp)) if os.path.exists(gp) else ({}, 0))
            au = auc_prebreak(load(fp))
            rundata.append((s, fl, gt, au))
        pertraj[tr] = rundata
        print(f"\n--- T{tr} ---")
        print(f"  {'seed':>4} {'HOLD%':>6} {'abort':>5} {'span':>5} {'meanSep':>7} {'minSep':>6} "
              f"{'rec_s':>6} {'satVx%':>6} {'satVy%':>6} {'satWz%':>6} | {'OUT_FOV':>7} {'TOO_FAR':>7} {'OCC/MISS':>8} (loss) | {'AUC ex/ey/ea':>16}")
        for s, fl, gt, au in rundata:
            if fl is None:
                print(f"  {s:>4}   --- (no CSV)"); continue
            cats, nloss = gt
            a = au[0] if au else {}
            def fmt(x): return f"{x:.2f}" if (x is not None and x == x) else " -- "
            print(f"  {s:>4} {fl['holdpct']:6.1f} {fl['aborted']:>5} {fl['span']:5.0f} "
                  f"{fl['mean_sep']:7.1f} {fl['min_sep']:6.1f} "
                  f"{(fl['rec'] if fl['rec']==fl['rec'] else 0):6.1f} "
                  f"{fl['sat_vx']:6.1f} {fl['sat_vy']:6.1f} {fl['sat_wz']:6.1f} | "
                  f"{cats.get('OUT_OF_FOV',0):7.0f} {cats.get('TOO_FAR',0):7.0f} {cats.get('OCCLUSION_OR_MISS',0):8.0f} ({nloss:4d}) | "
                  f"{fmt(a.get('ex'))}/{fmt(a.get('ey'))}/{fmt(a.get('ea'))}")

    # ── weakness map ──────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("WEAKNESS MAP (mean +/- spread over 3 seeds)")
    print("=" * 96)
    print(f"  {'traj':>4} {'HOLD% (mean[min-max])':>22} {'abrt':>4} {'meanSep':>8} {'OUT/FAR/OCC %':>16} {'AUC ex/ey/ea':>16}  dom-fail / owner")
    for tr in TRAJS:
        rd = [(s, fl, gt, au) for s, fl, gt, au in pertraj[tr] if fl is not None]
        if not rd:
            print(f"  T{tr}: no data"); continue
        holds = [fl['holdpct'] for _, fl, _, _ in rd]
        ab = sum(fl['aborted'] for _, fl, _, _ in rd)
        msep = st.mean([fl['mean_sep'] for _, fl, _, _ in rd if fl['mean_sep'] == fl['mean_sep']])
        # pooled gt cats
        cc = {'OUT_OF_FOV': 0.0, 'TOO_FAR': 0.0, 'OCCLUSION_OR_MISS': 0.0}; ntot = 0
        for _, _, (cats, nloss), _ in rd:
            for k in cc: cc[k] += cats.get(k, 0) * nloss
            ntot += nloss
        cc = {k: (v / ntot if ntot else 0) for k, v in cc.items()}
        # pooled AUC (mean of available)
        def meanauc(key):
            vs = [au[0][key] for _, _, _, au in rd if au and au[0].get(key) is not None]
            return st.mean(vs) if vs else None
        aex, aey, aea = meanauc('ex'), meanauc('ey'), meanauc('ea')
        owner = max((('ex', aex), ('ey', aey), ('ea', aea)),
                    key=lambda kv: (kv[1] if kv[1] is not None else -1))[0]
        # dominant failure mode heuristic
        sep_grew = msep > 20
        if cc['OUT_OF_FOV'] >= 45: dom = 'FOV-angular'
        elif cc['TOO_FAR'] >= 45 or sep_grew: dom = 'closure-runaway'
        elif cc['OCCLUSION_OR_MISS'] >= 45: dom = 'detection'
        else: dom = 'centering-error'
        passfail = 'PASS' if ab == 0 and st.mean(holds) >= 50 else ('FAIL' if ab >= 2 else 'MIXED')
        def fm(x): return f"{x:.2f}" if x is not None else "--"
        star = '  (T9 stress)' if tr == 9 else ''
        print(f"  T{tr} {passfail:>4} {st.mean(holds):6.1f}[{min(holds):4.1f}-{max(holds):4.1f}]  {ab:>4} "
              f"{msep:8.1f} {cc['OUT_OF_FOV']:4.0f}/{cc['TOO_FAR']:3.0f}/{cc['OCCLUSION_OR_MISS']:3.0f}   "
              f"{fm(aex)}/{fm(aey)}/{fm(aea)}  {dom} / {owner}{star}")

    print("\n(ea/ex/ey consistency verdict: compare the owner column across T2/T4/T5/T8 vs T3/T6/T7.)")


if __name__ == "__main__":
    main()
