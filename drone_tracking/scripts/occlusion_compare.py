#!/usr/bin/env python3
"""
occlusion_compare.py — M10.3 occlusion-confound side-by-side.
READ-ONLY. Places the Zone-1 (obstacle-sparse) hard-trajectory results directly
beside the Zone-5 weakness-map baseline for T3/T6/T7:
  HOLD% [min-max], aborts, OUT/FAR/OCC %, mean sep, owner axis + AUC (ex/ey).
Headline: does OCC% collapse toward ~0 while OUT_OF_FOV persists? The HOLD%
delta is the occlusion ceiling (the slice no controller change recovers in
cluttered zones). No control change — descriptive only.
"""
import os, csv, glob, math, bisect, statistics as st

Z5 = os.path.expanduser("~/fyp/Results/diagnostics/baseline_sweep")   # Zone 5
Z1 = os.path.expanduser("~/fyp/Results/diagnostics/occlusion")        # Zone 1
SEEDS = [42, 43, 45]; TRAJS = [3, 6, 7]
MIN_DET_W = 8.0; DUR = 200.0


def num(r, c):
    try:
        v = float(r.get(c, '')); return None if math.isnan(v) else v
    except (ValueError, TypeError): return None
def load(f):
    with open(f) as fh: return list(csv.DictReader(fh))


def classify_gt(g):
    cats = {'OUT_OF_FOV': 0, 'TOO_FAR': 0, 'OCCLUSION_OR_MISS': 0}; loss = 0
    for r in g:
        if r.get('yolo_det') != 'NONE': continue
        loss += 1
        if r.get('in_fov') != '1': cats['OUT_OF_FOV'] += 1
        else:
            pw = num(r, 'proj_w_px')
            if pw is not None and pw < MIN_DET_W: cats['TOO_FAR'] += 1
            else: cats['OCCLUSION_OR_MISS'] += 1
    return cats, loss


def flight_metrics(rows):
    mission = [r for r in rows if r.get('phase') in ('SEARCH', 'APPROACH', 'HOLD')]
    hold = [r for r in rows if r.get('phase') == 'HOLD']
    holdpct = 100.0 * len(hold) / len(mission) if mission else 0.0
    times = [num(r, 'sim_time') for r in rows if num(r, 'sim_time') is not None]
    span = (max(times) - min(times)) if times else 0.0
    last = rows[-1].get('phase') if rows else ''
    aborted = 1 if (last == 'SEARCH' and span < 190.0) else 0
    seps = [num(r, 'true_dist_3d') for r in mission if num(r, 'true_dist_3d') is not None]
    return holdpct, aborted, (st.mean(seps) if seps else float('nan'))


def auc_axes(rows, win_s=2.0):
    t = [num(r, 'sim_time') for r in rows]
    onset = []
    for i in range(1, len(rows)):
        ph0, ph1 = rows[i-1].get('phase'), rows[i].get('phase')
        d0, d1 = rows[i-1].get('raw_det'), rows[i].get('raw_det')
        if ph0 == 'HOLD' and ph1 in ('APPROACH', 'SEARCH') and t[i] is not None: onset.append(t[i])
        elif ph1 in ('APPROACH', 'HOLD') and d0 == 'REAL' and d1 in ('PRED', 'NONE') and t[i] is not None: onset.append(t[i])
    fine = {'ex': [], 'ey': []}; degr = {'ex': [], 'ey': []}
    for i, r in enumerate(rows):
        ti = t[i]
        if ti is None: continue
        ex, ey = num(r, 'ex'), num(r, 'ey')
        if ex is None or ey is None: continue
        if r.get('phase') == 'HOLD' and r.get('raw_det') == 'REAL' and abs(ex) < 0.10 and abs(ey) < 0.10:
            fine['ex'].append(abs(ex)); fine['ey'].append(abs(ey))
        elif r.get('phase') in ('APPROACH', 'HOLD') and any(0 <= (o - ti) <= win_s for o in onset):
            degr['ex'].append(abs(ex)); degr['ey'].append(abs(ey))
    out = {}
    for k in ('ex', 'ey'):
        fv, dv = sorted(fine[k]), degr[k]
        if not fv or not dv: out[k] = None; continue
        n = len(fv); tot = 0
        for x in dv:
            lo = bisect.bisect_left(fv, x); hi = bisect.bisect_right(fv, x)
            tot += lo + (hi - lo) * 0.5
        out[k] = tot / (len(dv) * n)
    return out


def summarize(root, tr):
    holds = []; aborts = 0; seps = []; cc = {'OUT_OF_FOV': 0.0, 'TOO_FAR': 0.0, 'OCCLUSION_OR_MISS': 0.0}; ntot = 0
    aex = []; aey = []
    for s in SEEDS:
        fp = os.path.join(root, f"T{tr}", f"T{tr}_s{s}_flight.csv")
        gp = os.path.join(root, f"T{tr}", f"T{tr}_s{s}_gt.csv")
        if not os.path.exists(fp): continue
        rows = load(fp)
        h, ab, ms = flight_metrics(rows)
        holds.append(h); aborts += ab
        if ms == ms: seps.append(ms)
        if os.path.exists(gp):
            cats, nloss = classify_gt(load(gp))
            for k in cc: cc[k] += cats.get(k, 0)
            ntot += nloss
        au = auc_axes(rows)
        if au.get('ex') is not None: aex.append(au['ex'])
        if au.get('ey') is not None: aey.append(au['ey'])
    if not holds: return None
    ccp = {k: (100.0 * v / ntot if ntot else 0) for k, v in cc.items()}
    mex = st.mean(aex) if aex else None; mey = st.mean(aey) if aey else None
    owner = 'ex' if (mex or 0) >= (mey or 0) else 'ey'
    return dict(holds=holds, aborts=aborts, sep=(st.mean(seps) if seps else float('nan')),
                cc=ccp, ex=mex, ey=mey, owner=owner, n=len(holds))


def main():
    print("\n" + "=" * 100)
    print("M10.3 OCCLUSION-CONFOUND — Zone 1 (sparse, clearance 69.5m) vs Zone 5 (treed, clearance 9.1m)")
    print("Config 2 baseline, gains untouched, seeds", SEEDS)
    print("=" * 100)
    hdr = f"  {'traj/zone':>10} {'HOLD% [min-max]':>18} {'abrt':>4} {'meanSep':>8} {'OUT/FAR/OCC %':>16} {'AUC ex/ey':>12}  owner"
    def fm(x): return f"{x:.2f}" if x is not None else "--"
    for tr in TRAJS:
        z5 = summarize(Z5, tr); z1 = summarize(Z1, tr)
        print("\n" + "-" * 100)
        for label, d in (("T%d Zone5" % tr, z5), ("T%d Zone1" % tr, z1)):
            if d is None:
                print(f"  {label:>10}   (no data)"); continue
            hs = d['holds']
            print(f"  {label:>10} {st.mean(hs):6.1f} [{min(hs):4.1f}-{max(hs):4.1f}]  {d['aborts']:>4}/{d['n']} "
                  f"{d['sep']:8.1f} {d['cc']['OUT_OF_FOV']:4.0f}/{d['cc']['TOO_FAR']:3.0f}/{d['cc']['OCCLUSION_OR_MISS']:3.0f}   "
                  f"{fm(d['ex'])}/{fm(d['ey'])}  {d['owner']}")
        if z5 and z1:
            dOCC = z1['cc']['OCCLUSION_OR_MISS'] - z5['cc']['OCCLUSION_OR_MISS']
            dFOV = z1['cc']['OUT_OF_FOV'] - z5['cc']['OUT_OF_FOV']
            dHOLD = st.mean(z1['holds']) - st.mean(z5['holds'])
            print(f"  >> T{tr} deltas (Z1-Z5): OCC {dOCC:+.0f}pp  OUT_OF_FOV {dFOV:+.0f}pp  HOLD {dHOLD:+.1f}pp "
                  f"(HOLD delta = occlusion ceiling)")
    print("\nHeadline check: OCC% -> ~0 in Zone1 while OUT_OF_FOV persists confirms FOV-departure is the")
    print("structural driver and occlusion is a separable, zone-dependent secondary loss.")


if __name__ == "__main__":
    main()
