#!/usr/bin/env python3
"""gt_loss_analyze.py — offline loss-onset classifier for the GT-overlay CSV.
==============================================================================
M10.3 FOV-loss mechanism. Reads the per-frame CSV from gt_overlay_record.py and,
for each trajectory, finds every loss onset (YOLO REAL->lost while the
ground-truth target is still geometrically valid) and classifies the mechanism:

  edge-departure   GT off-frame at loss (or crosses off within ~window), bbox
                   area roughly steady before loss -> target exited the FOV.
  too-far / shrink GT still in-frame at loss, bbox area collapsing / depth
                   growing -> target receded below the detector size floor.

A loss onset = a frame where yolo_detected goes 1->0 AND the GT is valid
(projected, not BEHIND). Initial-acquisition gap (before the FIRST detection)
is ignored. Tiny dropouts shorter than MIN_GAP_S are skipped as detector jitter.

Per onset it reports: GT in/out of frame, exit edge + axis, du/dv magnitude,
separation, depth, and the discriminator over the PRE_S seconds before loss
(bbox-area trend, depth trend). Then a per-trajectory verdict: dominant axis,
edge, departure-vs-shrink, one-line mechanism.

Usage:
  gt_loss_analyze.py --label T3 --csv watch_T3_frames.csv [more --label/--csv...]
  gt_loss_analyze.py --auto DIR     # pairs watch_T*_frames.csv by name
"""
import argparse, csv, glob, math, os, re

MIN_GAP_S = 0.4     # ignore sub-0.4 s dropouts (detector jitter, not a real loss)
PRE_S     = 1.0     # discriminator window before the loss onset
SHRINK_FRAC = 0.25  # >25% area drop over PRE_S => "collapsing"
DEPTH_FRAC  = 0.10  # >10% depth growth over PRE_S => "receding"


def fnum(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(
                t=fnum(r['t']),
                det=int(r['yolo_detected']),
                u=fnum(r['gt_u']), v=fnum(r['gt_v']),
                inf=int(r['gt_in_frame']),
                edge=r['exit_edge'], axis=r['axis'],
                du=fnum(r['du']), dv=fnum(r['dv']),
                sep=fnum(r['separation']), depth=fnum(r['depth']),
                area=fnum(r['bbox_area'])))
    return rows


def gt_valid(row):
    # GT is "valid" if it projected (u/v present) and not BEHIND camera
    return row['edge'] != 'BEHIND' and row['u'] is not None


def find_onsets(rows):
    """Return list of indices i where det 1->0 with GT valid, after 1st detect,
    coalescing dropouts separated by < MIN_GAP_S of recovered detection."""
    first_det = next((i for i, r in enumerate(rows) if r['det'] == 1), None)
    if first_det is None:
        return []
    onsets = []
    i = first_det
    n = len(rows)
    while i < n - 1:
        if rows[i]['det'] == 1 and rows[i + 1]['det'] == 0:
            j = i + 1
            # extend through the dropout
            while j < n and rows[j]['det'] == 0:
                j += 1
            gap = (rows[min(j, n - 1)]['t'] - rows[i + 1]['t']) if rows[i + 1]['t'] else 0
            # only count gaps that are real losses (>= MIN_GAP_S) or run-ending
            if gap >= MIN_GAP_S or j >= n:
                if gt_valid(rows[i + 1]):
                    onsets.append(i + 1)
            i = j
        else:
            i += 1
    return onsets


def pre_window(rows, idx):
    t0 = rows[idx]['t']
    if t0 is None:
        return []
    return [r for r in rows[max(0, idx - 120):idx + 1]
            if r['t'] is not None and 0 <= (t0 - r['t']) <= PRE_S]


def trend(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None, None
    return vals[0], vals[-1]


def classify(rows, idx):
    r = rows[idx]
    win = pre_window(rows, idx)
    areas = [w['area'] for w in win]
    depths = [w['depth'] for w in win]
    a0, a1 = trend(areas)
    d0, d1 = trend(depths)
    area_drop = (a0 - a1) / a0 if (a0 and a1 is not None and a0 > 0) else None
    depth_grow = (d1 - d0) / d0 if (d0 and d1 is not None and d0 > 0) else None
    shrinking = area_drop is not None and area_drop > SHRINK_FRAC
    receding = depth_grow is not None and depth_grow > DEPTH_FRAC
    off = (r['inf'] == 0)
    # mechanism
    if off and not shrinking:
        mech = 'edge-departure'
    elif shrinking and not off:
        mech = 'too-far/shrink'
    elif off and shrinking:
        mech = 'edge-departure(+shrink)'
    else:
        mech = 'edge-departure' if off else 'detector-miss(in-frame,steady)'
    return dict(idx=idx, t=r['t'], inf=r['inf'], edge=r['edge'], axis=r['axis'],
                du=r['du'], dv=r['dv'], sep=r['sep'], depth=r['depth'],
                area0=a0, area1=a1, area_drop=area_drop,
                depth0=d0, depth1=d1, depth_grow=depth_grow,
                shrinking=shrinking, receding=receding, mech=mech)


def axis_of(o):
    if o['inf'] == 1:
        # in-frame: owner = larger |offset|
        du, dv = abs(o['du'] or 0), abs(o['dv'] or 0)
        return 'ey(vert)' if dv >= du else 'ex(horiz)'
    return o['axis'] or '?'


def report(label, rows):
    onsets = find_onsets(rows)
    span = (rows[0]['t'], rows[-1]['t']) if rows else (None, None)
    print("\n" + "=" * 70)
    print(" %s  — %d frames, sim %.1f..%.1f s, %d loss onset(s)"
          % (label, len(rows), span[0] or 0, span[1] or 0, len(onsets)))
    print("=" * 70)
    if not onsets:
        det_total = sum(r['det'] for r in rows)
        print("  no qualifying loss onset (det frames=%d). "
              "Either never acquired or never lost while GT valid." % det_total)
        return None

    classed = [classify(rows, i) for i in onsets]
    # primary = first sustained onset
    first = classed[0]

    def fmt(x, p='%.1f'):
        return 'nan' if x is None else (p % x)

    print("  FIRST loss onset @ t=%.1fs:" % first['t'])
    loc = "OUTSIDE frame" if first['inf'] == 0 else "INSIDE frame"
    print("    GT location ....... %s" % loc)
    if first['inf'] == 0:
        print("    exit edge ......... %s   axis=%s" % (first['edge'], first['axis']))
    print("    off-center ........ du=%s dv=%s px  (|du|=%s |dv|=%s)"
          % (fmt(first['du'], '%+.0f'), fmt(first['dv'], '%+.0f'),
             fmt(abs(first['du']) if first['du'] is not None else None, '%.0f'),
             fmt(abs(first['dv']) if first['dv'] is not None else None, '%.0f')))
    print("    separation ........ %s m    depth=%s m" % (fmt(first['sep']), fmt(first['depth'])))
    print("    pre-loss (%.1fs): bbox_area %s -> %s (drop %s%%);  depth %s -> %s (grow %s%%)"
          % (PRE_S, fmt(first['area0'], '%.0f'), fmt(first['area1'], '%.0f'),
             fmt((first['area_drop'] or 0) * 100, '%.0f') if first['area_drop'] is not None else 'nan',
             fmt(first['depth0']), fmt(first['depth1']),
             fmt((first['depth_grow'] or 0) * 100, '%.0f') if first['depth_grow'] is not None else 'nan'))
    print("    discriminator ..... %s%s -> MECHANISM: %s"
          % ('shrinking ' if first['shrinking'] else 'steady-size ',
             'receding' if first['receding'] else 'stable-depth', first['mech']))

    # aggregate over all onsets
    n = len(classed)
    n_out = sum(1 for o in classed if o['inf'] == 0)
    n_in = n - n_out
    axes = [axis_of(o) for o in classed]
    n_ey = sum(1 for a in axes if a.startswith('ey'))
    n_ex = sum(1 for a in axes if a.startswith('ex'))
    edges = {}
    for o in classed:
        if o['inf'] == 0:
            for e in (o['edge'] or '').split('/'):
                if e:
                    edges[e] = edges.get(e, 0) + 1
    n_shrink = sum(1 for o in classed if o['shrinking'])
    dom_axis = 'ey(vert)' if n_ey >= n_ex else 'ex(horiz)'
    dom_edge = max(edges, key=edges.get) if edges else '(in-frame)'
    cls = 'edge-departure' if n_out >= n_in and n_shrink <= n / 2 else \
          ('too-far/shrink' if n_shrink > n / 2 else 'mixed')
    print("  ALL onsets (%d): off-frame=%d in-frame=%d | axis ey=%d ex=%d | "
          "shrinking=%d | edges=%s" % (n, n_out, n_in, n_ey, n_ex, n_shrink,
                                       edges or '{}'))
    print("  VERDICT: dominant axis=%s, edge=%s, class=%s"
          % (dom_axis, dom_edge, cls))
    return dict(label=label, n=n, n_out=n_out, n_in=n_in, dom_axis=dom_axis,
                dom_edge=dom_edge, cls=cls, n_shrink=n_shrink,
                first_t=first['t'], first_inf=first['inf'],
                first_edge=first['edge'], first_axis=first['axis'],
                first_sep=first['sep'], first_depth=first['depth'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--csv', action='append', default=[])
    ap.add_argument('--auto', default=None, help='dir of watch_T*_frames.csv')
    args = ap.parse_args()

    pairs = []
    if args.auto:
        for p in sorted(glob.glob(os.path.join(args.auto, 'watch_T*_frames.csv'))):
            m = re.search(r'watch_(T\d)_frames', p)
            pairs.append((m.group(1) if m else os.path.basename(p), p))
    else:
        pairs = list(zip(args.label, args.csv))

    results = []
    for label, path in pairs:
        if not os.path.exists(path):
            print("\n[MISSING] %s -> %s" % (label, path)); continue
        results.append(report(label, load(path)))

    print("\n" + "#" * 70)
    print("# RESULT TABLE")
    print("#" * 70)
    print("%-5s %-10s %-12s %-22s %s" %
          ("traj", "axis", "edge", "classification", "first-loss sep/depth"))
    for r in results:
        if r is None:
            continue
        print("%-5s %-10s %-12s %-22s sep=%.1fm depth=%.1fm @t=%.0fs (%s)" % (
            r['label'], r['dom_axis'], r['dom_edge'], r['cls'],
            r['first_sep'] or 0, r['first_depth'] or 0, r['first_t'] or 0,
            'OFF' if r['first_inf'] == 0 else 'IN'))


if __name__ == '__main__':
    main()
