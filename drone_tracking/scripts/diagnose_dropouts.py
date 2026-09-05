#!/usr/bin/env python3
"""
diagnose_dropouts.py — classify why each YOLO dropout happened.
==============================================================================
DELIVERABLE 2. Post-processes a gt_projection log (gt_projection.py). For every
frame where the YOLO detection == NONE, classifies the cause using the
ground-truth projection:

  OUT_OF_FOV   : projected target outside the image bounds / behind the camera.
  TOO_FAR      : in FOV but projected box width < detection floor (MIN_DET_W_PX).
  OCCLUSION_OR_MISS : in FOV, above the floor → the target SHOULD be detectable
                 but YOLO produced nothing. Baylands trees are a single static
                 mesh (model://baylands) with NO individually-queryable tree
                 models, so occlusion approach (a) is infeasible; this tool uses
                 approach (b): such frames are flagged "should-detect-but-missed"
                 and CANNOT be split into occlusion vs genuine detector weakness
                 without per-tree geometry. A [GATED] sub-tag marks frames where
                 the detector had a candidate (conf>0) that the persistence/conf
                 gate withheld — those are pipeline-suppressed, not blind misses.

Detection floor: a box narrower than MIN_DET_W_PX (default 8 px) is treated as
too small to detect (the v3.4 plausibility gate hard-rejects <3 px; 8 px adds
margin for YOLOv8 small-object limits). Reported with its alpha-equivalent.

Usage:
  diagnose_dropouts.py GT_LOG.csv [--flight FLIGHT.csv] [--min_det_w 8]
                                  [--png OUT.png] [--focus 79]
"""
import sys, csv, argparse, math

MIN_DET_W_PX_DEFAULT = 8.0


def fnum(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def classify(row, min_det_w):
    in_fov = row.get('in_fov') == '1'
    pw = fnum(row.get('proj_w_px'))
    if not in_fov:
        return 'OUT_OF_FOV'
    if pw is None or pw < min_det_w:
        return 'TOO_FAR'
    return 'OCCLUSION_OR_MISS'


def nearest_phase(flight_rows, t):
    if not flight_rows:
        return ''
    best = min(flight_rows, key=lambda fr: abs(fr[0] - t))
    return best[1] if abs(best[0] - t) < 0.5 else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gt_log')
    ap.add_argument('--flight', default=None, help='flight CSV for phase breakdown')
    ap.add_argument('--min_det_w', type=float, default=MIN_DET_W_PX_DEFAULT)
    ap.add_argument('--png', default=None)
    ap.add_argument('--focus', type=float, default=None,
                    help='print frames around this sim_time (e.g. the loss instant)')
    args = ap.parse_args()

    floor_alpha = args.min_det_w * args.min_det_w        # square-box alpha equiv (px^2)
    rows = list(csv.DictReader(open(args.gt_log)))

    flight_rows = []
    if args.flight:
        for fr in csv.DictReader(open(args.flight)):
            t = fnum(fr.get('sim_time'))
            if t is not None:
                flight_rows.append((t, fr.get('phase', '')))

    cats = ['OUT_OF_FOV', 'TOO_FAR', 'OCCLUSION_OR_MISS']
    counts = {c: 0 for c in cats}
    gated = {c: 0 for c in cats}
    by_phase = {}              # phase -> {cat: n}
    timeline = []              # (t, cat, gated, phase)
    n_none = n_real = 0

    for r in rows:
        det = r.get('yolo_det')
        if det == 'REAL':
            n_real += 1
            continue
        if det != 'NONE':
            continue
        n_none += 1
        c = classify(r, args.min_det_w)
        counts[c] += 1
        dc = fnum(r.get('det_conf')) or 0.0
        is_gated = dc > 0.0           # detector had a candidate it withheld
        if is_gated:
            gated[c] += 1
        t = fnum(r.get('sim_time'))
        ph = nearest_phase(flight_rows, t) if flight_rows else ''
        if ph:
            by_phase.setdefault(ph, {x: 0 for x in cats})[c] += 1
        timeline.append((t, c, is_gated, ph))

    print("=" * 80)
    print(" YOLO dropout cause diagnosis")
    print(" gt_log : %s" % args.gt_log)
    print(" frames : %d NONE (dropouts) | %d REAL | floor: proj_w < %.0f px "
          "(alpha < %.0f px^2)" % (n_none, n_real, args.min_det_w, floor_alpha))
    print(" occlusion: approach (b) — baylands trees not queryable; "
          "OCCLUSION_OR_MISS = occlusion AND/OR detector weakness (see header)")
    print("=" * 80)
    if n_none == 0:
        print(" No dropouts — YOLO detected on every frame.")
        return

    print("\n DROPOUT CATEGORY BREAKDOWN:")
    for c in cats:
        pct = 100.0 * counts[c] / n_none
        g = ("  (%d [GATED] — candidate withheld by the gate)" % gated[c]) if gated[c] else ""
        bar = '#' * int(round(pct / 2))
        print("   %-18s %5d  %5.1f%%  %s%s" % (c, counts[c], pct, bar, g))

    if by_phase:
        print("\n BY PHASE (dropout frames):")
        print("   %-10s %10s %8s %18s" % ('phase', 'OUT_FOV', 'TOO_FAR', 'OCCL_OR_MISS'))
        for ph in sorted(by_phase):
            d = by_phase[ph]
            print("   %-10s %10d %8d %18d"
                  % (ph, d['OUT_OF_FOV'], d['TOO_FAR'], d['OCCLUSION_OR_MISS']))

    # ── 10 s timeline of dominant category
    print("\n TIMELINE (10 s bins — dominant dropout category):")
    tl = [t for t in timeline if t[0] is not None]
    if tl:
        t0 = min(t[0] for t in tl); t1 = max(t[0] for t in tl)
        b = t0
        while b < t1:
            seg = [x for x in tl if b <= x[0] < b + 10]
            if seg:
                cc = {c: 0 for c in cats}
                for x in seg:
                    cc[x[1]] += 1
                dom = max(cc, key=cc.get)
                print("   %5.0f-%-5.0f s : %3d drops  dominant=%-18s (%s)"
                      % (b, b + 10, len(seg), dom,
                         ' '.join('%s:%d' % (k[:4], v) for k, v in cc.items() if v)))
            b += 10

    # ── focused window (e.g. the loss instant)
    if args.focus is not None:
        lo, hi = args.focus - 2.0, args.focus + 4.0
        print("\n FOCUS WINDOW %.0f-%.0f s (per-frame around the loss):" % (lo, hi))
        print("   %-8s %-6s %-8s %-9s %-9s %-18s %-8s"
              % ('t', 'det', 'in_fov', 'true_d', 'proj_w', 'category', 'gatestate'))
        for r in rows:
            t = fnum(r.get('sim_time'))
            if t is None or not (lo <= t <= hi):
                continue
            det = r.get('yolo_det')
            cat = classify(r, args.min_det_w) if det == 'NONE' else '-'
            d = fnum(r.get('true_dist')); pw = fnum(r.get('proj_w_px'))
            print("   %-8.1f %-6s %-8s %-9s %-9s %-18s %s"
                  % (t, det, r.get('in_fov'),
                     ('%.2f' % d) if d is not None else 'nan',
                     ('%.1f' % pw) if pw is not None else 'nan',
                     cat, r.get('det_state', '')))

    # ── optional PNG (histogram + timeline scatter)
    if args.png:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
            vals = [counts[c] for c in cats]
            ax1.bar(cats, vals, color=['#888', '#39c', '#e55'])
            ax1.set_ylabel('dropout frames')
            ax1.set_title('Dropout cause histogram — %s' % args.gt_log.split('/')[-1])
            for i, v in enumerate(vals):
                ax1.text(i, v, ' %d (%.0f%%)' % (v, 100.0*v/n_none), va='bottom')
            cmap = {'OUT_OF_FOV': '#888', 'TOO_FAR': '#39c', 'OCCLUSION_OR_MISS': '#e55'}
            for c in cats:
                ts = [x[0] for x in tl if x[1] == c]
                ax2.scatter(ts, [c]*len(ts), s=8, c=cmap[c], label=c)
            ax2.set_xlabel('sim_time (s)'); ax2.set_title('Dropout timeline')
            if args.focus is not None:
                ax2.axvline(args.focus, color='k', ls='--', lw=1, label='focus')
            plt.tight_layout(); plt.savefig(args.png, dpi=110)
            print("\n PNG saved: %s" % args.png)
        except Exception as e:
            print("\n (PNG skipped: %s)" % e)


if __name__ == '__main__':
    main()
