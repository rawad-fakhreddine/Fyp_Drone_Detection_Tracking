#!/usr/bin/env python3
"""
measure_yolo_noise.py — empirical R characterization for the Kalman filter.
==============================================================================
DELIVERABLE 1. Post-processes a gt_projection log (see gt_projection.py) — the
live-capture-then-offline-analysis split is deliberate: capturing the
synchronized model_states + intrinsics + YOLO output MUST be live, but the
statistics are computed offline so bins / filters can be re-run freely without
re-flying. Offline analysis is the more reliable half and is what this tool does.

For every frame with a REAL YOLO detection AND the ground-truth target in FOV:
    residual_cx    = yolo_cx        - proj_u          [px]
    residual_cy    = yolo_cy        - proj_v          [px]
    residual_alpha = yolo_alpha_pix - proj_alpha_pix  [px^2 area]
Variances are reported DISTANCE-BINNED. cx/cy residuals are geometry-only (size
-model-independent) → trustworthy. alpha residual VARIANCE is the area noise R
needs; its MEAN is dominated by the 0.5 m point-size model (an effective-size
diagnostic is printed). Units match the Kalman: z = [cx_px, cy_px, area_px],
R = diag([6, 6, 5]).

Usage:  measure_yolo_noise.py GT_LOG.csv [--out residuals.csv] [--bins 4,7,10]
"""
import sys, csv, argparse, math

CURRENT_R = (6.0, 6.0, 5.0)            # Kalman M9.8 R = diag([6,6,5])


def fnum(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def var(xs):
    n = len(xs)
    if n < 2:
        return float('nan'), float('nan'), n
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)     # sample variance
    return v, m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gt_log')
    ap.add_argument('--out', default=None, help='residuals CSV (default: <gt_log>.residuals.csv)')
    ap.add_argument('--bins', default='4,7,10', help='distance bin edges (m)')
    args = ap.parse_args()

    edges = [float(x) for x in args.bins.split(',')]
    labels = (['<%g m' % edges[0]]
              + ['%g-%g m' % (edges[i], edges[i+1]) for i in range(len(edges)-1)]
              + ['>%g m' % edges[-1]])

    def bin_of(d):
        for i, e in enumerate(edges):
            if d < e:
                return i
        return len(edges)

    rows = list(csv.DictReader(open(args.gt_log)))
    bins = {i: {'ru': [], 'rv': [], 'ra': [], 'ya': [], 'pa': []}
            for i in range(len(labels))}
    used = skipped = 0
    out_path = args.out or (args.gt_log.rsplit('.', 1)[0] + '.residuals.csv')
    fout = open(out_path, 'w', newline='')
    w = csv.writer(fout)
    w.writerow(['sim_time', 'true_dist', 'bin', 'residual_cx', 'residual_cy',
                'residual_alpha', 'yolo_cx', 'proj_u', 'yolo_cy', 'proj_v',
                'yolo_alpha_pix', 'proj_alpha_pix'])

    for r in rows:
        if r.get('yolo_det') != 'REAL' or r.get('in_fov') != '1':
            skipped += 1
            continue
        ycx, ycy, yal = fnum(r['yolo_cx']), fnum(r['yolo_cy']), fnum(r['yolo_alpha_pix'])
        pu, pv, pa = fnum(r['proj_u']), fnum(r['proj_v']), fnum(r['proj_alpha_pix'])
        d = fnum(r['true_dist'])
        if None in (ycx, ycy, yal, pu, pv, pa, d):
            skipped += 1
            continue
        ru, rv, ra = ycx - pu, ycy - pv, yal - pa
        b = bin_of(d)
        bins[b]['ru'].append(ru); bins[b]['rv'].append(rv); bins[b]['ra'].append(ra)
        bins[b]['ya'].append(yal); bins[b]['pa'].append(pa)
        used += 1
        w.writerow(['%.3f' % fnum(r['sim_time']), '%.2f' % d, labels[b],
                    '%.2f' % ru, '%.2f' % rv, '%.1f' % ra,
                    '%.2f' % ycx, '%.2f' % pu, '%.2f' % ycy, '%.2f' % pv,
                    '%.1f' % yal, '%.1f' % pa])
    fout.close()

    print("=" * 78)
    print(" YOLO measurement-noise (R) characterization")
    print(" gt_log : %s" % args.gt_log)
    print(" frames : %d used (REAL det & in-FOV) | %d skipped" % (used, skipped))
    print("=" * 78)
    if used == 0:
        print(" No usable frames — cannot characterize R.")
        return

    # ── sanity check: mean cx/cy residual should be ~0 if the frame chain is right
    allru = [x for b in bins.values() for x in b['ru']]
    allrv = [x for b in bins.values() for x in b['rv']]
    mu = sum(allru) / len(allru)
    mv = sum(allrv) / len(allrv)
    print("\n FRAME-CHAIN SANITY (mean cx/cy residual — should be near 0):")
    print("   mean residual_cx = %+.2f px | mean residual_cy = %+.2f px" % (mu, mv))
    if abs(mu) > 25 or abs(mv) > 25:
        print("   ** WARNING: large systematic offset — possible optical-frame /")
        print("      intrinsics error in the transform chain. Treat R with caution. **")
    else:
        print("   OK — projection chain consistent with YOLO centroids.")

    # ── per-bin table
    print("\n PER-DISTANCE-BIN measured R (variance) and bias (mean residual):")
    hdr = " %-9s %6s | %9s %8s | %9s %8s | %11s %9s"
    print(hdr % ("bin", "n", "R_cx", "bias_cx", "R_cy", "bias_cy", "R_alpha", "bias_a"))
    print(" " + "-" * 74)
    pooled = {'ru': [], 'rv': [], 'ra': []}
    for i in range(len(labels)):
        ru, rv, ra = bins[i]['ru'], bins[i]['rv'], bins[i]['ra']
        if not ru:
            print(" %-9s %6d |    (no frames in this bin)" % (labels[i], 0))
            continue
        vcx, mcx, n = var(ru); vcy, mcy, _ = var(rv); vca, mca, _ = var(ra)
        print(" %-9s %6d | %9.2f %+8.2f | %9.2f %+8.2f | %11.1f %+9.1f"
              % (labels[i], n, vcx, mcx, vcy, mcy, vca, mca))
        # effective-size diagnostic for alpha bias
        mya = sum(bins[i]['ya']) / len(bins[i]['ya'])
        mpa = sum(bins[i]['pa']) / len(bins[i]['pa'])
        if mpa > 0:
            s_eff = TARGET_SIZE * math.sqrt(mya / mpa)
            print(" %-9s        |   alpha effective-size that zeroes bias: "
                  "S_eff=%.2f m (proj used %.2f m)" % ("", s_eff, TARGET_SIZE))
        pooled['ru'] += ru; pooled['rv'] += rv; pooled['ra'] += ra

    vcx, _, ncx = var(pooled['ru']); vcy, _, _ = var(pooled['rv']); vca, _, _ = var(pooled['ra'])
    print(" " + "-" * 74)
    print(" %-9s %6d | %9.2f %8s | %9.2f %8s | %11.1f %9s"
          % ("POOLED", ncx, vcx, "", vcy, "", vca, ""))

    print("\n RECOMMENDED R vs CURRENT:")
    print("   current Kalman R = [%.1f, %.1f, %.1f]  (cx, cy, alpha; px,px,px^2)"
          % CURRENT_R)
    print("   measured  R_pool = [%.2f, %.2f, %.1f]" % (vcx, vcy, vca))
    print("   cx/cy: trustworthy (geometry-only). alpha: VARIANCE is the area")
    print("   noise; its absolute level also reflects the smoothing + size model.")
    rec_a = vca
    print("\n   >> Suggested R ≈ [%.1f, %.1f, %.0f]" % (vcx, vcy, rec_a))
    if rec_a > CURRENT_R[2] * 3:
        print("      (R_alpha is %.0fx the current 5.0 — current value under-models"
              % (rec_a / CURRENT_R[2]))
        print("       YOLO area noise; the Kalman currently over-trusts alpha.)")
    print("\n residuals saved: %s" % out_path)


# TARGET_SIZE mirrors gt_projection default (for the effective-size diagnostic)
TARGET_SIZE = 0.5

if __name__ == '__main__':
    main()
