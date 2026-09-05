#!/usr/bin/env python3
"""
analyze_estimation.py — per-run Kalman estimation-quality metrics.
Reads an estimation_logger.py CSV and reports raw-vs-truth and filtered-vs-truth
error, restricted to in-FOV frames (where the ground-truth projection is valid).

  raw_err_*  = RMS(raw - true)      over DETECTION frames (raw_det==REAL, in_fov)
  kf_err_*   = RMS(filt - true)     over ALL in_fov frames (incl. predicted)
  improvement= raw_err - kf_err     (>0 = filter helps)
  dropout_kf = RMS(filt - true)     over is_dropout==1 in_fov frames
  std_*      = STD(est - true)      (bias-removed; the meaningful alpha metric,
               since proj_alpha carries a 0.5 m size-model bias)

Usage: analyze_estimation.py --csv RUN.csv [--label STR]
"""
import csv, argparse, math


def fnum(s):
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (ValueError, TypeError):
        return None


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float('nan')


def std(xs):
    n = len(xs)
    if n < 2:
        return float('nan')
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))

    # in-FOV frames with valid truth = the comparison domain
    fov = [r for r in rows if r.get('in_fov') == '1'
           and None not in (fnum(r['proj_u']), fnum(r['proj_v']), fnum(r['proj_alpha']))]

    # raw residuals: REAL detections only
    rd = {'cx': [], 'cy': [], 'a': []}
    for r in fov:
        if r.get('raw_det') != 'REAL':
            continue
        rcx, rcy, ra = fnum(r['raw_cx']), fnum(r['raw_cy']), fnum(r['raw_alpha'])
        if None in (rcx, rcy, ra):
            continue
        rd['cx'].append(rcx - fnum(r['proj_u']))
        rd['cy'].append(rcy - fnum(r['proj_v']))
        rd['a'].append(ra - fnum(r['proj_alpha']))

    # filtered residuals: all in_fov frames with a finite estimate
    fd = {'cx': [], 'cy': [], 'a': []}
    dd = {'cx': [], 'cy': [], 'a': []}          # dropout-only
    for r in fov:
        fcx, fcy, fa = fnum(r['filt_cx']), fnum(r['filt_cy']), fnum(r['filt_alpha'])
        if None in (fcx, fcy, fa):
            continue
        ecx = fcx - fnum(r['proj_u']); ecy = fcy - fnum(r['proj_v']); ea = fa - fnum(r['proj_alpha'])
        fd['cx'].append(ecx); fd['cy'].append(ecy); fd['a'].append(ea)
        if r.get('is_dropout') == '1':
            dd['cx'].append(ecx); dd['cy'].append(ecy); dd['a'].append(ea)

    def emit(k, v):
        print("%s=%s" % (k, ('%.3f' % v) if v == v else 'nan'))   # nan-safe

    print("== estimation metrics: %s ==" % (a.label or a.csv))
    print("frames_total=%d  frames_in_fov=%d  raw_det_frames=%d  filt_frames=%d  dropout_frames=%d"
          % (len(rows), len(fov), len(rd['cx']), len(fd['cx']), len(dd['cx'])))
    for comp, key in (('cx', 'cx'), ('cy', 'cy'), ('a', 'alpha')):
        emit('raw_err_%s' % key, rms(rd[comp]))
        emit('kf_err_%s' % key, rms(fd[comp]))
        emit('improvement_%s' % key, rms(rd[comp]) - rms(fd[comp]))
        emit('dropout_kf_%s' % key, rms(dd[comp]))
        emit('raw_std_%s' % key, std(rd[comp]))
        emit('kf_std_%s' % key, std(fd[comp]))
        emit('dropout_kf_std_%s' % key, std(dd[comp]))


if __name__ == '__main__':
    main()
