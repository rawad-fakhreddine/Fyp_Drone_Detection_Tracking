#!/usr/bin/env python3
"""recover_imgsz_test.py — M10.3 Part C: offline imgsz/tiling recovery test.
==============================================================================
Read-only. Takes the clean-frame capture (clean_capture_record.py output),
isolates the IN-FRAME RAW-MISS frames (gt_in_frame=1, live yolo_detected=0,
det_state=LOST), and re-runs the SAME model (best_v41s.pt) at several input
resolutions to measure how many misses each setting recovers — and at what
target-px / range. Mirrors the live detector gate exactly (conf + plausibility
w in [3,300], aspect in [0.8,6.0]) so a "recovery" means the live pipeline
would have detected it.

IMPORTANT interpretation (Part A): the camera is 640x480 NATIVE and the live
node already feeds full-native frames to YOLO at imgsz=640. So imgsz=1280 here
is UPSAMPLING (interpolation) of 640x480 pixels — no new optical detail. Any
recovery is interpolation/receptive-field gain, NOT recovered real detail; the
optical ceiling is the 640x480 sensor.

Usage:
  recover_imgsz_test.py --dir ~/fyp/Results/diagnostics/cap_T7 \
      --model ~/drone_detection/models/best_v41s.pt
"""
import argparse, csv, os, time
import numpy as np

MIN_BOX, MAX_BOX = 3.0, 300.0
ASP_MIN, ASP_MAX = 0.8, 6.0


def plausible(w, h):
    if w < MIN_BOX or w > MAX_BOX:
        return False
    if h <= 0 or not (ASP_MIN <= w / h <= ASP_MAX):
        return False
    return True


def best_box(boxes_xywh, confs, conf_th):
    """Return (w,h,conf) of the best PLAUSIBLE box clearing conf_th, else None."""
    order = np.argsort(-confs)
    for i in order:
        if confs[i] < conf_th:
            break
        w, h = boxes_xywh[i][2], boxes_xywh[i][3]
        if plausible(w, h):
            return (w, h, float(confs[i]))
    return None


def infer_full(model, img, imgsz, conf_th, cls):
    r = model(img, imgsz=imgsz, conf=conf_th, classes=[cls],
              device='cuda', verbose=False)[0]
    b = r.boxes
    if len(b) == 0:
        return None
    xywh = b.xywh.cpu().numpy()
    confs = b.conf.cpu().numpy()
    return best_box(xywh, confs, conf_th)


def infer_tiled(model, img, conf_th, cls, nx=2, ny=2, overlap=0.2):
    """2x2 tiling: crop overlapping tiles, infer each at imgsz=640 (tile is
    upscaled ~2x internally), map boxes back to full-image px. Plausibility is
    applied in ORIGINAL px (tile crops preserve scale)."""
    H, W = img.shape[:2]
    tw, th = int(W / nx), int(H / ny)
    ox, oy = int(tw * overlap), int(th * overlap)
    best = None
    for iy in range(ny):
        for ix in range(nx):
            x0 = max(0, ix * tw - ox); x1 = min(W, (ix + 1) * tw + ox)
            y0 = max(0, iy * th - oy); y1 = min(H, (iy + 1) * th + oy)
            tile = img[y0:y1, x0:x1]
            r = model(tile, imgsz=640, conf=conf_th, classes=[cls],
                      device='cuda', verbose=False)[0]
            b = r.boxes
            if len(b) == 0:
                continue
            xywh = b.xywh.cpu().numpy()
            confs = b.conf.cpu().numpy()
            cand = best_box(xywh, confs, conf_th)
            if cand is not None and (best is None or cand[2] > best[2]):
                best = cand
    return best


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def dist_summary(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "n=0"
    a = np.array(vals)
    return "n=%d  px:min=%.1f med=%.1f max=%.1f" % (len(a), a.min(),
                                                    np.median(a), a.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--model', default=os.path.expanduser(
        '~/drone_detection/models/best_v41s.pt'))
    ap.add_argument('--conf', type=float, default=0.55,
                    help='acquire-mode conf (LOST state); the miss happened here')
    ap.add_argument('--cls', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='cap miss frames (0=all)')
    args = ap.parse_args()
    d = os.path.expanduser(args.dir)
    import cv2
    from ultralytics import YOLO

    rows = list(csv.DictReader(open(os.path.join(d, 'clean_capture_meta.csv'))))
    in_fov = [r for r in rows if r['gt_in_frame'] == '1']
    miss = [r for r in in_fov
            if r['yolo_detected'] == '0' and r['det_state'].startswith('LOST')]
    gate = [r for r in in_fov
            if r['yolo_detected'] == '0' and r['det_state'].startswith('CONFIRM')]
    print("=" * 68)
    print(" Part C — imgsz recovery test   dir=%s" % d)
    print("=" * 68)
    print(" in-FOV frames .............. %d" % len(in_fov))
    print(" in-frame RAW-MISS (LOST) ... %d   <- recovery target" % len(miss))
    print(" in-frame gate-withheld ..... %d   (CONFIRMING, not raw)" % len(gate))
    rng = [float(r['separation']) for r in miss]
    px = [float(r['est_target_px']) for r in miss]
    if rng:
        print(" miss range: %.1f..%.1f m   est px: %.1f..%.1f"
              % (min(rng), max(rng), min(px), max(px)))
    if args.limit and len(miss) > args.limit:
        idx = np.linspace(0, len(miss) - 1, args.limit).astype(int)
        miss = [miss[i] for i in idx]
        print(" (sampled %d miss frames evenly)" % len(miss))
    if not miss:
        print(" no raw-miss frames — nothing to recover."); return

    print("\n loading model %s ..." % args.model)
    model = YOLO(args.model)
    fdir = os.path.join(d, 'frames')

    settings = [('imgsz640 (=live)', 'full', 640),
                ('imgsz1280 (upsample)', 'full', 1280),
                ('tiled2x2', 'tile', 0)]
    results = {}
    for name, kind, sz in settings:
        rec_px, miss_px, dt = [], [], 0.0
        for r in miss:
            img = cv2.imread(os.path.join(fdir, r['png']))
            if img is None:
                continue
            t0 = time.time()
            if kind == 'full':
                got = infer_full(model, img, sz, args.conf, args.cls)
            else:
                got = infer_tiled(model, img, args.conf, args.cls)
            dt += time.time() - t0
            (rec_px if got else miss_px).append(float(r['est_target_px']))
        n = len(rec_px) + len(miss_px)
        results[name] = (len(rec_px), n, rec_px, miss_px, dt / max(n, 1))
        print("\n  [%s]  recovered %d/%d = %.0f%%   (%.1f ms/frame)"
              % (name, len(rec_px), n, pct(len(rec_px), n), 1000 * dt / max(n, 1)))
        print("      recovered  : %s" % dist_summary(rec_px))
        print("      still-miss  : %s" % dist_summary(miss_px))

    print("\n" + "#" * 68)
    print("# RECOVERY SUMMARY (in-frame raw-miss frames, conf=%.2f)" % args.conf)
    print("#" * 68)
    print(" %-22s %8s %8s" % ("setting", "recov%", "ms/frame"))
    for name, _, _ in settings:
        rc, n, _, _, ms = results[name]
        print(" %-22s %7.0f%% %8.1f" % (name, pct(rc, n), 1000 * ms))


if __name__ == '__main__':
    main()
