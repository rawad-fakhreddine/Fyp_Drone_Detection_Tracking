#!/usr/bin/env python3
"""
sort_v4_by_scenario.py — v2.0 (content-based, no trajectory tags)
==================================================================
AI-Based Drone-to-Drone Detection and Tracking
Rawad Fakhredine | FYP Masters in Robotics | Supervisor: Ibrahim Sammour

v2.0: _hard = low-confidence YOLO (not T9_ trajectory prefix)
      filename format: f000000.jpg (capture v3.0 compatible)

BUCKET LOGIC:
  _empty   → no GT label (target outside FOV)
  _missed  → YOLO conf < CONF_THRESH      (complete failure)
  _hard    → CONF_THRESH ≤ conf < HARD_CONF_THRESH  (barely detects)
  _far     → area < FAR_MAX px²           (d > ~10m)
  _mid     → FAR_MAX ≤ area < MID_MAX px²
  _close   → area ≥ MID_MAX px²           (d < ~5m)
"""
import os, glob, shutil

BASE_DIR   = os.path.expanduser('~/drone_detection/capture_v4')
DEDUP_IMG  = os.path.join(BASE_DIR, 'deduped', 'images')
DEDUP_LBL  = os.path.join(BASE_DIR, 'deduped', 'labels')
SORTED_DIR = os.path.join(BASE_DIR, 'sorted')
MODEL_PATH = os.path.expanduser('~/drone_detection/models/best.pt')
BUCKETS    = ['_far', '_mid', '_close', '_missed', '_empty', '_hard']

CONF_THRESH      = 0.35
HARD_CONF_THRESH = 0.55
FAR_MAX          = 600
MID_MAX          = 3500
IMG_W = 640; IMG_H = 480

def read_gt_area(lbl_path):
    if not os.path.exists(lbl_path): return None
    try:
        with open(lbl_path) as f: line = f.readline().strip()
        if not line: return None
        parts = line.split()
        if len(parts) < 5: return None
        return float(parts[3]) * IMG_W * float(parts[4]) * IMG_H
    except Exception: return None

def run_yolo_on_frame(model, jpg_path):
    results = model(jpg_path, conf=0.01, classes=[0], verbose=False)
    boxes   = results[0].boxes
    if len(boxes) == 0: return 0.0
    return float(boxes.conf.cpu().numpy().max())

def assign_bucket(area, yolo_conf):
    if area is None:                   return '_empty'
    if yolo_conf < CONF_THRESH:        return '_missed'
    if yolo_conf < HARD_CONF_THRESH:   return '_hard'
    if area < FAR_MAX:                 return '_far'
    if area < MID_MAX:                 return '_mid'
    return '_close'

def main():
    for b in BUCKETS:
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(SORTED_DIR, b, sub), exist_ok=True)

    jpegs = sorted(glob.glob(os.path.join(DEDUP_IMG, '*.jpg')))
    if not jpegs:
        print("ERROR: No frames in %s — run dedup_v4.py first" % DEDUP_IMG); return

    print("\n=== sort_v4_by_scenario.py v2.0 ===")
    print("Input : %d frames from %s" % (len(jpegs), DEDUP_IMG))

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("YOLO v3 loaded | missed<%.2f | hard<%.2f | far<%dpx² | mid<%dpx²" %
          (CONF_THRESH, HARD_CONF_THRESH, FAR_MAX, MID_MAX))

    counts = {b: 0 for b in BUCKETS}
    for i, jpg_path in enumerate(jpegs):
        stem    = os.path.splitext(os.path.basename(jpg_path))[0]
        lbl_src = os.path.join(DEDUP_LBL, stem + '.txt')
        area    = read_gt_area(lbl_src)
        yolo_conf = run_yolo_on_frame(model, jpg_path) if area is not None else 0.0
        bucket  = assign_bucket(area, yolo_conf)
        shutil.copy2(jpg_path, os.path.join(SORTED_DIR, bucket, 'images', os.path.basename(jpg_path)))
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(SORTED_DIR, bucket, 'labels', stem + '.txt'))
        counts[bucket] += 1
        if (i+1) % 100 == 0 or (i+1) == len(jpegs):
            print("  [%d/%d] %s → %s (conf=%.3f area=%s)" %
                  (i+1, len(jpegs), stem, bucket, yolo_conf,
                   "%.0fpx²" % area if area else "None"))

    print("\n=== Results ===")
    total = sum(counts.values())
    for b in BUCKETS:
        print("  %-10s %5d  (%5.1f%%)" % (b, counts[b], 100*counts[b]/total if total else 0))
    print("  Total    %5d\nDone → next: python3 assemble_v4_training_set.py" % total)

if __name__ == '__main__': main()
