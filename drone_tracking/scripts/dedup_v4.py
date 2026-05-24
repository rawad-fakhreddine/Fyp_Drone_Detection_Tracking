#!/usr/bin/env python3
"""
dedup_v4.py — Perceptual hash deduplication for capture_v4/raw/
================================================================
Rawad Fakhredine | FYP Masters in Robotics

Same philosophy as v3 dedup_captures.py (M6.3), adapted for
the new capture_v4 folder structure (images/ + labels/ are separate).

WHAT IT DOES:
  Scans capture_v4/raw/images/ for near-identical frames using
  perceptual hashing. Within a sliding window of WINDOW_SIZE frames,
  if two frames have hash distance ≤ HASH_THRESHOLD, the later one
  is a duplicate and gets moved to a 'dedup_rejects/' subfolder
  (NOT deleted — you can inspect them).

  Output lands in capture_v4/deduped/ (safe copy, raw/ untouched).

WHY DEDUP:
  At ~7 fps capture rate during HOLD, a static hover produces
  10-15 nearly identical frames per second. Training on duplicates
  causes YOLO to memorize specific frames rather than generalize.
  Expected rejection: 50-65% (similar to v3's 58.6%).

PARAMS:
  HASH_THRESHOLD = 4  → same as v3 (sweet spot: catches obvious dups,
                         keeps frames with real scene change)
  WINDOW_SIZE    = 15 → same as v3 (only compare within 15-frame window)

USAGE:
  python3 dedup_v4.py

GITHUB: Push this script ✓. Never push the data folders ✗.
"""

import os, glob, shutil
from PIL import Image
import imagehash
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.expanduser('~/drone_detection/capture_v4')
SRC_IMG_DIR = os.path.join(BASE_DIR, 'raw', 'images')
SRC_LBL_DIR = os.path.join(BASE_DIR, 'raw', 'labels')
OUT_IMG_DIR = os.path.join(BASE_DIR, 'deduped', 'images')
OUT_LBL_DIR = os.path.join(BASE_DIR, 'deduped', 'labels')
REJ_IMG_DIR = os.path.join(BASE_DIR, 'dedup_rejects', 'images')  # NOT deleted
REJ_LBL_DIR = os.path.join(BASE_DIR, 'dedup_rejects', 'labels')

# ── Tuning (same as v3) ────────────────────────────────────────────────────────
HASH_THRESHOLD = 4   # ≤ 4: duplicate. Raise to 6 if rejection >85%, lower to 3 if <20%
WINDOW_SIZE    = 15  # Only compare with the last N frames (sliding window, not global)


def main():
    for d in [OUT_IMG_DIR, OUT_LBL_DIR, REJ_IMG_DIR, REJ_LBL_DIR]:
        os.makedirs(d, exist_ok=True)

    jpegs = sorted(glob.glob(os.path.join(SRC_IMG_DIR, '*.jpg')))
    if not jpegs:
        print("ERROR: No .jpg files in %s" % SRC_IMG_DIR)
        print("       Run capture_training_data.py first.")
        return

    print("\n=== dedup_v4.py ===")
    print("Source: %s (%d frames)" % (SRC_IMG_DIR, len(jpegs)))
    print("HASH_THRESHOLD=%d  WINDOW_SIZE=%d" % (HASH_THRESHOLD, WINDOW_SIZE))

    kept = 0; rejected = 0
    recent_hashes = []   # sliding window of (hash, filename)

    for jpg_path in tqdm(jpegs, desc="Deduping"):
        fname = os.path.basename(jpg_path)
        stem  = os.path.splitext(fname)[0]
        lbl_src = os.path.join(SRC_LBL_DIR, stem + '.txt')

        try:
            img  = Image.open(jpg_path)
            h    = imagehash.phash(img)
        except Exception as e:
            print("\nWARN: Could not hash %s: %s" % (fname, e))
            continue

        # Check against window
        is_dup = any(
            abs(h - prev_h) <= HASH_THRESHOLD
            for prev_h, _ in recent_hashes
        )

        if is_dup:
            shutil.copy2(jpg_path, os.path.join(REJ_IMG_DIR, fname))
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(REJ_LBL_DIR, stem + '.txt'))
            rejected += 1
        else:
            shutil.copy2(jpg_path, os.path.join(OUT_IMG_DIR, fname))
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(OUT_LBL_DIR, stem + '.txt'))
            kept += 1
            recent_hashes.append((h, fname))
            if len(recent_hashes) > WINDOW_SIZE:
                recent_hashes.pop(0)

    total = kept + rejected
    pct_kept = 100 * kept / total if total else 0
    pct_rej  = 100 * rejected / total if total else 0

    print("\n=== Dedup results ===")
    print("  Total input:  %d frames" % total)
    print("  Kept:         %d frames (%.1f%%)" % (kept, pct_kept))
    print("  Rejected:     %d frames (%.1f%%)" % (rejected, pct_rej))
    print("  Rejects kept in: %s (safe to inspect)" % REJ_IMG_DIR)
    print("  Deduped set:     %s" % OUT_IMG_DIR)

    if pct_rej > 85:
        print("\n  ⚠ WARN: Rejection >85%% — HASH_THRESHOLD may be too loose (lower to 3)")
    elif pct_rej < 20:
        print("\n  ⚠ WARN: Rejection <20%% — HASH_THRESHOLD may be too strict (raise to 6)")
    else:
        print("\n  ✓ Rejection rate normal (expected 50-65%% for simulation data)")

    print("\nNext: run sort_v4_by_scenario.py")


if __name__ == '__main__':
    main()