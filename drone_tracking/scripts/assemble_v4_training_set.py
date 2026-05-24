#!/usr/bin/env python3
"""
assemble_v4_training_set.py — Build YOLO v4 training set from reviewed buckets
===============================================================================
Rawad Fakhredine | FYP Masters in Robotics

Combines v4_buckets/ (after manual label review) with:
  - Oversampling of hard cases
  - 12% rehearsal from v3_training_set (prevents catastrophic forgetting)

Oversampling strategy (confirmed):
  _missed  : 3× (current model failures — highest priority)
  _far     : 3× (small bbox — hardest to detect)
  _hard    : 3× (T9 evasion — fastest motion)
  _close   : 1× (v3 handles this well)
  _mid     : 1× (main zone, well covered)
  _empty   : 1× (negative examples — don't overweight)

Rehearsal: 12% from v3_training_set/train/ (ensures model keeps v3 knowledge)
  Uses class remapping to ensure all labels are class 0 (drone).

Split: 85% train / 15% valid (same philosophy as v3's 80/20, slightly larger train)

Output: v4_training_set/ in Ultralytics format ready for Colab.

GITHUB: Push this script ✓. Never push data or training set ✗.
"""

import os, glob, shutil, random, yaml

BUCKETS_DIR  = os.path.expanduser('~/drone_detection/capture_v4/sorted')
V3_TRAIN_DIR = os.path.expanduser('~/drone_detection/v3_training_set/train')
OUTPUT_DIR   = os.path.expanduser('~/drone_detection/v4_training_set')
SEED = 42; random.seed(SEED)

VAL_SPLIT      = 0.15
REHEARSAL_FRAC = 0.12

OVERSAMPLE = {
    '_missed': 3,
    '_far':    3,
    '_hard':   3,
    '_mid':    1,
    '_close':  1,
    '_empty':  1,
}


def remap_to_class0(src, dst):
    """Ensure all label lines use class 0 (drone). Fixes multi-class v3 labels."""
    try:
        with open(src) as f: lines = f.readlines()
        with open(dst, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = '0'; f.write(' '.join(parts) + '\n')
    except Exception:
        open(dst, 'w').close()


def main():
    for split in ('train', 'valid'):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

    all_entries = []   # (jpg_path, lbl_path, dest_prefix, source_tag)

    # ── 1. Add v4 bucket frames with oversampling ──────────────────────
    print("=== Collecting v4 bucket frames ===")
    for bucket, mult in OVERSAMPLE.items():
        bdir = os.path.join(BUCKETS_DIR, bucket)
        if not os.path.exists(bdir):
            print("  %-10s NOT FOUND — skipped" % bucket); continue
        jpegs = sorted(glob.glob(os.path.join(bdir, 'images', '*.jpg')))
        for rep in range(mult):
            for jpg in jpegs:
                stem = os.path.splitext(os.path.basename(jpg))[0]
                lbl  = os.path.join(bdir, 'labels', stem + '.txt')
                pfx  = 'v4%s_x%d_' % (bucket, rep)
                all_entries.append((jpg, lbl, pfx, 'v4'))
        print("  %-10s %3d × %d = %4d" % (bucket, len(jpegs), mult, len(jpegs)*mult))

    # ── 2. Add v3 rehearsal ────────────────────────────────────────────
    if os.path.exists(V3_TRAIN_DIR):
        v3_imgs = glob.glob(os.path.join(V3_TRAIN_DIR, 'images', '*.jpg'))
        n_rehearsal = int(len(v3_imgs) * REHEARSAL_FRAC)
        sample = random.sample(v3_imgs, n_rehearsal)
        for jpg in sample:
            stem = os.path.splitext(os.path.basename(jpg))[0]
            lbl  = os.path.join(V3_TRAIN_DIR, 'labels', stem + '.txt')
            all_entries.append((jpg, lbl, 'v3r_', 'v3'))
        print("  v3 rehearsal: %d/%d (%.0f%%)"
              % (n_rehearsal, len(v3_imgs), REHEARSAL_FRAC*100))
    else:
        print("  v3_training_set NOT FOUND — skipping rehearsal")

    # ── 3. Shuffle and split ───────────────────────────────────────────
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * (1 - VAL_SPLIT))
    train_set = all_entries[:split_idx]
    valid_set = all_entries[split_idx:]

    # ── 4. Copy files ──────────────────────────────────────────────────
    print("\n=== Copying files ===")
    for entries, split in [(train_set, 'train'), (valid_set, 'valid')]:
        img_dir = os.path.join(OUTPUT_DIR, split, 'images')
        lbl_dir = os.path.join(OUTPUT_DIR, split, 'labels')
        for jpg, lbl, pfx, src in entries:
            out_img = os.path.join(img_dir, pfx + os.path.basename(jpg))
            out_lbl = os.path.join(lbl_dir,
                                   pfx + os.path.splitext(os.path.basename(jpg))[0] + '.txt')
            shutil.copy2(jpg, out_img)
            if src == 'v3' and os.path.exists(lbl):
                remap_to_class0(lbl, out_lbl)
            elif os.path.exists(lbl):
                shutil.copy2(lbl, out_lbl)
            else:
                open(out_lbl, 'w').close()
        print("  %s: %d frames" % (split, len(entries)))

    # ── 5. data.yaml ──────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump({'path': OUTPUT_DIR, 'train': 'train/images',
                   'val': 'valid/images', 'nc': 1, 'names': ['drone']}, f)

    print("\n=== v4 training set ready ===")
    print("  Total: %d | Train: %d | Valid: %d"
          % (len(all_entries), len(train_set), len(valid_set)))
    print("  Output: %s" % OUTPUT_DIR)
    print("\nNext:")
    print("  cd ~/drone_detection")
    print("  zip -r v4_training_set.zip v4_training_set/")
    print("  # Upload to Colab and fine-tune from v3 weights")


if __name__ == '__main__':
    main()