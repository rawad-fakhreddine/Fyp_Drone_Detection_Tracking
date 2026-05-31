#!/usr/bin/env python3
"""
dedup_phash_v4.py — Remove near-duplicate frames via perceptual hashing
========================================================================
Walks frames in capture order, drops any frame within --threshold
Hamming distance of the last KEPT frame. Moves image + its label
together to a _dedup_rejects/ folder (nothing is deleted).

Usage:
  python3 dedup_phash_v4.py --dry-run        # preview only, moves nothing
  python3 dedup_phash_v4.py                  # threshold=5 (default)
  python3 dedup_phash_v4.py --threshold 3    # stricter (keeps more)
  python3 dedup_phash_v4.py --threshold 8    # looser (removes more)
"""
import os, glob, shutil, argparse
from PIL import Image
import imagehash

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', default=os.path.expanduser(
        '~/drone_detection/capture_v4/raw/images'))
    p.add_argument('--labels', default=os.path.expanduser(
        '~/drone_detection/capture_v4/labels'))
    p.add_argument('--threshold', type=int, default=5,
                   help='Hamming distance; lower=stricter (default 5)')
    p.add_argument('--hash-size', type=int, default=10)
    p.add_argument('--dry-run', action='store_true',
                   help='Preview counts without moving any files')
    args = p.parse_args()

    rej_root = os.path.join(os.path.dirname(args.images), '..', '_dedup_rejects')
    rej_img  = os.path.abspath(os.path.join(rej_root, 'images'))
    rej_lbl  = os.path.abspath(os.path.join(rej_root, 'labels'))
    if not args.dry_run:
        os.makedirs(rej_img, exist_ok=True)
        os.makedirs(rej_lbl, exist_ok=True)

    images = sorted(glob.glob(f"{args.images}/*.jpg") +
                    glob.glob(f"{args.images}/*.png"))
    print(f"Scanning {len(images)} images | threshold={args.threshold} "
          f"| dry_run={args.dry_run}")

    last_hash = None
    kept = removed = errors = 0
    for i, path in enumerate(images):
        try:
            h = imagehash.phash(Image.open(path), hash_size=args.hash_size)
        except Exception:
            errors += 1
            continue

        if last_hash is not None and (h - last_hash) <= args.threshold:
            removed += 1
            if not args.dry_run:
                fname = os.path.basename(path)
                stem  = os.path.splitext(fname)[0]
                shutil.move(path, os.path.join(rej_img, fname))
                lbl = os.path.join(args.labels, f"{stem}.txt")
                if os.path.exists(lbl):
                    shutil.move(lbl, os.path.join(rej_lbl, f"{stem}.txt"))
        else:
            last_hash = h
            kept += 1

        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(images)}] kept={kept} removed={removed}")

    pct = 100 * removed / max(len(images), 1)
    print(f"\nResult: kept={kept}  removed={removed} ({pct:.1f}%)  errors={errors}")
    if args.dry_run:
        print("DRY RUN — nothing moved. Re-run without --dry-run to apply.")
    else:
        print(f"Rejects moved to: {os.path.abspath(rej_root)}")
        print("Review the reduced set in labelImg, then run sort + assemble.")

if __name__ == '__main__':
    main()
