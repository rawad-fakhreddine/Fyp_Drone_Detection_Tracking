#!/usr/bin/env python3
"""
relabel_empties.py — Re-run YOLO at lower conf on empty-label frames only.
Doesn't touch any file that already has labels (manual or auto).
"""
import os, glob, argparse, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.path.expanduser(
        '~/drone_detection/models/best.pt'))
    p.add_argument('--images', default=os.path.expanduser(
        '~/drone_detection/capture_v4/raw/images'))
    p.add_argument('--labels', default=os.path.expanduser(
        '~/drone_detection/capture_v4/labels'))
    p.add_argument('--conf', type=float, default=0.20,
                   help='Lower confidence threshold (default 0.20)')
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)

    # Find empty label files
    empties = [f for f in glob.glob(f"{args.labels}/*.txt")
               if os.path.getsize(f) == 0]
    print(f"Found {len(empties)} empty label files")
    print(f"Model: {args.model}  conf: {args.conf}")

    newly_labeled = still_empty = errors = 0
    t0 = time.time()

    for i, label_path in enumerate(empties):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = None
        for ext in ['.jpg', '.png']:
            candidate = os.path.join(args.images, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            continue

        try:
            results = model(img_path, verbose=False, conf=args.conf)
            boxes = results[0].boxes
            if len(boxes) > 0:
                with open(label_path, 'w') as f:
                    for box in boxes:
                        xywhn = box.xywhn[0].tolist()
                        f.write("0 %.6f %.6f %.6f %.6f\n"
                                % (xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
                newly_labeled += 1
            else:
                still_empty += 1
        except Exception:
            errors += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(empties)}] new_labels={newly_labeled} "
                  f"still_empty={still_empty}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Newly labeled (conf={args.conf}): {newly_labeled}")
    print(f"  Still empty (truly no drone):     {still_empty}")
    print(f"  Errors: {errors}")
    print(f"\nNow spot-check the newly labeled frames in labelImg "
          f"(low-conf detections may include false positives)")

if __name__ == '__main__':
    main()
