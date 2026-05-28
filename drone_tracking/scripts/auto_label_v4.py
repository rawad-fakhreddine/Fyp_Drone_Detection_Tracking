#!/usr/bin/env python3
"""
auto_label_v4.py — Auto-label captured frames using YOLO v3
=============================================================
Runs YOLO inference on all images in capture_v4/raw/, saves
YOLO-format .txt label files in capture_v4/labels/.

After running, review labels in labelImg:
  labelImg ~/drone_detection/capture_v4/raw \
           ~/drone_detection/capture_v4/labels/classes.txt \
           ~/drone_detection/capture_v4/labels

Usage:
  python3 auto_label_v4.py                    # default paths
  python3 auto_label_v4.py --conf 0.3         # lower threshold
  python3 auto_label_v4.py --model path.pt    # custom model
"""
import os, glob, argparse, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.expanduser(
        '~/drone_detection/models/best.pt'))
    parser.add_argument('--input', default=os.path.expanduser(
        '~/drone_detection/capture_v4/raw'))
    parser.add_argument('--output', default=os.path.expanduser(
        '~/drone_detection/capture_v4/labels'))
    parser.add_argument('--conf', type=float, default=0.40,
                        help='Confidence threshold (default 0.40)')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.output, exist_ok=True)

    # Write classes.txt for labelImg
    with open(os.path.join(args.output, 'classes.txt'), 'w') as f:
        f.write('drone\n')

    model = YOLO(args.model)
    images = sorted(
        glob.glob(f"{args.input}/*.jpg") +
        glob.glob(f"{args.input}/*.png"))
    print(f"Model:  {args.model}")
    print(f"Images: {len(images)} in {args.input}")
    print(f"Output: {args.output}")
    print(f"Conf:   {args.conf}")

    labeled = empty = errors = 0
    t0 = time.time()

    for i, img_path in enumerate(images):
        try:
            results = model(img_path, verbose=False, conf=args.conf)
            boxes = results[0].boxes
            fname = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(args.output, f"{fname}.txt")

            if len(boxes) > 0:
                with open(label_path, 'w') as f:
                    for box in boxes:
                        xywhn = box.xywhn[0].tolist()
                        f.write("0 %.6f %.6f %.6f %.6f\n"
                                % (xywhn[0], xywhn[1], xywhn[2], xywhn[3]))
                labeled += 1
            else:
                open(label_path, 'w').close()
                empty += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {os.path.basename(img_path)}: {e}")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(images)}] {rate:.0f} img/s "
                  f"| labeled={labeled} empty={empty} errors={errors}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({len(images)/elapsed:.0f} img/s)")
    print(f"  Labeled:  {labeled}")
    print(f"  Empty:    {empty} (no detection — background frames)")
    print(f"  Errors:   {errors}")
    print(f"\nLabels saved to: {args.output}")
    print(f"\nNext: review in labelImg, then run assemble_v4_training_set.py")

if __name__ == '__main__':
    main()
