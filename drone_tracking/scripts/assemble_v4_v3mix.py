#!/usr/bin/env python3
"""
assemble_v4_v3mix.py — Build training set from v4 (post-dedup) + v3 mix
=======================================================================
- v4 images split 85/15 train/val
- ALL v3 train images added to train (not val — val stays pure v4)
- Creates data.yaml + MANIFEST.txt for per-scenario eval
"""
import os, glob, random, shutil

V4_IMAGES  = os.path.expanduser("~/drone_detection/capture_v4/raw/images")
V4_LABELS  = os.path.expanduser("~/drone_detection/capture_v4/labels")
V3_TRAIN_I = os.path.expanduser("~/drone_detection/v3_training_set/images/train")
V3_TRAIN_L = os.path.expanduser("~/drone_detection/v3_training_set/labels/train")
OUTPUT     = os.path.expanduser("~/drone_detection/v4_training_set")
VAL_RATIO  = 0.15
SEED       = 42

random.seed(SEED)

# ── 1. Collect v4 pairs ─────────────────────────────────────────
v4_imgs = sorted(glob.glob(f"{V4_IMAGES}/*.jpg") +
                 glob.glob(f"{V4_IMAGES}/*.png"))
v4_pairs = []
for img in v4_imgs:
    stem = os.path.splitext(os.path.basename(img))[0]
    lbl  = os.path.join(V4_LABELS, f"{stem}.txt")
    if os.path.exists(lbl):
        v4_pairs.append((img, lbl, "v4"))

print(f"v4 pairs: {len(v4_pairs)}")

# ── 2. Collect v3 train pairs ───────────────────────────────────
v3_pairs = []
if os.path.isdir(V3_TRAIN_I):
    for img in sorted(glob.glob(f"{V3_TRAIN_I}/*.jpg") +
                      glob.glob(f"{V3_TRAIN_I}/*.png")):
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl  = os.path.join(V3_TRAIN_L, f"{stem}.txt")
        if os.path.exists(lbl):
            v3_pairs.append((img, lbl, "v3"))
print(f"v3 train pairs: {len(v3_pairs)}")

# ── 3. Split v4 into train/val ──────────────────────────────────
random.shuffle(v4_pairs)
n_val = int(len(v4_pairs) * VAL_RATIO)
v4_val   = v4_pairs[:n_val]
v4_train = v4_pairs[n_val:]

# ── 4. Combine: v4_train + ALL v3 for training ──────────────────
train_all = v4_train + v3_pairs
random.shuffle(train_all)

print(f"\nFinal split:")
print(f"  Train: {len(train_all)} ({len(v4_train)} v4 + {len(v3_pairs)} v3)")
print(f"  Val:   {len(v4_val)} (pure v4)")

# ── 5. Create output structure ───────────────────────────────────
if os.path.exists(OUTPUT):
    shutil.rmtree(OUTPUT)
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(f"{OUTPUT}/{sub}", exist_ok=True)

manifest = []
for split, dataset in [("train", train_all), ("val", v4_val)]:
    for img_path, lbl_path, source in dataset:
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]
        # Prefix v3 files to avoid name collisions
        if source == "v3":
            fname = f"v3_{fname}"
            stem  = f"v3_{stem}"
        dst_img = f"{OUTPUT}/images/{split}/{fname}"
        dst_lbl = f"{OUTPUT}/labels/{split}/{stem}.txt"
        shutil.copy(img_path, dst_img)
        shutil.copy(lbl_path, dst_lbl)
        manifest.append(f"{source}\t{stem}\t{split}")

# ── 6. data.yaml ────────────────────────────────────────────────
with open(f"{OUTPUT}/data.yaml", "w") as f:
    f.write(f"path: {OUTPUT}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 1\n")
    f.write("names:\n  0: drone\n")

# ── 7. MANIFEST.txt (for per-scenario eval in Colab) ────────────
with open(f"{OUTPUT}/MANIFEST.txt", "w") as f:
    f.write("source\tfilename\tsplit\n")
    for line in manifest:
        f.write(line + "\n")

# ── 8. Summary ───────────────────────────────────────────────────
train_v3 = sum(1 for _, _, s in train_all if s == "v3")
train_v4 = sum(1 for _, _, s in train_all if s == "v4")
empty_train = sum(1 for _, l, _ in train_all if os.path.getsize(l) == 0)
empty_val   = sum(1 for _, l, _ in v4_val   if os.path.getsize(l) == 0)

print(f"\nDataset created at: {OUTPUT}")
print(f"  Train: {len(train_all)} total ({train_v4} v4 + {train_v3} v3)")
print(f"  Val:   {len(v4_val)} (pure v4)")
print(f"  Empty labels: train={empty_train} val={empty_val} (background frames)")
print(f"  v3 fraction in train: {100*train_v3/len(train_all):.1f}%")
print(f"\nNext: zip -r v4_training_set.zip v4_training_set/")
