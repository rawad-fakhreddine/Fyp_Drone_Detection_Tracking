# ============================================================
# CELL 1 — Install and check GPU
# ============================================================
!pip install ultralytics pyyaml -q

import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT FOUND")
print("Torch:", torch.__version__)


# ============================================================
# CELL 2 — Mount Drive, unzip dataset
# ============================================================
from google.colab import drive
import os, shutil

drive.mount('/content/drive')

zip_src = "/content/drive/MyDrive/FYP_Rawad/v4_training_set.zip"
pt_src  = "/content/drive/MyDrive/FYP_Rawad/best_v3.pt"

assert os.path.exists(zip_src), f"Not found: {zip_src}"
assert os.path.exists(pt_src),  f"Not found: {pt_src}"

shutil.copy(zip_src, "/content/v4_training_set.zip")
os.makedirs("/content/weights", exist_ok=True)
shutil.copy(pt_src, "/content/weights/best_v3.pt")

!unzip -qo /content/v4_training_set.zip -d /content/

DATASET_DIR = "/content/v4_training_set"
print(f"Train images: {len(os.listdir(f'{DATASET_DIR}/images/train'))}")
print(f"Val images:   {len(os.listdir(f'{DATASET_DIR}/images/val'))}")


# ============================================================
# CELL 3 — Write data.yaml
# ============================================================
import yaml

yaml_path = f"{DATASET_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump({
        "path":  DATASET_DIR,
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "drone"},
        "nc":    1,
    }, f)
with open(yaml_path) as f:
    print(f.read())


# ============================================================
# CELL 4 — Train all 3 models (n, s, m)
# ============================================================
from ultralytics import YOLO
import time

MODELS = [
    {"name": "n", "weights": "yolov8n.pt",          "batch": 32},
    {"name": "s", "weights": "/content/weights/best_v3.pt", "batch": 16},
    {"name": "m", "weights": "yolov8m.pt",          "batch": 8},
]

COMMON = dict(
    data           = yaml_path,
    epochs         = 80,
    imgsz          = 640,
    lr0            = 0.001,
    lrf            = 0.01,
    cos_lr         = True,
    warmup_epochs  = 3,
    warmup_bias_lr = 0.01,
    mosaic         = 0.3,
    mixup          = 0.0,
    hsv_h          = 0.015,
    hsv_s          = 0.5,
    hsv_v          = 0.3,
    degrees        = 5.0,
    translate      = 0.1,
    scale          = 0.5,
    fliplr         = 0.5,
    flipud         = 0.0,
    patience       = 20,
    close_mosaic   = 15,
    seed           = 42,
    plots          = True,
    cache          = True,
    workers        = 4,
    project        = "/content/runs/detect",
    exist_ok       = True,
)

results_summary = {}
for cfg in MODELS:
    tag = f"yolov8{cfg['name']}_drone_v4"
    print("\n" + "=" * 60)
    print(f"TRAINING YOLOv8{cfg['name'].upper()}  (batch={cfg['batch']})")
    print("=" * 60)
    t0 = time.time()
    model = YOLO(cfg["weights"])
    model.train(name=tag, batch=cfg["batch"], **COMMON)
    train_min = (time.time() - t0) / 60.0

    best = f"/content/runs/detect/{tag}/weights/best.pt"
    metrics = YOLO(best).val(data=yaml_path, imgsz=640,
                             batch=16, verbose=False, plots=False)
    n_params = sum(p.numel() for p in YOLO(best).model.parameters())
    results_summary[cfg["name"]] = {
        "params_M":  n_params / 1e6,
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "P":         float(metrics.box.mp),
        "R":         float(metrics.box.mr),
        "train_min": train_min,
        "best":      best,
    }

print("\n" + "=" * 60)
print("3-MODEL COMPARISON (v4 dataset)")
print("=" * 60)
print(f"{'Model':<8}{'Params(M)':>10}{'mAP50':>9}"
      f"{'mAP50-95':>10}{'P':>8}{'R':>8}{'Train(min)':>12}")
print("-" * 60)
for k in ["n", "s", "m"]:
    r = results_summary[k]
    print(f"v8{k:<6}{r['params_M']:>10.1f}{r['mAP50']:>9.3f}"
          f"{r['mAP50-95']:>10.3f}{r['P']:>8.3f}{r['R']:>8.3f}"
          f"{r['train_min']:>12.1f}")


# ============================================================
# CELL 5 — Per-scenario evaluation
# ============================================================
import glob
from collections import defaultdict

val_imgs = set(os.path.basename(p).replace(".jpg", "")
               for p in glob.glob(f"{DATASET_DIR}/images/val/*.jpg"))

val_img_to_source = {}
manifest = f"{DATASET_DIR}/MANIFEST.txt"
if os.path.exists(manifest):
    with open(manifest) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1] in val_imgs:
                val_img_to_source[parts[1]] = parts[0]

by_source = defaultdict(list)
for fname, src in val_img_to_source.items():
    by_source[src].append(fname)

def eval_buckets(best_path, model_tag):
    mdl = YOLO(best_path)
    root = f"/content/eval_{model_tag}"
    if os.path.exists(root): shutil.rmtree(root)
    out = {}
    for src, fnames in sorted(by_source.items()):
        if len(fnames) < 5: continue
        bdir = f"{root}/{src.lstrip('_')}"
        for s in ["images/val", "labels/val"]:
            os.makedirs(f"{bdir}/{s}", exist_ok=True)
        for fn in fnames:
            for ext, sub in [(".jpg","images"),(".txt","labels")]:
                sp = f"{DATASET_DIR}/{sub}/val/{fn}{ext}"
                if os.path.exists(sp):
                    shutil.copy(sp, f"{bdir}/{sub}/val/{fn}{ext}")
        byaml = f"{bdir}/data.yaml"
        with open(byaml,"w") as f:
            yaml.dump({"path":bdir,"train":"images/val","val":"images/val",
                       "names":{0:"drone"},"nc":1},f)
        m = mdl.val(data=byaml,imgsz=640,batch=16,verbose=False,plots=False)
        out[src] = {"n":len(fnames),
                    "mAP50":float(m.box.map50),
                    "R":float(m.box.mr)}
    return out

if by_source:
    print("\nPER-SCENARIO mAP50 / Recall")
    per_model = {k: eval_buckets(results_summary[k]["best"], k)
                 for k in ["n","s","m"]}
    buckets = sorted(by_source.keys())
    print(f"{'Bucket':<16}{'n:mAP50/R':>14}{'s:mAP50/R':>14}{'m:mAP50/R':>14}")
    print("-" * 58)
    for b in buckets:
        row = f"{b:<16}"
        for k in ["n","s","m"]:
            r = per_model[k].get(b)
            row += f"{(r['mAP50'] if r else 0):>7.3f}/{(r['R'] if r else 0):<6.3f}"
        print(row)


# ============================================================
# CELL 6 — Save to Drive + download
# ============================================================
from google.colab import files

for k in ["n","s","m"]:
    dst = f"/content/drive/MyDrive/FYP_Rawad/best_v4{k}.pt"
    shutil.copy(results_summary[k]["best"], dst)
    print(f"Saved best_v4{k}.pt to Drive")

files.download(results_summary["s"]["best"])
print("Done — best_v4s.pt downloaded, all 3 saved to Drive")
