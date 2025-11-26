import os
from pathlib import Path

# === CONFIG ===
DATASET_YAML = "data/mydataset.yaml"  # Your dataset YAML (with nc=2 and names)
MODEL = "yolov5n.pt"                  # Pretrained model: yolov5n.pt or yolov5m.pt
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = "cpu"  # or "0" for GPU

# === PATHS ===
YOLOV5_DIR = Path("yolov5")           # Path to cloned YOLOv5 repo
os.chdir(YOLOV5_DIR)

# === TRAIN ===
os.system(
    f"python train.py "
    f"--img {IMG_SIZE} "
    f"--batch {BATCH_SIZE} "
    f"--epochs {EPOCHS} "
    f"--data {DATASET_YAML} "
    f"--weights {MODEL} "
    f"--device {DEVICE} "
    f"--cache"
)
