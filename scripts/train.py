"""
Fine-tune a pretrained YOLOv8n model on the coin dataset (runs on the Jetson GPU).

Usage:
    python scripts/train.py --data data/coins/data.yaml --epochs 30
The best model is copied to models/coinsense_best.pt
"""
import argparse
import shutil
from pathlib import Path
import _path  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/coins/data.yaml")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=8)      # small batch: Jetson memory
parser.add_argument("--device", default="0")
parser.add_argument("--base", default="yolov8n.pt")      # pretrained starting point
args = parser.parse_args()

from ultralytics import YOLO

model = YOLO(args.base)
results = model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=2,
    project="runs",
    name="coinsense",
    exist_ok=True,
)

best = Path(results.save_dir) / "weights" / "best.pt"
Path("models").mkdir(exist_ok=True)
shutil.copy(best, "models/coinsense_best.pt")
print("\nSaved trained model to models/coinsense_best.pt")
print(f"Training plots and metrics are in: {results.save_dir}")
