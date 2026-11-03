"""
Run the full backend on one photo and print the decision.

Usage:
    python scripts/run_image.py path/to/coins.jpg --target 2.50
Saves the annotated image to outputs/
"""
import argparse
import json
from pathlib import Path
import _path  # noqa: F401

import cv2
from backend.detector import CoinDetector
from backend.pipeline import CoinPipeline
from backend import config

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--target", type=float, default=None, help="amount needed in dollars, e.g. 2.50")
parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
parser.add_argument("--device", default=config.DEVICE)
args = parser.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    raise SystemExit(f"Could not read image: {args.image}")

target = round(args.target * 100) if args.target else None
pipe = CoinPipeline(CoinDetector(args.model, device=args.device), target_cents=target)
annotated, detections, decision = pipe.process_frame(frame, use_stabilizer=False)

Path("outputs").mkdir(exist_ok=True)
out_path = Path("outputs") / f"result_{Path(args.image).stem}.jpg"
cv2.imwrite(str(out_path), annotated)

print(json.dumps(decision.to_dict(), indent=2))
print(f"\nAnnotated image saved to {out_path}")
