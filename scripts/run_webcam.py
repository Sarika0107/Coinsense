"""
Run the backend live on the Jetson USB webcam.

Usage:
    python scripts/run_webcam.py --target 2.50            # prints results in the terminal
    python scripts/run_webcam.py --target 2.50 --show     # also opens a window (needs a display)
Press Ctrl+C to stop. A snapshot of the last frame is saved to outputs/webcam_last.jpg
"""
import argparse
import json
import time
from pathlib import Path
import _path  # noqa: F401

import cv2
from backend.detector import CoinDetector
from backend.pipeline import CoinPipeline
from backend import config

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--target", type=float, default=None)
parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

target = round(args.target * 100) if args.target else None
pipe = CoinPipeline(CoinDetector(args.model), target_cents=target)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("No camera signal - check the webcam is plugged in.")

Path("outputs").mkdir(exist_ok=True)
last_print, frames, start = 0.0, 0, time.time()
annotated = None
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No camera signal")
            break
        annotated, _, decision = pipe.process_frame(frame)
        frames += 1

        if time.time() - last_print >= 1.0:          # print once per second
            fps = frames / (time.time() - start)
            info = decision.to_dict()
            print(f"[{fps:4.1f} FPS] {info['total']:>7} | {info['status']:<10} | "
                  f"{info['message']} | final={info['is_final']} | {info['counts']}")
            last_print = time.time()

        if args.show:
            cv2.imshow("CoinSense", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()
    if annotated is not None:
        cv2.imwrite("outputs/webcam_last.jpg", annotated)
        print("Last frame saved to outputs/webcam_last.jpg")
    cv2.destroyAllWindows()
