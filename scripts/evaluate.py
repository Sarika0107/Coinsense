"""
Evaluate CoinSense (proposal section 6).

1) Model accuracy on the dataset's test split (mAP, precision, recall per coin):
    python scripts/evaluate.py --data data/coins/data.yaml
2) Decision accuracy on your own known coin piles:
    python scripts/evaluate.py --piles data/my_piles.csv
   CSV columns: image,true_total   (true_total in dollars, e.g. 1.36)
"""
import argparse
import csv
import _path  # noqa: F401

from backend import config

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None)
parser.add_argument("--piles", default=None)
parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
args = parser.parse_args()

if args.data:
    from ultralytics import YOLO
    metrics = YOLO(args.model).val(data=args.data, split="test", device=config.DEVICE)
    print(f"\nmAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"Precision    : {metrics.box.mp:.3f}")
    print(f"Recall       : {metrics.box.mr:.3f}")

if args.piles:
    import cv2
    from backend.detector import CoinDetector
    from backend.pipeline import CoinPipeline

    pipe = CoinPipeline(CoinDetector(args.model))
    correct = total = 0
    with open(args.piles) as f:
        for row in csv.DictReader(f):
            frame = cv2.imread(row["image"])
            _, _, d = pipe.process_frame(frame, use_stabilizer=False)
            truth = round(float(row["true_total"]) * 100)
            ok = d.total_cents == truth
            correct += ok
            total += 1
            print(f"{row['image']}: predicted ${d.total_dollars:.2f}, "
                  f"true ${truth/100:.2f} -> {'OK' if ok else 'WRONG'}")
    if total:
        print(f"\nExact-total accuracy: {correct}/{total} = {correct/total:.0%} (goal: 80-90%)")

if not args.data and not args.piles:
    parser.print_help()
