"""
CoinSense - the AI part. Loads the fine-tuned YOLOv8 model and turns
its raw output into a clean list of Detection objects.
"""

from pathlib import Path

from backend import config
from backend.decisions import Detection


def normalize_name(label):
    """Map any dataset label ("1c", "Penny", "pennies"...) to our 4 names."""
    return config.CLASS_ALIASES.get(str(label).strip().lower())


class CoinDetector:
    def __init__(self, model_path=config.DEFAULT_MODEL_PATH,
                 device=config.DEVICE, image_size=config.IMAGE_SIZE,
                 min_confidence=config.MIN_CONFIDENCE):
        from ultralytics import YOLO   # imported here so tests run without it

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train it first with "
                f"'python scripts/train.py' (see README.md).")

        self.model = YOLO(model_path)
        self.device = device
        self.image_size = image_size
        self.min_confidence = min_confidence

        # label id -> our coin name (None if the dataset has extra classes)
        self.class_map = {i: normalize_name(n) for i, n in self.model.names.items()}
        unknown = [self.model.names[i] for i, c in self.class_map.items() if c is None]
        if unknown:
            print(f"[CoinDetector] Ignoring unknown classes: {unknown}")

    def detect(self, frame_bgr):
        """Run the model on one image (numpy BGR array) -> list of Detection."""
        result = self.model.predict(
            source=frame_bgr,
            imgsz=self.image_size,
            conf=self.min_confidence,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            coin = self.class_map.get(int(box.cls[0]))
            conf = float(box.conf[0])
            if coin is None or conf < self.min_confidence:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(Detection(coin, conf, (x1, y1, x2, y2)))
        return detections
