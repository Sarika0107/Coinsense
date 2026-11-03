"""
CoinSense - decision-making logic.

This file does NOT use the AI model. It takes the list of detected coins
and turns it into counts, a total amount and decisions (the rules from the
proposal). Keeping it separate means it can be tested without a camera.
"""

import time
from dataclasses import dataclass, field

from backend import config


@dataclass
class Detection:
    """One coin found by the model."""
    coin: str            # "penny", "nickel", "dime" or "quarter"
    confidence: float    # 0.0 - 1.0
    box: tuple           # (x1, y1, x2, y2) in pixels

    @property
    def is_sure(self):
        return self.confidence >= config.SURE_CONFIDENCE


@dataclass
class Decision:
    """Everything the frontend needs to show for one frame."""
    counts: dict
    total_cents: int
    status: str                      # NO_COINS, SPREAD_OUT, ENOUGH, NEED_MORE, COUNTED
    color: str                       # green / yellow / red / grey
    message: str
    roll_ready: list = field(default_factory=list)
    unsure_count: int = 0
    overlapping_pairs: int = 0
    is_final: bool = True

    @property
    def total_dollars(self):
        return self.total_cents / 100

    def to_dict(self):
        return {
            "counts": self.counts,
            "total": f"${self.total_dollars:.2f}",
            "total_cents": self.total_cents,
            "status": self.status,
            "color": self.color,
            "message": self.message,
            "roll_ready": self.roll_ready,
            "unsure_count": self.unsure_count,
            "overlapping_pairs": self.overlapping_pairs,
            "is_final": self.is_final,
        }


# ---------------------------------------------------------------- helpers

def count_coins(detections):
    """How many of each coin type were found."""
    counts = {coin: 0 for coin in config.COIN_VALUES}
    for d in detections:
        if d.coin in counts:
            counts[d.coin] += 1
    return counts


def total_in_cents(counts):
    """Add up the money: count x value for every coin type."""
    return sum(counts[c] * config.COIN_VALUES[c] for c in counts)


def iou(a, b):
    """Intersection-over-Union of two boxes: 0 = no overlap, 1 = same box."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def count_overlaps(detections, threshold=config.OVERLAP_IOU):
    """Number of coin pairs whose boxes overlap too much (coins touching/stacked)."""
    pairs = 0
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if iou(detections[i].box, detections[j].box) > threshold:
                pairs += 1
    return pairs


def rolls_ready(counts):
    """Rule 2: coin types that have enough coins for a full bank roll."""
    return [
        f"{coin} roll ready ({counts[coin]}/{config.ROLL_SIZES[coin]})"
        for coin in config.ROLL_SIZES
        if counts[coin] >= config.ROLL_SIZES[coin]
    ]


# ------------------------------------------------------------- main rules

def make_decision(detections, target_cents=None):
    """
    Apply the proposal's rules to one frame's detections.

    detections   : list of Detection (already filtered to >= MIN_CONFIDENCE)
    target_cents : amount the user needs, e.g. 250 for $2.50 (or None)
    """
    counts = count_coins(detections)
    total = total_in_cents(counts)
    unsure = sum(1 for d in detections if not d.is_sure)
    overlaps = count_overlaps(detections)
    rolls = rolls_ready(counts)

    # Missing detections: never show $0.00 as if it were a real answer
    if not detections:
        return Decision(counts, 0, "NO_COINS", "grey",
                        "No coins detected", [], 0, 0, is_final=False)

    # Rule 3: coins overlap or the model is unsure -> warn instead of a final total
    if overlaps > 0 or unsure > 0:
        reasons = []
        if overlaps:
            reasons.append(f"{overlaps} overlapping pair(s)")
        if unsure:
            reasons.append(f"{unsure} unsure coin(s)")
        return Decision(counts, total, "SPREAD_OUT", "yellow",
                        "Spread coins out - " + ", ".join(reasons),
                        rolls, unsure, overlaps, is_final=False)

    # Rule 1: compare with the target amount
    if target_cents is not None and target_cents > 0:
        if total >= target_cents:
            return Decision(counts, total, "ENOUGH", "green",
                            f"You have enough! (${total/100:.2f} of ${target_cents/100:.2f})",
                            rolls, unsure, overlaps)
        missing = target_cents - total
        return Decision(counts, total, "NEED_MORE", "red",
                        f"You need ${missing/100:.2f} more",
                        rolls, unsure, overlaps)

    return Decision(counts, total, "COUNTED", "green",
                    f"Total: ${total/100:.2f}", rolls, unsure, overlaps)


# --------------------------------------------------- flicker smoothing

class TotalStabilizer:
    """
    Detections flicker from frame to frame. The total is only treated as
    final after it has stayed the same for STABLE_SECONDS.
    """

    def __init__(self, stable_seconds=config.STABLE_SECONDS, clock=time.monotonic):
        self.stable_seconds = stable_seconds
        self.clock = clock
        self._value = None
        self._since = None
        self.stable_value = None

    def update(self, total_cents):
        now = self.clock()
        if total_cents != self._value:
            self._value = total_cents
            self._since = now
        if now - self._since >= self.stable_seconds:
            self.stable_value = self._value
        return self.is_stable

    @property
    def is_stable(self):
        return self.stable_value is not None and self.stable_value == self._value
