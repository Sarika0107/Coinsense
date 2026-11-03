"""
CoinSense - full backend pipeline for one frame:
image -> detect coins -> make decision -> draw results on the image.
The frontend (next stage) will call process_frame().
"""

import cv2

from backend.decisions import make_decision, TotalStabilizer

COLORS = {                      # BGR colors
    "green": (60, 180, 75),
    "yellow": (0, 200, 255),
    "red": (60, 60, 230),
    "grey": (160, 160, 160),
}
COIN_COLORS = {
    "penny": (40, 90, 200),
    "nickel": (200, 200, 200),
    "dime": (230, 160, 40),
    "quarter": (60, 180, 75),
}


def draw_results(frame, detections, decision):
    """Draw a box + label on each coin and a status bar at the top."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = COIN_COLORS[d.coin] if d.is_sure else COLORS["grey"]
        label = f"{d.coin} {d.confidence:.2f}" + ("" if d.is_sure else " ?")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    bar = COLORS[decision.color]
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), bar, -1)
    cv2.putText(out, f"${decision.total_dollars:.2f} | {decision.message}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


class CoinPipeline:
    def __init__(self, detector, target_cents=None):
        self.detector = detector
        self.target_cents = target_cents
        self.stabilizer = TotalStabilizer()

    def process_frame(self, frame_bgr, use_stabilizer=True):
        detections = self.detector.detect(frame_bgr)
        decision = make_decision(detections, self.target_cents)

        # A total is only final once it has been steady for ~1 second (video)
        if use_stabilizer and decision.is_final:
            decision.is_final = self.stabilizer.update(decision.total_cents)

        annotated = draw_results(frame_bgr, detections, decision)
        return annotated, detections, decision
