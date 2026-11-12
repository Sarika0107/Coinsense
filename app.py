"""
CoinSense - Streamlit frontend.

Run on the Jetson (inside the yolo_gpu environment):
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Then open http://192.168.55.1:8501 in the laptop's browser.
"""

import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from backend import config
from backend.detector import CoinDetector
from backend.pipeline import CoinPipeline

st.set_page_config(page_title="CoinSense", page_icon="🪙", layout="wide")

STATUS_STYLE = {
    "green": ("#1e7b34", "#e6f4ea"),
    "yellow": ("#8a6100", "#fff4ce"),
    "red": ("#a3201b", "#fde7e6"),
    "grey": ("#555555", "#eeeeee"),
}


# ------------------------------------------------------------------ helpers

@st.cache_resource(show_spinner="Loading the coin model...")
def load_detector(model_path):
    """Load the YOLO model once and keep it in memory."""
    return CoinDetector(model_path)


def read_uploaded_image(file):
    """Streamlit upload / camera photo -> OpenCV BGR image."""
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def grab_jetson_frame(camera_index):
    """Take one frame from the USB webcam plugged into the Jetson."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    for _ in range(5):              # first frames are often dark, skip them
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def status_banner(decision):
    text_color, bg_color = STATUS_STYLE[decision.color]
    st.markdown(
        f"""<div style="background:{bg_color};color:{text_color};padding:14px 18px;
        border-radius:10px;border-left:8px solid {text_color};font-size:1.25rem;
        font-weight:600;">{decision.message}</div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ sidebar

st.sidebar.title("⚙️ Settings")

source = st.sidebar.radio(
    "Input",
    ["Upload a photo", "Laptop camera", "Jetson USB webcam"],
    help="Where the coin image comes from.",
)

use_target = st.sidebar.checkbox("I need a target amount", value=True)
target_dollars = st.sidebar.number_input(
    "Target amount ($)", min_value=0.01, max_value=500.0, value=2.50, step=0.25,
    format="%.2f", disabled=not use_target,
)
target_cents = round(target_dollars * 100) if use_target else None

min_conf = st.sidebar.slider(
    "Minimum confidence", 0.10, 0.90, float(config.MIN_CONFIDENCE), 0.05,
    help="Detections below this are ignored.",
)
sure_conf = st.sidebar.slider(
    "'Sure' confidence", min_conf, 0.95, max(min_conf, float(config.SURE_CONFIDENCE)), 0.05,
    help="Coins between the two sliders are marked as unsure (grey box).",
)

model_path = st.sidebar.text_input("Model file", config.DEFAULT_MODEL_PATH)
camera_index = st.sidebar.number_input("Jetson camera index", 0, 5, 0, 1,
                                       disabled=source != "Jetson USB webcam")

# ------------------------------------------------------------------ header

st.title("🪙 CoinSense – Smart Coin Counter")
st.caption("Fine-tuned YOLOv8n on an NVIDIA Jetson · detects pennies, nickels, dimes "
           "and quarters, adds up the total and tells you what to do next.")

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------ model

try:
    detector = load_detector(model_path)
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()

detector.min_confidence = min_conf
config.SURE_CONFIDENCE = sure_conf          # used by Detection.is_sure

# ------------------------------------------------------------------ input

frame = None
if source == "Upload a photo":
    file = st.file_uploader("Upload a photo of coins on a table",
                            type=["jpg", "jpeg", "png"])
    if file is not None:
        frame = read_uploaded_image(file)

elif source == "Laptop camera":
    shot = st.camera_input("Point the camera down at the coins and take a photo")
    if shot is not None:
        frame = read_uploaded_image(shot)

else:
    st.write("Place the coins under the Jetson webcam, then press the button.")
    if st.button("📸 Capture from Jetson webcam", type="primary"):
        frame = grab_jetson_frame(int(camera_index))
        if frame is None:
            st.error("No camera signal – check the webcam is plugged into the Jetson.")
            st.stop()
        st.session_state.last_frame = frame
    elif "last_frame" in st.session_state:
        frame = st.session_state.last_frame

if frame is None:
    st.info("👈 Choose an input on the left and add a photo of coins to start.")
    st.stop()

# ------------------------------------------------------------------ run backend

pipeline = CoinPipeline(detector, target_cents=target_cents)
start = time.time()
annotated, detections, decision = pipeline.process_frame(frame, use_stabilizer=False)
elapsed_ms = (time.time() - start) * 1000

# ------------------------------------------------------------------ results

left, right = st.columns([3, 2])

with left:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption=f"Detections (processed in {elapsed_ms:.0f} ms)",
             width="stretch")

with right:
    st.metric("Total", f"${decision.total_dollars:.2f}",
              delta=None if target_cents is None
              else f"{(decision.total_cents - target_cents) / 100:+.2f} vs target")
    status_banner(decision)
    st.write("")

    counts = decision.counts
    table = pd.DataFrame({
        "Coin": [c.capitalize() for c in counts],
        "Count": [counts[c] for c in counts],
        "Value": [f"${counts[c] * config.COIN_VALUES[c] / 100:.2f}" for c in counts],
    })
    st.dataframe(table, hide_index=True, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Coins", len(detections))
    c2.metric("Unsure", decision.unsure_count)
    c3.metric("Overlaps", decision.overlapping_pairs)

    for roll in decision.roll_ready:
        st.success(f"🎉 {roll}")

    if not decision.is_final and decision.status != "NO_COINS":
        st.caption("The total is not final yet – fix the warning above and try again.")

# ------------------------------------------------------------------ history

st.divider()
h1, h2 = st.columns([3, 1])
with h1:
    st.subheader("Count history")
with h2:
    if st.button("💾 Save this count"):
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Total": f"${decision.total_dollars:.2f}",
            "Pennies": counts["penny"], "Nickels": counts["nickel"],
            "Dimes": counts["dime"], "Quarters": counts["quarter"],
            "Status": decision.status,
        })

if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    st.dataframe(hist, hide_index=True, width="stretch")
    st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False),
                       "coinsense_history.csv", "text/csv")
else:
    st.caption("No saved counts yet. Press **Save this count** to keep a result.")

with st.expander("Raw backend output (JSON)"):
    st.json(decision.to_dict())
