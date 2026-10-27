"""
config.py — shared constants for the entire Gesture Genix pipeline.
Edit GESTURE_LABELS to add/remove gestures at any time.
"""

from pathlib import Path

# ── Directory layout ───────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent   # gesture-genix/
DATA_DIR  = ROOT_DIR / "data"
RAW_DIR   = DATA_DIR / "raw_landmarks"
MODEL_DIR = DATA_DIR / "models"

# Ensure directories exist at import time
RAW_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── File paths ─────────────────────────────────────────────────────────────────
LANDMARKS_CSV  = RAW_DIR   / "landmarks.csv"
MODEL_PATH     = MODEL_DIR / "gesture_classifier.pkl"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"

# ── Gesture vocabulary ─────────────────────────────────────────────────────────
# Keys are the digit keys pressed during data collection (0–9).
# Values are the token strings fed to the NLP layer.
GESTURE_LABELS = {
    0: "hello",
    1: "yes",
    2: "no",
    3: "please",
    4: "thank_you",
    5: "i_love_you",
    6: "peace",
    7: "thumbs_up",
    8: "thumbs_down",
    9: "stop",
}

# ── MediaPipe settings ─────────────────────────────────────────────────────────
NUM_LANDMARKS            = 21
# We use x and y only — z is unreliable on standard webcams
NUM_FEATURES             = NUM_LANDMARKS * 2   # 42 features per frame
MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE  = 0.75

# ── Data collection ────────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 200   # recommended target per gesture

# ── Inference / prediction ─────────────────────────────────────────────────────
# Majority-vote over the last N frames for stability
PREDICTION_BUFFER_SIZE = 10
# Drop predictions below this probability (model.predict_proba)
CONFIDENCE_THRESHOLD   = 0.70
# Seconds of hand-absent signal before the buffered sentence is spoken
PAUSE_TRIGGER_SECONDS  = 1.2

# ── Display ────────────────────────────────────────────────────────────────────
WINDOW_TITLE = "Gesture Genix"
FONT_SCALE   = 0.9
FONT_COLOR   = (0, 255, 120)    # green-ish
FONT_COLOR_DIM = (180, 180, 180)
