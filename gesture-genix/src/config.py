"""
Shared constants used across the pipeline.
Keep this as the single source of truth for gesture labels and paths,
so data_collector.py, train_classifier.py, and inference.py all stay in sync.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_LANDMARKS_DIR = os.path.join(DATA_DIR, "raw_landmarks")
MODELS_DIR = os.path.join(DATA_DIR, "models")

LANDMARKS_CSV = os.path.join(RAW_LANDMARKS_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_classifier.pkl")

# ---- Gesture set ----
# Start small (8-10 static gestures). Add more once the pipeline works end-to-end.
GESTURE_LABELS = [
    "HELLO",
    "YES",
    "NO",
    "WATER",
    "HELP",
    "THANK_YOU",
    "I",
    "WANT",
]

# ---- Capture settings ----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

# ---- MediaPipe settings ----
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ---- Data collection ----
SAMPLES_PER_GESTURE = 200

# ---- Classification / smoothing ----
CONFIDENCE_THRESHOLD = 0.8
SMOOTHING_WINDOW = 10       # frames considered
SMOOTHING_AGREEMENT = 8     # of those frames, how many must agree to "commit"

# ---- Sentence buffering ----
PAUSE_FRAMES_TO_END_SENTENCE = 30  # ~1 second at 30fps with no hand detected
