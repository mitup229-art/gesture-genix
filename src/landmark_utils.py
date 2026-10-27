"""
landmark_utils.py — hand landmark extraction using the MediaPipe Tasks API.

MediaPipe 0.10.35+ removed mp.solutions entirely. This module uses the
Tasks API (mediapipe.tasks.vision.HandLandmarker) which is the supported
replacement. The model file is downloaded automatically on first run.

Public interface (unchanged for all callers):
    extract_landmarks(frame_rgb) → np.ndarray (42,) or None
    get_hand_results(frame_rgb)  → TasksResult
    draw_landmarks(frame_bgr, hand_landmarks)  → None (mutates in place)
"""

import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_DIR,
)

# ── Model file (downloaded once, cached in data/models/) ──────────────────────
_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model() -> None:
    if _MODEL_PATH.exists():
        return
    print("[INFO] First run — downloading hand landmarker model (~30 MB)…")
    print(f"       Saving to: {_MODEL_PATH}")

    def _progress(block_num, block_size, total_size):
        pct = min(block_num * block_size / total_size * 100, 100)
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r       [{bar}] {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH, reporthook=_progress)
    print("\n[INFO] Model ready.")


_ensure_model()

# ── Detector singleton ─────────────────────────────────────────────────────────
_options = _mp_vision.HandLandmarkerOptions(
    base_options=_mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH)),
    running_mode=_mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
_detector = _mp_vision.HandLandmarker.create_from_options(_options)

# ── Hand skeleton connection pairs (index pairs into the 21 landmarks) ─────────
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (0,9),(9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),    # ring
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm cross-links
]


# ── Internal runner ────────────────────────────────────────────────────────────
def _run(frame_rgb: np.ndarray):
    """Convert frame to mp.Image and run the detector."""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    return _detector.detect(mp_img)


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_landmarks(frame_rgb: np.ndarray) -> np.ndarray | None:
    """
    Run detection and return a normalised (42,) feature vector, or None.

    Normalisation
    ─────────────
    1. Translate  → wrist (landmark 0) becomes the origin.
    2. Scale      → divide by bounding-box diagonal → size-invariant.
    3. Flatten    → [x0,y0, x1,y1, … x20,y20]  shape (42,).
    """
    result = _run(frame_rgb)
    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]   # list[NormalizedLandmark] for first hand
    raw = np.array([[lm.x, lm.y] for lm in lms], dtype=np.float32)

    # 1. Translate
    raw -= raw[0]
    # 2. Scale
    x_min, y_min = raw.min(axis=0)
    x_max, y_max = raw.max(axis=0)
    diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    if diag > 1e-6:
        raw /= diag
    # 3. Flatten
    return raw.flatten()


def get_hand_results(frame_rgb: np.ndarray):
    """Return the raw Tasks-API result (used by capture, collector, inference)."""
    return _run(frame_rgb)


def draw_landmarks(frame_bgr: np.ndarray, hand_landmarks: list) -> None:
    """
    Draw the 21-point skeleton on a BGR frame in-place.

    Parameters
    ----------
    frame_bgr     : np.ndarray  BGR image — mutated in place.
    hand_landmarks: list        One hand's NormalizedLandmark list,
                                i.e. result.hand_landmarks[i].
    """
    h, w = frame_bgr.shape[:2]

    # Skeleton connections
    for a, b in _CONNECTIONS:
        x1 = int(hand_landmarks[a].x * w)
        y1 = int(hand_landmarks[a].y * h)
        x2 = int(hand_landmarks[b].x * w)
        y2 = int(hand_landmarks[b].y * h)
        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 210, 110), 2, cv2.LINE_AA)

    # Landmark dots
    for lm in hand_landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame_bgr, (cx, cy), 5, (255, 255, 255), -1)
        cv2.circle(frame_bgr, (cx, cy), 5, (0, 160, 90), 1)
