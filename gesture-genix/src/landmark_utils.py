"""
Shared helpers for extracting and normalizing MediaPipe hand landmarks.
Used by data_collector.py (to build training data) and inference.py
(to build the same feature vector at prediction time) -- keeping this
logic in one place guarantees both stages agree on the feature format.
"""

import numpy as np


def extract_landmark_coords(hand_landmarks):
    """Given a MediaPipe hand_landmarks object, return a list of 21 (x, y, z) tuples."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


def normalize_landmarks(coords):
    """
    Normalize a set of 21 (x, y, z) landmarks so the feature vector is
    invariant to hand position in frame and distance from camera.

    Steps:
      1. Translate so the wrist (landmark 0) becomes the origin.
      2. Scale by the wrist-to-middle-finger-MCP distance (landmark 9).

    Returns a flattened 63-length numpy array.
    """
    coords = np.array(coords, dtype=np.float32)
    wrist = coords[0].copy()
    coords -= wrist

    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale

    return coords.flatten()


def build_feature_vector(multi_hand_landmarks, max_hands=2):
    """
    Build a fixed-length feature vector from however many hands MediaPipe detected.
    Zero-pads if fewer than max_hands are present, so the classifier always
    receives a consistent input size.
    """
    feature_len_per_hand = 63
    vector = np.zeros(feature_len_per_hand * max_hands, dtype=np.float32)

    if not multi_hand_landmarks:
        return vector

    for i, hand_landmarks in enumerate(multi_hand_landmarks[:max_hands]):
        coords = extract_landmark_coords(hand_landmarks)
        normalized = normalize_landmarks(coords)
        start = i * feature_len_per_hand
        vector[start:start + feature_len_per_hand] = normalized

    return vector
