"""
Phase 2: Data collection tool.

Lets you select a gesture label, then press 'r' to record samples.
Each recorded frame's normalized landmark vector + label is appended
to data/raw_landmarks/landmarks.csv.

Run: python src/data_collector.py
Controls:
  0-9 keys : select gesture from GESTURE_LABELS by index (shown on screen)
  r        : toggle recording on/off for the selected gesture
  q        : quit
"""

import csv
import os

import cv2
import mediapipe as mp

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS,
    MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    GESTURE_LABELS, LANDMARKS_CSV, RAW_LANDMARKS_DIR, SAMPLES_PER_GESTURE,
)
from landmark_utils import build_feature_vector

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def ensure_csv_exists():
    os.makedirs(RAW_LANDMARKS_DIR, exist_ok=True)
    if not os.path.exists(LANDMARKS_CSV):
        with open(LANDMARKS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(63 * MAX_NUM_HANDS)] + ["label"]
            writer.writerow(header)


def main():
    ensure_csv_exists()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    selected_idx = 0
    recording = False
    sample_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands, open(LANDMARKS_CSV, "a", newline="") as f:

        writer = csv.writer(f)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                if recording:
                    vector = build_feature_vector(results.multi_hand_landmarks, MAX_NUM_HANDS)
                    label = GESTURE_LABELS[selected_idx]
                    writer.writerow(list(vector) + [label])
                    sample_count += 1

            # ---- overlay UI ----
            label_text = GESTURE_LABELS[selected_idx]
            status = f"RECORDING ({sample_count}/{SAMPLES_PER_GESTURE})" if recording else "PAUSED"
            color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(frame, f"Label: {label_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "0-9: select label | r: record | q: quit", (10, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Gesture Genix - Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                sample_count = 0
            elif ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(GESTURE_LABELS):
                    selected_idx = idx
                    sample_count = 0
                    recording = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {LANDMARKS_CSV}")


if __name__ == "__main__":
    main()
