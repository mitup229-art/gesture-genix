"""
data_collector.py — Phase 2: Interactive labeled data collection tool.

Run:
    python data_collector.py

Controls
────────
  0–9   Select the gesture you want to record (maps to GESTURE_LABELS).
  r     Toggle recording ON / OFF for the selected gesture.
  s     Print a per-gesture sample count summary to the terminal.
  q     Quit (data is already saved to disk after every sample).

Output
──────
  Appends rows to  data/raw_landmarks/landmarks.csv
  Columns: label, x0, y0, x1, y1, … x20, y20   (1 + 42 = 43 cols)
"""

import sys
import csv
import cv2
import numpy as np
import pandas as pd

import config
from landmark_utils import get_hand_results, draw_landmarks, extract_landmarks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_sample_counts() -> dict[int, int]:
    """Return {gesture_id: sample_count} from the existing CSV (if any)."""
    counts = {k: 0 for k in config.GESTURE_LABELS}
    if config.LANDMARKS_CSV.exists():
        df = pd.read_csv(config.LANDMARKS_CSV)
        for label_str, grp in df.groupby("label"):
            # find the key whose value matches this label string
            for k, v in config.GESTURE_LABELS.items():
                if v == label_str:
                    counts[k] = len(grp)
    return counts


def _append_row(label: str, features: np.ndarray) -> None:
    """Append one sample row to the CSV (create with header if new)."""
    write_header = not config.LANDMARKS_CSV.exists()
    with open(config.LANDMARKS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            coord_cols = [f"{axis}{i}" for i in range(config.NUM_LANDMARKS)
                          for axis in ("x", "y")]
            writer.writerow(["label"] + coord_cols)
        writer.writerow([label] + features.tolist())


def _draw_hud(frame, selected_id, selected_label, recording, counts):
    """Overlay the HUD on the frame (mutates frame in-place)."""
    h, w = frame.shape[:2]

    # Recording indicator
    rec_text  = "● REC" if recording else "○ READY"
    rec_color = (0, 0, 220) if recording else (160, 160, 160)
    cv2.putText(frame, rec_text, (w - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2, cv2.LINE_AA)

    # Selected gesture
    cv2.putText(frame, f"Gesture [{selected_id}]: {selected_label}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                config.FONT_COLOR, 2, cv2.LINE_AA)

    # Sample count for this gesture
    n = counts[selected_id]
    bar_pct = min(n / config.SAMPLES_PER_GESTURE, 1.0)
    cv2.putText(frame, f"Samples: {n} / {config.SAMPLES_PER_GESTURE}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                config.FONT_COLOR, 1, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 10, 88, 260, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * bar_pct), bar_y + bar_h),
                  config.FONT_COLOR, -1)

    # Key hints
    hint = "0-9: select  |  r: record  |  s: summary  |  q: quit"
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                config.FONT_COLOR_DIM, 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    counts      = _load_sample_counts()
    selected_id = 0
    recording   = False

    print("\n[INFO] Data collector ready.")
    print(f"       Target: {config.SAMPLES_PER_GESTURE} samples/gesture")
    print(f"       Output: {config.LANDMARKS_CSV}\n")
    for k, v in config.GESTURE_LABELS.items():
        print(f"  [{k}] {v:20s}  — {counts[k]} samples collected")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = get_hand_results(frame_rgb)
        features  = None

        if results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                draw_landmarks(frame, hand_lm)
            features = extract_landmarks(frame_rgb)

        selected_label = config.GESTURE_LABELS[selected_id]

        # Record if active and we have a valid feature vector
        if recording and features is not None:
            _append_row(selected_label, features)
            counts[selected_id] += 1

        _draw_hud(frame, selected_id, selected_label, recording, counts)

        cv2.imshow(config.WINDOW_TITLE + " — Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            state = "STARTED" if recording else "STOPPED"
            print(f"[REC {state}] gesture={selected_label}  "
                  f"samples={counts[selected_id]}")
        elif key == ord("s"):
            print("\n── Sample summary ──────────────────────────────")
            for k, v in config.GESTURE_LABELS.items():
                bar = "█" * (counts[k] // 10)
                print(f"  [{k}] {v:20s}  {counts[k]:4d}  {bar}")
            print()
        elif ord("0") <= key <= ord("9"):
            new_id = key - ord("0")
            if new_id in config.GESTURE_LABELS:
                selected_id = new_id
                recording   = False   # safety: stop recording on gesture switch
                print(f"[SELECT] gesture={config.GESTURE_LABELS[selected_id]}  "
                      f"samples={counts[selected_id]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Data collection finished.")
    print(f"       CSV saved to: {config.LANDMARKS_CSV}")


if __name__ == "__main__":
    main()
