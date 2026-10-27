"""
capture.py — Phase 1: Confirm camera + MediaPipe landmark tracking works.

Run:
    python capture.py

You should see your hand with the 21-point skeleton drawn in real time.
Press  q  to quit.
"""

import sys
import cv2

import config
from landmark_utils import get_hand_results, draw_landmarks



def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that a camera is connected.")
        sys.exit(1)

    print("[INFO] Camera opened. Press  q  to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Empty frame received — retrying…")
            continue

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = get_hand_results(frame_rgb)

        # Draw skeleton if a hand is found
        if results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                draw_landmarks(frame, hand_lm)

            # Show handedness label (Left / Right)
            if results.handedness:
                label = results.handedness[0][0].category_name
                cv2.putText(
                    frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                    config.FONT_COLOR, 2, cv2.LINE_AA,
                )
        else:
            cv2.putText(
                frame, "No hand detected", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                config.FONT_COLOR_DIM, 2, cv2.LINE_AA,
            )

        cv2.putText(
            frame, "Phase 1 — Landmark check  |  q: quit",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            config.FONT_COLOR_DIM, 1, cv2.LINE_AA,
        )

        cv2.imshow(config.WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
