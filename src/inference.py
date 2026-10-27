"""
inference.py — Phase 4-5: Live gesture recognition + sentence + TTS.

Run:
    python inference.py

What happens
────────────
1. Loads the trained model from  data/models/gesture_classifier.pkl
2. Opens the webcam and runs MediaPipe hand tracking on every frame.
3. Extracts normalised landmarks  →  classifier  →  prediction.
4. Majority-votes the last N predictions (reduces jitter).
5. Pushes confident predictions into a SentenceBuffer (NLP layer).
6. When the hand leaves the frame for PAUSE_TRIGGER_SECONDS:
       • the buffered sentence is displayed on screen
       • spoken aloud via pyttsx3 (text-to-speech)
       • buffer is cleared ready for the next sentence

Keys
────
  q     Quit
  c     Manually clear the sentence buffer (without speaking)
  s     Immediately speak + flush the current buffer
"""

import sys
import time
import threading
from collections import deque

import cv2
import joblib
import numpy as np

import config
from landmark_utils import get_hand_results, draw_landmarks, extract_landmarks
from nlp_layer import SentenceBuffer

# ── TTS (non-blocking) ─────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)   # words per minute
    TTS_AVAILABLE = True
except Exception as e:
    print(f"[WARN] TTS unavailable ({e}). Audio output disabled.")
    TTS_AVAILABLE = False


def _speak_async(text: str) -> None:
    """Run TTS in a background thread so the video loop never blocks."""
    if not TTS_AVAILABLE or not text:
        return
    def _run():
        eng = pyttsx3.init()
        eng.setProperty("rate", 160)
        eng.say(text)
        eng.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model():
    if not config.MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {config.MODEL_PATH}")
        print("        Run  train_classifier.py  first.")
        sys.exit(1)
    bundle = joblib.load(config.MODEL_PATH)
    pipeline = bundle["pipeline"]
    le       = bundle["label_encoder"]
    print(f"[INFO] Model loaded — {len(le.classes_)} gestures: {le.classes_.tolist()}")
    return pipeline, le


# ── Prediction buffer (majority-vote) ─────────────────────────────────────────

class PredictionBuffer:
    """Smooths predictions with a majority-vote over a sliding window."""

    def __init__(self, size: int = config.PREDICTION_BUFFER_SIZE) -> None:
        self._buf: deque[str] = deque(maxlen=size)

    def push(self, label: str) -> None:
        self._buf.append(label)

    def majority(self) -> str | None:
        if not self._buf:
            return None
        from collections import Counter
        return Counter(self._buf).most_common(1)[0][0]

    def clear(self) -> None:
        self._buf.clear()


# ── HUD drawing ────────────────────────────────────────────────────────────────

def _draw_hud(frame, prediction, confidence, sentence, hand_present, last_spoken):
    h, w = frame.shape[:2]

    # Current gesture prediction
    if prediction:
        conf_pct = int(confidence * 100)
        cv2.putText(frame, f"{prediction}  ({conf_pct}%)",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    config.FONT_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "— waiting for hand —",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    config.FONT_COLOR_DIM, 1, cv2.LINE_AA)

    # Live sentence buffer
    cv2.putText(frame, f"Buffer: {sentence if sentence else '(empty)'}",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 200, 0), 1, cv2.LINE_AA)

    # Last spoken sentence
    if last_spoken:
        cv2.putText(frame, f"Spoken: {last_spoken}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (180, 180, 255), 1, cv2.LINE_AA)

    # Key hints
    hint = "q: quit  |  s: speak now  |  c: clear buffer"
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                config.FONT_COLOR_DIM, 1, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline, le = _load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    pred_buf     = PredictionBuffer()
    sentence_buf = SentenceBuffer(max_tokens=20)

    hand_absent_since: float | None = None   # timestamp when hand last left
    last_spoken: str = ""
    last_pred:   str | None = None
    last_conf:   float = 0.0

    print("\n[INFO] Live inference running. Press  q  to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = get_hand_results(frame_rgb)
        features  = None
        hand_present = False

        if results.hand_landmarks:
            hand_present = True
            hand_absent_since = None
            for hand_lm in results.hand_landmarks:
                draw_landmarks(frame, hand_lm)
            features = extract_landmarks(frame_rgb)

        # ── Predict ────────────────────────────────────────────────────────
        if features is not None:
            proba  = pipeline.predict_proba([features])[0]
            idx    = int(np.argmax(proba))
            conf   = float(proba[idx])
            label  = le.classes_[idx]

            if conf >= config.CONFIDENCE_THRESHOLD:
                pred_buf.push(label)
                stable = pred_buf.majority()
                if stable:
                    last_pred = stable
                    last_conf = conf
                    sentence_buf.push(stable)
            else:
                last_pred = None   # low confidence — don't show anything
        else:
            last_pred = None

        # ── Pause detection → speak ───────────────────────────────────────
        if not hand_present:
            now = time.time()
            if hand_absent_since is None:
                hand_absent_since = now
            elif (now - hand_absent_since) >= config.PAUSE_TRIGGER_SECONDS:
                if not sentence_buf.is_empty():
                    sentence = sentence_buf.flush()
                    pred_buf.clear()
                    last_spoken = sentence
                    hand_absent_since = None   # reset so we don't re-trigger
                    print(f"[SPEAK] "{sentence}"")
                    _speak_async(sentence)

        # ── Draw HUD ──────────────────────────────────────────────────────
        _draw_hud(
            frame,
            prediction   = last_pred,
            confidence   = last_conf,
            sentence     = sentence_buf.peek(),
            hand_present = hand_present,
            last_spoken  = last_spoken,
        )

        cv2.imshow(config.WINDOW_TITLE + " — Live Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence_buf.flush()
            pred_buf.clear()
            print("[INFO] Buffer cleared.")
        elif key == ord("s"):
            if not sentence_buf.is_empty():
                sentence = sentence_buf.flush()
                pred_buf.clear()
                last_spoken = sentence
                print(f"[SPEAK] "{sentence}"")
                _speak_async(sentence)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")


if __name__ == "__main__":
    main()
