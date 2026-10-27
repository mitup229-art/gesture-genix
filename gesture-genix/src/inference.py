"""
Phase 4-5: Live inference loop.

Webcam -> landmarks -> classify -> temporal smoothing -> gesture buffer
-> sentence construction -> on-screen display + text-to-speech.

Run: python src/inference.py
Press 'q' to quit.
"""

from collections import deque

import cv2
import joblib
import mediapipe as mp
import pyttsx3

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS,
    MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MODEL_PATH, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW, SMOOTHING_AGREEMENT,
    PAUSE_FRAMES_TO_END_SENTENCE,
)
from landmark_utils import build_feature_vector
from nlp_layer import construct_sentence

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def speak(engine, text):
    engine.say(text)
    engine.runAndWait()


def main():
    clf = joblib.load(MODEL_PATH)
    tts_engine = pyttsx3.init()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
    sentence_tokens = []
    last_committed = None
    no_hand_frames = 0
    current_sentence_display = ""

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            committed_gesture = None

            if results.multi_hand_landmarks:
                no_hand_frames = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                vector = build_feature_vector(results.multi_hand_landmarks, MAX_NUM_HANDS)
                probs = clf.predict_proba([vector])[0]
                best_idx = probs.argmax()
                confidence = probs[best_idx]
                pred = clf.classes_[best_idx]

                pred_buffer.append(pred if confidence > CONFIDENCE_THRESHOLD else None)

                if len(pred_buffer) == SMOOTHING_WINDOW:
                    top_pred = pred_buffer[-1]
                    if top_pred and pred_buffer.count(top_pred) >= SMOOTHING_AGREEMENT:
                        committed_gesture = top_pred
            else:
                no_hand_frames += 1
                pred_buffer.clear()

            if committed_gesture and committed_gesture != last_committed:
                sentence_tokens.append(committed_gesture)
                last_committed = committed_gesture

            if no_hand_frames > PAUSE_FRAMES_TO_END_SENTENCE and sentence_tokens:
                current_sentence_display = construct_sentence(sentence_tokens)
                print("Sentence:", current_sentence_display)
                speak(tts_engine, current_sentence_display)
                sentence_tokens = []
                last_committed = None
                no_hand_frames = 0

            # ---- overlay UI ----
            cv2.putText(frame, f"Buffer: {' '.join(sentence_tokens)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Last sentence: {current_sentence_display}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Gesture Genix - Live Inference (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
