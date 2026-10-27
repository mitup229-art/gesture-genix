# Gesture Genix

AI-based hand sign interpreter — webcam + MediaPipe + lightweight ML pipeline.
MCA capstone project.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Build order

Run these in sequence — each step produces something demoable.

### 1. Confirm camera + landmark tracking works
```bash
cd src
python capture.py
```
You should see your hand with 21 landmarks drawn on it in real time. Press `q` to quit.

### 2. Collect training data
```bash
python data_collector.py
```
- Press `0`-`9` to select which gesture you're recording (see `GESTURE_LABELS` in `config.py`).
- Press `r` to start/stop recording samples for the selected gesture.
- Aim for 150-300 samples per gesture, varying hand angle/distance/position.
- Data is appended to `data/raw_landmarks/landmarks.csv`.

### 3. Train the classifier
```bash
python train_classifier.py
```
Prints accuracy and a confusion matrix, then saves the model to `data/models/gesture_classifier.pkl`.
Check the confusion matrix for gestures that get mixed up — you may need more/cleaner data for those.

### 4. Run live inference
```bash
python inference.py
```
Live webcam feed with recognized gestures buffered into a sentence, spoken aloud via text-to-speech
when you pause (drop your hand) for about a second.

## Project structure

```
gesture-genix/
├── data/
│   ├── raw_landmarks/    # collected training CSVs
│   └── models/           # trained classifier (.pkl)
├── src/
│   ├── config.py          # shared constants: paths, gesture labels, thresholds
│   ├── landmark_utils.py  # landmark extraction + normalization (shared logic)
│   ├── capture.py          # Phase 1: camera + landmark visualization
│   ├── data_collector.py  # Phase 2: labeled data collection tool
│   ├── train_classifier.py # Phase 3: train + validate the classifier
│   ├── nlp_layer.py        # sentence construction from gesture tokens
│   └── inference.py        # Phase 4-5: live inference + sentence + TTS
├── requirements.txt
└── README.md
```

## Next steps (capstone extensions)

Once the above works end-to-end:
- NLP fluency polish (small transformer pass over `nlp_layer.py` output)
- Web dashboard + WebSocket bridge (live video/text over the network)
- History logging (SQLite)
- Custom gesture training portal (record → retrain → hot-reload)
- TFLite/edge optimization (stretch goal)
