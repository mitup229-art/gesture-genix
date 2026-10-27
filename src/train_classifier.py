"""
train_classifier.py — Phase 3: Train and validate the gesture classifier.

Run:
    python train_classifier.py

What it does
────────────
1. Loads  data/raw_landmarks/landmarks.csv
2. Encodes gesture labels → integer class indices
3. Splits 80/20 train/test (stratified)
4. Trains a scikit-learn pipeline:
       StandardScaler  →  MLPClassifier (128-64-32)
5. Prints per-class accuracy and a confusion matrix
6. Saves the pipeline to  data/models/gesture_classifier.pkl
7. Saves the label→class mapping to  data/models/label_map.json

The saved pipeline is loaded by  inference.py  for live prediction.
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import config


# ── Utilities ──────────────────────────────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    """Pretty-print a confusion matrix to the terminal."""
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{col_w}}" + "".join(
            f"{cm[i, j]:>{col_w}}" for j in range(len(labels))
        )
        print(row)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load dataset ───────────────────────────────────────────────────
    if not config.LANDMARKS_CSV.exists():
        print(f"[ERROR] Dataset not found: {config.LANDMARKS_CSV}")
        print("        Run  data_collector.py  first.")
        sys.exit(1)

    df = pd.read_csv(config.LANDMARKS_CSV)
    print(f"[INFO] Loaded {len(df)} samples from {config.LANDMARKS_CSV}")

    # Check for gestures with too few samples
    counts = df["label"].value_counts()
    print("\n── Sample counts ──────────────────────────────────────")
    for label, n in counts.items():
        warn = "  ⚠ low (<50)" if n < 50 else ""
        print(f"  {label:20s}  {n:4d}{warn}")
    print()

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_str = df["label"].values

    # ── 2. Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    classes = le.classes_.tolist()
    print(f"[INFO] Classes: {classes}")

    # ── 3. Train / test split (stratified) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── 4. Build & train pipeline ─────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
            verbose=False,
        )),
    ])

    print("[INFO] Training …")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.\n")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"Test accuracy : {acc * 100:.2f}%\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    _print_confusion_matrix(cm, classes)

    # 5-fold cross-validation on full dataset (bonus check)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean() * 100:.2f}%"
          f"  ±  {cv_scores.std() * 100:.2f}%\n")

    # ── 6. Save model ─────────────────────────────────────────────────────
    model_bundle = {"pipeline": pipeline, "label_encoder": le}
    joblib.dump(model_bundle, config.MODEL_PATH)
    print(f"[SAVED] Model → {config.MODEL_PATH}")

    # Save label map (int index → gesture string) for quick look-ups
    label_map = {int(i): cls for i, cls in enumerate(classes)}
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[SAVED] Label map → {config.LABEL_MAP_PATH}")

    print("\n✓ Done. Run  inference.py  to test live recognition.")


if __name__ == "__main__":
    main()
