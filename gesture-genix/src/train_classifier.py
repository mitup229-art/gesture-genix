"""
Phase 3: Train and validate the gesture classifier.

Reads data/raw_landmarks/landmarks.csv, trains a Random Forest,
prints accuracy + confusion matrix, and saves the trained model.

Run: python src/train_classifier.py
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from config import LANDMARKS_CSV, MODEL_PATH, MODELS_DIR
import os


def main():
    df = pd.read_csv(LANDMARKS_CSV)
    if df.empty:
        raise ValueError("No data found. Run data_collector.py first.")

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Loaded {len(df)} samples across {y.nunique()} gesture classes.")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print("Labels order:", list(clf.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
