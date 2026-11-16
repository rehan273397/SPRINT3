
import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model_artifacts"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename):
    path = DATA_DIR / filename

    df = pd.read_csv(path)

    if df["text"].isna().any():
        raise ValueError(f"Missing text in {filename}")

    if df["label"].isna().any():
        raise ValueError(f"Missing labels in {filename}")

    df["label"] = df["label"].astype(int)

    return df


def main():

    print("=" * 50)
    print("PHISHGUARD - BASELINE MODEL")
    print("=" * 50)

    print("\nLoading datasets...")

    train = load_data("train.csv")
    validation = load_data("validation.csv")
    test = load_data("test.csv")

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Testing records:", len(test))

    print("\nExtracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.98
    )

    X_train = vectorizer.fit_transform(train["text"])
    X_validation = vectorizer.transform(validation["text"])
    X_test = vectorizer.transform(test["text"])

    y_train = train["label"]
    y_validation = validation["label"]
    y_test = test["label"]

    print("Training feature shape:", X_train.shape)

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nVALIDATION RESULTS")
    print("-" * 50)

    validation_predictions = model.predict(X_validation)

    print(classification_report(
        y_validation,
        validation_predictions,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    print("\nFINAL TEST RESULTS")
    print("-" * 50)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    print(json.dumps(metrics, indent=4))

    print("\nDetailed classification report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    print("\nSaving trained models...")

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / "logistic_regression.joblib"
    )

    with (REPORT_DIR / "baseline_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=4)

    print("\nPHISHGUARD BASELINE TRAINING COMPLETED!")
    print("Models saved in:", MODEL_DIR)
    print("Metrics saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
