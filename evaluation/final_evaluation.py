
import json
import joblib
import torch
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.lstm_model import PhishGuardLSTM

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model_artifacts"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 8


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist()
    }


def main():
    print("=" * 60)
    print("PHISHGUARD - FINAL TEST EVALUATION")
    print("=" * 60)

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label"].astype(int).tolist()

    results = {}

    # 1. Baseline
    print("\nEvaluating Logistic Regression...")

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )
    baseline = joblib.load(
        MODEL_DIR / "logistic_regression.joblib"
    )

    baseline_predictions = baseline.predict(
        vectorizer.transform(texts)
    )

    results["Baseline"] = calculate_metrics(
        y_true, baseline_predictions
    )

    # 2. LSTM
    print("Evaluating LSTM...")

    lstm_tokenizer = joblib.load(
        MODEL_DIR / "lstm_tokenizer.joblib"
    )

    lstm = PhishGuardLSTM(
        vocab_size=lstm_tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )

    lstm.load_state_dict(
        torch.load(
            MODEL_DIR / "lstm_model.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    lstm.eval()

    lstm_predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

            sequences = [
                lstm_tokenizer.encode(text, max_length=128)
                for text in batch_texts
            ]

            inputs = torch.tensor(sequences, dtype=torch.long)
            probabilities = torch.sigmoid(lstm(inputs))

            predictions = (probabilities >= 0.5).int()
            lstm_predictions.extend(predictions.tolist())

    results["LSTM"] = calculate_metrics(
        y_true, lstm_predictions
    )

    # 3. DistilBERT
    print("Evaluating DistilBERT...")

    distilbert_dir = MODEL_DIR / "distilbert_phishguard"

    db_tokenizer = AutoTokenizer.from_pretrained(
        distilbert_dir
    )

    distilbert = AutoModelForSequenceClassification.from_pretrained(
        distilbert_dir
    ).to(DEVICE)

    distilbert.eval()

    db_predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

            encoded = db_tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )

            outputs = distilbert(**encoded)
            predictions = outputs.logits.argmax(dim=1)

            db_predictions.extend(predictions.tolist())

    results["DistilBERT"] = calculate_metrics(
        y_true, db_predictions
    )

    # Print reports
    for name, predictions in [
        ("Baseline", baseline_predictions),
        ("LSTM", lstm_predictions),
        ("DistilBERT", db_predictions)
    ]:
        print(f"\n{name.upper()} CLASSIFICATION REPORT")
        print("-" * 50)
        print(classification_report(
            y_true,
            predictions,
            target_names=["Legitimate", "Phishing"],
            zero_division=0
        ))

    # Save complete results
    with (REPORT_DIR / "final_test_results.json").open("w") as file:
        json.dump(results, file, indent=4)

    comparison = pd.DataFrame(results).T
    comparison = comparison.drop(columns=["confusion_matrix"])
    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv"
    )

    print("\nFINAL MODEL COMPARISON")
    print("=" * 60)
    print(comparison.round(4).to_string())

    print("\nCONFUSION MATRICES")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(metrics["confusion_matrix"])

    print("\nResults saved in:", REPORT_DIR)
    print("PHISHGUARD FINAL EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
