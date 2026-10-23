
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/security_evidence.csv"
SPLIT_PATH = ROOT / "reports/dataset_splits.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 8
RANDOM_STATE = 42
CANDIDATE_C_VALUES = [0.1, 1.0, 3.0, 10.0]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    data = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    required_data = {"evidence_text", "label", "source_id"}
    required_splits = {"source_id", "label", "split"}

    if not required_data.issubset(data.columns):
        raise ValueError(
            f"Dataset is missing columns: "
            f"{required_data - set(data.columns)}"
        )

    if not required_splits.issubset(splits.columns):
        raise ValueError(
            f"Split file is missing columns: "
            f"{required_splits - set(splits.columns)}"
        )

    if data["source_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate source_id values.")

    if splits.duplicated(["source_id", "label"]).any():
        raise ValueError("Split file contains duplicate source_id/label pairs.")

    data["source_id"] = data["source_id"].astype(str)
    splits["source_id"] = splits["source_id"].astype(str)

    merged = data.merge(
        splits[["source_id", "label", "split"]],
        on=["source_id", "label"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        examples = merged.loc[unmatched, "source_id"].head().tolist()
        raise ValueError(
            f"{int(unmatched.sum())} dataset rows have no saved split. "
            f"Examples: {examples}"
        )

    merged = merged.drop(columns="_merge")

    allowed_splits = {"train", "validation", "test"}
    actual_splits = set(merged["split"].dropna().unique())
    if not actual_splits.issubset(allowed_splits):
        raise ValueError(f"Unexpected split names: {actual_splits}")

    if actual_splits != allowed_splits:
        raise ValueError(f"Expected train, validation and test; got {actual_splits}")

    merged["evidence_text"] = (
        merged["evidence_text"].fillna("").astype(str).str.strip()
    )
    merged["label"] = merged["label"].astype(str)

    if (merged["evidence_text"] == "").any():
        raise ValueError("Empty evidence_text rows found; clean them before training.")

    print("Dataset loaded:", len(merged), "rows")
    print(merged["split"].value_counts().to_string())

    return merged


def load_transformer():
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    print("The first load may download model files if they are not cached.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Frozen feature extraction: no gradients or weight updates.
    model.eval()
    model.to("cpu")

    return tokenizer, model


def extract_embeddings(texts, tokenizer, model):
    embeddings = []
    started = time.time()

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()

            # Mean-pool token embeddings, ignoring padding tokens.
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts

        embeddings.append(pooled.cpu().numpy().astype(np.float32))

        completed = min(start + BATCH_SIZE, len(texts))
        if completed % 128 == 0 or completed == len(texts):
            print(f"Embedded {completed}/{len(texts)} texts")

    result = np.concatenate(embeddings, axis=0)
    print(f"Embedding extraction took {time.time() - started:.1f} seconds")
    print("Embedding matrix shape:", result.shape)

    return result


def main():
    total_started = time.time()
    data = load_data()

    # Preserve the dataset's row order when creating the embedding matrix.
    tokenizer, transformer = load_transformer()
    all_embeddings = extract_embeddings(
        data["evidence_text"].tolist(),
        tokenizer,
        transformer,
    )

    train_mask = data["split"].eq("train").to_numpy()
    val_mask = data["split"].eq("validation").to_numpy()
    test_mask = data["split"].eq("test").to_numpy()

    X_train = all_embeddings[train_mask]
    X_val = all_embeddings[val_mask]
    X_test = all_embeddings[test_mask]

    y_train = data.loc[train_mask, "label"].to_numpy()
    y_val = data.loc[val_mask, "label"].to_numpy()
    y_test = data.loc[test_mask, "label"].to_numpy()

    classes = sorted(data["label"].unique())
    if set(y_train) != set(classes):
        raise ValueError("Some labels are missing from the training split.")

    # Select regularization strength using validation data only.
    best_c = None
    best_val_f1 = -1.0

    print("\nSelecting Logistic Regression C using validation macro-F1:")
    for c_value in CANDIDATE_C_VALUES:
        classifier = LogisticRegression(
            C=c_value,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        classifier.fit(X_train, y_train)
        val_predictions = classifier.predict(X_val)

        _, _, val_macro_f1, _ = precision_recall_fscore_support(
            y_val,
            val_predictions,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        print(f"C={c_value:<5} validation macro-F1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = float(val_macro_f1)
            best_c = c_value

    # Refit on the training split only for comparison with the existing
    # baseline/LSTM setup. Validation is used for model selection, not fitting.
    final_classifier = LogisticRegression(
        C=best_c,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    fit_started = time.time()
    final_classifier.fit(X_train, y_train)
    classifier_fit_seconds = time.time() - fit_started

    test_started = time.time()
    predictions = final_classifier.predict(X_test)
    test_prediction_seconds = time.time() - test_started

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment": "Frozen DistilBERT embeddings + Logistic Regression",
        "model_name": MODEL_NAME,
        "training_method": "Frozen pretrained feature extractor; no fine-tuning",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "selected_logistic_regression_C": best_c,
        "validation_macro_f1_at_selected_C": best_val_f1,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "embedding_dimensions": int(all_embeddings.shape[1]),
        "classifier_fit_seconds": float(classifier_fit_seconds),
        "test_prediction_seconds": float(test_prediction_seconds),
        "total_seconds": float(time.time() - total_started),
        "test_samples": int(len(y_test)),
        "note": (
            "Experimental classification of dataset labels, not proof that "
            "a real website is vulnerable. Dataset examples are not "
            "independently verified."
        ),
    }

    with open(REPORT_DIR / "frozen_transformer_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    with open(
        REPORT_DIR / "frozen_transformer_classification_report.json", "w"
    ) as file:
        json.dump(report, file, indent=2)

    matrix = confusion_matrix(y_test, predictions, labels=classes)
    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    ).to_csv(REPORT_DIR / "frozen_transformer_confusion_matrix.csv")

    test_rows = data.loc[test_mask, ["source_id", "evidence_text", "label"]].copy()
    test_rows["predicted_label"] = predictions
    test_rows["correct"] = test_rows["label"] == test_rows["predicted_label"]
    test_rows.loc[~test_rows["correct"]].to_csv(
        REPORT_DIR / "frozen_transformer_errors.csv",
        index=False,
    )

    joblib.dump(
        {
            "classifier": final_classifier,
            "classes": classes,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "training_method": "frozen embeddings",
        },
        MODEL_DIR / "frozen_transformer_logreg.joblib",
    )

    print("\n=== FROZEN TRANSFORMER TEST RESULTS ===")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall:    {recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Selected C:      {best_c}")
    print(f"Classifier fit:  {classifier_fit_seconds:.2f} seconds")
    print(f"Total time:      {metrics['total_seconds']:.2f} seconds")
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions, labels=classes, zero_division=0
    ))
    print("\nSaved metrics, error analysis and classifier.")
    print("Next: compare results against the baseline and LSTM.")


if __name__ == "__main__":
    main()
