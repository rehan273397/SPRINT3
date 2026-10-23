from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/security_evidence.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main():
    if not DATA.exists():
        raise SystemExit("Run: python -m ml.prepare_dataset")

    df = pd.read_csv(DATA).dropna(
        subset=["evidence_text", "label", "group_id"]
    )
    df["evidence_text"] = df["evidence_text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["group_id"] = df["group_id"].astype(str).str.strip()
    df = df[df["evidence_text"] != ""].copy()

    counts = df["label"].value_counts()
    if len(counts) < 2 or (counts < 10).any():
        raise SystemExit(
            "Each label needs at least 10 records for this evaluation.\n"
            + counts.to_string()
        )

    # First reserve 30% for validation/test.
    # Group-aware splitting prevents a source group appearing on both sides.
    if df["group_id"].duplicated().any():
        first = GroupShuffleSplit(
            n_splits=1, test_size=0.30, random_state=42
        )
        train_idx, rest_idx = next(
            first.split(df, df["label"], groups=df["group_id"])
        )
        train_df = df.iloc[train_idx].copy()
        rest_df = df.iloc[rest_idx].copy()

        second = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=43
        )
        val_idx, test_idx = next(
            second.split(
                rest_df,
                rest_df["label"],
                groups=rest_df["group_id"],
            )
        )
        val_df = rest_df.iloc[val_idx].copy()
        test_df = rest_df.iloc[test_idx].copy()
        split_method = "GroupShuffleSplit"
    else:
        train_df, rest_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df["label"],
        )
        val_df, test_df = train_test_split(
            rest_df,
            test_size=0.50,
            random_state=42,
            stratify=rest_df["label"],
        )
        split_method = "Stratified random split; source groups unique"

    if set(train_df["label"]) != set(df["label"]):
        raise SystemExit(
            "Training split lost one or more classes. "
            "Review group sizes and adjust the split."
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    start = time.perf_counter()
    model.fit(train_df["evidence_text"], train_df["label"])
    training_seconds = time.perf_counter() - start

    predictions = model.predict(test_df["evidence_text"])
    labels = sorted(df["label"].unique())

    results = {
        "dataset_records": int(len(df)),
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df["label"], predictions)),
        "macro_precision": float(precision_score(
            test_df["label"], predictions, labels=labels,
            average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            test_df["label"], predictions, labels=labels,
            average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            test_df["label"], predictions, labels=labels,
            average="macro", zero_division=0
        )),
        "training_seconds": round(training_seconds, 3),
        "random_state": 42,
        "split_method": split_method,
        "labels": labels,
        "synthetic_records_present": True,
        "interpretation": (
            "Experimental category classification on synthetic/unverified "
            "data; not validated real-world vulnerability detection."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "baseline_tfidf_logreg_clean.joblib")
    with open(REPORT_DIR / "baseline_clean_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    report = classification_report(
        test_df["label"], predictions, labels=labels,
        output_dict=True, zero_division=0
    )
    with open(REPORT_DIR / "baseline_clean_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    matrix = confusion_matrix(
        test_df["label"], predictions, labels=labels
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        REPORT_DIR / "baseline_clean_confusion_matrix.csv"
    )

    errors = test_df[["source_id", "evidence_text", "label"]].copy()
    errors["predicted_label"] = predictions
    errors = errors[errors["label"] != errors["predicted_label"]]
    errors.to_csv(REPORT_DIR / "baseline_clean_errors.csv", index=False)

    split_rows = []
    for name, subset in [
        ("train", train_df), ("validation", val_df), ("test", test_df)
    ]:
        for idx, row in subset.iterrows():
            split_rows.append({
                "row_index": int(idx),
                "source_id": row["source_id"],
                "group_id": row["group_id"],
                "label": row["label"],
                "split": name,
            })

    pd.DataFrame(split_rows).to_csv(
        REPORT_DIR / "dataset_splits.csv", index=False
    )

    print("\nGHOSTTRACE CLEAN BASELINE")
    print("=" * 32)
    print(json.dumps(results, indent=2))
    print("\nPer-class evaluation:")
    print(classification_report(
        test_df["label"], predictions, labels=labels, zero_division=0
    ))
    print(f"\nReports saved under {REPORT_DIR}")


if __name__ == "__main__":
    main()
