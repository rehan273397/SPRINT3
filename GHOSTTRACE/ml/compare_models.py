
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

EXPERIMENTS = {
    "TF-IDF + Logistic Regression": "baseline_clean_metrics.json",
    "Bidirectional LSTM": "lstm_metrics.json",
    "Frozen DistilBERT + Logistic Regression": "frozen_transformer_metrics.json",
}

rows = []

for model_name, filename in EXPERIMENTS.items():
    path = REPORTS / filename
    if not path.exists():
        print(f"Skipping missing report: {path}")
        continue

    with path.open() as file:
        metrics = json.load(file)

    rows.append({
        "model": model_name,
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "test_samples": metrics.get("test_samples"),
        "training_seconds": metrics.get(
            "training_seconds",
            metrics.get("classifier_fit_seconds")
        ),
        "total_seconds": metrics.get("total_seconds"),
    })

if not rows:
    raise SystemExit("No experiment metrics were found.")

comparison = pd.DataFrame(rows).sort_values(
    "macro_f1", ascending=False
)

comparison.to_csv(REPORTS / "model_comparison.csv", index=False)

with (REPORTS / "model_comparison.json").open("w") as file:
    json.dump(comparison.to_dict(orient="records"), file, indent=2)

print("\n=== MODEL COMPARISON ===")
print(comparison.to_string(index=False))
print(f"\nSaved: {REPORTS / 'model_comparison.csv'}")
print(f"Saved: {REPORTS / 'model_comparison.json'}")
