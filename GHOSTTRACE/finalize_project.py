from pathlib import Path
from datetime import datetime
import json
import sys
import traceback
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"
REPORTS.mkdir(exist_ok=True)

checks = []

def check(name, fn):
    try:
        detail = fn()
        checks.append({
            "check": name,
            "status": "PASS",
            "detail": str(detail),
        })
        print(f"[PASS] {name}: {detail}")
    except Exception as exc:
        checks.append({
            "check": name,
            "status": "FAIL",
            "detail": str(exc),
        })
        print(f"[FAIL] {name}: {exc}")

def check_app_syntax():
    import py_compile
    app = ROOT / "app.py"
    if not app.exists():
        raise FileNotFoundError("app.py is missing")
    py_compile.compile(str(app), doraise=True)
    return "app.py compiles successfully"

def check_model():
    import joblib
    model_path = MODELS / "baseline_tfidf_logreg_clean.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    sample = (
        "A user changes an account identifier in a request "
        "and accesses another user's private records."
    )
    prediction = model.predict([sample])
    if len(prediction) != 1:
        raise AssertionError("Model did not return exactly one prediction")
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([sample]).max())
        if not 0 <= confidence <= 1:
            raise AssertionError("Invalid confidence value")
    return f"Prediction={prediction[0]}; confidence={confidence}"

def check_model_experiments():
    expected = [
        "baseline_clean_metrics.json",
        "lstm_metrics.json",
        "frozen_transformer_metrics.json",
        "model_comparison.csv",
    ]
    missing = [f for f in expected if not (REPORTS / f).exists()]
    if missing:
        raise FileNotFoundError("Missing experiment reports: " + ", ".join(missing))
    with (REPORTS / "model_comparison.csv").open(encoding="utf-8") as f:
        comparison = pd.read_csv(f)
    if comparison.empty:
        raise AssertionError("Model comparison report is empty")
    return f"{len(comparison)} model comparison row(s) found"

def check_dataset():
    path = DATA / "security_evidence.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"evidence_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing dataset columns: {sorted(missing)}")
    if df.empty:
        raise AssertionError("Dataset is empty")
    return f"{len(df)} records; {df['label'].nunique()} labels"

def check_split_report():
    path = REPORTS / "dataset_splits.csv"
    if not path.exists():
        raise FileNotFoundError("reports/dataset_splits.csv is missing")
    df = pd.read_csv(path)
    required = {"row_index", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing split columns: {sorted(missing)}")
    counts = df["split"].value_counts().to_dict()
    return f"Split counts: {counts}"

def check_csv_and_report_helpers():
    # Test common CSV shapes and report generation without changing project data.
    import io
    sample = pd.DataFrame({
        "evidence_text": [
            "An attacker accesses another user's private account information."
        ],
        "source_id": ["test-001"],
    })
    csv_bytes = sample.to_csv(index=False).encode("utf-8-sig")
    loaded = pd.read_csv(io.BytesIO(csv_bytes))
    if "evidence_text" not in loaded.columns:
        raise AssertionError("CSV round-trip lost evidence_text")
    if loaded.empty or not loaded["evidence_text"].iloc[0]:
        raise AssertionError("CSV round-trip lost evidence")
    return "CSV export/import round-trip passed"

def check_requirements():
    path = ROOT / "requirements.txt"
    if not path.exists():
        raise FileNotFoundError("requirements.txt is missing")
    required = {"streamlit", "pandas", "scikit-learn", "joblib"}
    present = {
        line.strip().split("==")[0].split(">=")[0].lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = required - present
    if missing:
        raise AssertionError(f"Add required packages to requirements.txt: {sorted(missing)}")
    return "Core application dependencies are listed"

print("=" * 64)
print("GHOSTTRACE SUBMISSION PREFLIGHT")
print("=" * 64)

check("Application syntax", check_app_syntax)
check("Trained baseline model loads and predicts", check_model)
check("Model experiment reports", check_model_experiments)
check("Processed dataset", check_dataset)
check("Dataset split report", check_split_report)
check("CSV round-trip", check_csv_and_report_helpers)
check("Requirements file", check_requirements)

passed = sum(r["status"] == "PASS" for r in checks)
failed = sum(r["status"] == "FAIL" for r in checks)

result = {
    "project": "GHOSTTRACE",
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "passed": passed,
    "failed": failed,
    "checks": checks,
}

(REPORTS / "submission_preflight.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)

lines = [
    "GHOSTTRACE SUBMISSION PREFLIGHT",
    "=" * 40,
    f"Generated: {result['generated_at']}",
    f"Passed: {passed}",
    f"Failed: {failed}",
    "",
]
for item in checks:
    lines.append(f"[{item['status']}] {item['check']}")
    lines.append(f"    {item['detail']}")
    lines.append("")

lines += [
    "INTERPRETATION",
    "This report checks project files, basic model inference, and CSV handling.",
    "It does not prove real-world vulnerability detection performance.",
    "Manually verify the Streamlit interface and generated downloads before submission.",
    "The current app classifies supplied evidence; it is not a live website scanner.",
]

(REPORTS / "submission_preflight.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

readme = """# GHOSTTRACE
## AI/ML Security Evidence Classification

GHOSTTRACE is an experimental Python and Streamlit application that classifies
supplied security-related text into likely OWASP Top 10:2021 categories.

### Features
- Single finding classification
- CSV batch classification
- Plain-English category explanations
- Review status and reviewer notes
- Downloadable CSV and text reports
- Model experiments and evaluation reports

### Requirements
- Python and the packages listed in `requirements.txt`
- The trained model file `models/baseline_tfidf_logreg_clean.joblib`

### Setup on Kali Linux

```bash
cd ~/Desktop/GHOSTTRACE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile app.py
streamlit run app.py
```

If `.venv` already exists, activate it instead of recreating it.

### Model methodology
The primary application model uses TF-IDF text features and Logistic Regression.
The project also contains LSTM and frozen DistilBERT experiments. Refer to the
JSON/CSV reports under `reports/` for the recorded evaluation results.

The best previously recorded baseline test macro-F1 is approximately 83.98%.
This is a result on the project's experimental dataset, not a guarantee of
real-world vulnerability detection accuracy.

### Limitations
- The app classifies supplied text; it does not scan live websites or networks.
- Predictions are not proof that a vulnerability exists.
- The dataset includes synthetic or unreviewed records; independent validation
  against suitable real-world evidence is needed.
- Human review is required before treating any result as confirmed.
- Do not upload secrets or data you are not authorized to process.

### Dataset attribution and license
The project used the `scthornton/securecode` dataset from Hugging Face.
Review the dataset's original license and attribution requirements before
redistributing the dataset or derived material, or using it commercially.
The reported license is CC BY-NC-SA 4.0; verify this against the source.

### Final verification
Run `python finalize_project.py` to check important project files, baseline
inference, experiment reports, dataset structure, and CSV round-trip behavior.
See `reports/submission_preflight.txt` for the latest check results.
"""

(ROOT / "README.md").write_text(readme, encoding="utf-8")

print()
print(f"Checks passed: {passed}")
print(f"Checks failed: {failed}")
print(f"Machine-readable report: {REPORTS / 'submission_preflight.json'}")
print(f"Text report: {REPORTS / 'submission_preflight.txt'}")
print(f"README created: {ROOT / 'README.md'}")

if failed:
    print("\nSome checks failed. Read the report and fix those issues before submission.")
    sys.exit(1)

print("\nAll preflight checks passed. Manually test the UI before submitting.")
