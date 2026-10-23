from pathlib import Path
import os
import sys
import json
import zipfile
import subprocess
import traceback

ROOT = Path.cwd()
REPORTS = ROOT / "reports"
MODEL = ROOT / "models" / "baseline_tfidf_logreg_clean.joblib"
APP = ROOT / "app.py"
FAILURES = []
CHECKS = []

def check(name, passed, detail=""):
    CHECKS.append({
        "check": name,
        "passed": bool(passed),
        "detail": str(detail)
    })
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" +
          (f" — {detail}" if detail else ""))
    if not passed:
        FAILURES.append(name)

print("\n=== GHOSTTRACE FINAL VERIFICATION ===\n")

# 1. Essential project files
for name in ["app.py", "requirements.txt", "README.md"]:
    check(f"Required file: {name}", (ROOT / name).is_file())

check("Best model exists", MODEL.is_file())
check("Reports directory exists", REPORTS.is_dir())
check("ML scripts directory exists", (ROOT / "ml").is_dir())

# 2. Compile application and ML Python files
python_files = []
for folder in [ROOT / "ml"]:
    if folder.is_dir():
        python_files.extend(folder.rglob("*.py"))
if APP.is_file():
    python_files.append(APP)

compile_ok = True
try:
    for file in python_files:
        # Compile without writing .pyc files.
        compile(file.read_text(encoding="utf-8"), str(file), "exec")
except Exception:
    compile_ok = False
    traceback.print_exc()

check("Python syntax", compile_ok, f"{len(python_files)} files checked")

# 3. Load model and make a sample prediction
model_ok = False
model_detail = ""
if MODEL.is_file():
    try:
        import joblib
        model = joblib.load(MODEL)
        sample = [
            "The application builds a SQL query using unsanitized user input."
        ]
        prediction = model.predict(sample)
        model_ok = len(prediction) == 1
        model_detail = f"Sample classification returned: {prediction[0]}"
    except Exception as exc:
        model_detail = repr(exc)

check("Model load and prediction", model_ok, model_detail)

# 4. Verify important dataset and experiment outputs
dataset_candidates = [
    ROOT / "data" / "processed" / "security_evidence.csv",
    REPORTS / "dataset_splits.csv",
]
for file in dataset_candidates:
    check(f"Data/report: {file.relative_to(ROOT)}", file.is_file())

comparison_candidates = [
    REPORTS / "model_comparison.csv",
    REPORTS / "model_comparison.json",
]
comparison = next((p for p in comparison_candidates if p.is_file()), None)
check("Model comparison report exists", comparison is not None)

# 5. Save final verification report
REPORTS.mkdir(exist_ok=True)
report = {
    "project": "GHOSTTRACE",
    "verification": "passed" if not FAILURES else "failed",
    "checks": CHECKS,
    "failed_checks": FAILURES,
    "note": (
        "Model classification is experimental. Passing these checks does not "
        "prove real-world vulnerability detection or replace manual UI testing."
    ),
}
(REPORTS / "final_submission_check.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

with (REPORTS / "final_submission_check.txt").open("w", encoding="utf-8") as f:
    f.write("GHOSTTRACE FINAL SUBMISSION CHECK\n")
    f.write("=" * 36 + "\n\n")
    for item in CHECKS:
        f.write(
            f"[{'PASS' if item['passed'] else 'FAIL'}] "
            f"{item['check']}: {item['detail']}\n"
        )
    f.write("\nOverall: " + report["verification"].upper() + "\n")
    f.write(report["note"] + "\n")

if FAILURES:
    print("\nSome checks failed. Fix those before submission.")
    print("Details: reports/final_submission_check.txt")
    sys.exit(1)

# 6. Build clean submission ZIP, excluding environments and caches
archive = ROOT.parent / "GHOSTTRACE_SUBMISSION.zip"
excluded_dirs = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", ".cache"
}
excluded_names = {
    ".DS_Store", "secrets.toml"
}

try:
    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for current, dirs, files in os.walk(ROOT):
            current_path = Path(current)
            dirs[:] = [
                d for d in dirs
                if d not in excluded_dirs
            ]

            for filename in files:
                if filename in excluded_names:
                    continue
                path = current_path / filename
                if path == archive:
                    continue
                relative = path.relative_to(ROOT)
                # Exclude common secret/config files.
                if any(part in excluded_dirs for part in relative.parts):
                    continue
                if filename.endswith((".pyc", ".pyo")):
                    continue
                zf.write(path, Path("GHOSTTRACE") / relative)

    check("Submission ZIP created", archive.is_file(), str(archive))
except Exception as exc:
    check("Submission ZIP created", False, repr(exc))

# Update report after packaging result
report["checks"] = CHECKS
report["failed_checks"] = FAILURES
report["verification"] = "passed" if not FAILURES else "failed"
(REPORTS / "final_submission_check.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print("\n=== RESULT ===")
if FAILURES:
    print("Verification failed. Check the messages above.")
    sys.exit(1)

print("Project checks passed.")
print(f"Submission archive: {archive}")
print("Do not include your .venv folder in the submission.")
print("Review the dataset licence and attribution before publishing or redistributing.")
