# GHOSTTRACE
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
