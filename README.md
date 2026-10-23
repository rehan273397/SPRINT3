
# GHOSTTRACE 🛡️
### AI/ML-Powered Cybersecurity Evidence Classification

GHOSTTRACE is an experimental cybersecurity project that uses machine learning to classify web application security evidence into **OWASP Top 10:2021** categories. Built with Python and Streamlit, it provides an interactive interface for analyzing security-related text, processing multiple findings from CSV files, reviewing predictions, and generating reports.

## 🚀 Features

- **AI/ML Classification:** Predicts the likely OWASP category of supplied security evidence.
- **Multiple ML Models:** Experiments with TF-IDF + Logistic Regression, an LSTM neural network, and frozen DistilBERT representations.
- **Batch Analysis:** Processes multiple security findings from CSV files.
- **Prediction Review:** Displays model predictions and confidence scores to support human review.
- **Reports and Exports:** Provides downloadable analysis results and reports.
- **Model Evaluation:** Includes experiment metrics, classification reports, and confusion matrices.
- **Cybersecurity Learning:** Includes educational content about web application security and OWASP categories.

## 🧠 How It Works

1. **Dataset preparation:** Security-related text and category labels are cleaned and prepared for supervised learning.
2. **Model training:** Multiple machine-learning approaches are trained and compared.
3. **Evaluation:** Models are assessed on a held-out test set using accuracy, macro-F1, and other classification metrics.
4. **Prediction:** The Streamlit application loads the saved baseline model and classifies user-provided evidence.
5. **Review:** Users can inspect predictions, process batches, and export results for further analysis.

## 📊 Dataset

GHOSTTRACE uses the **SecureCode dataset**, published by `scthornton` on Hugging Face, specifically its `web` configuration.

Dataset: https://huggingface.co/datasets/scthornton/securecode

The original web configuration contains 1,625 examples. The project's prepared dataset contains 1,439 records used for the reported experiments.

The records contain security-related text and labels associated with OWASP Top 10:2021 categories. Dataset verification is limited, and the examples should not be treated as independently confirmed real-world vulnerabilities.

**Dataset licence:** The published dataset identifies CC BY-NC-SA 4.0. Review its attribution, non-commercial, and share-alike requirements before redistributing the data or derived materials.

## 🏆 Model Results

The best recorded model on the project's prepared test split was **TF-IDF + Logistic Regression**.

| Metric | Result |
|---|---:|
| Test accuracy | 83.80% |
| Macro-F1 | 83.98% |
| Test records | 216 |

These results measure classification performance against the dataset's labels. They do not establish real-world vulnerability-detection accuracy.

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Joblib
- Matplotlib and other visualization libraries

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/GHOSTTRACE.git
cd GHOSTTRACE
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch GHOSTTRACE

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## 📁 Project Structure

```text
GHOSTTRACE/
├── app.py
├── requirements.txt
├── README.md
├── ml/
│   ├── prepare_dataset.py
│   ├── train_baseline.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── train_frozen_transformer.py
│   └── compare_models.py
├── models/
│   ├── baseline_tfidf_logreg_clean.joblib
│   ├── lstm_security_classifier.pt
│   └── frozen_transformer_logreg.joblib
├── data/
│   └── processed/
└── reports/
    ├── dataset_splits.csv
    ├── model_comparison.csv
    └── evaluation reports
```

The exact files present may vary with the version of the project.

## ⚠️ Limitations

- GHOSTTRACE classifies supplied security evidence; it does **not** automatically scan live websites or APIs.
- Predictions are suggestions for human review, not proof that a vulnerability exists.
- Model confidence is not the probability that a real vulnerability is present.
- The dataset is relatively small and contains synthetic or insufficiently verified records.
- Performance on the prepared test set may not generalize to real-world security findings.
- The project is experimental and is not a replacement for professional security testing.

## 🔮 Future Improvements

Potential future extensions include authorized API endpoint testing, OpenAPI specification import, evidence collection, stronger validation against independently verified findings, improved rare-category classification, and integration with established security testing tools.

These are proposed enhancements, not features currently implemented.

## 🎯 Project Objective

GHOSTTRACE demonstrates an end-to-end applied machine-learning workflow for cybersecurity: dataset preparation, model training, model comparison, evaluation, and deployment through an interactive application.

Its goal is to explore how AI/ML can assist security analysts in organizing and reviewing web application security findings.

## 📜 Acknowledgements

- **Dataset:** SecureCode by `scthornton` on Hugging Face.
- **Security taxonomy:** OWASP Top 10:2021.
- **Frameworks:** The open-source Python and machine-learning ecosystem.

