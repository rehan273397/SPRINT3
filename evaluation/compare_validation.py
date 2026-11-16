
import joblib
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import classification_report

from models.lstm_model import PhishGuardLSTM

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model_artifacts"

df = pd.read_csv(DATA_DIR / "validation.csv")
texts = df["text"].astype(str).tolist()
y_true = df["label"].astype(int)

# Baseline predictions
vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)
baseline = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)

baseline_predictions = baseline.predict(
    vectorizer.transform(texts)
)

# LSTM predictions
tokenizer = joblib.load(
    MODEL_DIR / "lstm_tokenizer.joblib"
)

model = PhishGuardLSTM(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    hidden_dim=64,
    num_layers=1,
    dropout=0.3
)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "lstm_model.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

sequences = [
    tokenizer.encode(text, max_length=128)
    for text in texts
]

X = torch.tensor(sequences, dtype=torch.long)

with torch.no_grad():
    probabilities = torch.sigmoid(model(X))
    lstm_predictions = (
        probabilities >= 0.5
    ).int().numpy()

print("\nBASELINE - VALIDATION RESULTS")
print("=" * 50)
print(classification_report(
    y_true,
    baseline_predictions,
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

print("\nLSTM - VALIDATION RESULTS")
print("=" * 50)
print(classification_report(
    y_true,
    lstm_predictions,
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))

