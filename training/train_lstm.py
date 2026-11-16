
import json
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from pathlib import Path

from core.tokenizer import TextTokenizer
from models.lstm_model import PhishGuardLSTM


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model_artifacts"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# CPU-friendly training settings
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001
PATIENCE = 2


def load_data(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"Missing values in {filename}")

    df["label"] = df["label"].astype(int)

    return df


def encode_dataset(df, tokenizer):
    sequences = [
        tokenizer.encode(text, MAX_LENGTH)
        for text in df["text"]
    ]

    X = torch.tensor(sequences, dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.float32)

    return TensorDataset(X, y)


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).int()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(
            all_labels, all_predictions, zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_predictions, zero_division=0
        ),
        "f1": f1_score(
            all_labels, all_predictions, zero_division=0
        )
    }


def main():
    device = torch.device("cpu")

    print("=" * 50)
    print("PHISHGUARD - LSTM TRAINING")
    print("=" * 50)

    print("\nLoading datasets...")

    train_df = load_data("train.csv")
    validation_df = load_data("validation.csv")

    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))

    # Build vocabulary using training data only
    print("\nBuilding vocabulary...")

    tokenizer = TextTokenizer(max_vocab_size=30000)
    tokenizer.fit(train_df["text"])

    print("Vocabulary size:", tokenizer.vocab_size)

    # Convert messages into numerical sequences
    print("\nEncoding messages...")

    train_dataset = encode_dataset(train_df, tokenizer)
    validation_dataset = encode_dataset(validation_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    model = PhishGuardLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    ).to(device)

    # Account for class imbalance
    labels = train_df["label"].values
    negatives = (labels == 0).sum()
    positives = (labels == 1).sum()

    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0
    epochs_without_improvement = 0

    print("\nStarting training on:", device)
    print("-" * 50)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            # Prevent excessively large gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        validation_metrics = evaluate(
            model, validation_loader, device
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.4f}")
        print(
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )
        print(
            f"Validation precision: "
            f"{validation_metrics['precision']:.4f}"
        )
        print(
            f"Validation recall: "
            f"{validation_metrics['recall']:.4f}"
        )
        print(
            f"Validation F1: "
            f"{validation_metrics['f1']:.4f}"
        )

        # Save the model with the best validation F1
        current_f1 = validation_metrics["f1"]

        if current_f1 > best_f1:

            best_f1 = current_f1
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "lstm_model.pt"
            )

            joblib.dump(
                tokenizer,
                MODEL_DIR / "lstm_tokenizer.joblib"
            )

            with (REPORT_DIR / "lstm_metrics.json").open("w") as file:
                json.dump(validation_metrics, file, indent=4)

            print("Best model saved!")

        else:
            epochs_without_improvement += 1
            print("No validation improvement.")

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 50)
    print("LSTM TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best validation F1: {best_f1:.4f}")
    print("Model:", MODEL_DIR / "lstm_model.pt")
    print("Tokenizer:", MODEL_DIR / "lstm_tokenizer.joblib")
    print("Metrics:", REPORT_DIR / "lstm_metrics.json")


if __name__ == "__main__":
    main()
