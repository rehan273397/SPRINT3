
import json
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
SAMPLES_PER_CLASS = 1500

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model_artifacts" / "distilbert_phishguard"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(filename):
    df = pd.read_csv(DATA_DIR / filename)

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"Missing values in {filename}")

    df["label"] = df["label"].astype(int)
    return df


def prepare_dataset(df, tokenizer):
    encoded = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    labels = torch.tensor(
        df["label"].values,
        dtype=torch.long
    )

    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        labels
    )


def evaluate(model, loader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = outputs.logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "precision": float(precision_score(
            all_labels, all_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            all_labels, all_predictions, zero_division=0
        )),
        "f1": float(f1_score(
            all_labels, all_predictions, zero_division=0
        ))
    }


def main():
    device = torch.device("cpu")

    print("=" * 55)
    print("PHISHGUARD - DISTILBERT TRAINING")
    print("=" * 55)

    print("\nLoading datasets...")

    train_df = load_data("train.csv")
    validation_df = load_data("validation.csv")

    # Balanced, reproducible training subset
    sample_size = min(
        SAMPLES_PER_CLASS,
        int(train_df["label"].value_counts().min())
    )

    train_df = (
        train_df.groupby("label", group_keys=False)
        .sample(n=sample_size, random_state=SEED)
        .reset_index(drop=True)
    )

    print("Training records used:", len(train_df))
    print("Validation records:", len(validation_df))

    print("\nLoading pretrained tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading pretrained DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Legitimate", 1: "Phishing"},
        label2id={"Legitimate": 0, "Phishing": 1}
    )

    # Freeze all layers first
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final Transformer layer and classification head
    for parameter in model.distilbert.transformer.layer[-1].parameters():
        parameter.requires_grad = True

    for parameter in model.pre_classifier.parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model.to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")

    print("\nTokenizing messages...")

    train_dataset = prepare_dataset(train_df, tokenizer)
    validation_dataset = prepare_dataset(validation_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0

    print("\nStarting CPU training...")
    print("-" * 55)

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for step, (input_ids, attention_mask, labels) in enumerate(
            train_loader, start=1
        ):

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device)
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = total_loss / len(train_loader)

        metrics = evaluate(model, validation_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.4f}")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation precision: {metrics['precision']:.4f}")
        print(f"Validation recall: {metrics['recall']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            with (REPORT_DIR / "distilbert_metrics.json").open("w") as file:
                json.dump(metrics, file, indent=4)

            print("Best DistilBERT model saved!")

    print("\n" + "=" * 55)
    print("DISTILBERT TRAINING COMPLETED")
    print("=" * 55)
    print(f"Best validation F1: {best_f1:.4f}")
    print("Saved model:", MODEL_DIR)
    print("Saved metrics:", REPORT_DIR / "distilbert_metrics.json")


if __name__ == "__main__":
    main()

