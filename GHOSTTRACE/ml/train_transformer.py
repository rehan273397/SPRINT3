
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/security_evidence.csv"
SPLIT_PATH = ROOT / "reports/dataset_splits.csv"
MODEL_DIR = ROOT / "models/distilbert_security_classifier"
REPORT_DIR = ROOT / "reports"

MODEL_NAME = "distilbert/distilbert-base-uncased"
SEED = 42
MAX_LENGTH = 64
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5
PATIENCE = 1


class EvidenceDataset(Dataset):
    def __init__(self, frame, label_to_id):
        self.texts = frame["evidence_text"].astype(str).tolist()
        self.labels = [
            label_to_id[label] for label in frame["label"].astype(str)
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def collate_batch(batch, tokenizer):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return encoded


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)

            total_loss += outputs.loss.item() * len(labels)
            total += len(labels)
            predicted.extend(outputs.logits.argmax(1).cpu().tolist())
            actual.extend(labels.cpu().tolist())

    return (
        total_loss / max(total, 1),
        f1_score(actual, predicted, average="macro", zero_division=0),
        actual,
        predicted,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not DATA_PATH.exists() or not SPLIT_PATH.exists():
        raise SystemExit("Processed dataset or saved split file is missing.")

    data = pd.read_csv(DATA_PATH).fillna("")
    splits = pd.read_csv(SPLIT_PATH).fillna("")

    for column in ("source_id", "label"):
        data[column] = data[column].astype(str)
        splits[column] = splits[column].astype(str)

    if splits.duplicated(["source_id", "label"]).any():
        raise SystemExit("Duplicate keys in saved split assignments.")

    merged = data.merge(
        splits[["source_id", "label", "split"]],
        on=["source_id", "label"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise SystemExit("Some records have no saved split assignment.")

    merged = merged.drop(columns="_merge")

    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "validation"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise SystemExit("A dataset split is empty.")

    labels = sorted(merged["label"].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        f"Records: train={len(train_df)}, validation={len(val_df)}, "
        f"test={len(test_df)}"
    )

    print("Loading tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id=label_to_id,
        use_safetensors=True,
    ).to(device)

    train_ds = EvidenceDataset(train_df, label_to_id)
    val_ds = EvidenceDataset(val_df, label_to_id)
    test_ds = EvidenceDataset(test_df, label_to_id)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda batch: collate_batch(batch, tokenizer),
        )

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, seen = 0.0, 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = len(batch["labels"])
            loss_sum += outputs.loss.item() * count
            seen += count

        train_loss = loss_sum / max(seen, 1)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | train loss={train_loss:.4f} | "
            f"validation loss={val_loss:.4f} | val macro-F1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(MODEL_DIR)
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print("Early stopping.")
            break

    # Evaluate the checkpoint selected using validation macro-F1.
    del model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, use_safetensors=True
    ).to(device)

    test_loss, _, y_true, y_pred = evaluate(model, test_loader, device)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(labels)))
    )
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "actual"
    confusion_df.to_csv(REPORT_DIR / "transformer_confusion_matrix.csv")

    errors = []
    test_rows = test_df.reset_index(drop=True)
    for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
        if actual != predicted:
            row = test_rows.iloc[i]
            errors.append({
                "source_id": row["source_id"],
                "actual_label": labels[actual],
                "predicted_label": labels[predicted],
                "evidence_text": row["evidence_text"],
            })

    pd.DataFrame(errors, columns=[
        "source_id", "actual_label", "predicted_label", "evidence_text"
    ]).to_csv(REPORT_DIR / "transformer_errors.csv", index=False)

    metrics = {
        "model": MODEL_NAME,
        "task": "OWASP category classification",
        "train_records": len(train_df),
        "validation_records": len(val_df),
        "test_records": len(test_df),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "training_seconds": round(time.perf_counter() - start, 2),
        "device": str(device),
        "random_state": SEED,
        "train_history": history,
        "warning": (
            "Synthetic/unverified dataset records; results do not establish "
            "real-world vulnerability detection performance."
        ),
    }

    (REPORT_DIR / "transformer_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    (REPORT_DIR / "transformer_classification_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\n=== DISTILBERT TEST RESULTS ===")
    print(f"Accuracy:        {metrics['test_accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Best epoch:      {best_epoch}")
    print(f"Misclassified:   {len(errors)} / {len(test_df)}")

    print("\nPer-class F1:")
    for i, label in enumerate(labels):
        print(f"{label}: F1={f1[i]:.3f}, support={support[i]}")

    print("\nModel:", MODEL_DIR)
    print("Metrics:", REPORT_DIR / "transformer_metrics.json")


if __name__ == "__main__":
    main()
