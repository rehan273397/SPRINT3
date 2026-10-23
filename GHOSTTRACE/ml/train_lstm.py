from pathlib import Path
from collections import Counter
import copy
import json
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
)
from torch.utils.data import Dataset, DataLoader
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
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "lstm_security_classifier.pt"
METRICS_PATH = REPORT_DIR / "lstm_metrics.json"
CLASS_REPORT_PATH = REPORT_DIR / "lstm_classification_report.json"
CONFUSION_PATH = REPORT_DIR / "lstm_confusion_matrix.csv"
ERRORS_PATH = REPORT_DIR / "lstm_errors.csv"

SEED = 42
MAX_VOCAB = 20000
MAX_LENGTH = 256
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 0.001

PAD_ID = 0
UNK_ID = 1

TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*|[^\w\s]"
)


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def load_data():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    if not SPLIT_PATH.exists():
        raise SystemExit(
            f"Split file not found: {SPLIT_PATH}. "
            "Run the baseline training script first."
        )

    data = pd.read_csv(DATA_PATH).fillna("")
    splits = pd.read_csv(SPLIT_PATH).fillna("")

    required_data = {"evidence_text", "label", "source_id"}
    required_splits = {"source_id", "label", "split"}

    if not required_data.issubset(data.columns):
        raise SystemExit(
            f"Dataset missing columns: "
            f"{sorted(required_data - set(data.columns))}"
        )

    if not required_splits.issubset(splits.columns):
        raise SystemExit(
            f"Split file missing columns: "
            f"{sorted(required_splits - set(splits.columns))}"
        )

    data["source_id"] = data["source_id"].astype(str)
    splits["source_id"] = splits["source_id"].astype(str)
    data["label"] = data["label"].astype(str)
    splits["label"] = splits["label"].astype(str)

    if splits.duplicated(["source_id", "label"]).any():
        raise SystemExit(
            "Duplicate source_id/label keys in the split file. "
            "Refusing to merge ambiguous split assignments."
        )

    if data.duplicated(["source_id", "label"]).any():
        raise SystemExit(
            "Duplicate source_id/label keys in the dataset. "
            "Inspect the dataset before training."
        )

    merged = data.merge(
        splits[["source_id", "label", "split"]],
        on=["source_id", "label"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())

    if unmatched:
        raise SystemExit(
            f"{unmatched} dataset records have no saved split assignment. "
            "Recreate the baseline split file before continuing."
        )

    merged = merged.drop(columns=["_merge"])

    allowed_splits = {"train", "validation", "test"}
    actual_splits = set(merged["split"].unique())

    if not actual_splits.issubset(allowed_splits):
        raise SystemExit(
            f"Unexpected split names: {sorted(actual_splits)}"
        )

    for split_name in ("train", "validation", "test"):
        count = int((merged["split"] == split_name).sum())
        if count == 0:
            raise SystemExit(f"The {split_name} split is empty.")

    if merged["evidence_text"].str.strip().eq("").any():
        raise SystemExit(
            "Empty evidence text found. Run dataset preparation again."
        )

    print(f"Dataset records: {len(merged)}")
    print("Saved split counts:")
    print(merged["split"].value_counts().to_string())

    return merged


def build_vocabulary(texts):
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}

    for token, _ in counts.most_common(MAX_VOCAB - len(vocab)):
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)[:MAX_LENGTH]

    ids = [vocab.get(token, UNK_ID) for token in tokens]

    if not ids:
        ids = [UNK_ID]

    return torch.tensor(ids, dtype=torch.long)


class EvidenceDataset(Dataset):
    def __init__(self, frame, vocab, label_to_id):
        self.texts = frame["evidence_text"].astype(str).tolist()
        self.labels = [
            label_to_id[label] for label in frame["label"].astype(str)
        ]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return (
            encode_text(self.texts[index], self.vocab),
            self.labels[index],
        )


def collate_batch(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long,
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )

    return padded, lengths, torch.tensor(labels, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # Final forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        return self.classifier(self.dropout(combined))


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    actual = []
    predicted = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for tokens, lengths, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)

            predictions = logits.argmax(dim=1)

            actual.extend(labels.cpu().tolist())
            predicted.extend(predictions.cpu().tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )

    return mean_loss, macro_f1, actual, predicted


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("CPU training selected; no CUDA GPU is available.")

    data = load_data()

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "validation"].copy()
    test_df = data[data["split"] == "test"].copy()

    # Build vocabulary using training text only.
    vocab = build_vocabulary(train_df["evidence_text"])

    labels = sorted(data["label"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}

    train_dataset = EvidenceDataset(train_df, vocab, label_to_id)
    val_dataset = EvidenceDataset(val_df, vocab, label_to_id)
    test_dataset = EvidenceDataset(test_df, vocab, label_to_id)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    # Calculate class weights from the training split only.
    train_counts = np.bincount(
        train_dataset.labels,
        minlength=len(labels),
    )

    class_weights = len(train_dataset) / (
        len(labels) * np.maximum(train_counts, 1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )

    model = LSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(labels),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    start_time = time.perf_counter()

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of classes: {len(labels)}")
    print("\nStarting LSTM training...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_f1, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "validation_loss": val_loss,
            "validation_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"train macro-F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise SystemExit("Training failed to produce a model checkpoint.")

    # Restore the checkpoint chosen by validation macro-F1.
    model.load_state_dict(best_state)
    model.to(device)

    # Evaluate on the test set only after model selection.
    test_loss, _, y_true, y_pred = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    confusion_df.index.name = "actual"
    confusion_df.to_csv(CONFUSION_PATH)

    errors = []
    test_rows = test_df.reset_index(drop=True)

    for index, (actual_id, predicted_id) in enumerate(
        zip(y_true, y_pred)
    ):
        if actual_id != predicted_id:
            row = test_rows.iloc[index]

            errors.append({
                "source_id": row["source_id"],
                "actual_label": labels[actual_id],
                "predicted_label": labels[predicted_id],
                "evidence_text": row["evidence_text"],
                "verification_status": row.get(
                    "verification_status", ""
                ),
                "is_synthetic": row.get("is_synthetic", True),
            })

    pd.DataFrame(errors, columns=[
        "source_id",
        "actual_label",
        "predicted_label",
        "evidence_text",
        "verification_status",
        "is_synthetic",
    ]).to_csv(ERRORS_PATH, index=False)

    elapsed = time.perf_counter() - start_time

    metrics = {
        "model": "Bidirectional LSTM",
        "task": "OWASP category classification from attack/impact descriptions",
        "dataset_records": int(len(data)),
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "vocabulary_size": len(vocab),
        "number_of_classes": len(labels),
        "max_sequence_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_macro_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "training_seconds": round(elapsed, 2),
        "device": str(device),
        "random_state": SEED,
        "split_file": str(SPLIT_PATH.relative_to(ROOT)),
        "train_history": history,
        "synthetic_records_present": bool(
            data["is_synthetic"].astype(str).str.lower()
            .isin(["true", "1", "yes"]).any()
        ) if "is_synthetic" in data.columns else True,
        "warning": (
            "Experimental results on synthetic/unverified dataset records. "
            "These metrics do not establish real-world vulnerability "
            "detection performance or prove exploitability."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "vocab": vocab,
        "labels": labels,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_length": MAX_LENGTH,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "model_type": "bidirectional_lstm",
        },
    }, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    CLASS_REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== LSTM TEST RESULTS ===")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro precision:  {macro_precision:.4f}")
    print(f"Macro recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Training time:    {elapsed:.2f} seconds")
    print(f"Misclassified:    {len(errors)} / {len(test_df)}")

    print("\nPer-class results:")
    for index, label in enumerate(labels):
        print(
            f"{label}: "
            f"precision={precision[index]:.3f}, "
            f"recall={recall[index]:.3f}, "
            f"F1={f1[index]:.3f}, "
            f"support={support[index]}"
        )

    print("\nSaved artifacts:")
    print(f"Model:             {MODEL_PATH}")
    print(f"Metrics:           {METRICS_PATH}")
    print(f"Classification:    {CLASS_REPORT_PATH}")
    print(f"Confusion matrix:  {CONFUSION_PATH}")
    print(f"Prediction errors: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
