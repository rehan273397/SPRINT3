
import csv
import json
import re
from collections import Counter
from pathlib import Path


RAW_DIR = Path("data/raw/difraud/phishing")
PROCESSED_DIR = Path("data/processed")

SPLITS = ("train", "validation", "test")


def normalize_text(text: str) -> str:
    """Normalize whitespace without removing useful message content."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: Path) -> list[dict]:
    """Load and validate every JSONL record."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at line {line_number}")

            text = record.get("text")
            label = record.get("label")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid text at line {line_number}")

            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"Invalid label at line {line_number}")

            records.append({
                "text": normalize_text(text),
                "label": label
            })

    return records


def save_csv(records: list[dict], path: Path) -> None:
    """Save processed records as CSV."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["text", "label"]
        )
        writer.writeheader()
        writer.writerows(records)


def prepare_dataset() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        source = RAW_DIR / f"{split}.jsonl"
        destination = PROCESSED_DIR / f"{split}.csv"

        if not source.exists():
            raise FileNotFoundError(source)

        records = load_jsonl(source)
        save_csv(records, destination)

        distribution = Counter(
            record["label"] for record in records
        )

        print(f"\n{split.upper()}")
        print("-" * 35)
        print("Total records:", len(records))
        print("Legitimate (0):", distribution[0])
        print("Phishing (1):", distribution[1])
        print("Saved:", destination)


if __name__ == "__main__":
    prepare_dataset()

