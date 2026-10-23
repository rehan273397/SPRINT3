from pathlib import Path
import json
import re

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RAW_DIR / "security_evidence.csv"
MANIFEST = RAW_DIR / "dataset_manifest.json"


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    if value.lower() in {"none", "null", "n/a", "unknown"}:
        return ""
    return value


print("Loading SecureCode web dataset...")
dataset = load_dataset("scthornton/securecode", "web")
source = dataset["train"]

rows = []

for record in source:
    metadata = record.get("metadata") or {}
    category = clean_text(metadata.get("owasp_2021"))

    if not category.startswith("A"):
        continue

    category_id = category.split(":", 1)[0]
    category_name = category.split("-", 1)[-1]

    label = (
        category_id + "_" + category_name.lower()
    ).replace(" ", "_").replace("-", "_")

    context = record.get("context") or {}

    # Deliberately exclude OWASP category, subcategory and CWE
    # from model input to reduce label shortcuts.
    parts = []

    attack = clean_text(context.get("attack_vector"))
    impact = clean_text(context.get("impact"))

    if attack:
        parts.append("Attack description: " + attack)

    if impact:
        parts.append("Potential impact: " + impact)

    # Use severity only as a separate metadata field, not in text.
    evidence_text = " ".join(parts).strip()

    if len(evidence_text) < 30:
        continue

    validation = record.get("validation") or {}
    source_id = clean_text(
        record.get("uid") or record.get("id")
    )

    rows.append({
        "evidence_text": evidence_text,
        "label": label,
        "source_type": "securecode_context",
        "verification_status": (
            clean_text(validation.get("security_review"))
            or "not_reviewed"
        ),
        "source_id": source_id,
        "group_id": source_id,
        "severity_metadata": clean_text(metadata.get("severity")),
        "cwe_metadata": clean_text(metadata.get("cwe")),
        "is_synthetic": True,
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit(
        "No usable records found. Check the source dataset schema."
    )

df["evidence_text"] = df["evidence_text"].fillna("").str.strip()
df = df[df["evidence_text"] != ""].copy()

# Remove exact duplicate text, even when labels differ.
df = df.drop_duplicates(subset=["evidence_text"])

df.to_csv(OUTPUT, index=False)

manifest = {
    "dataset_name": "SecureCode",
    "dataset_config": "web",
    "source": "https://huggingface.co/datasets/scthornton/securecode",
    "source_split": "train",
    "records_imported": len(df),
    "task": "OWASP category classification from attack and impact descriptions",
    "input_fields": ["attack_vector", "impact"],
    "excluded_from_text": [
        "OWASP category",
        "OWASP subcategory",
        "CWE",
        "severity",
        "conversation transcripts",
    ],
    "warning": (
        "All rows are conservatively marked synthetic/unverified. "
        "These results do not establish real-world detection performance."
    ),
    "license_to_review": "CC BY-NC-SA 4.0",
}

MANIFEST.write_text(json.dumps(manifest, indent=2))

print(f"\nImported records: {len(df)}")
print(f"Saved: {OUTPUT}")
print(f"Manifest: {MANIFEST}")
print("\nRecords per label:")
print(df["label"].value_counts().sort_index().to_string())
print("\nAll rows marked synthetic/unverified for this experiment.")
