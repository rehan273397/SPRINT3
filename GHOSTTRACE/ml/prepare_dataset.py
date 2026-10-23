from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/security_evidence.csv"
PROCESSED = ROOT / "data/processed/security_evidence.csv"

REQUIRED_COLUMNS = [
    "evidence_text",
    "label",
    "source_type",
    "verification_status",
    "source_id",
    "group_id",
    "severity_metadata",
    "cwe_metadata",
    "is_synthetic",
]

ALLOWED_LABELS = {
    "A01_broken_access_control",
    "A02_cryptographic_failures",
    "A03_injection",
    "A04_insecure_design",
    "A05_security_misconfiguration",
    "A06_vulnerable_and_outdated_components",
    "A07_identification_and_authentication_failures",
    "A08_software_and_data_integrity_failures",
    "A09_security_logging_and_monitoring_failures",
    "A10_server_side_request_forgery",
}


def main():
    if not RAW.exists():
        raise SystemExit(f"Raw dataset not found: {RAW}")

    df = pd.read_csv(RAW)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(
            f"Missing columns: {sorted(missing)}. "
            "Run python -m ml.import_securecode first."
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["evidence_text"] = (
        df["evidence_text"].fillna("").astype(str).str.strip()
    )
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df["group_id"] = df["group_id"].fillna("").astype(str).str.strip()

    df = df[
        (df["evidence_text"] != "")
        & (df["label"] != "")
        & (df["group_id"] != "")
    ].copy()

    unknown = set(df["label"].unique()) - ALLOWED_LABELS
    if unknown:
        raise SystemExit(f"Unknown labels: {sorted(unknown)}")

    df = df.drop_duplicates(subset=["evidence_text"])

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)

    print(f"Valid records: {len(df)}")
    print(f"Saved: {PROCESSED}")
    print("\nRecords per label:")
    print(df["label"].value_counts().sort_index().to_string())
    print("\nUnique groups:", df["group_id"].nunique())

    print("\nVerification status:")
    print(df["verification_status"].value_counts(dropna=False).to_string())

    print("\nNote: all imported records are treated as synthetic/unverified.")


if __name__ == "__main__":
    main()
