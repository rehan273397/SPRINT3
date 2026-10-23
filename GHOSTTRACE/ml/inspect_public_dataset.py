from datasets import load_dataset

print("Loading public web-security dataset...")
dataset = load_dataset("scthornton/securecode", "web")

print("\nDataset splits:")
print(dataset)

train = dataset["train"]

print("\nColumn names:")
print(train.column_names)

print("\nFirst example:")
print(train[0])

print("\nOWASP category counts:")
counts = {}

for row in train:
    metadata = row.get("metadata") or {}
    category = metadata.get("owasp_2021", "UNKNOWN")
    counts[category] = counts.get(category, 0) + 1

for category, count in sorted(counts.items()):
    print(f"{category}: {count}")

print("\nInspection complete. No training data was modified.")
