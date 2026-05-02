from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login

login(token="HF_TOKEN")

dev  = load_dataset("diarizers-community/voxconverse", split="dev")
test = load_dataset("diarizers-community/voxconverse", split="test")
full = concatenate_datasets([dev, test])

# Filter >= 3 speakers
filtered = full.filter(lambda s: len(set(s["speakers"])) >= 3)
print(f"Samples with >= 3 speakers: {len(filtered)}")

# Take 30
subset = filtered.select(range(min(30, len(filtered))))
print(f"Subset size: {len(subset)}")
print(f"Columns: {subset.column_names}")

# Split and push
split    = subset.train_test_split(test_size=0.3, seed=42)
val_test = split["test"].train_test_split(test_size=0.5, seed=42)

final_ds = DatasetDict({
    "train"     : split["train"],
    "validation": val_test["train"],
    "test"      : val_test["test"]
})

print(final_ds)
final_ds.push_to_hub("aminulpalash506/voxconverse-30-subset")
print("Pushed!")