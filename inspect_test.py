import pandas as pd

# Load test data
source1 = pd.read_csv(
    "student_resource/dataset/test/test_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "student_resource/dataset/test/test_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "student_resource/dataset/test/test_source3.tsv",
    sep="\t"
)

print("\n===== TEST DATASET SIZES =====")
print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))

print("\n===== COLUMNS =====")
print("Source 1:", source1.columns.tolist())
print("Source 2:", source2.columns.tolist())
print("Source 3:", source3.columns.tolist())

print("\n===== COUNTRIES =====")
print("\nSource 1:")
print(source1["country"].value_counts())

print("\nSource 2:")
print(source2["country"].value_counts())

print("\nSource 3:")
print(source3["country"].value_counts())

print("\n===== MISSING VALUES =====")

print("\nSource 1:")
print(source1.isna().sum())

print("\nSource 2:")
print(source2.isna().sum())

print("\nSource 3:")
print(source3.isna().sum())

print("\n===== SAMPLE SOURCE 1 =====")
print(source1.head(5).to_string(index=False))

print("\n===== SAMPLE SOURCE 2 =====")
print(source2.head(5).to_string(index=False))

print("\n===== SAMPLE SOURCE 3 =====")
print(source3.head(5).to_string(index=False))