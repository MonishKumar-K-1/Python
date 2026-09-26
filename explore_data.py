import pandas as pd

# Load ground truth
ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# Make sure empty values are treated as empty strings
matches = ground_truth["matched_entity_ids"].fillna("").astype(str)

# Count how many matches each Source 1 entity has
match_counts = matches.apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)

print("=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)

print("Total Source 1 entities:", len(ground_truth))

print("Entities with NO matches:", (match_counts == 0).sum())

print("Entities with at least 1 match:", (match_counts > 0).sum())

print("Total known matches:", match_counts.sum())


print("\n" + "=" * 60)
print("MATCH COUNT DISTRIBUTION")
print("=" * 60)

print(match_counts.value_counts().sort_index())


print("\n" + "=" * 60)
print("MATCH COUNT STATISTICS")
print("=" * 60)

print("Minimum matches:", match_counts.min())
print("Maximum matches:", match_counts.max())
print("Average matches:", round(match_counts.mean(), 2))
print("Median matches:", match_counts.median())


print("\n" + "=" * 60)
print("EXAMPLES OF SINGLETONS")
print("=" * 60)

singletons = ground_truth[match_counts == 0]

print(singletons.head(10).to_string(index=False))


print("\n" + "=" * 60)
print("EXAMPLES WITH MANY MATCHES")
print("=" * 60)

many_matches = ground_truth[match_counts >= 5]

print(
    many_matches.head(10).to_string(index=False)
)