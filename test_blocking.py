import pandas as pd
import re
from collections import Counter, defaultdict


# --------------------------------------------------
# 1. TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).casefold()

    cleaned = []

    for char in text:
        if char.isalnum() or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_tokens(text):
    normalized = normalize_text(text)

    # Remove very short tokens
    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


# --------------------------------------------------
# 2. LOAD TRAINING DATA
# --------------------------------------------------

print("Loading training data...")

source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

source3 = pd.read_csv(
    "student_resource/dataset/train/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# --------------------------------------------------
# 3. COUNT TOKENS IN SOURCE 2 + SOURCE 3
# --------------------------------------------------

print("Counting tokens...")

token_frequency = Counter()

for df in [source2, source3]:

    for name in df["business_name"].dropna():

        tokens = get_tokens(name)

        # Count once per record
        for token in tokens:
            token_frequency[token] += 1


print("Unique tokens:", len(token_frequency))


# --------------------------------------------------
# 4. CREATE RARE TOKEN INDEX
# --------------------------------------------------

MAX_FREQUENCY = 100

print("Building rare-token index...")

token_index = defaultdict(list)

for df in [source2, source3]:

    for entity_id, name, country in zip(
        df["entity_id"],
        df["business_name"],
        df["country"]
    ):

        tokens = get_tokens(name)

        for token in tokens:

            if token_frequency[token] <= MAX_FREQUENCY:

                token_index[(country, token)].append(entity_id)


print("Index created.")
print("Index keys:", len(token_index))


# --------------------------------------------------
# 5. CREATE GROUND-TRUTH LOOKUP
# --------------------------------------------------

ground_truth_lookup = {}

for _, row in ground_truth.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched) or str(matched).strip() == "":
        ground_truth_lookup[row["source1_entity_id"]] = set()

    else:
        ground_truth_lookup[row["source1_entity_id"]] = set(
            str(matched).split(",")
        )


# --------------------------------------------------
# 6. TEST ON FIRST 10,000 SOURCE 1 RECORDS
# --------------------------------------------------

TEST_SIZE = 10_000

test_source1 = source1.head(TEST_SIZE)

print()
print("=" * 60)
print("TESTING BLOCKING ON", TEST_SIZE, "SOURCE 1 RECORDS")
print("=" * 60)


total_true_matches = 0
total_recovered_matches = 0

entities_with_all_matches_found = 0
entities_with_at_least_one_candidate = 0

total_candidates = 0
max_candidates = 0


for _, row in test_source1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]
    name = row["business_name"]

    true_matches = ground_truth_lookup.get(s1_id, set())

    tokens = get_tokens(name)

    candidates = set()

    for token in tokens:

        if token_frequency[token] <= MAX_FREQUENCY:

            key = (country, token)

            candidates.update(
                token_index.get(key, [])
            )

    # Remove accidental Source 1 IDs (shouldn't happen)
    candidates = {
        candidate
        for candidate in candidates
        if candidate.startswith("S2-")
        or candidate.startswith("S3-")
    }

    total_candidates += len(candidates)

    max_candidates = max(
        max_candidates,
        len(candidates)
    )

    if len(candidates) > 0:
        entities_with_at_least_one_candidate += 1

    recovered = true_matches.intersection(candidates)

    total_true_matches += len(true_matches)
    total_recovered_matches += len(recovered)

    if true_matches.issubset(candidates):
        entities_with_all_matches_found += 1


# --------------------------------------------------
# 7. RESULTS
# --------------------------------------------------

print()
print("=" * 60)
print("BLOCKING RESULTS")
print("=" * 60)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "Recovered true matches:",
    total_recovered_matches
)

if total_true_matches > 0:

    recall = (
        total_recovered_matches /
        total_true_matches
    )

    print(
        f"Pair recall: {recall:.4%}"
    )

print(
    "Entities with all matches recovered:",
    entities_with_all_matches_found
)

print(
    "Entities with at least one candidate:",
    entities_with_at_least_one_candidate
)

print(
    "Average candidates per Source 1:",
    total_candidates / TEST_SIZE
)

print(
    "Maximum candidates for one Source 1:",
    max_candidates
)

print()
print("Experiment complete.")