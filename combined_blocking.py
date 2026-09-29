import pandas as pd
import re
from collections import Counter, defaultdict


# ============================================================
# 1. NORMALIZATION
# ============================================================

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


def get_tokens(text, min_length=3):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= min_length
    }


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

source3 = pd.read_csv(
    "student_resource/dataset/train/train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ============================================================
# 3. COUNT NAME TOKENS
# ============================================================

print("\nCounting name tokens...")

name_frequency = Counter()

for df in [source2, source3]:

    for name in df["business_name"].dropna():

        tokens = get_tokens(name)

        for token in tokens:
            name_frequency[token] += 1


# ============================================================
# 4. COUNT ADDRESS TEXT TOKENS
# ============================================================

print("Counting address tokens...")

address_frequency = Counter()

for df in [source2, source3]:

    for address in df["business_address"].dropna():

        tokens = get_tokens(address)

        for token in tokens:
            address_frequency[token] += 1


print("\nUnique name tokens:", len(name_frequency))
print("Unique address tokens:", len(address_frequency))


# ============================================================
# 5. BUILD NAME INDEX
# ============================================================

print("\nBuilding name index...")

NAME_MAX_FREQUENCY = 100

name_index = defaultdict(list)

for df in [source2, source3]:

    for entity_id, name, country in zip(
        df["entity_id"],
        df["business_name"],
        df["country"]
    ):

        tokens = get_tokens(name)

        for token in tokens:

            if name_frequency[token] <= NAME_MAX_FREQUENCY:

                name_index[(country, token)].append(entity_id)


# ============================================================
# 6. BUILD ADDRESS INDEX
# ============================================================

print("Building address index...")

ADDRESS_MAX_FREQUENCY = 100

address_index = defaultdict(list)

for df in [source2, source3]:

    for entity_id, address, country in zip(
        df["entity_id"],
        df["business_address"],
        df["country"]
    ):

        tokens = get_tokens(address)

        for token in tokens:

            if address_frequency[token] <= ADDRESS_MAX_FREQUENCY:

                address_index[(country, token)].append(entity_id)


print("Name index keys:", len(name_index))
print("Address index keys:", len(address_index))


# ============================================================
# 7. GROUND TRUTH LOOKUP
# ============================================================

print("\nBuilding ground-truth lookup...")

ground_truth_lookup = {}

for _, row in ground_truth.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched) or str(matched).strip() == "":
        ground_truth_lookup[row["source1_entity_id"]] = set()

    else:
        ground_truth_lookup[row["source1_entity_id"]] = set(
            str(matched).split(",")
        )


# ============================================================
# 8. TEST BLOCKING
# ============================================================

TEST_SIZE = 10_000

test_source1 = source1.head(TEST_SIZE)

print()
print("=" * 70)
print("TESTING COMBINED BLOCKING ON", TEST_SIZE, "SOURCE 1 RECORDS")
print("=" * 70)


total_true_matches = 0

name_recovered = 0
address_recovered = 0
combined_recovered = 0

name_all_found = 0
address_all_found = 0
combined_all_found = 0

name_candidate_total = 0
address_candidate_total = 0
combined_candidate_total = 0

max_combined_candidates = 0


# ============================================================
# 9. PROCESS EACH SOURCE 1
# ============================================================

for _, row in test_source1.iterrows():

    s1_id = row["entity_id"]
    country = row["country"]

    name = row["business_name"]
    address = row["business_address"]

    true_matches = ground_truth_lookup.get(
        s1_id,
        set()
    )


    # --------------------------------------------------------
    # NAME CANDIDATES
    # --------------------------------------------------------

    name_candidates = set()

    name_tokens = get_tokens(name)

    for token in name_tokens:

        if name_frequency[token] <= NAME_MAX_FREQUENCY:

            key = (country, token)

            name_candidates.update(
                name_index.get(key, [])
            )


    # --------------------------------------------------------
    # ADDRESS CANDIDATES
    # --------------------------------------------------------

    address_candidates = set()

    address_tokens = get_tokens(address)

    for token in address_tokens:

        if address_frequency[token] <= ADDRESS_MAX_FREQUENCY:

            key = (country, token)

            address_candidates.update(
                address_index.get(key, [])
            )


    # --------------------------------------------------------
    # COMBINED CANDIDATES
    # --------------------------------------------------------

    combined_candidates = (
        name_candidates |
        address_candidates
    )


    # --------------------------------------------------------
    # REMOVE INVALID IDs
    # --------------------------------------------------------

    name_candidates = {
        x for x in name_candidates
        if x.startswith("S2-") or x.startswith("S3-")
    }

    address_candidates = {
        x for x in address_candidates
        if x.startswith("S2-") or x.startswith("S3-")
    }

    combined_candidates = {
        x for x in combined_candidates
        if x.startswith("S2-") or x.startswith("S3-")
    }


    # --------------------------------------------------------
    # RECOVER TRUE MATCHES
    # --------------------------------------------------------

    name_found = true_matches.intersection(
        name_candidates
    )

    address_found = true_matches.intersection(
        address_candidates
    )

    combined_found = true_matches.intersection(
        combined_candidates
    )


    # --------------------------------------------------------
    # UPDATE METRICS
    # --------------------------------------------------------

    total_true_matches += len(true_matches)

    name_recovered += len(name_found)
    address_recovered += len(address_found)
    combined_recovered += len(combined_found)

    if true_matches.issubset(name_candidates):
        name_all_found += 1

    if true_matches.issubset(address_candidates):
        address_all_found += 1

    if true_matches.issubset(combined_candidates):
        combined_all_found += 1

    name_candidate_total += len(name_candidates)
    address_candidate_total += len(address_candidates)
    combined_candidate_total += len(combined_candidates)

    max_combined_candidates = max(
        max_combined_candidates,
        len(combined_candidates)
    )


# ============================================================
# 10. RESULTS
# ============================================================

print()
print("=" * 70)
print("BLOCKING COMPARISON")
print("=" * 70)

print("\nTRUE MATCHES:", total_true_matches)


print("\n--- NAME BLOCKING ---")

print(
    "Recovered:",
    name_recovered
)

print(
    "Pair recall:",
    f"{name_recovered / total_true_matches:.4%}"
)

print(
    "Entities with all matches:",
    name_all_found
)

print(
    "Average candidates:",
    f"{name_candidate_total / TEST_SIZE:.2f}"
)


print("\n--- ADDRESS BLOCKING ---")

print(
    "Recovered:",
    address_recovered
)

print(
    "Pair recall:",
    f"{address_recovered / total_true_matches:.4%}"
)

print(
    "Entities with all matches:",
    address_all_found
)

print(
    "Average candidates:",
    f"{address_candidate_total / TEST_SIZE:.2f}"
)


print("\n--- COMBINED BLOCKING ---")

print(
    "Recovered:",
    combined_recovered
)

print(
    "Pair recall:",
    f"{combined_recovered / total_true_matches:.4%}"
)

print(
    "Entities with all matches:",
    combined_all_found
)

print(
    "Average candidates:",
    f"{combined_candidate_total / TEST_SIZE:.2f}"
)

print(
    "Maximum candidates:",
    max_combined_candidates
)


print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)