import pandas as pd
import re
from collections import Counter, defaultdict


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

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading data...")

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


# ---------------------------------------------------------
# LOOKUP SOURCE 2/3 RECORDS
# ---------------------------------------------------------

lookup = {}

for df in [source2, source3]:

    for _, row in df.iterrows():

        lookup[row["entity_id"]] = (
            row["business_name"],
            row["business_address"],
            row["country"]
        )


# ---------------------------------------------------------
# TOKEN FREQUENCIES
# ---------------------------------------------------------

name_frequency = Counter()
address_frequency = Counter()

for df in [source2, source3]:

    for name in df["business_name"].dropna():

        for token in get_tokens(name):
            name_frequency[token] += 1

    for address in df["business_address"].dropna():

        for token in get_tokens(address):
            address_frequency[token] += 1


# ---------------------------------------------------------
# INDEXES
# ---------------------------------------------------------

NAME_MAX = 100
ADDRESS_MAX = 100

name_index = defaultdict(set)
address_index = defaultdict(set)


for df in [source2, source3]:

    for _, row in df.iterrows():

        entity_id = row["entity_id"]
        country = row["country"]

        # Name index
        for token in get_tokens(row["business_name"]):

            if name_frequency[token] <= NAME_MAX:

                name_index[
                    (country, token)
                ].add(entity_id)

        # Address index
        for token in get_tokens(row["business_address"]):

            if address_frequency[token] <= ADDRESS_MAX:

                address_index[
                    (country, token)
                ].add(entity_id)


# ---------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------

truth = {}

for _, row in ground_truth.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        truth[row["source1_entity_id"]] = set()

    else:
        truth[row["source1_entity_id"]] = set(
            str(value).split(",")
        )


# ---------------------------------------------------------
# INSPECT MISSED MATCHES
# ---------------------------------------------------------

TEST_SIZE = 10_000

shown = 0

for _, row in source1.head(TEST_SIZE).iterrows():

    s1_id = row["entity_id"]
    country = row["country"]

    true_matches = truth.get(s1_id, set())

    if not true_matches:
        continue


    # Name candidates
    name_candidates = set()

    for token in get_tokens(row["business_name"]):

        if name_frequency[token] <= NAME_MAX:

            name_candidates.update(
                name_index.get(
                    (country, token),
                    set()
                )
            )


    # Address candidates
    address_candidates = set()

    for token in get_tokens(row["business_address"]):

        if address_frequency[token] <= ADDRESS_MAX:

            address_candidates.update(
                address_index.get(
                    (country, token),
                    set()
                )
            )


    combined = name_candidates | address_candidates


    missed = true_matches - combined


    if not missed:
        continue


    # -----------------------------------------------------
    # SHOW ONE MISSED CASE
    # -----------------------------------------------------

    print("\n" + "=" * 80)

    print("SOURCE 1")
    print("ID      :", s1_id)
    print("NAME    :", row["business_name"])
    print("ADDRESS :", row["business_address"])
    print("COUNTRY :", country)

    print("\nMISSED TRUE MATCHES:")

    for match_id in list(missed)[:5]:

        match = lookup.get(match_id)

        if match:

            print("\nID      :", match_id)
            print("NAME    :", match[0])
            print("ADDRESS :", match[1])
            print("COUNTRY :", match[2])


    shown += 1

    if shown >= 20:
        break


print("\n" + "=" * 80)
print("Displayed", shown, "missed Source 1 examples.")
print("=" * 80)