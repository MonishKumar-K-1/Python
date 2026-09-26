import pandas as pd


# --------------------------------------------------
# 1. Load the data
# --------------------------------------------------

source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "student_resource/dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# --------------------------------------------------
# 2. Create quick lookup tables using entity_id
# --------------------------------------------------

source1_lookup = source1.set_index("entity_id")
source2_lookup = source2.set_index("entity_id")
source3_lookup = source3.set_index("entity_id")


# --------------------------------------------------
# 3. Select a few Source 1 businesses
#    that have multiple known matches
# --------------------------------------------------

ground_truth["matched_entity_ids"] = (
    ground_truth["matched_entity_ids"]
    .fillna("")
    .astype(str)
)

ground_truth["match_count"] = ground_truth["matched_entity_ids"].apply(
    lambda x: 0 if x.strip() == "" else len(x.split(","))
)

examples = ground_truth[
    ground_truth["match_count"] >= 3
].head(3)


# --------------------------------------------------
# 4. Display the actual matches
# --------------------------------------------------

for _, gt in examples.iterrows():

    s1_id = gt["source1_entity_id"]

    print("\n" + "=" * 80)
    print("SOURCE 1 BUSINESS")
    print("=" * 80)

    print(source1_lookup.loc[s1_id].to_string())

    matched_ids = gt["matched_entity_ids"].split(",")

    print("\n" + "-" * 80)
    print("KNOWN MATCHES")
    print("-" * 80)

    for entity_id in matched_ids:

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            print("\n[SOURCE 2]")
            print(source2_lookup.loc[entity_id].to_string())

        elif entity_id.startswith("S3-"):
            print("\n[SOURCE 3]")
            print(source3_lookup.loc[entity_id].to_string())