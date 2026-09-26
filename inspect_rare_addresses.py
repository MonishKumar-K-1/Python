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
        if len(token) >= 2
    }


counter = Counter()
examples = defaultdict(list)


files = [
    "student_resource/dataset/train/train_source1.tsv",
    "student_resource/dataset/train/train_source2.tsv",
    "student_resource/dataset/train/train_source3.tsv"
]


for filename in files:

    print("Processing:", filename)

    df = pd.read_csv(
        filename,
        sep="\t",
        usecols=["business_address"]
    )

    for address in df["business_address"].dropna():

        tokens = get_tokens(address)

        for token in tokens:

            counter[token] += 1

            if len(examples[token]) < 3:
                examples[token].append(address)


print("\n===== SAMPLE RARE ADDRESS TOKENS =====")

rare_tokens = [
    token
    for token, count in counter.items()
    if count <= 5
]

# Sort rare tokens alphabetically just to make
# the output reproducible.
rare_tokens.sort()

for token in rare_tokens[:100]:

    print(
        f"\nToken: {token}"
        f"\nFrequency: {counter[token]}"
    )

    for address in examples[token]:
        print("  Example:", address)