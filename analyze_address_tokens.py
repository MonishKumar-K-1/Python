import pandas as pd
import re
from collections import Counter


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


for filename in [
    "student_resource/dataset/train/train_source1.tsv",
    "student_resource/dataset/train/train_source2.tsv",
    "student_resource/dataset/train/train_source3.tsv"
]:

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


print("\n===== MOST COMMON ADDRESS TOKENS =====")

for token, count in counter.most_common(60):
    print(f"{token:30} {count:,}")


print("\n===== TOKEN STATISTICS =====")

print("Unique tokens:", len(counter))

print(
    "Tokens appearing only once:",
    sum(1 for count in counter.values() if count == 1)
)

print(
    "Tokens appearing <= 5 times:",
    sum(1 for count in counter.values() if count <= 5)
)

print(
    "Tokens appearing <= 100 times:",
    sum(1 for count in counter.values() if count <= 100)
)