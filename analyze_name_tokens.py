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


# Load only the name column to reduce memory usage
source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t",
    usecols=["business_name"]
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t",
    usecols=["business_name"]
)

source3 = pd.read_csv(
    "student_resource/dataset/train/train_source3.tsv",
    sep="\t",
    usecols=["business_name"]
)


counter = Counter()


for df in [source1, source2, source3]:

    names = df["business_name"].dropna()

    for name in names:

        normalized = normalize_text(name)

        tokens = set(normalized.split())

        for token in tokens:
            if len(token) >= 3:
                counter[token] += 1


print("\n===== MOST COMMON NAME TOKENS =====")

for token, count in counter.most_common(50):
    print(f"{token:30} {count:,}")


print("\n===== TOKEN STATISTICS =====")

print("Unique tokens:", len(counter))

print("\nTokens appearing only once:", 
      sum(1 for count in counter.values() if count == 1))

print("Tokens appearing <= 5 times:",
      sum(1 for count in counter.values() if count <= 5))

print("Tokens appearing <= 100 times:",
      sum(1 for count in counter.values() if count <= 100))