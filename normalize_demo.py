import re
import unicodedata


def normalize_text(text):
    """
    Basic Unicode-safe normalization.

    We:
    1. Handle missing values
    2. Normalize Unicode representation
    3. Convert to lowercase using casefold()
    4. Replace punctuation with spaces
    5. Normalize repeated whitespace

    We DO NOT remove non-English characters.
    """

    if text is None:
        return ""

    text = str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Better than lower() for multilingual text
    text = text.casefold()

    # Replace punctuation while preserving Unicode letters/numbers
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(char)

    text = "".join(cleaned)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


examples = [
    "Dahlia Power Reliable Scientific LLC",
    "Dahlia Power Reliable Scientific",
    "Dahlia Ponr Reliable Scientific LLC",
    "Maure Williams Colombier Inc",
    "Maure Wilblims Colombier Inc",
    "Raj Investments LLP",
    "Raj Investments எல்எல்பி",
    "मॉडर्न फाइनेंस",
    "No. 10, Enkay Square, 448A",
    "R. DE DIEPPE, LILLE",
]


for text in examples:
    print("Original   :", text)
    print("Normalized :", normalize_text(text))
    print("-" * 70)