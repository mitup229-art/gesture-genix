"""
NLP sentence construction layer.

Baseline: rule-based slot filling (capitalization + punctuation).
Extension point: swap construct_sentence() to call a lightweight
grammar-correction model (e.g., a small HuggingFace transformer)
for fluency polishing once the baseline pipeline works end-to-end.
"""


def construct_sentence(tokens):
    """
    Convert a list of recognized gesture tokens into a simple sentence.
    e.g. ["I", "WANT", "WATER"] -> "I want water."
    """
    if not tokens:
        return ""

    words = [t.replace("_", " ").lower() for t in tokens]
    text = " ".join(words)
    text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text
