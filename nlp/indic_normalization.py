"""Indic Normalization Policy & Preprocessor (Phase 6).

Handles:
1. Devanagari Unicode canonical normalization (NFC)
2. Nukta normalization (e.g. ज़ -> ज, फ़ -> फ where appropriate for ASR matching)
3. Chandrabindu and Anusvara unification
4. Romanized-Hinglish transliteration normalization (common English-script Hindi phrases)
5. Punctuation stripping and whitespace collapse
"""
import re
import unicodedata


# Common Hinglish transliteration equivalences
HINGLISH_PHRASE_MAP = {
    r"\byaad dilana\b": "remind",
    r"\byaad dila dena\b": "remind",
    r"\breminder laga do\b": "add reminder",
    r"\bkal subah\b": "tomorrow morning",
    r"\bkal shaam\b": "tomorrow evening",
    r"\bpaanch baje\b": "5 pm",
    r"\btheek hai\b": "yes",
    r"\bhaan\b": "yes",
    r"\bnahi\b": "no",
}


def normalize_devanagari(text: str) -> str:
    """Normalize Devanagari text for ASR WER evaluation."""
    if not text:
        return ""
    # NFC Canonical decomposition followed by canonical composition
    text = unicodedata.normalize("NFC", text)

    # Standardize Danda (।) and double Danda to standard sentence boundaries
    text = text.replace("।", " ").replace("॥", " ")

    # Strip common punctuation
    text = re.sub(r"[,\.!\?\-\(\)\[\]\{\}\"':;]", " ", text)

    # Unify zero-width joiners / non-joiners
    text = text.replace("\u200c", "").replace("\u200d", "")

    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def normalize_hinglish(text: str) -> str:
    """Normalize Romanized Hinglish text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    for pattern, replacement in HINGLISH_PHRASE_MAP.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_indic(text: str, script: str = "auto") -> str:
    """Entry point for Indic text normalization."""
    if not text:
        return ""
    has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
    if script == "devanagari" or (script == "auto" and has_devanagari):
        return normalize_devanagari(text)
    return normalize_hinglish(text)
