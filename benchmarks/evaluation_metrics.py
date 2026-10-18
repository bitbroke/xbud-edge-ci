"""Shared scoring policy: NFC, lowercase and collapsed whitespace; punctuation retained."""
import unicodedata
from datetime import datetime, timezone

import numpy as np
from jiwer import process_words, process_characters


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def compute_wer_cer_metrics(references, hypotheses):
    if not references or len(references) != len(hypotheses):
        raise ValueError("Nonempty, equally sized reference/hypothesis lists required")
    refs = [normalize_text(x) for x in references]
    hyps = [normalize_text(x) for x in hypotheses]
    if any(not x for x in refs):
        raise ValueError("Non-speech references must be scored separately, not as WER")
    # Do not conceal scoring errors or replace corpus statistics with clip means.
    words = process_words(refs, hyps)
    chars = process_characters(refs, hyps)
    clip_wers = [process_words(r, h).wer for r, h in zip(refs, hyps)]
    clip_cers = [process_characters(r, h).cer for r, h in zip(refs, hyps)]
    return {
        "normalization": "NFC + lowercase + collapse whitespace; punctuation retained",
        "mean_clip_wer": float(np.mean(clip_wers)), "corpus_wer": float(words.wer),
        "mean_clip_cer": float(np.mean(clip_cers)), "corpus_cer": float(chars.cer),
        "clip_wers": clip_wers, "clip_cers": clip_cers,
        "substitutions": words.substitutions, "deletions": words.deletions,
        "insertions": words.insertions, "hits": words.hits,
    }


def assert_exact_utc(actual, expected):
    """Compare complete instants, rejecting naive timestamps on either side."""
    a, e = (datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (actual, expected))
    if any(x.tzinfo is None or x.utcoffset() is None for x in (a, e)):
        raise ValueError("Timezone-aware timestamps required")
    if a.astimezone(timezone.utc) != e.astimezone(timezone.utc):
        raise AssertionError(f"UTC timestamp mismatch: expected {expected}, got {actual}")
