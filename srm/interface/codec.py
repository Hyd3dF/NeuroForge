"""Codec: input byte normalization (03 §6).

The output vocabulary belongs to the Mouth (later batch); this module provides the
input side used by Perception and by SEF conversion.
"""

from __future__ import annotations

import unicodedata


def normalize_text(data: str | bytes, form: str = "NFC") -> str:
    """UTF-8 decode (with replacement), Unicode normalization and newline normalization."""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    text = unicodedata.normalize(form, text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def to_bytes(text: str, form: str = "NFC") -> bytes:
    return normalize_text(text, form).encode("utf-8")


def normalize_surface(s: str) -> str:
    """Normalized surface form for alias lookup (02 §7.1 step 1)."""
    s = unicodedata.normalize("NFC", s).casefold()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in s]
    return " ".join("".join(kept).split())
