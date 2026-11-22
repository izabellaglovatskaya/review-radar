"""Text normalization and hashing used for deduplication and the analysis cache."""

import datetime as dt
import hashlib
import json
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Return NFC-normalized, whitespace-collapsed, casefolded text."""
    nfc = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", nfc).strip().casefold()


def text_hash(text: str) -> str:
    """Return the sha256 hex digest of the normalized text."""
    return _sha256_hex(normalize_text(text))


def _require_non_blank(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def cache_key(text: str, *, provider: str, model: str, prompt_version: str) -> str:
    """Return the cache key for one analysis.

    The parts are encoded as a JSON array, so no choice of values can make two
    different (text, provider, model, prompt_version) tuples produce the same input.
    """
    _require_non_blank("provider", provider)
    _require_non_blank("model", model)
    _require_non_blank("prompt_version", prompt_version)
    return _sha256_hex(json.dumps([text_hash(text), provider, model, prompt_version]))


def content_hash(review_date: dt.date, text: str, source: str | None) -> str:
    """Return a digest identifying a review by its content (date, normalized text, source).

    Used as the dedup key for reviews without an id. The parts are encoded as a JSON
    array, so no choice of values can make two different tuples produce the same input.
    """
    return _sha256_hex(json.dumps([review_date.isoformat(), text_hash(text), source]))
