"""Demo-mode classifier: canned answers keyed by text hash plus a deterministic fallback.

No key, no network, no cost. Privacy rule: logs and error messages contain counts and file
names only, never review text.

The `prompt_version` is derived from the fixture content and `FALLBACK_RULES_VERSION`, so
changing either one invalidates the cached demo analyses instead of serving stale answers.
"""

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.hashing import normalize_text, text_hash
from review_radar.models import MAX_TOPICS, Analysis, Sentiment, Topic

logger = logging.getLogger(__name__)

FIXTURE_FORMAT = 1
# Bump this only when the LOGIC of `_fallback` changes. The rule data (word lists, topic
# keywords, confidence) is hashed into `prompt_version` automatically, so editing it needs no
# bump; both are part of `FixtureClassifier.prompt_version`.
FALLBACK_RULES_VERSION = 1
FALLBACK_CONFIDENCE = 0.5
_VERSION_PREFIX = "fixture-1"
_FIXTURE_KEY = re.compile(r"[0-9a-f]{64}")
_ENTRY_KEYS = frozenset(["sentiment", "topics"])

_POSITIVE_WORDS = frozenset(
    [
        "great",
        "good",
        "excellent",
        "love",
        "loved",
        "perfect",
        "amazing",
        "fast",
        "happy",
        "recommend",
        "superb",
        "fantastic",
        "pleased",
        "wonderful",
        "awesome",
    ]
)
_NEGATIVE_WORDS = frozenset(
    [
        "bad",
        "terrible",
        "awful",
        "worst",
        "broken",
        "late",
        "slow",
        "disappointed",
        "poor",
        "horrible",
        "rude",
        "useless",
        "overpriced",
    ]
)

# Phrases instead of risky single words ("double-check", "will return", "used it for weeks").
_TOPIC_KEYWORDS: dict[Topic, tuple[str, ...]] = {
    Topic.DELIVERY_DELAY: ("took weeks", "still waiting", "late", "delayed", "delay"),
    Topic.PRODUCT_QUALITY: (
        "broken",
        "crack",
        "cracked",
        "defect",
        "defective",
        "poor quality",
        "fell apart",
    ),
    Topic.CUSTOMER_SUPPORT: ("support", "no reply", "rude", "ignored"),
    Topic.PRICING: ("expensive", "overpriced", "price"),
    Topic.BILLING: ("charged twice", "double charged", "overcharged", "invoice", "billing"),
    Topic.APP_USABILITY: (
        "the app",
        "app crashes",
        "app crashed",
        "login",
        "log in",
        "confusing",
    ),
    Topic.PACKAGING: (
        "packaging",
        "crushed box",
        "damaged box",
        "box was crushed",
        "box was damaged",
    ),
    Topic.REFUND_PROCESS: ("refund", "money back", "return label", "returned it"),
}
_TOPIC_PATTERNS: dict[Topic, re.Pattern[str]] = {
    topic: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in words) + r")(?!\w)")
    for topic, words in _TOPIC_KEYWORDS.items()
}
_WORD = re.compile(r"\w+")


class FixtureClassifier:
    """Classifier for demo mode; never fails on any text."""

    provider = "fixture"
    model = "demo"

    def __init__(self, fixtures: Mapping[str, Analysis]) -> None:
        self._fixtures = fixtures
        self.prompt_version = f"{_VERSION_PREFIX}+{_content_digest(fixtures)}"

    @classmethod
    def from_file(cls, path: Path) -> "FixtureClassifier":
        """Load `{"format": 1, "analyses": {"<text_hash>": {"sentiment", "topics"}}}`.

        Fixtures are trusted input, so validation is strict: unknown sentiments or topics,
        duplicate topics or keys, more than three topics, malformed keys and an empty file
        are all errors. Raises ClassifierError("fixture_load_failed") naming only the file.
        """
        try:
            raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
            return cls(_parse_fixtures(raw))
        except (OSError, UnicodeDecodeError):
            raise _load_error(path, "the file cannot be read") from None
        except json.JSONDecodeError:
            raise _load_error(path, "the file is not valid JSON") from None
        except ValueError:  # includes pydantic.ValidationError
            raise _load_error(path, "the content is invalid") from None

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        analyses: dict[str, Analysis] = {}
        hits = 0
        for item in items:
            canned = self._fixtures.get(text_hash(item.text))
            if canned is not None:
                hits += 1
                analyses[item.item_id] = canned
            else:
                analyses[item.item_id] = _fallback(item)
        misses = len(items) - hits
        logger.info("Fixture classifier: %d fixture hits, %d fallback", hits, misses)
        return ClassifyResult(analyses=analyses, fallback=misses)


def _content_digest(fixtures: Mapping[str, Analysis]) -> str:
    """Return 8 hex chars identifying the fixtures and the complete fallback rules."""
    payload = {
        "fixtures": {key: fixtures[key].model_dump(mode="json") for key in sorted(fixtures)},
        "rules": {
            "version": FALLBACK_RULES_VERSION,
            "confidence": FALLBACK_CONFIDENCE,
            "positive": sorted(_POSITIVE_WORDS),
            "negative": sorted(_NEGATIVE_WORDS),
            "topics": {topic.value: sorted(words) for topic, words in _TOPIC_KEYWORDS.items()},
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _load_error(path: Path, problem: str) -> ClassifierError:
    return ClassifierError(
        "fixture_load_failed", f"Cannot load fixtures from {path.name}: {problem}."
    )


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _parse_fixtures(raw: object) -> dict[str, Analysis]:
    if not isinstance(raw, dict) or raw.get("format") != FIXTURE_FORMAT:
        raise ValueError("unsupported format")
    if set(raw) - {"format", "analyses"}:
        raise ValueError("unexpected top-level keys")
    entries = raw.get("analyses")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("analyses must be a non-empty object")
    parsed: dict[str, Analysis] = {}
    for key, entry in entries.items():
        if not _FIXTURE_KEY.fullmatch(key):
            raise ValueError("fixture keys must be lowercase sha256 hex")
        parsed[key] = parse_strict_entry(entry)
    return parsed


def parse_strict_entry(entry: object) -> Analysis:
    """Validate one `{"sentiment", "topics"}` entry exactly and return it as an Analysis.

    Strict on purpose (fixtures and ground truth are trusted input): only the keys
    `sentiment` and `topics`, an exact sentiment value, exact topic values without duplicates,
    at most `MAX_TOPICS` of them. Raises ValueError whose message names counts and positions
    only, never the offending values. The confidence is always None.
    """
    if not isinstance(entry, Mapping):
        raise ValueError("entry must be an object")
    if set(entry) - _ENTRY_KEYS:
        raise ValueError(f"entry has {len(set(entry) - _ENTRY_KEYS)} unexpected key(s)")
    sentiment, topics = entry.get("sentiment"), entry.get("topics")
    if not isinstance(sentiment, str) or not isinstance(topics, list):
        raise ValueError("entry needs a sentiment string and a topics list")
    if len(topics) > MAX_TOPICS:
        raise ValueError(f"too many topics ({len(topics)}, at most {MAX_TOPICS})")
    parsed = [_exact_topic(t, position) for position, t in enumerate(topics)]
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate topics")
    return Analysis(sentiment=_exact_sentiment(sentiment), topics=tuple(parsed), confidence=None)


def _exact_sentiment(value: str) -> Sentiment:
    try:
        return Sentiment(value)
    except ValueError:
        raise ValueError("sentiment is not a valid value") from None


def _exact_topic(value: object, position: int) -> Topic:
    try:
        if isinstance(value, str):
            return Topic(value)
    except ValueError:
        pass
    raise ValueError(f"topic at position {position} is not a valid value")


def _fallback(item: ClassifyItem) -> Analysis:
    """Deterministic answer from the rating, or from word counts when there is no rating.

    Only complaint topics exist: positive reviews never get topics. A negative review without
    a keyword hit gets `other`; a neutral one gets none. Bump `FALLBACK_RULES_VERSION` when
    this function or the keyword lists change.
    """
    normalized = normalize_text(item.text)
    sentiment = _sentiment(normalized, item.rating)
    topics = _topics(normalized) if sentiment is not Sentiment.POSITIVE else []
    if not topics and sentiment is Sentiment.NEGATIVE:
        topics = [Topic.OTHER]
    return Analysis(sentiment=sentiment, topics=tuple(topics), confidence=FALLBACK_CONFIDENCE)


def _sentiment(normalized: str, rating: int | None) -> Sentiment:
    if rating is not None:
        if rating >= 4:
            return Sentiment.POSITIVE
        return Sentiment.NEGATIVE if rating <= 2 else Sentiment.NEUTRAL
    words = _WORD.findall(normalized)
    positive = sum(w in _POSITIVE_WORDS for w in words)
    negative = sum(w in _NEGATIVE_WORDS for w in words)
    if positive > negative:
        return Sentiment.POSITIVE
    return Sentiment.NEGATIVE if negative > positive else Sentiment.NEUTRAL


def _topics(normalized: str) -> list[Topic]:
    found = [topic for topic, pattern in _TOPIC_PATTERNS.items() if pattern.search(normalized)]
    return found[:MAX_TOPICS]
