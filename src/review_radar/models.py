"""Domain models shared by ingest, analysis, storage and reporting."""

import datetime as dt
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
)

from review_radar.hashing import content_hash

logger = logging.getLogger(__name__)

MAX_TOPICS = 3
MAX_EXTERNAL_ID_LENGTH = 128
MAX_SOURCE_LENGTH = 64


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Topic(StrEnum):
    """Closed list of complaint topics."""

    DELIVERY_DELAY = "delivery_delay"
    PRODUCT_QUALITY = "product_quality"
    CUSTOMER_SUPPORT = "customer_support"
    PRICING = "pricing"
    BILLING = "billing"
    APP_USABILITY = "app_usability"
    PACKAGING = "packaging"
    REFUND_PROCESS = "refund_process"
    OTHER = "other"


def _lookup_topic(value: Any) -> Topic | None:
    """Return the Topic for a known value (case- and whitespace-insensitive), else None."""
    if isinstance(value, Topic):
        return value
    if isinstance(value, str):
        try:
            return Topic(value.strip().lower())
        except ValueError:
            return None
    return None


def _warn_unknown(value: Any) -> None:
    # Never log the value itself: model output can quote review text.
    detail = f"str, length {len(value)}" if isinstance(value, str) else type(value).__name__
    logger.warning("Unknown topic (%s) mapped to %r", detail, Topic.OTHER.value)


def parse_topic(value: Any) -> Topic:
    """Map a raw value to a Topic; unknown values become Topic.OTHER (with a warning).

    Known values match case- and whitespace-insensitively.
    """
    topic = _lookup_topic(value)
    if topic is None:
        _warn_unknown(value)
        return Topic.OTHER
    return topic


@dataclass(frozen=True)
class TopicNormalization:
    """Result of `normalize_topics`.

    `unknown` counts raw values that were not valid topics (each occurrence, repeats included);
    `truncated` counts unique topics dropped to stay within `MAX_TOPICS`.
    """

    topics: tuple[Topic, ...]
    unknown: int
    truncated: int


def normalize_topics(raw: Iterable[object]) -> TopicNormalization:
    """Apply the topic rules to raw classifier output.

    Unknown values become `other`, duplicates are removed keeping order, and more than
    `MAX_TOPICS` unique topics are truncated, preferring known topics over `other` (a warning
    is logged for each problem, never with the raw value). Every classifier must pass its raw
    topics through this function and report the counts in `ClassifyResult`.
    """
    parsed: list[Topic] = []
    unknown = 0
    for value in raw:
        topic = _lookup_topic(value)
        if topic is None:
            unknown += 1
            _warn_unknown(value)
            topic = Topic.OTHER
        parsed.append(topic)
    unique = tuple(dict.fromkeys(parsed))
    truncated = 0
    if len(unique) > MAX_TOPICS:
        logger.warning("Truncating %d topics to %d", len(unique), MAX_TOPICS)
        known = [t for t in unique if t is not Topic.OTHER]
        # After dedup there is at most one OTHER, so more than MAX_TOPICS unique
        # topics means at least MAX_TOPICS known ones: OTHER never gets a slot.
        kept = tuple(known[:MAX_TOPICS])
        truncated = len(unique) - len(kept)
        unique = kept
    return TopicNormalization(topics=unique, unknown=unknown, truncated=truncated)


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    external_id: Annotated[str, Field(max_length=MAX_EXTERNAL_ID_LENGTH)] | None
    review_date: dt.date
    rating: Annotated[StrictInt, Field(ge=1, le=5)] | None
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    source: Annotated[str, Field(max_length=MAX_SOURCE_LENGTH)] | None

    @field_validator("external_id", "source", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value


class Analysis(BaseModel):
    """Classification result for one review.

    Topics are normalized with `parse_topic` (unknown -> other), deduplicated
    preserving order, and truncated to 3 with a warning. Truncating (instead of
    rejecting) keeps one over-eager model answer from failing a batch. Without
    truncation the original order is kept; when truncating, known topics win (in
    original order) and `other` is kept only if a slot is left.
    """

    model_config = ConfigDict(frozen=True)

    sentiment: Sentiment
    topics: tuple[Topic, ...]
    confidence: Annotated[float, Field(ge=0, le=1)] | None

    @field_validator("topics", mode="before")
    @classmethod
    def _normalize_topics(cls, value: Any) -> tuple[Topic, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("topics must be a list or tuple")
        return normalize_topics(value).topics


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]

    def __add__(self, other: object) -> Self:
        if not isinstance(other, Usage):
            return NotImplemented
        return type(self)(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


UNKNOWN_SOURCE = "unknown"


def dedup_key(review: Review) -> str:
    """Return the idempotency key.

    With an external id: the id scoped by source (casefolded; a missing source counts as
    `UNKNOWN_SOURCE`), encoded as a JSON array so no id/source pair can collide with another.
    Without one: a hash of date + normalized text + source.
    """
    if review.external_id is not None:
        source = (review.source or UNKNOWN_SOURCE).casefold()
        return "id:" + json.dumps([source, review.external_id], separators=(",", ":"))
    return "hash:" + content_hash(review.review_date, review.text, review.source)
