"""Domain models shared by ingest, analysis, storage and reporting."""

import datetime as dt
import logging
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


def parse_topic(value: Any) -> Topic:
    """Map a raw value to a Topic; unknown values become Topic.OTHER (with a warning).

    Known values match case- and whitespace-insensitively.
    """
    if isinstance(value, Topic):
        return value
    if isinstance(value, str):
        try:
            return Topic(value.strip().lower())
        except ValueError:
            pass
    logger.warning("Unknown topic %r mapped to %r", value, Topic.OTHER.value)
    return Topic.OTHER


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
        if isinstance(value, str) or not hasattr(value, "__iter__"):
            raise ValueError("topics must be a sequence")
        unique = tuple(dict.fromkeys(parse_topic(v) for v in value))
        if len(unique) > MAX_TOPICS:
            logger.warning("Truncating %d topics to %d", len(unique), MAX_TOPICS)
            known = [t for t in unique if t is not Topic.OTHER]
            # After dedup there is at most one OTHER, so more than MAX_TOPICS unique
            # topics means at least MAX_TOPICS known ones: OTHER never gets a slot.
            unique = tuple(known[:MAX_TOPICS])
        return unique


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


def dedup_key(review: Review) -> str:
    """Return the idempotency key: the external id if set, else a content hash."""
    if review.external_id is not None:
        return f"id:{review.external_id}"
    return "hash:" + content_hash(review.review_date, review.text, review.source)
