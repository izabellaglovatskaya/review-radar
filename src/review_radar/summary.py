"""Daily summary: aggregates over a period and the Telegram message built from them.

Privacy rule: a summary holds counts and names from the closed topic list only. Review text can
contain harmful links and mentions, so it is never loaded into a `Summary` or a message.
"""

import datetime as dt
import html
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from review_radar.models import Sentiment, Topic
from review_radar.storage import load_period_analyses

MAX_MESSAGE_LENGTH = 4096  # Telegram's limit, counted in UTF-16 code units
TOP_TOPICS = 3

TOPIC_LABELS: dict[Topic, str] = {
    Topic.DELIVERY_DELAY: "Delivery delay",
    Topic.PRODUCT_QUALITY: "Product quality",
    Topic.CUSTOMER_SUPPORT: "Customer support",
    Topic.PRICING: "Pricing",
    Topic.BILLING: "Billing",
    Topic.APP_USABILITY: "App usability",
    Topic.PACKAGING: "Packaging",
    Topic.REFUND_PROCESS: "Refund process",
    Topic.OTHER: "Other",
}


class MessageTooLong(ValueError):
    """The rendered message is over Telegram's limit. Never carries the message."""


@dataclass(frozen=True)
class Period:
    """An inclusive range of review dates."""

    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("start must not be after end")

    @classmethod
    def day(cls, date: dt.date) -> Self:
        """Return the period covering one day."""
        return cls(date, date)

    @property
    def label(self) -> str:
        """The period as shown to people."""
        if self.start == self.end:
            return self.start.isoformat()
        return f"{self.start.isoformat()} to {self.end.isoformat()}"


@dataclass(frozen=True)
class TopicCount:
    """A complaint topic and how many negative reviews mention it."""

    label: str
    count: int


@dataclass(frozen=True)
class Summary:
    """Aggregates for one period. Sentiment counts cover analyzed reviews only."""

    period: Period
    total: int
    analyzed: int
    positive: int
    neutral: int
    negative: int
    top_topics: tuple[TopicCount, ...]

    @property
    def not_analyzed(self) -> int:
        """Reviews in the period that have no analysis for the current classifier."""
        return self.total - self.analyzed


def build_summary(
    conn: sqlite3.Connection, period: Period, *, provider: str, model: str, prompt_version: str
) -> Summary:
    """Aggregate the period's reviews using the analyses of one classifier key."""
    analyses = load_period_analyses(
        conn,
        period.start,
        period.end,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
    )
    done = [analysis for analysis in analyses if analysis is not None]
    sentiments = Counter(analysis.sentiment for analysis in done)
    topics = Counter(
        TOPIC_LABELS[topic]
        for analysis in done
        if analysis.sentiment is Sentiment.NEGATIVE
        for topic in analysis.topics
    )
    ranked = sorted(topics.items(), key=lambda item: (-item[1], item[0]))
    return Summary(
        period=period,
        total=len(analyses),
        analyzed=len(done),
        positive=sentiments[Sentiment.POSITIVE],
        neutral=sentiments[Sentiment.NEUTRAL],
        negative=sentiments[Sentiment.NEGATIVE],
        top_topics=tuple(TopicCount(label, count) for label, count in ranked[:TOP_TOPICS]),
    )


def percentages(counts: Sequence[int]) -> tuple[int, ...]:
    """Return whole percentages that sum to 100 (largest remainder), or zeros for no data."""
    total = sum(counts)
    if total == 0:
        return tuple(0 for _ in counts)
    shares = [count * 100 // total for count in counts]
    remainders = [count * 100 % total for count in counts]
    by_remainder = sorted(range(len(counts)), key=lambda index: (-remainders[index], index))
    for index in by_remainder[: 100 - sum(shares)]:
        shares[index] += 1
    return tuple(shares)


def render_message(summary: Summary, *, demo: bool = False) -> str:
    """Render the summary as Telegram HTML; every dynamic value is escaped.

    `demo` labels numbers that come from canned analyses, so they cannot pass for real ones.
    """
    prefix = "[DEMO] " if demo else ""
    positive, neutral, negative = percentages((summary.positive, summary.neutral, summary.negative))
    lines = [
        f"<b>{prefix}Review Radar: {_escape(summary.period.label)}</b>",
        f"Reviews: {summary.total}",
        f"Positive: {summary.positive} ({positive}%)",
        f"Neutral: {summary.neutral} ({neutral}%)",
        f"Negative: {summary.negative} ({negative}%)",
        "",
        "<b>Top complaint topics</b>",
    ]
    if summary.top_topics:
        lines += [
            f"{rank}. {_escape(topic.label)}: {topic.count}"
            for rank, topic in enumerate(summary.top_topics, start=1)
        ]
    else:
        lines.append("No complaint topics")
    if summary.not_analyzed:
        lines += ["", f"Not analyzed: {summary.not_analyzed} (shares cover analyzed reviews only)"]
    return "\n".join(lines)


def check_length(message: str) -> None:
    """Raise MessageTooLong when the message is over Telegram's limit."""
    length = len(message.encode("utf-16-le")) // 2
    if length > MAX_MESSAGE_LENGTH:
        raise MessageTooLong(f"The message has {length} characters, over {MAX_MESSAGE_LENGTH}.")


def _escape(value: str) -> str:
    return html.escape(value, quote=False)
