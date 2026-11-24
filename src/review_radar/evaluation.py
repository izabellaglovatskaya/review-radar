"""Evaluation: score classifier answers against ground truth and guard what it costs.

Scoring is pure. A missing prediction is an error (it is never dropped), and a metric with a zero
denominator is `None`, never 0 or 1. Reports hold ids, labels and numbers only, never review text.

`GuardedClient` enforces a spending limit in front of every real API attempt, including retries
and batch splits: a call whose worst case could pass the limit is not made. With a `SpendLedger`
the worst case is written to the database before the call and replaced by the real cost after
it. A crash, a timeout or an unreadable response therefore leaves the worst case on the books:
the total can be too high, never too low. An error response from the API releases it.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import Message

from review_radar.anthropic_classifier import MessagesClient
from review_radar.classifier import FatalClassifierError
from review_radar.config import PROJECT_ROOT
from review_radar.models import Sentiment, Topic, Usage
from review_radar.pricing import PRICES_PER_MTOK, estimate_cost
from review_radar.storage import StorageError, reserve_spend, settle_spend

logger = logging.getLogger(__name__)

INPUT_TOKENS_PER_TEXT = 135  # measured on the first live run
OUTPUT_TOKENS_PER_TEXT = 36  # measured on the first live run
SAFETY_MARGIN = Decimal("1.5")


class EvalError(Exception):
    """The evaluation cannot run on this input. Never carries review text."""


@dataclass(frozen=True)
class Dataset:
    """A reviews CSV and its ground truth."""

    csv: Path
    truth: Path


DATASETS = {
    "regular": Dataset(
        csv=PROJECT_ROOT / "data" / "sample" / "reviews.csv",
        truth=PROJECT_ROOT / "data" / "sample" / "ground_truth.json",
    ),
    "hard": Dataset(
        csv=PROJECT_ROOT / "data" / "eval" / "hard_reviews.csv",
        truth=PROJECT_ROOT / "data" / "eval" / "hard_ground_truth.json",
    ),
}


@dataclass(frozen=True)
class Truth:
    """The expected labels of one review."""

    sentiment: Sentiment
    topics: frozenset[Topic]
    category: str | None


@dataclass(frozen=True)
class Prediction:
    """A classifier's labels for one review."""

    sentiment: Sentiment
    topics: frozenset[Topic]


@dataclass(frozen=True)
class ClassMetric:
    """Precision and recall of one sentiment class; `None` when undefined."""

    precision: float | None
    recall: float | None
    support: int


@dataclass(frozen=True)
class Scores:
    """Quality on one set of reviews. Rates are fractions from 0 to 1."""

    n: int
    predicted: int
    missing: int
    sentiment_accuracy: float
    sentiment: dict[str, ClassMetric]
    macro_precision: float | None
    macro_recall: float | None
    topic_precision: float | None
    topic_recall: float | None
    exact_topics: float


@dataclass(frozen=True)
class ErrorRow:
    """One review the classifier got wrong or did not answer (labels only, no text)."""

    review_id: str
    category: str | None
    true_sentiment: Sentiment
    predicted_sentiment: Sentiment | None
    true_topics: tuple[str, ...]
    predicted_topics: tuple[str, ...] | None


@dataclass(frozen=True)
class CostEstimate:
    """What classifying `texts` distinct texts is expected to cost on `model`."""

    model: str
    texts: int
    input_tokens: int
    output_tokens: int
    expected_usd: Decimal
    upper_usd: Decimal


@dataclass(frozen=True)
class ModelResult:
    """One model's quality per dataset and category, and its measured cost."""

    model: str
    dataset_scores: dict[str, Scores]
    category_scores: dict[str, Scores]
    usage: Usage
    cost_usd: Decimal
    reviews: int
    cost_per_1000: Decimal
    failed: int


@dataclass(frozen=True)
class EvalReport:
    """A whole evaluation; `errors` maps a model name to its wrong reviews."""

    models: tuple[ModelResult, ...]
    mode: str = "live"
    errors: Mapping[str, tuple[ErrorRow, ...]] = field(default_factory=dict)
    incomplete: str | None = None


def load_truth(path: Path) -> dict[str, Truth]:
    """Read and strictly validate a ground truth file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise EvalError(f"Cannot read the ground truth file {path.name}.") from None
    reviews = raw.get("reviews") if isinstance(raw, dict) else None
    if not isinstance(reviews, dict) or not reviews:
        raise EvalError(f"{path.name} has no reviews.")
    try:
        return {review_id: _truth(entry) for review_id, entry in reviews.items()}
    except (KeyError, TypeError, ValueError):
        raise EvalError(f"{path.name} has an invalid entry.") from None


def _truth(entry: Any) -> Truth:
    if not isinstance(entry["topics"], list) or not isinstance(entry["sentiment"], str):
        raise ValueError("bad entry")
    category = entry.get("category")
    if category is not None and not isinstance(category, str):
        raise ValueError("bad category")
    return Truth(
        sentiment=Sentiment(entry["sentiment"]),
        topics=frozenset(Topic(topic) for topic in entry["topics"]),
        category=category,
    )


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _mean(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def score(truth: Mapping[str, Truth], predictions: Mapping[str, Prediction]) -> Scores:
    """Score the ids in `truth`; a missing prediction counts as wrong."""
    if not truth:
        raise EvalError("There is nothing to score.")
    answered = {key: predictions[key] for key in truth if key in predictions}
    correct = sum(answered[key].sentiment is truth[key].sentiment for key in answered)
    classes = {
        sentiment.value: _class_metric(sentiment, truth, answered) for sentiment in Sentiment
    }
    found = sum(len(answered[key].topics & truth[key].topics) for key in answered)
    predicted_labels = sum(len(item.topics) for item in answered.values())
    true_labels = sum(len(item.topics) for item in truth.values())
    exact = sum(answered[key].topics == truth[key].topics for key in answered)
    return Scores(
        n=len(truth),
        predicted=len(answered),
        missing=len(truth) - len(answered),
        sentiment_accuracy=correct / len(truth),
        sentiment=classes,
        macro_precision=_mean(_macro_precision(metric) for metric in classes.values()),
        macro_recall=_mean(metric.recall for metric in classes.values()),
        topic_precision=_ratio(found, predicted_labels),
        topic_recall=_ratio(found, true_labels),
        exact_topics=exact / len(truth),
    )


def _macro_precision(metric: ClassMetric) -> float | None:
    """A class with support that is never predicted counts as 0, so ignoring it is not rewarded."""
    if metric.precision is None and metric.support:
        return 0.0
    return metric.precision


def _class_metric(
    sentiment: Sentiment, truth: Mapping[str, Truth], answered: Mapping[str, Prediction]
) -> ClassMetric:
    support = sum(item.sentiment is sentiment for item in truth.values())
    chosen = [key for key, item in answered.items() if item.sentiment is sentiment]
    hits = sum(truth[key].sentiment is sentiment for key in chosen)
    return ClassMetric(_ratio(hits, len(chosen)), _ratio(hits, support), support)


def list_errors(
    truth: Mapping[str, Truth], predictions: Mapping[str, Prediction]
) -> list[ErrorRow]:
    """Every wrong or unanswered review, ordered by id."""
    rows = []
    for review_id in sorted(truth):
        expected = truth[review_id]
        answer = predictions.get(review_id)
        if answer == Prediction(expected.sentiment, expected.topics):
            continue
        rows.append(
            ErrorRow(
                review_id=review_id,
                category=expected.category,
                true_sentiment=expected.sentiment,
                predicted_sentiment=None if answer is None else answer.sentiment,
                true_topics=_names(expected.topics),
                predicted_topics=None if answer is None else _names(answer.topics),
            )
        )
    return rows


def _names(topics: Iterable[Topic]) -> tuple[str, ...]:
    return tuple(sorted(topic.value for topic in topics))


def scores_by_category(
    truth: Mapping[str, Truth], predictions: Mapping[str, Prediction]
) -> dict[str, Scores]:
    """Score each truth category separately; reviews without a category form no group."""
    groups: dict[str, dict[str, Truth]] = defaultdict(dict)
    for review_id, item in truth.items():
        if item.category is not None:
            groups[item.category][review_id] = item
    return {name: score(group, predictions) for name, group in groups.items()}


def estimate_cost_for(model: str, texts: Sequence[str]) -> CostEstimate:
    """Estimate from the measured tokens per text; `upper_usd` adds the safety margin."""
    usage = Usage(
        input_tokens=len(texts) * INPUT_TOKENS_PER_TEXT,
        output_tokens=len(texts) * OUTPUT_TOKENS_PER_TEXT,
    )
    expected = estimate_cost(model, usage)
    if expected is None:
        raise EvalError(f"No price is known for the model {model}.")
    return CostEstimate(
        model=model,
        texts=len(texts),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        expected_usd=expected,
        upper_usd=expected * SAFETY_MARGIN,
    )


class SpendLedger:
    """Writes reservations and real costs to the database, one transaction per call."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def reserve(self, model: str, amount: Decimal) -> int:
        return reserve_spend(self._conn, model, amount)

    def settle(self, entry: int, amount: Decimal, usage: Usage | None = None) -> None:
        settle_spend(
            self._conn,
            entry,
            amount,
            input_tokens=0 if usage is None else usage.input_tokens,
            output_tokens=0 if usage is None else usage.output_tokens,
        )


@dataclass
class SpendGuard:
    """A spending limit in USD shared by every client of one evaluation."""

    limit: Decimal
    spent: Decimal = Decimal(0)
    ledger: SpendLedger | None = None

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must not be negative")

    def reserve(self, model: str, amount: Decimal) -> int | None:
        """Book the worst case before a call; a call that cannot be booked is not made."""
        entry = None
        if self.ledger is not None:
            try:
                entry = self.ledger.reserve(model, amount)
            except StorageError:
                raise FatalClassifierError(
                    "spend_ledger_failed", "The spend could not be recorded; no call was made."
                ) from None
        self.spent += amount
        return entry

    def settle(
        self, entry: int | None, reserved: Decimal, actual: Decimal, usage: Usage | None = None
    ) -> None:
        """Replace the reservation; if that fails it stays booked, so the total stays safe."""
        if self.ledger is not None and entry is not None:
            try:
                self.ledger.settle(entry, actual, usage)
            except StorageError as exc:
                logger.warning("The reserved spend stays on the books: %s", exc)
                return
        self.spent += actual - reserved


class GuardedClient:
    """Wraps an Anthropic client: refuses any call whose worst case could pass the limit."""

    def __init__(
        self, client: anthropic.Anthropic | MessagesClient, guard: SpendGuard, model: str
    ) -> None:
        if model not in PRICES_PER_MTOK:
            raise EvalError(f"No price is known for the model {model}.")
        self._client = client
        self._guard = guard
        self._model = model

    @property
    def messages(self) -> "GuardedClient":
        return self

    def create(self, **kwargs: Any) -> Message:
        """Check the limit, forward the call unchanged and count what it really used."""
        worst: Decimal = _price(
            self._model,
            Usage(input_tokens=_input_bound(kwargs), output_tokens=int(kwargs["max_tokens"])),
        )
        if self._guard.spent + worst > self._guard.limit:
            raise FatalClassifierError(
                "budget_exceeded", "The evaluation spend limit stopped this call."
            )
        entry: int | None = self._guard.reserve(self._model, worst)
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError:  # the API answered with an error: nothing to bill
            self._guard.settle(entry, worst, Decimal(0))
            raise
        # Any other exception (timeout, broken connection, unreadable reply) may have been
        # billed, so the reservation stays.
        reported: Any = getattr(response, "usage", None)
        if reported is None:
            logger.warning("A response had no usage; its worst-case cost stays on the books")
            self._guard.settle(entry, worst, worst)
            return response
        usage: Usage = Usage(
            input_tokens=reported.input_tokens, output_tokens=reported.output_tokens
        )
        self._guard.settle(entry, worst, _price(self._model, usage), usage)
        return response


def _price(model: str, usage: Usage) -> Decimal:
    cost = estimate_cost(model, usage)
    if cost is None:  # pragma: no cover (GuardedClient checks the model first)
        raise EvalError(f"No price is known for the model {model}.")
    return cost


def _input_bound(request: Mapping[str, Any]) -> int:
    """An upper bound for the input tokens: one token per UTF-8 byte, schema included."""
    total = _text_bytes(request.get("system", ""))
    for message in request.get("messages", []):
        total += _text_bytes(message.get("content", ""))
    for name in ("output_config", "tools"):
        if name in request:
            total += len(json.dumps(request[name]).encode("utf-8"))
    return total


def _text_bytes(content: Any) -> int:
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return sum(len(block.get("text", "").encode("utf-8")) for block in content)


def render_markdown(report: EvalReport, *, include_errors: bool = True) -> str:
    """The report as Markdown tables; every value is escaped for a table cell."""
    lines = ["# Model evaluation", ""]
    if report.incomplete is not None:
        lines += [f"**INCOMPLETE: {_cell(report.incomplete)}**", ""]
    lines += [
        f"Mode: {_cell(report.mode)}. Rates cover every review; a missing answer counts as wrong.",
        "",
        "| Model | Dataset | Reviews | Sentiment accuracy | Macro precision | Macro recall "
        "| Topic precision | Topic recall | Exact topics |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report.models:
        for name, scores in result.dataset_scores.items():
            lines.append(
                f"| {_cell(result.model)} | {_cell(name)} | {scores.n} | {_metrics(scores)} |"
            )
    lines += [
        "",
        "## Cost",
        "",
        "Measured per model over every review it analyzed, across the datasets.",
        "",
        "| Model | Reviews analyzed | Input tokens | Output tokens | Cost "
        "| Cost per 1,000 reviews |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in report.models:
        lines.append(
            f"| {_cell(result.model)} | {result.reviews} | {result.usage.input_tokens} "
            f"| {result.usage.output_tokens} "
            f"| ${result.cost_usd.quantize(Decimal('0.000001'), ROUND_HALF_UP)} "
            f"| ${result.cost_per_1000.quantize(Decimal('0.0001'), ROUND_HALF_UP)} |"
        )
    if any(result.category_scores for result in report.models):
        lines += [
            "",
            "## By category",
            "",
            "| Model | Category | Reviews | Sentiment accuracy | Macro precision | Macro recall "
            "| Topic precision | Topic recall | Exact topics |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for result in report.models:
            for name, scores in result.category_scores.items():
                lines.append(
                    f"| {_cell(result.model)} | {_cell(name)} | {scores.n} | {_metrics(scores)} |"
                )
    if include_errors:
        lines += _error_lines(report)
    return "\n".join(lines) + "\n"


def _error_lines(report: EvalReport) -> list[str]:
    if not any(report.errors.values()):
        return []
    lines = [
        "",
        "## Errors",
        "",
        "| Model | Review | Category | Expected | Predicted |",
        "|---|---|---|---|---|",
    ]
    for model, rows in report.errors.items():
        for row in rows:
            expected = _labels(row.true_sentiment, row.true_topics)
            predicted = _labels(row.predicted_sentiment, row.predicted_topics)
            lines.append(
                f"| {_cell(model)} | {_cell(row.review_id)} | {_cell(row.category or 'regular')} "
                f"| {_cell(expected)} | {_cell(predicted)} |"
            )
    return lines


def _labels(sentiment: Sentiment | None, topics: tuple[str, ...] | None) -> str:
    if sentiment is None or topics is None:
        return "no answer"
    return f"{sentiment.value}: {', '.join(topics) or 'no topics'}"


def _metrics(scores: Scores) -> str:
    values = (
        scores.sentiment_accuracy,
        scores.macro_precision,
        scores.macro_recall,
        scores.topic_precision,
        scores.topic_recall,
        scores.exact_topics,
    )
    return " | ".join(_percent(value) for value in values)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
