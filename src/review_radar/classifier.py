"""The `Classifier` protocol: turns review texts into sentiment and topics.

Privacy rule: a `ClassifierError` message never contains review text. The closed vocabulary
(sentiment positive/neutral/negative, nine topics, at most three per review) is enforced by
`models.Analysis`; every classifier returns `Analysis` objects, so the rules hold for all.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from review_radar.models import Analysis, Usage


@dataclass(frozen=True)
class ClassifyItem:
    """One text to classify; `item_id` is a short id (`r1`, `r2`, ...) local to a batch."""

    item_id: str
    text: str
    rating: int | None


@dataclass(frozen=True)
class ClassifyResult:
    """Analyses keyed by `item_id`, plus token usage and anomaly counters.

    `fallback` counts answers produced by built-in rules instead of the real source;
    `unknown_topics` and `truncated_topics` count topics that were mapped to `other` or
    dropped while building the analyses. All counters are per call.
    """

    analyses: Mapping[str, Analysis]
    usage: Usage | None = None
    fallback: int = 0
    unknown_topics: int = 0
    truncated_topics: int = 0


class ClassifierError(Exception):
    """The whole batch failed. `reason` is a short snake_case code; no review text in messages."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


class Classifier(Protocol):
    """Anything that can classify a batch of items; also identifies itself for the cache.

    Classifiers that get raw topic values (e.g. from a model) must normalize them with
    `models.normalize_topics` and report the resulting `unknown` and `truncated` counts in
    `ClassifyResult.unknown_topics` / `truncated_topics`, so anomalies stay visible.
    """

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def prompt_version(self) -> str: ...

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult: ...
