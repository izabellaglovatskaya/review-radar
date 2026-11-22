"""The `Classifier` protocol: turns review texts into sentiment and topics.

Privacy rule: a `ClassifierError` message never contains review text, model output or secrets.
The closed vocabulary (sentiment positive/neutral/negative, nine topics, at most three per
review) is enforced by `models.Analysis`; every classifier returns `Analysis` objects, so the
rules hold for all.
"""

import contextlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    dropped while building the analyses; `unexpected_results` counts ids that were returned
    but not asked for (they are never in `analyses`). All counters are per call.

    `failures` maps the `item_id` of an item that could not be classified to a reason code;
    such an item is not in `analyses`. An asked-for id in neither mapping counts as missing.
    """

    analyses: Mapping[str, Analysis]
    usage: Usage | None = None
    fallback: int = 0
    unknown_topics: int = 0
    truncated_topics: int = 0
    failures: Mapping[str, str] = field(default_factory=dict)
    unexpected_results: int = 0


UNKNOWN_REASON = "unknown_error"
_REASON_PATTERN = re.compile(r"[a-z0-9_]{1,64}")


def safe_reason(reason: str) -> str:
    """Return `reason` if it is a snake_case code (`[a-z0-9_]{1,64}`), else `unknown_error`."""
    if isinstance(reason, str) and _REASON_PATTERN.fullmatch(reason):
        return reason
    return UNKNOWN_REASON


class ClassifierError(Exception):
    """The whole batch failed. `reason` is a short snake_case code; no review text in messages.

    A reason that is not `[a-z0-9_]{1,64}` is replaced by `unknown_error`, so a reason is always
    safe to print.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = safe_reason(reason)


class FatalClassifierError(ClassifierError):
    """An error that no retry or later batch can fix (bad key, no credit, unknown model).

    The analysis service stops at once. `usage` is what earlier calls of the same batch already
    cost, if anything; `partial` is the result gathered before the error (valid analyses, item
    failures and counters), which the service still saves because it was paid for.
    """

    def __init__(
        self,
        reason: str,
        message: str = "",
        *,
        usage: Usage | None = None,
        partial: "ClassifyResult | None" = None,
    ) -> None:
        super().__init__(reason, message)
        self.usage = usage
        if partial is not None:
            attach_partial(self, partial)

    @property
    def partial(self) -> "ClassifyResult | None":
        return get_partial(self)


PARTIAL_ATTRIBUTE = "review_radar_partial"
TRANSIENT_REASONS = frozenset({"rate_limited", "server_error", "network_error"})


def attach_partial(exc: BaseException, result: "ClassifyResult") -> None:
    """Attach what a failed `classify` call had already produced (and paid for) to `exc`."""
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, PARTIAL_ATTRIBUTE, result)


def get_partial(exc: BaseException) -> "ClassifyResult | None":
    """Return the partial result attached to `exc`, if any."""
    partial = getattr(exc, PARTIAL_ATTRIBUTE, None)
    return partial if isinstance(partial, ClassifyResult) else None


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
