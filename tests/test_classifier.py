import dataclasses

import pytest

from review_radar.classifier import (
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
)
from review_radar.models import Analysis, Sentiment, Usage


def test_error_carries_a_reason_and_message() -> None:
    exc = ClassifierError("rate_limited", "Too many requests.")

    assert exc.reason == "rate_limited"
    assert str(exc) == "Too many requests."


def test_error_message_defaults_to_the_reason() -> None:
    exc = ClassifierError("bad_output")

    assert exc.reason == "bad_output"
    assert str(exc) == "bad_output"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Rate Limited",
        "rate-limited",
        "a" * 65,
        "reason\nwith newline",
        "ключ",
        "sk-ant-X",
        " x",
    ],
)
def test_invalid_reason_becomes_unknown_error(bad: str) -> None:
    exc = ClassifierError(bad, "Some message.")

    assert exc.reason == "unknown_error"
    assert str(exc) == "Some message."


@pytest.mark.parametrize("good", ["a", "rate_limited", "x9_", "a" * 64, "0"])
def test_valid_reason_is_kept(good: str) -> None:
    assert ClassifierError(good).reason == good


def test_reason_with_a_trailing_newline_is_not_accepted() -> None:
    assert ClassifierError("abc\n").reason == "unknown_error"


def test_fatal_error_is_a_classifier_error() -> None:
    exc = FatalClassifierError("authentication_failed", "Key rejected.")

    assert isinstance(exc, ClassifierError)
    assert exc.reason == "authentication_failed"
    assert str(exc) == "Key rejected."
    assert exc.usage is None


def test_fatal_error_can_carry_the_usage_spent_before_it() -> None:
    usage = Usage(input_tokens=3, output_tokens=1)

    assert FatalClassifierError("billing_error", "x", usage=usage).usage == usage


def test_item_is_frozen() -> None:
    item = ClassifyItem(item_id="r1", text="x", rating=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        item.text = "y"  # type: ignore[misc]


def test_result_usage_defaults_to_none() -> None:
    analysis = Analysis(sentiment=Sentiment.NEUTRAL, topics=(), confidence=None)

    assert ClassifyResult(analyses={"r1": analysis}).usage is None
    assert ClassifyResult(analyses={}, usage=Usage(input_tokens=1, output_tokens=2)).usage == Usage(
        input_tokens=1, output_tokens=2
    )


def test_result_failures_default_to_empty_and_are_not_shared() -> None:
    first = ClassifyResult(analyses={})
    second = ClassifyResult(analyses={})

    assert first.failures == {}
    assert first.failures is not second.failures


def test_result_carries_per_item_failures() -> None:
    result = ClassifyResult(analyses={}, failures={"r2": "refusal"})

    assert result.failures == {"r2": "refusal"}
