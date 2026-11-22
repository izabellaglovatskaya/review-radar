import dataclasses

import pytest

from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.models import Analysis, Sentiment, Usage


def test_error_carries_a_reason_and_message() -> None:
    exc = ClassifierError("rate_limited", "Too many requests.")

    assert exc.reason == "rate_limited"
    assert str(exc) == "Too many requests."


def test_error_message_defaults_to_the_reason() -> None:
    exc = ClassifierError("bad_output")

    assert exc.reason == "bad_output"
    assert str(exc) == "bad_output"


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
