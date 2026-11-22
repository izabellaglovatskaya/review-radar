"""Opt-in test that calls the real Anthropic API (costs a fraction of a cent).

Run it explicitly with `uv run pytest -m live`; the default run deselects it. It is skipped
when no API key is configured. The reviews are fictional; the key is never printed.
"""

import pytest

from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.classifier import ClassifyItem
from review_radar.config import Settings
from review_radar.models import Analysis


@pytest.mark.live
def test_two_fictional_reviews_are_classified_by_the_default_model() -> None:
    settings = Settings()
    if settings.anthropic_api_key is None:
        pytest.skip("ANTHROPIC_API_KEY is not set")
    classifier = AnthropicClassifier(
        api_key=settings.anthropic_api_key, model=settings.anthropic_model
    )
    items = [
        ClassifyItem("r1", "Lovely service, the order arrived a day early.", 5),
        ClassifyItem("r2", "The parcel was two weeks late and nobody answered my emails.", 1),
    ]

    outcome = classifier.classify(items)

    assert outcome.failures == {}
    assert set(outcome.analyses) == {"r1", "r2"}
    assert all(isinstance(a, Analysis) for a in outcome.analyses.values())
    assert outcome.usage is not None
    assert outcome.usage.input_tokens > 0
    assert outcome.usage.output_tokens > 0
