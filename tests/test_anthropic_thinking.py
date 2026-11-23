"""The `thinking` request setting depends on the model; the prompt and the cache key do not."""

import pytest
from pydantic import SecretStr

from anthropic_fakes import FAKE_KEY, FakeClient, make_message, result
from review_radar.anthropic_classifier import (
    PROMPT_VERSION,
    SCHEMA,
    SYSTEM_PROMPT,
    AnthropicClassifier,
    compute_prompt_version,
    thinking_config,
)
from review_radar.classifier import ClassifyItem

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


def send_one(model: str) -> dict:
    """Classify one item with a fake client and return the request that was sent."""
    client = FakeClient(make_message([result("r1")]))
    classifier = AnthropicClassifier(api_key=SecretStr(FAKE_KEY), model=model, client=client)

    classifier.classify([ClassifyItem(item_id="r1", text="fictional review", rating=3)])

    assert len(client.calls) == 1
    return client.calls[0]


class TestThinkingConfig:
    def test_sonnet_uses_between_tools(self) -> None:
        assert thinking_config(SONNET) == {"type": "between_tools"}

    def test_haiku_disables_thinking(self) -> None:
        assert thinking_config(HAIKU) == {"type": "disabled"}

    @pytest.mark.parametrize("model", ["some-other-model", "claude-sonnet-5-5-preview", "", "x"])
    def test_any_other_model_disables_thinking(self, model: str) -> None:
        assert thinking_config(model) == {"type": "disabled"}

    def test_the_result_is_a_fresh_dict_each_time(self) -> None:
        first = thinking_config(SONNET)
        first["type"] = "mutated"

        assert thinking_config(SONNET) == {"type": "between_tools"}


class TestRequest:
    def test_a_sonnet_request_uses_between_tools(self) -> None:
        assert send_one(SONNET)["thinking"] == {"type": "between_tools"}

    def test_a_haiku_request_disables_thinking(self) -> None:
        assert send_one(HAIKU)["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize("model", [HAIKU, SONNET])
    def test_effort_stays_low(self, model: str) -> None:
        assert send_one(model)["output_config"]["effort"] == "low"

    @pytest.mark.parametrize("model", [HAIKU, SONNET])
    def test_the_request_names_the_model(self, model: str) -> None:
        assert send_one(model)["model"] == model


class TestPromptVersionIsModelIndependent:
    def test_the_prompt_version_is_computed_from_prompt_and_schema_only(self) -> None:
        assert compute_prompt_version(SYSTEM_PROMPT, SCHEMA) == PROMPT_VERSION

    def test_the_classifier_reports_the_same_version_for_both_models(self) -> None:
        versions = {
            AnthropicClassifier(
                api_key=SecretStr(FAKE_KEY), model=model, client=FakeClient()
            ).prompt_version
            for model in (HAIKU, SONNET)
        }

        assert versions == {PROMPT_VERSION}

    def test_the_system_prompt_and_schema_do_not_differ_by_model(self) -> None:
        haiku, sonnet = send_one(HAIKU), send_one(SONNET)

        assert haiku["system"] == sonnet["system"] == SYSTEM_PROMPT
        assert haiku["output_config"]["format"] == sonnet["output_config"]["format"]
        assert haiku["output_config"]["format"]["schema"] == SCHEMA
        assert haiku["messages"] == sonnet["messages"]
