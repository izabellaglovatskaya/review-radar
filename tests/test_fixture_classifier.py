import json
import logging
import re
from pathlib import Path

import pytest

from review_radar import fixture_classifier
from review_radar.classifier import ClassifierError, ClassifyItem
from review_radar.fixture_classifier import FixtureClassifier, parse_strict_entry
from review_radar.hashing import text_hash
from review_radar.ingest import read_reviews
from review_radar.models import Analysis, Sentiment, Topic

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_FILE = ROOT / "fixtures" / "demo_analyses.json"
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews.csv"


KEY = "a" * 64


def item(text: str, rating: int | None = None, item_id: str = "r1") -> ClassifyItem:
    return ClassifyItem(item_id=item_id, text=text, rating=rating)


def one(text: str, rating: int | None = None) -> Analysis:
    result = FixtureClassifier({}).classify([item(text, rating)])
    return result.analyses["r1"]


def write_fixture(tmp_path: Path, payload: object, name: str = "fixtures.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def entry(sentiment: str = "negative", topics: list[str] | None = None) -> dict[str, object]:
    return {"sentiment": sentiment, "topics": ["pricing"] if topics is None else topics}


class TestIdentity:
    def test_provider_model_prompt_version(self) -> None:
        classifier = FixtureClassifier({})

        assert classifier.provider == "fixture"
        assert classifier.model == "demo"
        assert re.fullmatch(r"fixture-1\+[0-9a-f]{8}", classifier.prompt_version)


class TestFixtureHits:
    def test_hit_returns_the_canned_analysis(self) -> None:
        canned = Analysis(sentiment=Sentiment.NEGATIVE, topics=(Topic.BILLING,), confidence=None)
        classifier = FixtureClassifier({text_hash("Charged twice"): canned})

        result = classifier.classify([item("Charged twice", rating=5)])

        assert result.analyses == {"r1": canned}
        assert result.usage is None

    def test_lookup_uses_the_normalized_text(self) -> None:
        canned = Analysis(sentiment=Sentiment.NEGATIVE, topics=(Topic.BILLING,), confidence=None)
        classifier = FixtureClassifier({text_hash("charged twice"): canned})

        result = classifier.classify([item("  CHARGED   Twice ")])

        assert result.analyses["r1"] is canned

    def test_results_are_keyed_by_item_id(self) -> None:
        items = [item("a", 5, "r1"), item("b", 1, "r2"), item("c", 3, "r3")]

        result = FixtureClassifier({}).classify(items)

        assert list(result.analyses) == ["r1", "r2", "r3"]

    def test_empty_batch(self) -> None:
        assert FixtureClassifier({}).classify([]).analyses == {}


class TestFallbackSentiment:
    @pytest.mark.parametrize(
        ("rating", "expected"),
        [
            (5, Sentiment.POSITIVE),
            (4, Sentiment.POSITIVE),
            (3, Sentiment.NEUTRAL),
            (2, Sentiment.NEGATIVE),
            (1, Sentiment.NEGATIVE),
        ],
    )
    def test_rating_decides(self, rating: int, expected: Sentiment) -> None:
        assert one("whatever", rating).sentiment is expected

    def test_rating_wins_over_words(self) -> None:
        assert one("terrible awful", 5).sentiment is Sentiment.POSITIVE

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Great product, I love it", Sentiment.POSITIVE),
            ("Terrible, awful and broken", Sentiment.NEGATIVE),
            ("It arrived on Tuesday", Sentiment.NEUTRAL),
            ("Good but terrible", Sentiment.NEUTRAL),
            ("GREAT, excellent; bad", Sentiment.POSITIVE),
            ("", Sentiment.NEUTRAL),
        ],
    )
    def test_words_decide_without_a_rating(self, text: str, expected: Sentiment) -> None:
        assert one(text).sentiment is expected

    def test_words_match_whole_words_only(self) -> None:
        assert one("a goodness badge").sentiment is Sentiment.NEUTRAL


class TestFallbackTopics:
    @pytest.mark.parametrize(
        ("text", "topic"),
        [
            ("It was late", Topic.DELIVERY_DELAY),
            ("Still waiting for it", Topic.DELIVERY_DELAY),
            ("Arrived broken", Topic.PRODUCT_QUALITY),
            ("It fell apart", Topic.PRODUCT_QUALITY),
            ("Support was rude", Topic.CUSTOMER_SUPPORT),
            ("Way too expensive", Topic.PRICING),
            ("I was charged twice", Topic.BILLING),
            ("The app keeps crashing, login fails", Topic.APP_USABILITY),
            ("The box was crushed", Topic.PACKAGING),
            ("I want my money back", Topic.REFUND_PROCESS),
        ],
    )
    def test_keywords_map_to_topics(self, text: str, topic: Topic) -> None:
        assert topic in one(text, 1).topics

    def test_ordered_by_topic_enum_order(self) -> None:
        analysis = one("refund please, the price is high and it was late", 1)

        assert analysis.topics == (Topic.DELIVERY_DELAY, Topic.PRICING, Topic.REFUND_PROCESS)

    def test_at_most_three_topics(self) -> None:
        text = "late, broken, support ignored me, expensive, charged, app, box, refund"

        analysis = one(text, 1)

        assert analysis.topics == (
            Topic.DELIVERY_DELAY,
            Topic.PRODUCT_QUALITY,
            Topic.CUSTOMER_SUPPORT,
        )

    def test_negative_without_hits_is_other(self) -> None:
        assert one("Meh.", 1).topics == (Topic.OTHER,)

    def test_neutral_without_hits_has_no_topics(self) -> None:
        assert one("Meh.", 3).topics == ()

    def test_neutral_with_hits_has_topics(self) -> None:
        assert one("A bit late.", 3).topics == (Topic.DELIVERY_DELAY,)

    def test_positive_never_has_topics(self) -> None:
        assert one("Late and broken but great", 5).topics == ()
        assert one("Late and broken but great and love it, excellent").topics == ()

    def test_keywords_match_whole_words(self) -> None:
        assert one("Happy with the apple", 3).topics == ()

    def test_matching_is_case_insensitive_and_whitespace_tolerant(self) -> None:
        assert one("STILL   WAITING", 1).topics == (Topic.DELIVERY_DELAY,)


class TestFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "I will double-check it",
            "Used it for weeks and it is fine",
            "I will return to this shop",
            "Happy with the apple",
            "Nice box of chocolates",
            "I called to charge my phone",
        ],
    )
    def test_risky_single_words_do_not_trigger_topics(self, text: str) -> None:
        assert one(text, 3).topics == ()

    @pytest.mark.parametrize(
        ("text", "topic"),
        [
            ("Charged twice for one order", Topic.BILLING),
            ("They double charged me", Topic.BILLING),
            ("Overcharged and no invoice", Topic.BILLING),
            ("The delivery took weeks", Topic.DELIVERY_DELAY),
            ("The box was damaged", Topic.PACKAGING),
            ("A crushed box on arrival", Topic.PACKAGING),
            ("The app crashes constantly", Topic.APP_USABILITY),
            ("I cannot log in", Topic.APP_USABILITY),
            ("Where is my return label", Topic.REFUND_PROCESS),
            ("I returned it a month ago", Topic.REFUND_PROCESS),
        ],
    )
    def test_phrases_trigger_topics(self, text: str, topic: Topic) -> None:
        assert topic in one(text, 1).topics


class TestRuleDataInVersion:
    """The rule data is part of the digest, so editing a word list can never serve stale rows."""

    FIXTURES: dict[str, Analysis] = {}

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("_POSITIVE_WORDS", frozenset(["splendid"])),
            ("_NEGATIVE_WORDS", frozenset(["dreadful"])),
            ("FALLBACK_CONFIDENCE", 0.9),
        ],
    )
    def test_changing_rule_data_changes_the_version(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: object
    ) -> None:
        before = FixtureClassifier(self.FIXTURES).prompt_version
        monkeypatch.setattr(fixture_classifier, name, value)

        assert FixtureClassifier(self.FIXTURES).prompt_version != before

    def test_changing_topic_keywords_changes_the_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = FixtureClassifier(self.FIXTURES).prompt_version
        changed = {**fixture_classifier._TOPIC_KEYWORDS, Topic.PRICING: ("costly",)}
        monkeypatch.setattr(fixture_classifier, "_TOPIC_KEYWORDS", changed)

        assert FixtureClassifier(self.FIXTURES).prompt_version != before

    def test_added_keyword_changes_the_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = FixtureClassifier(self.FIXTURES).prompt_version
        words = fixture_classifier._TOPIC_KEYWORDS
        changed = {**words, Topic.PRICING: (*words[Topic.PRICING], "costly")}
        monkeypatch.setattr(fixture_classifier, "_TOPIC_KEYWORDS", changed)

        assert FixtureClassifier(self.FIXTURES).prompt_version != before

    def test_word_order_does_not_matter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = FixtureClassifier(self.FIXTURES).prompt_version
        words = fixture_classifier._TOPIC_KEYWORDS
        shuffled = {t: tuple(reversed(w)) for t, w in reversed(list(words.items()))}
        monkeypatch.setattr(fixture_classifier, "_TOPIC_KEYWORDS", shuffled)

        assert FixtureClassifier(self.FIXTURES).prompt_version == before


class TestStrictEntry:
    def test_valid_entry(self) -> None:
        analysis = parse_strict_entry({"sentiment": "negative", "topics": ["pricing", "other"]})

        assert analysis.sentiment is Sentiment.NEGATIVE
        assert analysis.topics == (Topic.PRICING, Topic.OTHER)
        assert analysis.confidence is None

    @pytest.mark.parametrize(
        "bad",
        [
            "negative",
            None,
            {"sentiment": "negative", "topics": [], "extra": 1},
            {"sentiment": "Negative", "topics": []},
            {"sentiment": "bogus-sentiment", "topics": []},
            {"sentiment": "negative", "topics": ["weather"]},
            {"sentiment": "negative", "topics": ["Pricing"]},
            {"sentiment": "negative", "topics": ["pricing", "pricing"]},
            {"sentiment": "negative", "topics": ["delivery_delay", "pricing", "billing", "other"]},
        ],
    )
    def test_rejections_never_quote_values(self, bad: object) -> None:
        with pytest.raises(ValueError) as excinfo:
            parse_strict_entry(bad)

        message = str(excinfo.value)
        for value in ("weather", "Pricing", "bogus-sentiment", "Negative"):
            assert value not in message

    def test_a_mapping_is_accepted(self) -> None:
        from types import MappingProxyType

        entry = MappingProxyType({"sentiment": "neutral", "topics": []})

        assert parse_strict_entry(entry).sentiment is Sentiment.NEUTRAL


class TestTopLevelKeys:
    def test_unknown_top_level_key_is_rejected(self, tmp_path: Path) -> None:
        path = write_fixture(
            tmp_path, {"format": 1, "analyses": {KEY: entry("negative")}, "extra": 1}
        )

        with pytest.raises(ClassifierError, match="fixtures.json"):
            FixtureClassifier.from_file(path)

    def test_unknown_entry_key_is_rejected(self, tmp_path: Path) -> None:
        bad = {**entry("negative"), "confidence": 0.9}
        path = write_fixture(tmp_path, {"format": 1, "analyses": {KEY: bad}})

        with pytest.raises(ClassifierError, match="fixtures.json"):
            FixtureClassifier.from_file(path)


class TestPromptVersion:
    HIT = Analysis(sentiment=Sentiment.NEGATIVE, topics=(Topic.BILLING,), confidence=None)

    def test_same_fixtures_give_the_same_version(self) -> None:
        a = FixtureClassifier({text_hash("a"): self.HIT, text_hash("b"): self.HIT})
        b = FixtureClassifier({text_hash("b"): self.HIT, text_hash("a"): self.HIT})

        assert a.prompt_version == b.prompt_version

    def test_a_changed_fixture_changes_the_version(self) -> None:
        other = Analysis(sentiment=Sentiment.NEGATIVE, topics=(Topic.PRICING,), confidence=None)

        assert (
            FixtureClassifier({text_hash("a"): self.HIT}).prompt_version
            != FixtureClassifier({text_hash("a"): other}).prompt_version
        )

    def test_an_added_fixture_changes_the_version(self) -> None:
        assert (
            FixtureClassifier({}).prompt_version
            != FixtureClassifier({text_hash("a"): self.HIT}).prompt_version
        )

    def test_a_new_rules_version_changes_the_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fixtures = {text_hash("a"): self.HIT}
        before = FixtureClassifier(fixtures).prompt_version
        monkeypatch.setattr(
            fixture_classifier,
            "FALLBACK_RULES_VERSION",
            fixture_classifier.FALLBACK_RULES_VERSION + 1,
        )

        assert FixtureClassifier(fixtures).prompt_version != before

    def test_loading_the_same_file_twice_gives_the_same_version(self, tmp_path: Path) -> None:
        path = write_fixture(tmp_path, {"format": 1, "analyses": {KEY: entry("negative")}})

        assert (
            FixtureClassifier.from_file(path).prompt_version
            == FixtureClassifier.from_file(path).prompt_version
        )


class TestFallbackCount:
    def test_misses_are_counted(self) -> None:
        canned = Analysis(sentiment=Sentiment.NEUTRAL, topics=(), confidence=None)
        classifier = FixtureClassifier({text_hash("known"): canned})
        items = [item("known", None, "r1"), item("x", 1, "r2"), item("y", 1, "r3")]

        result = classifier.classify(items)

        assert result.fallback == 2
        assert result.unknown_topics == 0
        assert result.truncated_topics == 0

    def test_all_hits_count_zero(self) -> None:
        canned = Analysis(sentiment=Sentiment.NEUTRAL, topics=(), confidence=None)

        result = FixtureClassifier({text_hash("known"): canned}).classify([item("known")])

        assert result.fallback == 0
        assert (result.unknown_topics, result.truncated_topics) == (0, 0)


class TestConfidence:
    def test_fixture_hits_have_no_confidence(self) -> None:
        canned = Analysis(sentiment=Sentiment.NEUTRAL, topics=(), confidence=None)
        result = FixtureClassifier({text_hash("x"): canned}).classify([item("x")])

        assert result.analyses["r1"].confidence is None

    def test_fallback_has_half_confidence(self) -> None:
        assert one("anything", 2).confidence == 0.5


class TestClosedVocabulary:
    @pytest.mark.parametrize("rating", [None, 1, 2, 3, 4, 5])
    def test_fallback_output_is_always_valid(self, rating: int | None) -> None:
        texts = ["", "x", "late broken refund app box price charged", "great", "ЯЗЫК \U0001f60d"]

        analyses = (
            FixtureClassifier({})
            .classify([item(t, rating, f"r{i}") for i, t in enumerate(texts)])
            .analyses
        )

        for analysis in analyses.values():
            assert isinstance(analysis, Analysis)
            assert analysis.sentiment in set(Sentiment)
            assert set(analysis.topics) <= set(Topic)
            assert len(analysis.topics) <= 3


class TestFromFile:
    def test_loads_entries(self, tmp_path: Path) -> None:
        path = write_fixture(
            tmp_path,
            {
                "format": 1,
                "analyses": {text_hash("h1"): entry("negative", ["pricing", "billing"])},
            },
        )

        result = FixtureClassifier.from_file(path).classify([item("h1", rating=5)])

        assert result.analyses["r1"] == Analysis(
            sentiment=Sentiment.NEGATIVE, topics=(Topic.PRICING, Topic.BILLING), confidence=None
        )

    def test_loaded_entry_is_used_by_classify(self, tmp_path: Path) -> None:
        path = write_fixture(
            tmp_path, {"format": 1, "analyses": {text_hash("hello"): entry("positive", [])}}
        )

        result = FixtureClassifier.from_file(path).classify([item("Hello", rating=1)])

        assert result.analyses["r1"].sentiment is Sentiment.POSITIVE
        assert result.analyses["r1"].confidence is None

    def test_invalid_sentiment_is_a_load_error(self, tmp_path: Path) -> None:
        path = write_fixture(
            tmp_path, {"format": 1, "analyses": {KEY: entry("bogus-sentiment", [])}}
        )

        with pytest.raises(ClassifierError) as excinfo:
            FixtureClassifier.from_file(path)

        assert excinfo.value.reason == "fixture_load_failed"
        assert path.name in str(excinfo.value)
        assert "bogus-sentiment" not in str(excinfo.value)

    @pytest.mark.parametrize(
        "topics",
        [
            ["weather"],
            ["Pricing"],
            [" pricing"],
            ["pricing", "pricing"],
            ["delivery_delay", "pricing", "billing", "packaging"],
            [5],
            [None],
            [["pricing"]],
        ],
    )
    def test_strict_topic_rules(self, tmp_path: Path, topics: list[object]) -> None:
        path = write_fixture(tmp_path, {"format": 1, "analyses": {KEY: entry("negative", topics)}})

        with pytest.raises(ClassifierError) as excinfo:
            FixtureClassifier.from_file(path)

        assert excinfo.value.reason == "fixture_load_failed"
        assert "weather" not in str(excinfo.value)

    def test_exactly_three_topics_are_accepted(self, tmp_path: Path) -> None:
        topics = ["delivery_delay", "pricing", "billing"]
        path = write_fixture(tmp_path, {"format": 1, "analyses": {KEY: entry("negative", topics)}})

        classifier = FixtureClassifier.from_file(path)

        assert len(classifier.classify([item("x")]).analyses) == 1

    @pytest.mark.parametrize("key", ["h", "ABCDEF" * 10 + "abcd", "a" * 63, "a" * 65, "g" * 64])
    def test_keys_must_be_lowercase_sha256_hex(self, tmp_path: Path, key: str) -> None:
        path = write_fixture(tmp_path, {"format": 1, "analyses": {key: entry("negative")}})

        with pytest.raises(ClassifierError, match="fixture"):
            FixtureClassifier.from_file(path)

    def test_empty_analyses_are_rejected(self, tmp_path: Path) -> None:
        path = write_fixture(tmp_path, {"format": 1, "analyses": {}})

        with pytest.raises(ClassifierError, match="fixture"):
            FixtureClassifier.from_file(path)

    def test_duplicate_keys_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.json"
        body = '{"sentiment": "neutral", "topics": []}'
        path.write_text(
            '{"format": 1, "analyses": {"' + KEY + '": ' + body + ', "' + KEY + '": ' + body + "}}",
            encoding="utf-8",
        )

        with pytest.raises(ClassifierError, match="dup.json"):
            FixtureClassifier.from_file(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClassifierError) as excinfo:
            FixtureClassifier.from_file(tmp_path / "nope.json")

        assert excinfo.value.reason == "fixture_load_failed"
        assert "nope.json" in str(excinfo.value)
        assert str(tmp_path) not in str(excinfo.value)

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClassifierError, match="fixture|Cannot"):
            FixtureClassifier.from_file(tmp_path)

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            "[]",
            '{"format": 2, "analyses": {}}',
            '{"analyses": {}}',
            '{"format": 1}',
            '{"format": 1, "analyses": []}',
            '{"format": 1, "analyses": {"' + KEY + '": "negative"}}',
            '{"format": 1, "analyses": {"' + KEY + '": {"topics": []}}}',
            '{"format": 1, "analyses": {"' + KEY + '": {"sentiment": "negative"}}}',
            '{"format": 1, "analyses": {"' + KEY + '": {"sentiment": 5, "topics": []}}}',
            '{"format": 1, "analyses": {"'
            + KEY
            + '": {"sentiment": "negative", "topics": "pricing"}}}',
        ],
    )
    def test_malformed_files(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "bad.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ClassifierError) as excinfo:
            FixtureClassifier.from_file(path)

        assert excinfo.value.reason == "fixture_load_failed"
        assert "bad.json" in str(excinfo.value)

    def test_non_utf8_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_bytes(b"\xff\xfe\x00bad")

        with pytest.raises(ClassifierError, match="bad.json"):
            FixtureClassifier.from_file(path)


class TestLogging:
    def test_logs_counts_only(self, caplog: pytest.LogCaptureFixture) -> None:
        fake_text = "ZEBRA-QUOKKA-9917 late"
        canned = Analysis(sentiment=Sentiment.NEUTRAL, topics=(), confidence=None)
        classifier = FixtureClassifier({text_hash("known text"): canned})

        with caplog.at_level(logging.DEBUG):
            classifier.classify([item("known text", None, "r1"), item(fake_text, 1, "r2")])

        assert "1 fixture hit" in caplog.text
        assert "1 fallback" in caplog.text
        assert "ZEBRA" not in caplog.text
        assert "known text" not in caplog.text


class TestCommittedFixtures:
    def test_every_sample_review_is_covered(self) -> None:
        classifier = FixtureClassifier.from_file(FIXTURE_FILE)
        reviews = read_reviews(SAMPLE_CSV).reviews

        result = classifier.classify(
            [item(r.text, r.rating, f"r{i}") for i, r in enumerate(reviews)]
        )

        assert len(result.analyses) == len(reviews)
        assert all(a.confidence is None for a in result.analyses.values())
