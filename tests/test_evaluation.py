"""Evaluation core: truth loading, scoring, error rows, cost estimates and the Markdown report.

Pure functions only: no network, no database. All reviews and labels are fictional.
"""

import csv
import dataclasses
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from review_radar.config import PROJECT_ROOT
from review_radar.evaluation import (
    DATASETS,
    INPUT_TOKENS_PER_TEXT,
    OUTPUT_TOKENS_PER_TEXT,
    SAFETY_MARGIN,
    ClassMetric,
    CostEstimate,
    Dataset,
    ErrorRow,
    EvalError,
    EvalReport,
    ModelResult,
    Prediction,
    Scores,
    Truth,
    estimate_cost_for,
    list_errors,
    load_truth,
    render_markdown,
    score,
    scores_by_category,
)
from review_radar.models import Sentiment, Topic, Usage
from review_radar.pricing import PRICES_PER_MTOK

POS, NEU, NEG = Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE
BILLING, DELIVERY = Topic.BILLING, Topic.DELIVERY_DELAY
PRICING, PACKAGING = Topic.PRICING, Topic.PACKAGING


def truth(sentiment: Sentiment, *topics: Topic, category: str | None = None) -> Truth:
    return Truth(sentiment=sentiment, topics=frozenset(topics), category=category)


def pred(sentiment: Sentiment, *topics: Topic) -> Prediction:
    return Prediction(sentiment=sentiment, topics=frozenset(topics))


def write_truth(path: Path, reviews: Any) -> Path:
    path.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    return path


@pytest.fixture
def mixed_case() -> tuple[dict[str, Truth], dict[str, Prediction]]:
    """Four reviews: two right on sentiment, one wrong, one without a prediction."""
    expected = {
        "a": truth(NEG, BILLING, DELIVERY),
        "b": truth(POS),
        "c": truth(NEU),
        "d": truth(NEG, PACKAGING),
    }
    predictions = {
        "a": pred(NEG, BILLING),
        "b": pred(NEG, PRICING),
        "c": pred(NEU),
    }
    return expected, predictions


class TestDatasets:
    def test_both_datasets_exist_with_paths_anchored_to_the_repo_root(self) -> None:
        assert set(DATASETS) == {"regular", "hard"}
        assert DATASETS["regular"] == Dataset(
            csv=PROJECT_ROOT / "data" / "sample" / "reviews.csv",
            truth=PROJECT_ROOT / "data" / "sample" / "ground_truth.json",
        )
        assert DATASETS["hard"] == Dataset(
            csv=PROJECT_ROOT / "data" / "eval" / "hard_reviews.csv",
            truth=PROJECT_ROOT / "data" / "eval" / "hard_ground_truth.json",
        )

    @pytest.mark.parametrize("name", ["regular", "hard"])
    def test_the_files_exist(self, name: str) -> None:
        assert DATASETS[name].csv.is_file()
        assert DATASETS[name].truth.is_file()

    def test_a_dataset_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            DATASETS["regular"].csv = Path("x")  # type: ignore[misc]

    @pytest.mark.parametrize("name", ["regular", "hard"])
    def test_truth_ids_match_the_csv_ids(self, name: str) -> None:
        with DATASETS[name].csv.open(encoding="utf-8-sig", newline="") as handle:
            ids = {row["review_id"] for row in csv.DictReader(handle)}

        assert set(load_truth(DATASETS[name].truth)) == ids


class TestLoadTruth:
    def test_reads_sentiment_topics_and_category(self, tmp_path: Path) -> None:
        path = write_truth(
            tmp_path / "t.json",
            {
                "r1": {"sentiment": "negative", "topics": ["billing", "pricing"], "category": "x"},
                "r2": {"sentiment": "positive", "topics": []},
            },
        )

        loaded = load_truth(path)

        assert loaded == {
            "r1": Truth(NEG, frozenset({BILLING, PRICING}), "x"),
            "r2": Truth(POS, frozenset(), None),
        }
        assert isinstance(loaded["r1"].topics, frozenset)

    def test_the_real_regular_file_has_300_reviews(self) -> None:
        assert len(load_truth(DATASETS["regular"].truth)) == 300

    def test_the_real_hard_file_has_60_reviews_in_four_equal_categories(self) -> None:
        loaded = load_truth(DATASETS["hard"].truth)

        assert len(loaded) == 60
        assert Counter(item.category for item in loaded.values()) == {
            "sarcasm": 15,
            "mixed": 15,
            "typos": 15,
            "short": 15,
        }

    @pytest.mark.parametrize(
        "entry",
        [
            {"sentiment": "angry", "topics": []},
            {"sentiment": "positive", "topics": ["not_a_topic"]},
            {"sentiment": "positive"},
            {"topics": []},
            {"sentiment": "positive", "topics": "billing"},
            {"sentiment": None, "topics": []},
            "not an object",
        ],
    )
    def test_a_bad_entry_is_an_eval_error(self, tmp_path: Path, entry: Any) -> None:
        path = write_truth(tmp_path / "t.json", {"r1": entry})

        with pytest.raises(EvalError):
            load_truth(path)

    @pytest.mark.parametrize("content", ["{not json", "", "[]", '{"other": {}}', '{"reviews": []}'])
    def test_bad_json_or_missing_keys_are_an_eval_error(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / "t.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(EvalError):
            load_truth(path)

    def test_a_missing_file_is_an_eval_error(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError):
            load_truth(tmp_path / "missing.json")


class TestValueObjectsAreFrozen:
    @pytest.mark.parametrize(
        ("obj", "field"),
        [
            (Truth(POS, frozenset(), None), "sentiment"),
            (Prediction(POS, frozenset()), "sentiment"),
            (ClassMetric(None, None, 0), "support"),
            (ErrorRow("a", None, POS, None, (), None), "review_id"),
        ],
    )
    def test_assignment_is_rejected(self, obj: object, field: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, "changed")

    def test_scores_and_cost_estimate_are_frozen(self, mixed_case: Any) -> None:
        scores = score(*mixed_case)
        estimate = estimate_cost_for("claude-haiku-5-5", ["one"])

        with pytest.raises(dataclasses.FrozenInstanceError):
            scores.n = 0  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            estimate.texts = 0  # type: ignore[misc]


class TestScore:
    def test_counts(self, mixed_case: Any) -> None:
        scores = score(*mixed_case)

        assert scores.n == 4
        assert scores.predicted == 3
        assert scores.missing == 1

    def test_sentiment_accuracy_counts_a_missing_prediction_as_wrong(self, mixed_case: Any) -> None:
        # a and c are right, b is wrong, d is missing: 2 of 4.
        assert score(*mixed_case).sentiment_accuracy == pytest.approx(0.5)

    def test_per_class_precision_and_recall(self, mixed_case: Any) -> None:
        by_class = score(*mixed_case).sentiment

        # negative: true {a, d}, predicted {a, b}, one hit.
        assert by_class["negative"].precision == pytest.approx(0.5)
        assert by_class["negative"].recall == pytest.approx(0.5)
        assert by_class["negative"].support == 2
        # positive: true {b}, never predicted: precision undefined, recall 0.
        assert by_class["positive"].precision is None
        assert by_class["positive"].recall == pytest.approx(0.0)
        assert by_class["positive"].support == 1
        # neutral: true {c}, predicted {c}.
        assert by_class["neutral"].precision == pytest.approx(1.0)
        assert by_class["neutral"].recall == pytest.approx(1.0)
        assert by_class["neutral"].support == 1

    def test_a_class_with_support_that_is_never_predicted_counts_as_zero_in_macro(
        self, mixed_case: Any
    ) -> None:
        scores = score(*mixed_case)

        # Positive has support but no predictions: its precision is undefined on its own,
        # but ignoring it would reward a model that never predicts the class.
        assert scores.sentiment["positive"].precision is None
        assert scores.macro_precision == pytest.approx((0.5 + 0.0 + 1.0) / 3)
        assert scores.macro_recall == pytest.approx((0.5 + 0.0 + 1.0) / 3)

    def test_topic_precision_and_recall_are_micro_over_labels(self, mixed_case: Any) -> None:
        scores = score(*mixed_case)

        # Predicted labels: billing (a), pricing (b) = 2, one correct.
        assert scores.topic_precision == pytest.approx(1 / 2)
        # True labels: billing, delivery_delay (a), packaging (d) = 3, one found; d is missing.
        assert scores.topic_recall == pytest.approx(1 / 3)

    def test_exact_topics_counts_only_equal_sets(self, mixed_case: Any) -> None:
        # Only c (no topics, none predicted) matches exactly.
        assert score(*mixed_case).exact_topics == pytest.approx(1 / 4)

    def test_a_perfect_run(self) -> None:
        expected = {"a": truth(NEG, BILLING), "b": truth(POS), "c": truth(NEU, PRICING)}
        predictions = {"a": pred(NEG, BILLING), "b": pred(POS), "c": pred(NEU, PRICING)}

        scores = score(expected, predictions)

        assert scores.sentiment_accuracy == pytest.approx(1.0)
        assert scores.macro_precision == pytest.approx(1.0)
        assert scores.macro_recall == pytest.approx(1.0)
        assert scores.topic_precision == pytest.approx(1.0)
        assert scores.topic_recall == pytest.approx(1.0)
        assert scores.exact_topics == pytest.approx(1.0)
        assert scores.missing == 0

    def test_only_ids_in_the_truth_are_scored(self) -> None:
        expected = {"a": truth(POS)}
        predictions = {"a": pred(POS), "zzz": pred(NEG, BILLING), "yyy": pred(NEG)}

        scores = score(expected, predictions)

        assert scores.n == 1
        assert scores.predicted == 1
        assert scores.missing == 0
        assert scores.sentiment_accuracy == pytest.approx(1.0)
        assert scores.topic_precision is None  # the extra predictions are ignored

    def test_no_predictions_at_all_is_zero_accuracy_and_undefined_precision(self) -> None:
        expected = {"a": truth(NEG, BILLING), "b": truth(POS)}

        scores = score(expected, {})

        assert scores.n == 2
        assert scores.predicted == 0
        assert scores.missing == 2
        assert scores.sentiment_accuracy == pytest.approx(0.0)
        assert scores.exact_topics == pytest.approx(0.0)
        assert scores.macro_precision == pytest.approx(0.0)
        assert scores.macro_recall == pytest.approx(0.0)
        assert scores.topic_precision is None
        assert scores.topic_recall == pytest.approx(0.0)

    def test_a_class_with_no_support_and_no_predictions_is_undefined_not_zero(self) -> None:
        expected = {"a": truth(NEG), "b": truth(NEG)}
        predictions = {"a": pred(NEG), "b": pred(NEG)}

        scores = score(expected, predictions)

        assert scores.sentiment["positive"] == ClassMetric(None, None, 0)
        assert scores.sentiment["neutral"] == ClassMetric(None, None, 0)
        assert scores.macro_precision == pytest.approx(1.0)  # only negative counts
        assert scores.macro_recall == pytest.approx(1.0)

    def test_without_any_topic_label_topic_scores_are_undefined(self) -> None:
        expected = {"a": truth(POS), "b": truth(NEU)}
        predictions = {"a": pred(POS), "b": pred(NEU)}

        scores = score(expected, predictions)

        assert scores.topic_precision is None
        assert scores.topic_recall is None
        assert scores.exact_topics == pytest.approx(1.0)

    def test_predicting_topics_when_there_are_none_gives_zero_precision(self) -> None:
        scores = score({"a": truth(POS)}, {"a": pred(POS, BILLING)})

        assert scores.topic_precision == pytest.approx(0.0)
        assert scores.topic_recall is None

    def test_the_sentiment_dict_has_all_three_classes(self, mixed_case: Any) -> None:
        assert set(score(*mixed_case).sentiment) == {"positive", "neutral", "negative"}

    @pytest.mark.parametrize("predictions", [{}, {"a": pred(POS)}])
    def test_empty_truth_is_an_eval_error(self, predictions: dict[str, Prediction]) -> None:
        with pytest.raises(EvalError):
            score({}, predictions)

    def test_scores_is_a_value_object(self, mixed_case: Any) -> None:
        assert isinstance(score(*mixed_case), Scores)
        assert score(*mixed_case) == score(*mixed_case)


class TestListErrors:
    def test_lists_wrong_and_missing_reviews_ordered_by_id(self, mixed_case: Any) -> None:
        expected, predictions = mixed_case
        expected = {"z-ok": truth(POS), **expected}
        predictions = {"z-ok": pred(POS), **predictions}

        rows = list_errors(expected, predictions)

        assert [row.review_id for row in rows] == ["a", "b", "d"]  # c is right, so it is absent

    def test_a_row_carries_labels_and_no_text(self, mixed_case: Any) -> None:
        rows = {row.review_id: row for row in list_errors(*mixed_case)}

        assert rows["a"] == ErrorRow(
            review_id="a",
            category=None,
            true_sentiment=NEG,
            predicted_sentiment=NEG,
            true_topics=("billing", "delivery_delay"),
            predicted_topics=("billing",),
        )
        assert rows["b"].true_sentiment == POS
        assert rows["b"].predicted_sentiment == NEG
        assert rows["b"].true_topics == ()
        assert rows["b"].predicted_topics == ("pricing",)

    def test_a_missing_prediction_has_none_predicted_fields(self, mixed_case: Any) -> None:
        rows = {row.review_id: row for row in list_errors(*mixed_case)}

        assert rows["d"].predicted_sentiment is None
        assert rows["d"].predicted_topics is None
        assert rows["d"].true_sentiment == NEG
        assert rows["d"].true_topics == ("packaging",)

    def test_a_topic_only_difference_is_an_error(self) -> None:
        rows = list_errors({"a": truth(NEG, BILLING)}, {"a": pred(NEG, PRICING)})

        assert [row.review_id for row in rows] == ["a"]

    def test_the_category_is_copied_from_the_truth(self) -> None:
        rows = list_errors({"a": truth(NEG, category="sarcasm")}, {"a": pred(POS)})

        assert rows[0].category == "sarcasm"

    def test_a_perfect_run_has_no_errors(self) -> None:
        assert list_errors({"a": truth(POS, BILLING)}, {"a": pred(POS, BILLING)}) == []

    def test_the_row_type_holds_no_review_text(self) -> None:
        assert {f.name for f in dataclasses.fields(ErrorRow)} == {
            "review_id",
            "category",
            "true_sentiment",
            "predicted_sentiment",
            "true_topics",
            "predicted_topics",
        }


class TestScoresByCategory:
    def test_each_category_is_scored_on_its_own_reviews(self) -> None:
        expected = {
            "a": truth(NEG, BILLING, category="sarcasm"),
            "b": truth(POS, category="sarcasm"),
            "c": truth(NEU, category="short"),
            "d": truth(NEG, PACKAGING, category="short"),
        }
        predictions = {"a": pred(NEG, BILLING), "b": pred(NEG), "c": pred(NEU)}

        by_category = scores_by_category(expected, predictions)

        assert set(by_category) == {"sarcasm", "short"}
        assert by_category["sarcasm"] == score(
            {k: expected[k] for k in ("a", "b")}, {k: predictions[k] for k in ("a", "b")}
        )
        assert by_category["short"] == score(
            {k: expected[k] for k in ("c", "d")}, {"c": predictions["c"]}
        )
        assert by_category["sarcasm"].sentiment_accuracy == pytest.approx(0.5)
        assert by_category["short"].missing == 1  # a missing prediction is never dropped

    def test_reviews_without_a_category_form_no_group(self) -> None:
        expected = {"a": truth(POS), "b": truth(NEG, category="mixed")}

        assert set(scores_by_category(expected, {})) == {"mixed"}


class TestEstimateCost:
    def test_the_constants(self) -> None:
        assert INPUT_TOKENS_PER_TEXT == 135
        assert OUTPUT_TOKENS_PER_TEXT == 36
        assert Decimal("1.5") == SAFETY_MARGIN

    def test_haiku_exact_amounts(self) -> None:
        estimate = estimate_cost_for("claude-haiku-5-5", ["one", "two", "three"])

        assert estimate.model == "claude-haiku-5-5"
        assert estimate.texts == 3
        assert estimate.input_tokens == 3 * 135
        assert estimate.output_tokens == 3 * 36
        # 405 * 0.10 / 1M + 108 * 0.50 / 1M
        assert estimate.expected_usd == Decimal("0.0000945")
        assert estimate.upper_usd == Decimal("0.00014175")

    def test_sonnet_exact_amounts(self) -> None:
        estimate = estimate_cost_for("claude-sonnet-5-5", ["one", "two", "three"])

        # 405 * 2.00 / 1M + 108 * 10.00 / 1M
        assert estimate.expected_usd == Decimal("0.00189")
        assert estimate.upper_usd == Decimal("0.002835")

    @pytest.mark.parametrize("model", sorted(PRICES_PER_MTOK))
    def test_matches_the_price_table_and_the_safety_margin(self, model: str) -> None:
        texts = [f"text {i}" for i in range(1000)]
        price_in, price_out = PRICES_PER_MTOK[model]

        estimate = estimate_cost_for(model, texts)

        expected = (Decimal(135_000) * price_in + Decimal(36_000) * price_out) / Decimal(1_000_000)
        assert estimate.expected_usd == expected
        assert estimate.upper_usd == expected * Decimal("1.5")
        assert isinstance(estimate, CostEstimate)

    def test_the_amounts_are_decimals(self) -> None:
        estimate = estimate_cost_for("claude-haiku-5-5", ["one"])

        assert isinstance(estimate.expected_usd, Decimal)
        assert isinstance(estimate.upper_usd, Decimal)

    def test_an_unknown_model_is_an_eval_error(self) -> None:
        with pytest.raises(EvalError):
            estimate_cost_for("claude-unknown-9", ["one"])


def metric_scores(*, accuracy: float | None = 0.5, precision: float | None = 0.25) -> Scores:
    return Scores(
        n=4,
        predicted=3,
        missing=1,
        sentiment_accuracy=accuracy if accuracy is not None else 0.0,
        sentiment={"negative": ClassMetric(0.5, 0.5, 2)},
        macro_precision=precision,
        macro_recall=0.5,
        topic_precision=None,
        topic_recall=1 / 3,
        exact_topics=0.25,
    )


def model_result(model: str = "claude-haiku-5-5", dataset: str = "regular") -> ModelResult:
    return ModelResult(
        model=model,
        dataset_scores={dataset: metric_scores()},
        category_scores={"sarcasm": metric_scores()},
        usage=Usage(input_tokens=405, output_tokens=108),
        cost_usd=Decimal("0.0000945"),
        reviews=3,
        cost_per_1000=Decimal("0.0315"),
        failed=0,
    )


def cells(line: str) -> list[str]:
    """Split a Markdown table row on unescaped pipes."""
    return [part.strip() for part in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def table_lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.lstrip().startswith("|")]


class TestRenderMarkdown:
    def test_has_a_table_with_the_required_columns(self) -> None:
        markdown = render_markdown(EvalReport(models=(model_result(),)))

        header = table_lines(markdown)[0].lower()
        for column in ("model", "dataset", "accuracy", "precision", "recall", "exact"):
            assert column in header
        assert "1,000" not in header  # the cost is per model, not per dataset

    def test_the_cost_is_one_row_per_model_in_its_own_table(self) -> None:
        both = dataclasses.replace(
            model_result(),
            dataset_scores={"regular": metric_scores(), "hard": metric_scores()},
        )

        markdown = render_markdown(EvalReport(models=(both,)))

        lines = table_lines(markdown)
        header = next(index for index, line in enumerate(lines) if "1,000" in line)
        assert "dataset" not in lines[header].lower()
        priced = [line for line in lines if "$0.0315" in line]
        assert len(priced) == 1
        assert "claude-haiku-5-5" in priced[0]

    def test_an_incomplete_report_says_so_at_the_top(self) -> None:
        report = EvalReport(models=(model_result(),), incomplete="7 reviews have no answer")

        markdown = render_markdown(report)

        first_table = markdown.index("|")
        assert "INCOMPLETE" in markdown[:first_table]
        assert "7 reviews have no answer" in markdown[:first_table]

    def test_a_complete_report_has_no_warning(self) -> None:
        assert "INCOMPLETE" not in render_markdown(EvalReport(models=(model_result(),)))

    def test_a_row_names_the_model_and_the_dataset(self) -> None:
        markdown = render_markdown(EvalReport(models=(model_result(),)))

        row = next(line for line in table_lines(markdown) if "claude-haiku-5-5" in line)
        assert "regular" in row
        assert "50" in row  # sentiment accuracy 0.5, shown as 0.5 / 0.50 / 50%

    def test_one_row_per_model_and_dataset(self) -> None:
        first = model_result("claude-haiku-5-5", "regular")
        second = dataclasses.replace(
            model_result("claude-sonnet-5-5", "hard"),
            dataset_scores={"regular": metric_scores(), "hard": metric_scores()},
        )

        markdown = render_markdown(EvalReport(models=(first, second)))

        assert sum("claude-haiku-5-5" in line for line in table_lines(markdown)) >= 1
        assert sum("claude-sonnet-5-5" in line for line in table_lines(markdown)) >= 2

    def test_none_renders_as_n_a_never_as_zero(self) -> None:
        markdown = render_markdown(EvalReport(models=(model_result(),)))

        row = next(line for line in table_lines(markdown) if "claude-haiku-5-5" in line)
        assert "n/a" in row  # topic_precision is None
        assert "None" not in markdown

    def test_pipes_are_escaped_in_every_value(self) -> None:
        nasty = dataclasses.replace(
            model_result("model|with|pipes", "data|set"),
            category_scores={"cat|egory": metric_scores()},
        )

        markdown = render_markdown(EvalReport(models=(nasty,)))

        header_cells = len(cells(table_lines(markdown)[0]))
        row = next(line for line in table_lines(markdown) if "model" in line and "pipes" in line)
        assert "model\\|with\\|pipes" in row
        assert "data\\|set" in row
        assert len(cells(row)) == header_cells
        assert "cat\\|egory" in markdown

    def test_the_report_is_text_and_holds_no_review_text_fields(self) -> None:
        markdown = render_markdown(EvalReport(models=(model_result(),)))

        assert isinstance(markdown, str)
        assert markdown.strip()
        assert {f.name for f in dataclasses.fields(ModelResult)} >= {
            "model",
            "dataset_scores",
            "category_scores",
            "usage",
            "cost_usd",
            "reviews",
            "cost_per_1000",
            "failed",
        }
