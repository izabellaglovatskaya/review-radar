import csv
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from generate_reviews import (
    COMPANY,
    PRODUCTS,
    RATING_WEIGHTS,
    TEMPLATES,
    GeneratedReview,
    _inject_duplicates,
    generate,
    main,
)
from review_radar.hashing import normalize_text, text_hash
from review_radar.models import Sentiment, Topic

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


@pytest.fixture(scope="module")
def reviews() -> list[GeneratedReview]:
    return generate()


class TestGenerate:
    def test_same_seed_gives_identical_output(self) -> None:
        assert generate(seed=7, n=50) == generate(seed=7, n=50)

    def test_different_seed_gives_different_output(self) -> None:
        assert generate(seed=1, n=50) != generate(seed=2, n=50)

    def test_does_not_touch_global_random(self) -> None:
        random.seed(123)
        expected = random.random()  # noqa: S311
        random.seed(123)
        generate(seed=5, n=20)
        assert random.random() == expected  # noqa: S311

    def test_count_and_ids(self, reviews: list[GeneratedReview]) -> None:
        assert len(reviews) == 300
        assert [r.review_id for r in reviews] == [f"r{i:04d}" for i in range(1, 301)]

    def test_custom_count(self) -> None:
        assert len(generate(n=10)) == 10

    def test_date_range(self, reviews: list[GeneratedReview]) -> None:
        dates = {r.review_date for r in reviews}
        assert min(dates) >= dt.date(2026, 11, 10)
        assert max(dates) <= dt.date(2026, 11, 18)
        assert len(dates) == 9

    def test_custom_start_and_days(self) -> None:
        out = generate(n=40, start=dt.date(2026, 1, 1), days=3)
        assert {r.review_date for r in out} <= {dt.date(2026, 1, d) for d in (1, 2, 3)}

    def test_sentiment_mix(self, reviews: list[GeneratedReview]) -> None:
        counts = Counter(r.sentiment for r in reviews)
        share = {s: counts[s] / len(reviews) for s in Sentiment}
        assert share[Sentiment.POSITIVE] == pytest.approx(0.45, abs=0.08)
        assert share[Sentiment.NEUTRAL] == pytest.approx(0.20, abs=0.08)
        assert share[Sentiment.NEGATIVE] == pytest.approx(0.35, abs=0.08)

    def test_topics_are_valid_and_bounded(self, reviews: list[GeneratedReview]) -> None:
        for r in reviews:
            assert len(r.topics) <= 3
            assert len(set(r.topics)) == len(r.topics)
            for t in r.topics:
                Topic(t)

    def test_negative_reviews_have_topics_positive_mostly_none(
        self, reviews: list[GeneratedReview]
    ) -> None:
        negatives = [r for r in reviews if r.sentiment is Sentiment.NEGATIVE]
        positives = [r for r in reviews if r.sentiment is Sentiment.POSITIVE]
        assert all(r.topics for r in negatives)
        assert sum(1 for r in positives if not r.topics) / len(positives) > 0.8

    def test_ratings_in_range_and_consistent_ish(self, reviews: list[GeneratedReview]) -> None:
        assert all(1 <= r.rating <= 5 for r in reviews)

        def mean(s: Sentiment) -> float:
            rs = [r.rating for r in reviews if r.sentiment is s]
            return sum(rs) / len(rs)

        assert mean(Sentiment.POSITIVE) > mean(Sentiment.NEUTRAL) > mean(Sentiment.NEGATIVE)
        # Noise exists: not every positive review is 4 or 5.
        assert any(r.rating <= 3 for r in reviews if r.sentiment is Sentiment.POSITIVE)

    def test_text_length_reasonable(self, reviews: list[GeneratedReview]) -> None:
        for r in reviews:
            assert r.text == r.text.strip()
            assert 5 <= len(r.text.split()) <= 120
            assert len(r.text) <= 2000

    def test_sources_from_small_set(self, reviews: list[GeneratedReview]) -> None:
        assert {r.source for r in reviews} == {"web", "app", "email"}

    def test_exact_duplicate_texts_on_different_dates(self, reviews: list[GeneratedReview]) -> None:
        by_text: dict[str, set[dt.date]] = {}
        for r in reviews:
            by_text.setdefault(r.text, set()).add(r.review_date)
        assert any(len(dates) >= 2 for dates in by_text.values())

    def test_contains_non_ascii(self, reviews: list[GeneratedReview]) -> None:
        assert any(not r.text.isascii() for r in reviews)

    def test_no_real_world_contact_details(self, reviews: list[GeneratedReview]) -> None:
        for r in reviews:
            assert "@" not in r.text
            assert "http" not in r.text.lower()


class TestGroundTruthInvariants:
    @pytest.mark.parametrize("seed", [1, 7, 42])
    @pytest.mark.parametrize("n", [50, 300])
    def test_each_text_hash_has_exactly_one_label(self, seed: int, n: int) -> None:
        labels: dict[str, set[tuple[Sentiment, tuple[Topic, ...]]]] = {}
        for r in generate(seed=seed, n=n):
            labels.setdefault(text_hash(r.text), set()).add((r.sentiment, r.topics))

        assert all(len(found) == 1 for found in labels.values())

    def test_rendered_template_texts_are_all_distinct(self) -> None:
        rendered = [normalize_text(t.text.format(product=p)) for t in TEMPLATES for p in PRODUCTS]

        assert len(set(rendered)) == len(TEMPLATES) * len(PRODUCTS)

    def test_injected_duplicates_copy_all_labels_and_rating(self) -> None:
        sentiments = list(Sentiment)
        before = [
            GeneratedReview(
                "",
                dt.date(2026, 11, 10 + i % 3),
                1 + i % 5,
                f"unique text {i}",
                ("web", "app", "email")[i % 3],
                sentiments[i % 3],
                (Topic.PRICING,) if i % 2 else (),
            )
            for i in range(20)
        ]
        rows = list(before)

        _inject_duplicates(rows, random.Random(1))  # noqa: S311

        groups: dict[str, list[GeneratedReview]] = {}
        for r in rows:
            groups.setdefault(r.text, []).append(r)
        dup_groups = [g for g in groups.values() if len(g) > 1]
        assert dup_groups
        for g in dup_groups:
            assert len({(r.sentiment, r.topics, r.rating) for r in g}) == 1
            assert len({r.review_date for r in g}) >= 2
        assert [(r.review_date, r.review_id, r.source) for r in rows] == [
            (r.review_date, r.review_id, r.source) for r in before
        ]

    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_every_sentiment_rating_pair_is_allowed(self, seed: int) -> None:
        for r in generate(seed=seed, n=300):
            assert RATING_WEIGHTS[r.sentiment].get(r.rating, 0) > 0


class TestMain:
    def test_writes_csv_and_ground_truth(self, tmp_path: Path) -> None:
        assert main(["--seed", "42", "--out", str(tmp_path)]) == 0

        raw = (tmp_path / "reviews.csv").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert b"\x0d" not in raw
        assert raw.startswith(b"review_id,review_date,rating,text,source\n")

        with (tmp_path / "reviews.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        truth = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))

        assert truth["seed"] == 42
        assert truth["company"] == COMPANY
        assert [r["review_id"] for r in rows] == list(truth["reviews"])
        for entry in truth["reviews"].values():
            assert set(entry) == {"sentiment", "topics"}
            Sentiment(entry["sentiment"])
            for t in entry["topics"]:
                Topic(t)

    def test_csv_matches_generated_reviews(self, tmp_path: Path) -> None:
        main(["--seed", "3", "--out", str(tmp_path)])
        with (tmp_path / "reviews.csv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        expected = generate(seed=3)
        assert [r["text"] for r in rows] == [g.text for g in expected]
        assert [r["rating"] for r in rows] == [str(g.rating) for g in expected]
        assert [r["review_date"] for r in rows] == [g.review_date.isoformat() for g in expected]

    def test_ground_truth_format(self, tmp_path: Path) -> None:
        main(["--out", str(tmp_path)])
        raw = (tmp_path / "ground_truth.json").read_text(encoding="utf-8")
        assert raw.endswith("}\n")
        assert (
            raw == json.dumps(json.loads(raw), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    def test_creates_missing_output_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b"
        main(["--out", str(out)])
        assert (out / "reviews.csv").is_file()

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        main(["--seed", "9", "--out", str(tmp_path / "x")])
        main(["--seed", "9", "--out", str(tmp_path / "y")])
        for name in ("reviews.csv", "ground_truth.json"):
            assert (tmp_path / "x" / name).read_bytes() == (tmp_path / "y" / name).read_bytes()


class TestCommittedSample:
    @pytest.mark.parametrize("name", ["reviews.csv", "ground_truth.json"])
    def test_sample_matches_fresh_output(self, tmp_path: Path, name: str) -> None:
        main(["--seed", "42", "--out", str(tmp_path)])
        assert (SAMPLE_DIR / name).read_bytes() == (tmp_path / name).read_bytes()
