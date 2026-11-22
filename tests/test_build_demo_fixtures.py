import json
from pathlib import Path

import pytest

from build_demo_fixtures import FixtureBuildError, build_fixtures, main, render
from review_radar.hashing import text_hash
from review_radar.ingest import read_reviews

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews.csv"
SAMPLE_TRUTH = ROOT / "data" / "sample" / "ground_truth.json"
COMMITTED = ROOT / "fixtures" / "demo_analyses.json"

HEADER = "review_id,review_date,rating,text,source\n"


def write_inputs(tmp_path: Path, rows: list[str], truth: dict[str, object]) -> tuple[Path, Path]:
    csv_path = tmp_path / "reviews.csv"
    csv_path.write_text(HEADER + "".join(rows), encoding="utf-8")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"reviews": truth}), encoding="utf-8")
    return csv_path, truth_path


def test_builds_entries_keyed_by_text_hash(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,1,Too  Expensive,web\n", "r2,2026-11-10,5,Lovely,web\n"],
        {
            "r1": {"sentiment": "negative", "topics": ["pricing", "billing"]},
            "r2": {"sentiment": "positive", "topics": []},
        },
    )

    payload = build_fixtures(csv_path, truth_path)

    assert payload == {
        "format": 1,
        "analyses": {
            text_hash("too expensive"): {"sentiment": "negative", "topics": ["pricing", "billing"]},
            text_hash("lovely"): {"sentiment": "positive", "topics": []},
        },
    }


def test_same_text_with_same_truth_is_one_entry(tmp_path: Path) -> None:
    truth = {"sentiment": "negative", "topics": ["pricing"]}
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,1,Pricey,web\n", "r2,2026-11-11,1,PRICEY,app\n"],
        {"r1": truth, "r2": truth},
    )

    assert len(build_fixtures(csv_path, truth_path)["analyses"]) == 1


def test_conflicting_truth_for_one_text_is_an_error(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,1,Pricey,web\n", "r2,2026-11-11,1,pricey,app\n"],
        {
            "r1": {"sentiment": "negative", "topics": ["pricing"]},
            "r2": {"sentiment": "neutral", "topics": []},
        },
    )

    with pytest.raises(FixtureBuildError, match="different ground truth"):
        build_fixtures(csv_path, truth_path)


def test_review_without_truth_is_an_error(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(tmp_path, ["r1,2026-11-10,1,Pricey,web\n"], {})

    with pytest.raises(FixtureBuildError, match="r1"):
        build_fixtures(csv_path, truth_path)


def test_review_without_an_id_is_an_error(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(tmp_path, [",2026-11-10,1,Pricey,web\n"], {})

    with pytest.raises(FixtureBuildError, match="review_id"):
        build_fixtures(csv_path, truth_path)


POSITIVE_TRUTH = {"sentiment": "positive", "topics": []}


def test_skipped_csv_rows_are_an_error(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,5,Lovely,web\n", "r2,not-a-date,5,Fine,web\n"],
        {"r1": POSITIVE_TRUTH, "r2": POSITIVE_TRUTH},
    )

    with pytest.raises(FixtureBuildError, match="skipped"):
        build_fixtures(csv_path, truth_path)


def test_unused_ground_truth_ids_are_an_error(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,5,Lovely,web\n"],
        {"r1": POSITIVE_TRUTH, "r9": POSITIVE_TRUTH, "r8": POSITIVE_TRUTH},
    )

    with pytest.raises(FixtureBuildError, match="unused") as excinfo:
        build_fixtures(csv_path, truth_path)

    assert "2" in str(excinfo.value)


@pytest.mark.parametrize("problem", ["skipped", "unused", "missing"])
def test_main_writes_nothing_on_input_problems(tmp_path: Path, problem: str) -> None:
    rows = {
        "skipped": ["r1,2026-11-10,5,Lovely,web\n", "r2,bad,5,Fine,web\n"],
        "unused": ["r1,2026-11-10,5,Lovely,web\n"],
        "missing": ["r1,2026-11-10,5,Lovely,web\n", "r2,2026-11-10,5,Fine,web\n"],
    }[problem]
    truth = {"r1": POSITIVE_TRUTH}
    if problem in {"skipped", "unused"}:
        truth["r2"] = POSITIVE_TRUTH
    csv_path, truth_path = write_inputs(tmp_path, rows, truth)
    out = tmp_path / "out.json"

    code = main(["--reviews", str(csv_path), "--truth", str(truth_path), "--out", str(out)])

    assert code == 1
    assert not out.exists()


def test_script_has_no_type_ignore_comments() -> None:
    source = (ROOT / "scripts" / "build_demo_fixtures.py").read_text(encoding="utf-8")

    assert "type: ignore" not in source


@pytest.mark.parametrize("content", ["{nope", "[]", '{"reviews": []}'])
def test_unreadable_truth_is_an_error(tmp_path: Path, content: str) -> None:
    csv_path, truth_path = write_inputs(tmp_path, ["r1,2026-11-10,1,Pricey,web\n"], {})
    truth_path.write_text(content, encoding="utf-8")

    with pytest.raises(FixtureBuildError):
        build_fixtures(csv_path, truth_path)


def test_render_is_sorted_deterministic_with_trailing_newline() -> None:
    payload = {"format": 1, "analyses": {"b": {"sentiment": "neutral", "topics": []}, "a": {}}}

    text = render(payload)

    assert text == render(payload)
    assert text.endswith("}\n")
    assert not text.endswith("\n\n")
    assert text.index('"a"') < text.index('"b"')
    assert "\r" not in text
    assert json.loads(text) == payload


def test_main_writes_the_file_and_creates_directories(tmp_path: Path) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,5,Lovely,web\n"],
        {"r1": {"sentiment": "positive", "topics": []}},
    )
    out = tmp_path / "nested" / "fixtures.json"

    code = main(["--reviews", str(csv_path), "--truth", str(truth_path), "--out", str(out)])

    assert code == 0
    assert out.read_bytes() == render(build_fixtures(csv_path, truth_path)).encode("utf-8")


def test_main_twice_is_byte_identical(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    args = ["--reviews", str(SAMPLE_CSV), "--truth", str(SAMPLE_TRUTH), "--out", str(out)]

    main(args)
    first = out.read_bytes()
    main(args)

    assert out.read_bytes() == first


def test_main_reports_conflicts_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path,
        ["r1,2026-11-10,1,Pricey,web\n", "r2,2026-11-11,1,pricey,app\n"],
        {
            "r1": {"sentiment": "negative", "topics": ["pricing"]},
            "r2": {"sentiment": "neutral", "topics": []},
        },
    )
    out = tmp_path / "out.json"

    code = main(["--reviews", str(csv_path), "--truth", str(truth_path), "--out", str(out)])

    assert code == 1
    assert "Error:" in capsys.readouterr().err
    assert not out.exists()


class TestCommittedFixtures:
    def test_file_matches_what_the_script_produces_now(self) -> None:
        expected = render(build_fixtures(SAMPLE_CSV, SAMPLE_TRUTH)).encode("utf-8")

        assert COMMITTED.read_bytes() == expected

    def test_every_sample_text_hash_is_present(self) -> None:
        analyses = json.loads(COMMITTED.read_text(encoding="utf-8"))["analyses"]

        for review in read_reviews(SAMPLE_CSV).reviews:
            assert text_hash(review.text) in analyses


BAD_ENTRIES = {
    "typo_topic": {"sentiment": "negative", "topics": ["pricng"]},
    "wrong_case_topic": {"sentiment": "negative", "topics": ["Pricing"]},
    "duplicate_topic": {"sentiment": "negative", "topics": ["pricing", "pricing"]},
    "four_topics": {
        "sentiment": "negative",
        "topics": ["delivery_delay", "pricing", "billing", "packaging"],
    },
    "extra_key": {"sentiment": "negative", "topics": [], "confidence": 0.5},
    "bad_sentiment": {"sentiment": "bogus-sentiment", "topics": []},
    "case_sentiment": {"sentiment": "Negative", "topics": []},
    "topics_not_a_list": {"sentiment": "negative", "topics": "pricing"},
    "entry_not_an_object": "negative",
}


@pytest.mark.parametrize("name", sorted(BAD_ENTRIES))
def test_strict_ground_truth_rules(tmp_path: Path, name: str) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path, ["r1,2026-11-10,1,Pricey,web\n"], {"r1": BAD_ENTRIES[name]}
    )

    with pytest.raises(FixtureBuildError, match="r1") as excinfo:
        build_fixtures(csv_path, truth_path)

    message = str(excinfo.value)
    for value in ("pricng", "Pricing", "bogus-sentiment", "Negative"):
        assert value not in message


@pytest.mark.parametrize("name", sorted(BAD_ENTRIES))
def test_main_writes_nothing_for_bad_ground_truth(
    tmp_path: Path, name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path, truth_path = write_inputs(
        tmp_path, ["r1,2026-11-10,1,Pricey,web\n"], {"r1": BAD_ENTRIES[name]}
    )
    out = tmp_path / "out.json"

    code = main(["--reviews", str(csv_path), "--truth", str(truth_path), "--out", str(out)])

    assert code == 1
    assert not out.exists()
    assert "Error:" in capsys.readouterr().err


def test_three_topics_are_accepted(tmp_path: Path) -> None:
    truth = {"sentiment": "negative", "topics": ["delivery_delay", "pricing", "other"]}
    csv_path, truth_path = write_inputs(tmp_path, ["r1,2026-11-10,1,Pricey,web\n"], {"r1": truth})

    payload = build_fixtures(csv_path, truth_path)

    assert next(iter(payload["analyses"].values())) == truth  # type: ignore[union-attr]


def test_generated_fixtures_pass_the_strict_loader(tmp_path: Path) -> None:
    from review_radar.fixture_classifier import FixtureClassifier

    out = tmp_path / "out.json"
    assert (
        main(["--reviews", str(SAMPLE_CSV), "--truth", str(SAMPLE_TRUTH), "--out", str(out)]) == 0
    )

    FixtureClassifier.from_file(out)
    assert out.read_bytes() == COMMITTED.read_bytes()
