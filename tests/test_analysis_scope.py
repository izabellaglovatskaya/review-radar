"""`analyze_pending(text_hashes=...)` classifies only the selected texts."""

import contextlib
import datetime as dt
from collections.abc import Sequence
from pathlib import Path

from review_radar.analysis import analyze_pending
from review_radar.classifier import ClassifyItem, ClassifyResult
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review, Sentiment
from review_radar.storage import connect, save_reviews


class Spy:
    provider = "spy"
    model = "spy-1"
    prompt_version = "p1"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.texts += [item.text for item in items]
        answer = Analysis(sentiment=Sentiment.POSITIVE, topics=[], confidence=0.9)
        return ClassifyResult(analyses={item.item_id: answer for item in items})


def review(review_id: str, text: str) -> Review:
    return Review(
        external_id=review_id,
        review_date=dt.date(2026, 11, 10),
        rating=3,
        text=text,
        source="web",
    )


def test_only_the_selected_texts_are_classified(tmp_path: Path) -> None:
    spy = Spy()
    with contextlib.closing(connect(tmp_path / "scope.db")) as conn:
        save_reviews(conn, [review("a", "first"), review("b", "second"), review("c", "third")])

        report = analyze_pending(conn, spy, text_hashes={text_hash("second")})

    assert spy.texts == ["second"]
    assert report.analyzed == 1
    assert report.failed == 0
    assert report.not_analyzed_limit == 2  # left for a later run, not a failure
    assert report.reviews_total == report.analyzed + report.cached + report.not_analyzed_limit


def test_an_empty_selection_classifies_nothing(tmp_path: Path) -> None:
    spy = Spy()
    with contextlib.closing(connect(tmp_path / "scope.db")) as conn:
        save_reviews(conn, [review("a", "first")])

        report = analyze_pending(conn, spy, text_hashes=set())

    assert spy.texts == []
    assert report.analyzed == 0
    assert report.not_analyzed_limit == 1


def test_the_limit_applies_within_the_selection(tmp_path: Path) -> None:
    spy = Spy()
    with contextlib.closing(connect(tmp_path / "scope.db")) as conn:
        save_reviews(conn, [review("a", "first"), review("b", "second"), review("c", "third")])

        report = analyze_pending(
            conn, spy, limit=1, text_hashes={text_hash("second"), text_hash("third")}
        )

    assert spy.texts == ["second"]
    assert report.not_analyzed_limit == 2
