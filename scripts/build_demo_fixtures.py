"""Build `fixtures/demo_analyses.json` from the sample reviews and their ground truth.

Demo mode (`FixtureClassifier`) answers from this file, keyed by `text_hash(text)`, so a demo
run on the sample is exact by construction. The output is deterministic: sorted keys, LF line
endings, one trailing newline. Two reviews that share a text hash must share their ground
truth, otherwise the script stops with an error.

Usage: uv run python scripts/build_demo_fixtures.py
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from review_radar.fixture_classifier import parse_strict_entry
from review_radar.hashing import text_hash
from review_radar.ingest import IngestError, read_reviews

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWS = ROOT / "data" / "sample" / "reviews.csv"
DEFAULT_TRUTH = ROOT / "data" / "sample" / "ground_truth.json"
DEFAULT_OUT = ROOT / "fixtures" / "demo_analyses.json"
FORMAT = 1


class FixtureBuildError(Exception):
    """The inputs cannot be turned into fixtures. Messages never contain review text."""


def _load_truth(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FixtureBuildError(f"Cannot read ground truth from {path.name}.") from None
    reviews = raw.get("reviews") if isinstance(raw, Mapping) else None
    if not isinstance(reviews, Mapping):
        raise FixtureBuildError(f"{path.name} has no 'reviews' object.")
    return reviews


def _entry(review_id: str, truth: object) -> dict[str, object]:
    """Validate one ground-truth entry strictly (same rules as the fixture loader)."""
    try:
        analysis = parse_strict_entry(truth)
    except ValueError as exc:  # messages carry counts and positions only
        raise FixtureBuildError(f"Invalid ground truth for review {review_id}: {exc}.") from None
    return {"sentiment": analysis.sentiment.value, "topics": [t.value for t in analysis.topics]}


def build_fixtures(reviews_path: Path, truth_path: Path) -> dict[str, object]:
    """Return the fixture payload for the reviews in `reviews_path`.

    Fails when any CSV row was skipped, a review has no id or no ground truth, or the ground
    truth holds ids that no review uses, so the fixtures always match the data exactly.
    """
    try:
        result = read_reviews(reviews_path)
    except IngestError as exc:
        raise FixtureBuildError(f"Cannot read reviews: {exc}") from None
    if result.errors:
        raise FixtureBuildError(f"{len(result.errors)} CSV row(s) were skipped; fix the file.")
    truth = _load_truth(truth_path)
    analyses: dict[str, dict[str, object]] = {}
    used: set[str] = set()
    for review in result.reviews:
        if review.external_id is None:
            raise FixtureBuildError("Every review needs a review_id to find its ground truth.")
        if review.external_id not in truth:
            raise FixtureBuildError(f"No ground truth for review {review.external_id}.")
        used.add(review.external_id)
        entry = _entry(review.external_id, truth[review.external_id])
        key = text_hash(review.text)
        if analyses.setdefault(key, entry) != entry:
            raise FixtureBuildError(
                f"Review {review.external_id} shares its text with another review "
                "that has different ground truth."
            )
    if unused := set(truth) - used:
        raise FixtureBuildError(f"{len(unused)} ground-truth id(s) are unused by any review.")
    return {"format": FORMAT, "analyses": analyses}


def render(payload: Mapping[str, object]) -> str:
    """Return the canonical JSON text for `payload`."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build demo fixtures from the sample ground truth."
    )
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        payload = build_fixtures(args.reviews, args.truth)
    except FixtureBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(render(payload).encode("utf-8"))
    analyses = payload["analyses"]
    count = len(analyses) if isinstance(analyses, Mapping) else 0
    print(f"Wrote {count} fixtures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
