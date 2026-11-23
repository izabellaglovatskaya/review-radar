"""Static dashboard: aggregates over all reviews rendered to one self-contained page.

Privacy rule: the page holds counts, dates and names from the closed topic list only. Review
text can contain harmful links and mentions, so it is never loaded here.

Security rule: Jinja2 autoescaping is on, chart data is embedded with `tojson` in a non-executing
`application/json` block, and the page has a strict CSP with no inline code or external requests.
"""

import contextlib
import datetime as dt
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from review_radar.config import PROJECT_ROOT
from review_radar.models import Sentiment
from review_radar.storage import load_daily_analyses
from review_radar.summary import TOPIC_LABELS, percentages

SITE_DIR = PROJECT_ROOT / "site"
CHART_JS = "assets/vendor/chart-4.5.1.umd.min.js"
REQUIRED_ASSETS = (
    CHART_JS,
    "assets/dashboard.js",
    "assets/dashboard.css",
    "assets/favicon.svg",
)
MAX_DAYS = 3660  # one mistyped date must not produce a page with tens of thousands of rows

_ENVIRONMENT = Environment(
    loader=PackageLoader("review_radar", "templates"),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class DashboardError(Exception):
    """The page cannot be built or written. Never carries review text."""


@dataclass(frozen=True)
class DayCounts:
    """Review counts for one day."""

    label: str
    positive: int
    neutral: int
    negative: int
    not_analyzed: int


@dataclass(frozen=True)
class TopicRow:
    """A complaint topic, its count and its whole-percent share of negative reviews."""

    label: str
    count: int
    share: int


@dataclass(frozen=True)
class DashboardData:
    """Aggregates for the page. Sentiment counts and shares cover analyzed reviews only."""

    generated_on: str
    demo: bool
    total: int
    analyzed: int
    positive: int
    neutral: int
    negative: int
    shares: tuple[int, int, int]
    days: tuple[DayCounts, ...]
    topics: tuple[TopicRow, ...]

    @property
    def not_analyzed(self) -> int:
        """Reviews that have no analysis for the current classifier."""
        return self.total - self.analyzed


def build_dashboard_data(
    conn: sqlite3.Connection,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    demo: bool,
    today: dt.date,
) -> DashboardData:
    """Aggregate all reviews using the analyses of one classifier key."""
    rows = load_daily_analyses(conn, provider=provider, model=model, prompt_version=prompt_version)
    if not rows:
        raise DashboardError("No reviews in the database; run ingest first.")
    per_day: dict[dt.date, Counter[str]] = {}
    topics: Counter[str] = Counter()
    for day, analysis in rows:
        counts = per_day.setdefault(day, Counter())
        if analysis is None:
            counts["not_analyzed"] += 1
            continue
        counts[analysis.sentiment.value] += 1
        if analysis.sentiment is Sentiment.NEGATIVE:
            topics.update(TOPIC_LABELS[topic] for topic in analysis.topics)
    days = _fill_days(per_day)
    positive = sum(day.positive for day in days)
    neutral = sum(day.neutral for day in days)
    negative = sum(day.negative for day in days)
    analyzed = positive + neutral + negative
    if analyzed == 0:
        raise DashboardError("No review is analyzed yet; run analyze first.")
    first, second, third = percentages((positive, neutral, negative))
    ranked = sorted(topics.items(), key=lambda item: (-item[1], item[0]))
    return DashboardData(
        generated_on=today.isoformat(),
        demo=demo,
        total=len(rows),
        analyzed=analyzed,
        positive=positive,
        neutral=neutral,
        negative=negative,
        shares=(first, second, third),
        days=days,
        topics=tuple(
            TopicRow(label, count, (count * 200 + negative) // (2 * negative))
            for label, count in ranked
        ),
    )


def _fill_days(per_day: dict[dt.date, Counter[str]]) -> tuple[DayCounts, ...]:
    """One row per calendar day from first to last, so quiet days show as zero on the axis."""
    first, last = min(per_day), max(per_day)
    span = (last - first).days + 1
    if span > MAX_DAYS:
        raise DashboardError(
            f"The review date range is over {MAX_DAYS} days; check the dates in the data."
        )
    rows = []
    for offset in range(span):
        day = first + dt.timedelta(days=offset)
        counts = per_day.get(day, Counter())
        rows.append(
            DayCounts(
                day.isoformat(),
                counts["positive"],
                counts["neutral"],
                counts["negative"],
                counts["not_analyzed"],
            )
        )
    return tuple(rows)


def chart_payload(data: DashboardData) -> dict[str, list[dict[str, str | int]]]:
    """The aggregates the charts need, as JSON-ready values."""
    counts = (data.positive, data.neutral, data.negative)
    names = ("Positive", "Neutral", "Negative")
    return {
        "days": [
            {
                "label": day.label,
                "positive": day.positive,
                "neutral": day.neutral,
                "negative": day.negative,
                "not_analyzed": day.not_analyzed,
            }
            for day in data.days
        ],
        "shares": [
            {"label": name, "count": count, "share": share}
            for name, count, share in zip(names, counts, data.shares, strict=True)
        ],
        "topics": [
            {"label": topic.label, "count": topic.count, "share": topic.share}
            for topic in data.topics
        ],
    }


def _daily_label(data: DashboardData) -> str:
    parts = [
        f"{day.label}: {day.positive} positive, {day.neutral} neutral, "
        f"{day.negative} negative, {day.not_analyzed} not analyzed"
        for day in data.days
    ]
    return "Line chart of reviews per day. " + "; ".join(parts) + "."


def _shares_label(data: DashboardData) -> str:
    return (
        f"Bar chart of sentiment shares: positive {data.shares[0]}%, "
        f"neutral {data.shares[1]}%, negative {data.shares[2]}%."
    )


def _topics_label(data: DashboardData) -> str:
    if not data.topics:
        return "Bar chart of complaint topics: there are no complaint topics."
    parts = [f"{topic.label}: {topic.count} ({topic.share}%)" for topic in data.topics]
    return "Bar chart of complaint topics in negative reviews. " + "; ".join(parts) + "."


def render_dashboard(data: DashboardData) -> str:
    """Render the page; a missing value raises instead of rendering blank."""
    template = _ENVIRONMENT.get_template("dashboard.html.j2")
    return template.render(
        data=data,
        chart_js=CHART_JS,
        chart_payload=chart_payload,
        daily_label=_daily_label,
        shares_label=_shares_label,
        topics_label=_topics_label,
    )


def write_dashboard(html: str, out_dir: Path) -> Path:
    """Write `index.html` atomically and make sure the assets are next to it."""
    missing = [name for name in REQUIRED_ASSETS if not (SITE_DIR / name).is_file()]
    if missing:
        raise DashboardError(
            f"Missing dashboard file(s): {', '.join(missing)}. Run from a repository checkout."
        )
    index = out_dir / "index.html"
    temporary: str | None = None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=out_dir, prefix=".index-", suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(html)
        os.chmod(temporary, 0o644)
        if out_dir.resolve() != SITE_DIR.resolve():
            shutil.copytree(SITE_DIR / "assets", out_dir / "assets", dirs_exist_ok=True)
        os.replace(temporary, index)
    except OSError as exc:
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
        reason = exc.strerror or type(exc).__name__
        raise DashboardError(f"Cannot write the dashboard: {reason}.") from None
    return index
