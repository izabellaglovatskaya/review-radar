"""Dashboard: aggregation, the chart payload, the rendered page contract and the atomic write."""

import contextlib
import dataclasses
import datetime as dt
import json
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from review_radar import dashboard
from review_radar.dashboard import (
    CHART_JS,
    SITE_DIR,
    DashboardData,
    DashboardError,
    DayCounts,
    TopicRow,
    build_dashboard_data,
    chart_payload,
    render_dashboard,
    write_dashboard,
)
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review
from review_radar.storage import PendingText, StorageError, connect, save_analyses, save_reviews
from review_radar.summary import percentages

KEY = {"provider": "spy", "model": "spy-1", "prompt_version": "p1"}
DAY = dt.date(2026, 11, 10)
NEXT_DAY = dt.date(2026, 11, 11)
TODAY = dt.date(2026, 11, 12)
MARKER = "SECRET-REVIEW-TEXT"
BAD_URL = "https://evil.example/steal?x=<script>"

HOSTILE = [
    "</script><img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "\"'&<>",
    "<!--",
    "<script>",
    "a b c",
    "&lt;script&gt;&#x3c;",
]


# ---------------------------------------------------------------- HTML parsing helpers


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None]
    parent: int | None
    text: list[str] = field(default_factory=list)


class PageParser(HTMLParser):
    """Collects elements with their parents and text, comments and visible text."""

    VOID = frozenset({"meta", "link", "img", "br", "hr", "input", "source", "area", "col"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self.stack: list[int] = []
        self.comments: list[str] = []
        self.visible: list[str] = []
        self.outside_tables: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self.stack[-1] if self.stack else None
        self.elements.append(Element(tag, dict(attrs), parent))
        if tag not in self.VOID:
            self.stack.append(len(self.elements) - 1)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self.stack[-1] if self.stack else None
        self.elements.append(Element(tag, dict(attrs), parent))

    def handle_endtag(self, tag: str) -> None:
        for position in range(len(self.stack) - 1, -1, -1):
            if self.elements[self.stack[position]].tag == tag:
                del self.stack[position:]
                return

    def handle_data(self, data: str) -> None:
        if not self.stack:
            return
        tags = [self.elements[index].tag for index in self.stack]
        if tags[-1] in ("script", "style"):
            self.elements[self.stack[-1]].text.append(data)
            return
        for index in self.stack:
            self.elements[index].text.append(data)
        self.visible.append(data)
        if "table" not in tags:
            self.outside_tables.append(data)

    def handle_comment(self, data: str) -> None:
        self.comments.append(data)


class Page:
    def __init__(self, html: str) -> None:
        parser = PageParser()
        parser.feed(html)
        parser.close()
        self.html = html
        self.parser = parser
        self.elements = parser.elements

    def all(self, tag: str) -> list[Element]:
        return [element for element in self.elements if element.tag == tag]

    def index(self, element: Element) -> int:
        return next(i for i, item in enumerate(self.elements) if item is element)

    def children(self, element: Element, tag: str | None = None) -> list[Element]:
        position = self.index(element)
        return [
            item
            for item in self.elements
            if item.parent == position and (tag is None or item.tag == tag)
        ]

    def descendants(self, element: Element, tag: str) -> list[Element]:
        position = self.index(element)
        found = []
        for item in self.elements:
            if item.tag != tag:
                continue
            cursor = item.parent
            while cursor is not None:
                if cursor == position:
                    found.append(item)
                    break
                cursor = self.elements[cursor].parent
        return found

    def text(self, element: Element) -> str:
        return "".join(element.text)

    def visible_text(self) -> str:
        return "".join(self.parser.visible)

    def text_outside_tables(self) -> str:
        return "".join(self.parser.outside_tables)

    def rows(self, table: Element) -> list[list[str]]:
        result = []
        for row in self.descendants(table, "tr"):
            cells = [item for item in self.children(row) if item.tag in ("td", "th")]
            result.append([self.text(cell).strip(" \t\r\n") for cell in cells])
        return result

    def data_script(self) -> Element:
        found = [s for s in self.all("script") if s.attrs.get("id") == "dashboard-data"]
        assert len(found) == 1
        return found[0]


def parse(html: str) -> Page:
    return Page(html)


# ---------------------------------------------------------------- data helpers


def make_data(
    days: list[DayCounts] | None = None,
    topics: list[TopicRow] | None = None,
    *,
    demo: bool = False,
) -> DashboardData:
    """Build a consistent DashboardData from day rows, deriving the totals."""
    if days is None:
        days = [
            DayCounts("2026-11-09", 2, 1, 1, 0),
            DayCounts("2026-11-10", 1, 1, 2, 4),
            DayCounts("2026-11-11", 0, 1, 1, 3),
        ]
    if topics is None:
        topics = [TopicRow("Delivery delay", 3, 75), TopicRow("Billing", 1, 25)]
    positive = sum(d.positive for d in days)
    neutral = sum(d.neutral for d in days)
    negative = sum(d.negative for d in days)
    analyzed = positive + neutral + negative
    shares = percentages((positive, neutral, negative))
    return DashboardData(
        generated_on="2026-11-12",
        demo=demo,
        total=analyzed + sum(d.not_analyzed for d in days),
        analyzed=analyzed,
        positive=positive,
        neutral=neutral,
        negative=negative,
        shares=(shares[0], shares[1], shares[2]),
        days=tuple(days),
        topics=tuple(topics),
    )


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "dashboard.db")) as connection:
        yield connection


def add(
    conn: sqlite3.Connection,
    review_id: str,
    day: dt.date = DAY,
    sentiment: str | None = None,
    topics: tuple[str, ...] = (),
    key: dict[str, str] | None = None,
) -> None:
    text = f"{MARKER} {BAD_URL} {review_id}"
    review = Review(external_id=review_id, review_date=day, rating=3, text=text, source="web")
    save_reviews(conn, [review])
    if sentiment is not None:
        pending = PendingText(text_hash(text), text, 3, 1)
        analysis = Analysis(sentiment=sentiment, topics=list(topics), confidence=0.9)
        save_analyses(conn, [(pending, analysis)], **(key or KEY))


def build(conn: sqlite3.Connection, *, demo: bool = False) -> DashboardData:
    return build_dashboard_data(conn, **KEY, demo=demo, today=TODAY)


# ---------------------------------------------------------------- build_dashboard_data


class TestBuildDashboardData:
    def test_days_are_ascending_and_quiet_days_are_zero(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", NEXT_DAY, sentiment="positive")
        add(conn, "r2", DAY, sentiment="neutral")
        add(conn, "r3", DAY + dt.timedelta(days=3), sentiment="negative", topics=("billing",))

        data = build(conn)

        assert data.days == (
            DayCounts("2026-11-10", 0, 1, 0, 0),
            DayCounts("2026-11-11", 1, 0, 0, 0),
            DayCounts("2026-11-12", 0, 0, 0, 0),
            DayCounts("2026-11-13", 0, 0, 1, 0),
        )

    def test_a_date_range_over_the_limit_is_a_clean_error(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", DAY, sentiment="positive")
        add(conn, "r2", DAY + dt.timedelta(days=dashboard.MAX_DAYS), sentiment="positive")

        with pytest.raises(DashboardError, match="(?i)date range"):
            build(conn)

    def test_a_date_range_at_the_limit_is_allowed(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", DAY, sentiment="positive")
        add(conn, "r2", DAY + dt.timedelta(days=dashboard.MAX_DAYS - 1), sentiment="positive")

        assert len(build(conn).days) == dashboard.MAX_DAYS

    def test_counts_per_day_include_not_analyzed(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", DAY, sentiment="positive")
        add(conn, "r2", DAY, sentiment="negative", topics=("billing",))
        add(conn, "r3", DAY)
        add(conn, "r4", NEXT_DAY, sentiment="neutral")

        data = build(conn)

        assert data.days == (
            DayCounts("2026-11-10", 1, 0, 1, 1),
            DayCounts("2026-11-11", 0, 1, 0, 0),
        )

    def test_totals_and_metadata(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2", sentiment="negative", topics=("billing",))
        add(conn, "r3")

        data = build(conn, demo=True)

        assert (data.total, data.analyzed, data.not_analyzed) == (3, 2, 1)
        assert (data.positive, data.neutral, data.negative) == (1, 0, 1)
        assert data.demo is True
        assert data.generated_on == "2026-11-12"

    def test_shares_cover_analyzed_reviews_and_sum_to_100(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2", sentiment="neutral")
        add(conn, "r3", sentiment="negative", topics=("billing",))
        add(conn, "r4")

        data = build(conn)

        assert data.shares == (34, 33, 33)
        assert sum(data.shares) == 100

    def test_topics_come_from_negative_reviews_only(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive", topics=("packaging",))
        add(conn, "r2", sentiment="neutral", topics=("packaging",))
        add(conn, "r3", sentiment="negative", topics=("billing",))

        assert build(conn).topics == (TopicRow("Billing", 1, 100),)

    def test_topics_are_sorted_with_human_labels_and_shares(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("delivery_delay", "pricing"))
        add(conn, "r2", sentiment="negative", topics=("delivery_delay", "billing"))
        add(conn, "r3", sentiment="negative", topics=("billing",))
        add(conn, "r4", sentiment="negative", topics=("pricing",))
        add(conn, "r5", sentiment="positive")

        topics = build(conn).topics

        assert topics == (
            TopicRow("Billing", 2, 50),
            TopicRow("Delivery delay", 2, 50),
            TopicRow("Pricing", 2, 50),
        )

    def test_a_higher_count_sorts_first(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("pricing", "billing"))
        add(conn, "r2", sentiment="negative", topics=("pricing",))
        add(conn, "r3", sentiment="negative", topics=("pricing",))
        add(conn, "r4", sentiment="negative", topics=("delivery_delay",))

        topics = build(conn).topics

        assert [(t.label, t.count, t.share) for t in topics] == [
            ("Pricing", 3, 75),
            ("Billing", 1, 25),
            ("Delivery delay", 1, 25),
        ]

    def test_every_listed_topic_has_a_positive_count(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        add(conn, "r2", sentiment="positive")

        assert all(topic.count > 0 for topic in build(conn).topics)

    def test_no_negative_reviews_means_no_topics(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")

        assert build(conn).topics == ()

    def test_reviews_sharing_a_text_each_count(self, conn: sqlite3.Connection) -> None:
        text = "Same text"
        reviews = [
            Review(external_id=i, review_date=DAY, rating=1, text=text, source="web")
            for i in ("a", "b")
        ]
        save_reviews(conn, reviews)
        analysis = Analysis(sentiment="negative", topics=["billing"], confidence=0.9)
        save_analyses(conn, [(PendingText(text_hash(text), text, 1, 2), analysis)], **KEY)

        data = build(conn)

        assert (data.total, data.negative) == (2, 2)
        assert data.topics == (TopicRow("Billing", 2, 100),)

    def test_analyses_of_another_key_do_not_count(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2", sentiment="negative", topics=("billing",), key={**KEY, "model": "other"})

        data = build(conn)

        assert (data.total, data.analyzed, data.negative) == (2, 1, 0)

    def test_no_reviews_is_an_error(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(DashboardError, match="(?i)ingest"):
            build(conn)

    def test_no_analyzed_reviews_is_an_error(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1")

        with pytest.raises(DashboardError, match="(?i)analyze"):
            build(conn)

    def test_reviews_only_analyzed_under_another_key_is_an_error(
        self, conn: sqlite3.Connection
    ) -> None:
        add(conn, "r1", sentiment="positive", key={**KEY, "model": "other"})

        with pytest.raises(DashboardError):
            build(conn)

    def test_corrupt_stored_topics_are_a_clean_error(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        conn.execute("UPDATE analyses SET topics = 'not json'")
        conn.commit()

        with pytest.raises(StorageError) as caught:
            build(conn)

        assert MARKER not in str(caught.value)


class TestDashboardData:
    def test_not_analyzed_is_total_minus_analyzed(self) -> None:
        assert make_data().not_analyzed == 7

    def test_the_value_objects_are_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_data().total = 1  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            DayCounts("d", 1, 1, 1, 1).positive = 2  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            TopicRow("t", 1, 1).count = 2  # type: ignore[misc]


# ---------------------------------------------------------------- chart_payload


def strings_in(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*strings_in(k), *strings_in(v))]
    if isinstance(value, list | tuple):
        return [s for item in value for s in strings_in(item)]
    return []


class TestChartPayload:
    def test_has_the_three_chart_keys(self) -> None:
        assert set(chart_payload(make_data())) == {"days", "shares", "topics"}

    def test_is_json_serializable_and_round_trips(self) -> None:
        payload = chart_payload(make_data())

        assert json.loads(json.dumps(payload)) == payload

    def test_carries_the_aggregates(self) -> None:
        text = json.dumps(chart_payload(make_data()))

        for label in ("2026-11-09", "2026-11-10", "2026-11-11", "Delivery delay", "Billing"):
            assert label in text

    def test_holds_aggregates_only(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))

        text = json.dumps(chart_payload(build(conn)))

        assert MARKER not in text
        assert "evil.example" not in text

    def test_hostile_labels_survive_a_round_trip(self) -> None:
        for hostile in HOSTILE:
            data = make_data([DayCounts(hostile, 1, 1, 1, 0)], [TopicRow(hostile, 1, 100)])

            payload = json.loads(json.dumps(chart_payload(data)))

            assert strings_in(payload).count(hostile) >= 2


# ---------------------------------------------------------------- render_dashboard: contract


@pytest.fixture
def data() -> DashboardData:
    return make_data()


@pytest.fixture
def page(data: DashboardData) -> Page:
    return parse(render_dashboard(data))


class TestDocumentBasics:
    def test_language_viewport_title_and_description(self, page: Page) -> None:
        (html,) = page.all("html")
        assert html.attrs.get("lang") == "en"
        viewports = [m for m in page.all("meta") if m.attrs.get("name") == "viewport"]
        assert len(viewports) == 1
        assert viewports[0].attrs.get("content")
        (title,) = page.all("title")
        assert page.text(title).strip()
        descriptions = [m for m in page.all("meta") if m.attrs.get("name") == "description"]
        assert len(descriptions) == 1
        assert (descriptions[0].attrs.get("content") or "").strip()

    def test_one_h1_and_a_main(self, page: Page) -> None:
        assert len(page.all("h1")) == 1
        assert page.text(page.all("h1")[0]).strip()
        assert len(page.all("main")) == 1

    def test_no_comments(self, page: Page) -> None:
        assert page.parser.comments == []

    def test_rendering_is_deterministic(self, data: DashboardData) -> None:
        assert render_dashboard(data) == render_dashboard(data)


class TestContentSecurityPolicy:
    def csp(self, page: Page) -> str:
        (head,) = page.all("head")
        first = page.children(head)[0]
        assert first.tag == "meta"
        assert (first.attrs.get("http-equiv") or "").lower() == "content-security-policy"
        return first.attrs.get("content") or ""

    def test_it_is_the_first_element_in_head(self, page: Page) -> None:
        assert self.csp(page)
        policies = [
            m
            for m in page.all("meta")
            if (m.attrs.get("http-equiv") or "").lower() == "content-security-policy"
        ]
        assert len(policies) == 1

    @pytest.mark.parametrize(
        "directive",
        [
            "default-src 'none'",
            "script-src 'self'",
            "style-src 'self'",
            "connect-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "object-src 'none'",
        ],
    )
    def test_required_directives(self, page: Page, directive: str) -> None:
        directives = [part.strip() for part in self.csp(page).split(";")]
        assert directive in directives

    @pytest.mark.parametrize("forbidden", ["unsafe-inline", "unsafe-eval", "*", "http:", "https:"])
    def test_forbidden_tokens(self, page: Page, forbidden: str) -> None:
        assert forbidden not in self.csp(page)


class TestScriptsStylesAndUrls:
    def test_exactly_one_json_data_script(self, page: Page) -> None:
        script = page.data_script()
        assert script.attrs.get("type") == "application/json"
        assert "src" not in script.attrs

    def test_other_scripts_are_relative_deferred_and_empty(self, page: Page) -> None:
        others = [s for s in page.all("script") if s.attrs.get("id") != "dashboard-data"]

        assert [s.attrs.get("src") for s in others] == [CHART_JS, "assets/dashboard.js"]
        for script in others:
            assert "defer" in script.attrs
            assert page.text(script).strip() == ""

    def test_the_stylesheet_is_a_relative_file(self, page: Page) -> None:
        sheets = [
            link
            for link in page.all("link")
            if (link.attrs.get("rel") or "").lower() == "stylesheet"
        ]
        assert [s.attrs.get("href") for s in sheets] == ["assets/dashboard.css"]

    def test_no_inline_style_and_no_event_handlers(self, page: Page) -> None:
        assert page.all("style") == []
        for element in page.elements:
            assert "style" not in element.attrs, element.tag
            for name in element.attrs:
                assert not name.startswith("on"), (element.tag, name)

    def test_no_javascript_urls(self, page: Page) -> None:
        for element in page.elements:
            for value in element.attrs.values():
                assert "javascript:" not in (value or "").casefold()

    def test_every_src_and_href_is_a_relative_path(self, page: Page) -> None:
        checked = 0
        for element in page.elements:
            for name in ("src", "href"):
                value = element.attrs.get(name)
                if value is None:
                    continue
                checked += 1
                assert not value.startswith(("/", "\\")), value
                assert not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value), value
        assert checked >= 3


class TestFigures:
    IDS = ["chart-daily", "chart-shares", "chart-topics"]

    def figures(self, page: Page) -> list[Element]:
        found = page.all("figure")
        assert len(found) == 3
        return found

    def test_each_figure_has_caption_canvas_and_table(self, page: Page) -> None:
        for figure, expected in zip(self.figures(page), self.IDS, strict=True):
            assert len(page.descendants(figure, "figcaption")) == 1
            assert page.text(page.descendants(figure, "figcaption")[0]).strip()
            (canvas,) = page.descendants(figure, "canvas")
            assert canvas.attrs.get("id") == expected
            assert canvas.attrs.get("role") == "img"
            assert len((canvas.attrs.get("aria-label") or "").strip()) > 20
            (table,) = page.descendants(figure, "table")
            (caption,) = page.descendants(table, "caption")
            assert page.text(caption).strip()
            headers = page.descendants(table, "th")
            assert headers
            assert all(h.attrs.get("scope") in ("col", "row") for h in headers)

    def label_of(self, page: Page, index: int) -> str:
        (canvas,) = page.descendants(self.figures(page)[index], "canvas")
        return (canvas.attrs.get("aria-label") or "").casefold()

    def test_the_daily_label_states_the_numbers(self, page: Page) -> None:
        label = self.label_of(page, 0)

        for word in ("positive", "neutral", "negative", "not analyzed"):
            assert word in label
        for day in ("2026-11-09", "2026-11-10", "2026-11-11"):
            assert day in label

    def test_the_shares_label_states_the_percentages(self, page: Page) -> None:
        label = self.label_of(page, 1)

        for word in ("positive", "neutral", "negative"):
            assert word in label
        assert "30%" in label
        assert "40%" in label

    def test_the_topics_label_states_the_topics(self, page: Page) -> None:
        label = self.label_of(page, 2)

        assert "delivery delay" in label
        assert "billing" in label
        assert "3" in label

    def tables(self, page: Page) -> list[list[list[str]]]:
        return [page.rows(page.descendants(f, "table")[0]) for f in self.figures(page)]

    def test_the_daily_table_holds_the_same_numbers(self, page: Page, data: DashboardData) -> None:
        rows = self.tables(page)[0]

        for day in data.days:
            expected = [
                day.label,
                str(day.positive),
                str(day.neutral),
                str(day.negative),
                str(day.not_analyzed),
            ]
            assert any(row[:5] == expected for row in rows), expected

    def test_the_shares_table_holds_the_same_numbers(self, page: Page, data: DashboardData) -> None:
        rows = self.tables(page)[1]
        expected = {
            "positive": (data.positive, data.shares[0]),
            "neutral": (data.neutral, data.shares[1]),
            "negative": (data.negative, data.shares[2]),
        }

        for name, (count, share) in expected.items():
            row = next(r for r in rows if r and r[0].casefold() == name)
            assert str(count) in row
            assert any(cell in (str(share), f"{share}%") for cell in row)

    def test_the_topics_table_holds_the_same_numbers(self, page: Page, data: DashboardData) -> None:
        rows = self.tables(page)[2]

        for topic in data.topics:
            row = next(r for r in rows if r and r[0] == topic.label)
            assert str(topic.count) in row
            assert any(cell in (str(topic.share), f"{topic.share}%") for cell in row)

    def test_a_page_without_topics_still_has_three_figures(self) -> None:
        page = parse(render_dashboard(make_data(topics=[])))

        assert len(page.all("figure")) == 3
        assert len(page.all("table")) == 3


class TestEmbeddedData:
    @pytest.mark.parametrize("demo", [False, True])
    def test_the_json_equals_the_chart_payload(self, demo: bool) -> None:
        data = make_data(demo=demo)

        page = parse(render_dashboard(data))
        raw = page.text(page.data_script())

        assert json.loads(raw) == chart_payload(data)


class TestNotices:
    def test_a_demo_page_shows_a_visible_notice(self) -> None:
        page = parse(render_dashboard(make_data(demo=True)))

        assert "Demo data" in page.visible_text()

    def test_a_live_page_has_no_demo_notice(self) -> None:
        page = parse(render_dashboard(make_data(demo=False)))

        assert "demo data" not in page.visible_text().casefold()

    def test_partial_data_shows_the_not_analyzed_count(self) -> None:
        page = parse(render_dashboard(make_data()))  # 7 reviews are not analyzed

        text = page.text_outside_tables().casefold()
        assert re.search(r"\b7\b[^.\n]*not analyzed|not analyzed[^0-9.\n]*\b7\b", text)


class TestTemplateStrictness:
    def test_a_missing_value_fails_instead_of_rendering_blank(self) -> None:
        incomplete = SimpleNamespace(generated_on="2026-11-12", demo=False)

        with pytest.raises(jinja2.UndefinedError):
            render_dashboard(incomplete)  # type: ignore[arg-type]


# ---------------------------------------------------------------- XSS and privacy


def hostile_data(value: str) -> DashboardData:
    return make_data([DayCounts(value, 1, 1, 1, 0)], [TopicRow(value, 2, 100)])


def benign_data() -> DashboardData:
    return make_data([DayCounts("Day A", 1, 1, 1, 0)], [TopicRow("Topic A", 2, 100)])


def structure(page: Page) -> tuple[Counter[str], set[tuple[str, tuple[str, ...]]]]:
    tags = Counter(element.tag for element in page.elements)
    shapes = {(e.tag, tuple(sorted(e.attrs))) for e in page.elements}
    return tags, shapes


@pytest.mark.parametrize("hostile", HOSTILE)
class TestHostileLabels:
    def test_the_document_gains_no_elements_or_handlers(self, hostile: str) -> None:
        baseline = parse(render_dashboard(benign_data()))

        page = parse(render_dashboard(hostile_data(hostile)))

        assert structure(page) == structure(baseline)
        assert page.parser.comments == []
        for element in page.elements:
            assert not any(name.startswith("on") for name in element.attrs)

    def test_the_embedded_json_is_inert(self, hostile: str) -> None:
        page = parse(render_dashboard(hostile_data(hostile)))

        raw = page.text(page.data_script())

        assert "</script" not in raw.lower()
        assert "<!--" not in raw
        assert " " not in raw
        assert " " not in raw
        assert len(re.findall(r"</script", page.html, re.IGNORECASE)) == 3

    def test_the_json_round_trips_to_the_exact_string(self, hostile: str) -> None:
        data = hostile_data(hostile)
        page = parse(render_dashboard(data))

        payload = json.loads(page.text(page.data_script()))

        assert payload == chart_payload(data)
        assert strings_in(payload).count(hostile) >= 2

    def test_the_tables_show_the_literal_string(self, hostile: str) -> None:
        page = parse(render_dashboard(hostile_data(hostile)))
        daily, _shares, topics = (page.rows(t) for t in page.all("table"))

        assert hostile in [row[0] for row in daily if row]
        assert hostile in [row[0] for row in topics if row]


class TestPrivacy:
    def test_review_text_never_reaches_the_page_or_payload(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        add(conn, "r2", sentiment="positive")
        add(conn, "r3")
        data = build(conn)

        html = render_dashboard(data)
        payload = json.dumps(chart_payload(data))

        for secret in (MARKER, "evil.example", BAD_URL):
            assert secret not in html
            assert secret not in payload


# ---------------------------------------------------------------- write_dashboard


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "out"


def make_site(root: Path) -> Path:
    """A fake site folder holding every required asset."""
    for name in dashboard.REQUIRED_ASSETS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return root


class TestRequiredAssets:
    @pytest.mark.parametrize("missing", dashboard.REQUIRED_ASSETS)
    @pytest.mark.parametrize("into_site", [True, False])
    def test_a_missing_asset_is_named_and_keeps_the_old_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str, into_site: bool
    ) -> None:
        site = make_site(tmp_path / "fake-site")
        target = site if into_site else tmp_path / "elsewhere"
        target.mkdir(exist_ok=True)
        (target / "index.html").write_text("old", encoding="utf-8")
        (site / missing).unlink()
        monkeypatch.setattr(dashboard, "SITE_DIR", site)

        with pytest.raises(DashboardError, match=re.escape(missing)) as caught:
            write_dashboard("new", target)

        assert "checkout" in str(caught.value)
        assert (target / "index.html").read_text(encoding="utf-8") == "old"
        assert not list(target.glob(".index-*"))

    def test_the_list_covers_every_file_the_page_loads(self) -> None:
        page = parse(render_dashboard(make_data()))
        loaded = {
            value
            for element in page.elements
            for name in ("src", "href")
            if (value := element.attrs.get(name)) and value.startswith("assets/")
        }

        assert loaded == set(dashboard.REQUIRED_ASSETS)


class TestWriteDashboard:
    def test_writes_index_html_and_returns_its_path(self, out: Path) -> None:
        result = write_dashboard("<html>one</html>", out)

        assert result == out / "index.html"
        assert result.read_text(encoding="utf-8") == "<html>one</html>"

    def test_creates_a_missing_output_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b"

        write_dashboard("<html></html>", target)

        assert (target / "index.html").is_file()

    def test_writes_utf8(self, out: Path) -> None:
        write_dashboard("<html>naive é中</html>", out)

        assert (out / "index.html").read_bytes() == "<html>naive é中</html>".encode()

    def test_copies_the_assets(self, out: Path) -> None:
        write_dashboard("<html></html>", out)

        assert (out / CHART_JS).read_bytes() == (SITE_DIR / CHART_JS).read_bytes()
        assert (out / "assets" / "dashboard.js").is_file()
        assert (out / "assets" / "dashboard.css").is_file()

    def test_a_second_write_replaces_the_page_and_assets(self, out: Path) -> None:
        write_dashboard("<html>one</html>", out)
        stale = out / "assets" / "dashboard.css"
        stale.write_text("stale", encoding="utf-8")

        write_dashboard("<html>two</html>", out)

        assert (out / "index.html").read_text(encoding="utf-8") == "<html>two</html>"
        assert stale.read_text(encoding="utf-8") != "stale"

    def test_no_temp_file_is_left_behind(self, out: Path) -> None:
        write_dashboard("<html></html>", out)

        assert {p.name for p in out.iterdir()} == {"index.html", "assets"}

    def test_writing_into_site_dir_does_not_copy_assets_onto_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        site = make_site(tmp_path / "fake-site")
        asset = site / "assets" / "dashboard.css"
        asset.write_text("body{}", encoding="utf-8")
        monkeypatch.setattr(dashboard, "SITE_DIR", site)

        result = write_dashboard("<html></html>", site)

        assert result == site / "index.html"
        assert asset.read_text(encoding="utf-8") == "body{}"
        assert {p.name for p in site.iterdir()} == {"index.html", "assets"}

    def test_copies_from_a_custom_site_dir_elsewhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        site = make_site(tmp_path / "fake-site")
        (site / "assets" / "vendor" / "lib.js").write_text("x", encoding="utf-8")
        monkeypatch.setattr(dashboard, "SITE_DIR", site)

        write_dashboard("<html></html>", tmp_path / "elsewhere")

        assert (tmp_path / "elsewhere" / "assets" / "vendor" / "lib.js").read_text() == "x"


class TestWriteFailures:
    def test_a_failed_replace_keeps_the_old_page_and_leaves_no_temp_file(
        self, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_dashboard("<html>old</html>", out)

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(DashboardError):
            write_dashboard("<html>new</html>", out)

        monkeypatch.undo()
        assert (out / "index.html").read_text(encoding="utf-8") == "<html>old</html>"
        assert {p.name for p in out.iterdir()} <= {"index.html", "assets"}

    def test_an_output_path_that_is_a_file_is_a_dashboard_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(DashboardError):
            write_dashboard("<html></html>", blocker)

        assert blocker.read_text(encoding="utf-8") == "not a directory"

    def test_an_index_that_is_a_directory_is_a_dashboard_error(self, out: Path) -> None:
        (out / "index.html").mkdir(parents=True)

        with pytest.raises(DashboardError):
            write_dashboard("<html></html>", out)

        assert (out / "index.html").is_dir()
        assert [p.name for p in out.iterdir() if p.name not in ("index.html", "assets")] == []

    def test_the_error_is_an_exception_type_of_its_own(self) -> None:
        assert issubclass(DashboardError, Exception)
        assert not issubclass(DashboardError, OSError)
