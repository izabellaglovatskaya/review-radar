"""The built page in a real headless Chrome. Opt-in: `uv run --group browser pytest -m browser`.

Needs a local Chrome. Skipped when Playwright or Chrome is missing, unless
`REVIEW_RADAR_REQUIRE_BROWSER=1` (set in CI) makes that a failure.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from browser_support import unavailable

try:
    from playwright import sync_api
except ImportError:
    unavailable("Playwright is not installed")

from review_radar.dashboard import (  # noqa: E402
    DashboardData,
    DayCounts,
    TopicRow,
    render_dashboard,
    write_dashboard,
)
from review_radar.summary import percentages  # noqa: E402

pytestmark = pytest.mark.browser

CHART_IDS = ["chart-daily", "chart-shares", "chart-topics"]
HOSTILE = "</script><img src=x onerror=window.__xss=1><svg onload=window.__xss=1>\"'&<>"
LOAD_TIMEOUT_MS = 15_000

RECORD_CSP = """
window.__csp = [];
document.addEventListener('securitypolicyviolation', (event) => {
    window.__csp.push(event.violatedDirective + ' ' + event.blockedURI);
});
"""

CHART_STATE = """
(ids) => ids.map((id) => {
    const chart = window.Chart && window.Chart.getChart(document.getElementById(id));
    if (!chart) return null;
    return {
        labels: chart.data.labels,
        datasets: chart.data.datasets.map((set) => set.data),
    };
})
"""

ALL_CHARTS_READY = """
(ids) => window.Chart && ids.every((id) => window.Chart.getChart(document.getElementById(id)))
"""


def make_data(
    days: list[DayCounts] | None = None, topics: list[TopicRow] | None = None
) -> DashboardData:
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
        demo=False,
        total=analyzed + sum(d.not_analyzed for d in days),
        analyzed=analyzed,
        positive=positive,
        neutral=neutral,
        negative=negative,
        shares=(shares[0], shares[1], shares[2]),
        days=tuple(days),
        topics=tuple(topics),
    )


def build_page(directory: Path, data: DashboardData) -> str:
    index = write_dashboard(render_dashboard(data), directory)
    return index.resolve().as_uri()


@pytest.fixture
def browser() -> Iterator["sync_api.Browser"]:
    with sync_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(channel="chrome", headless=True)
        except sync_api.Error as exc:
            unavailable(f"Chrome is not available: {str(exc).splitlines()[0]}")
        try:
            yield launched
        finally:
            launched.close()


class Session:
    """One open page with every console message, page error and request recorded."""

    def __init__(
        self,
        browser: "sync_api.Browser",
        url: str,
        *,
        width: int = 1280,
        js: bool = True,
        wait_for_charts: bool = True,
    ) -> None:
        self.context = browser.new_context(
            viewport={"width": width, "height": 900}, java_script_enabled=js
        )
        self.page = self.context.new_page()
        self.console_errors: list[str] = []
        self.page_errors: list[str] = []
        self.requests: list[str] = []
        self.page.on(
            "console",
            lambda message: (
                self.console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.on("request", lambda request: self.requests.append(request.url))
        self.page.add_init_script(RECORD_CSP)
        self.page.goto(url)
        if js and wait_for_charts:
            self.page.wait_for_function(ALL_CHARTS_READY, arg=CHART_IDS, timeout=LOAD_TIMEOUT_MS)

    def close(self) -> None:
        self.context.close()


Opener = Callable[..., Session]


@pytest.fixture
def open_page(browser: "sync_api.Browser") -> Iterator[Opener]:
    sessions: list[Session] = []

    def opener(
        url: str, *, width: int = 1280, js: bool = True, wait_for_charts: bool = True
    ) -> Session:
        session = Session(browser, url, width=width, js=js, wait_for_charts=wait_for_charts)
        sessions.append(session)
        return session

    yield opener
    for session in sessions:
        session.close()


def flatten(values: list[list[float]]) -> list[float]:
    return [value for series in values for value in series]


def test_chart_js_builds_three_charts_with_the_right_data(
    tmp_path: Path, open_page: Opener
) -> None:
    data = make_data()
    session = open_page(build_page(tmp_path, data))

    daily, shares, topics = session.page.evaluate(CHART_STATE, CHART_IDS)

    assert daily["labels"] == [d.label for d in data.days]
    series = daily["datasets"]
    assert [d.positive for d in data.days] in series
    assert [d.neutral for d in data.days] in series
    assert [d.negative for d in data.days] in series
    assert flatten(shares["datasets"]) == list(data.shares)
    assert topics["labels"] == [t.label for t in data.topics]
    assert flatten(topics["datasets"]) == [t.count for t in data.topics]


def test_the_page_loads_cleanly_with_no_csp_violations(tmp_path: Path, open_page: Opener) -> None:
    session = open_page(build_page(tmp_path, make_data()))

    assert session.console_errors == []
    assert session.page_errors == []
    assert session.page.evaluate("window.__csp") == []


def test_every_request_uses_the_file_scheme(tmp_path: Path, open_page: Opener) -> None:
    session = open_page(build_page(tmp_path, make_data()))

    assert session.requests
    assert all(url.startswith("file:") for url in session.requests), session.requests


def test_hostile_labels_run_nothing_and_inject_nothing(tmp_path: Path, open_page: Opener) -> None:
    data = make_data([DayCounts(HOSTILE, 1, 1, 1, 0)], [TopicRow(HOSTILE, 2, 100)])
    session = open_page(build_page(tmp_path, data))
    page = session.page

    assert page.evaluate("typeof window.__xss") == "undefined"
    assert page.locator("img[src='x']").count() == 0
    assert page.locator("svg").count() == 0
    assert page.evaluate("document.querySelectorAll('[onerror], [onload]').length") == 0
    assert page.evaluate("window.__csp") == []
    assert session.page_errors == []
    daily, _shares, topics = page.evaluate(CHART_STATE, CHART_IDS)
    assert daily["labels"] == [HOSTILE]
    assert topics["labels"] == [HOSTILE]
    cells = page.locator("table td, table th").all_inner_texts()
    assert HOSTILE in [cell.strip() for cell in cells]


def test_there_is_no_horizontal_scroll_on_a_phone(tmp_path: Path, open_page: Opener) -> None:
    session = open_page(build_page(tmp_path, make_data()), width=375)

    overflow = session.page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )

    assert overflow <= 0


def test_table_words_and_dates_do_not_break_on_a_phone(tmp_path: Path, open_page: Opener) -> None:
    session = open_page(build_page(tmp_path, make_data()), width=375)

    broken = session.page.evaluate(
        """() => [...document.querySelectorAll('table')[0].querySelectorAll('th')]
            .filter((cell) => !cell.textContent.includes(' '))
            .filter((cell) => {
                const range = document.createRange();
                range.selectNodeContents(cell);
                return range.getClientRects().length > 1;
            })
            .map((cell) => cell.textContent)"""
    )

    assert broken == []


def test_drawing_the_charts_does_not_shift_the_layout(tmp_path: Path, open_page: Opener) -> None:
    url = build_page(tmp_path, make_data())
    last_table_top = "() => document.querySelectorAll('table')[2].getBoundingClientRect().top"
    drawn = open_page(url, width=375).page.evaluate(last_table_top)

    (tmp_path / "assets" / "dashboard.js").unlink()  # the state before the script has run
    before_script = open_page(url, width=375, wait_for_charts=False).page.evaluate(last_table_top)

    assert abs(drawn - before_script) < 1


def test_a_missing_chart_library_shows_the_notice_and_keeps_the_tables(
    tmp_path: Path, browser: "sync_api.Browser"
) -> None:
    url = build_page(tmp_path, make_data())
    (tmp_path / "assets" / "vendor" / "chart-4.5.1.umd.min.js").unlink()
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()

    page.goto(url)
    page.wait_for_selector("#chart-error:not([hidden])")

    assert page.locator("table").count() == 3
    assert page.locator("table").first.is_visible()
    assert all(box is None for box in (c.bounding_box() for c in page.locator("canvas").all()))
    context.close()


def test_the_tables_stay_visible_without_javascript(tmp_path: Path, open_page: Opener) -> None:
    session = open_page(build_page(tmp_path, make_data()), js=False)

    tables = session.page.locator("table")

    assert tables.count() == 3
    for index in range(3):
        assert tables.nth(index).is_visible()
    assert session.page.locator("h1").is_visible()
