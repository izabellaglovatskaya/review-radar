"""Guards for the documentation: links that resolve, required sections, an unfilled passport."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"
PASSPORT = ROOT / "PASSPORT.md"
DOCS = list(dict.fromkeys([README, PASSPORT, ARCHITECTURE, *sorted((ROOT / "docs").glob("*.md"))]))

# [text](target) and ![alt](target); the target ends at whitespace or ")".
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
CI_BADGE = (
    "https://github.com/izabellaglovatskaya/review-radar/actions/workflows/ci.yml/badge.svg"
    "?branch=main"
)
LICENSE = ROOT / "LICENSE"
AGENTS_LICENSE = ".claude/agents/LICENSE-everything-claude-code.txt"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _links(path: Path) -> list[str]:
    return LINK.findall(FENCE.sub("", _text(path)))


def _anchors(path: Path) -> set[str]:
    """GitHub-style heading anchors: lower case, punctuation dropped, spaces to hyphens."""
    headings = re.findall(r"^#{1,6} (.+)$", FENCE.sub("", _text(path)), re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in headings}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_point_to_existing_files_and_headings(doc: Path) -> None:
    assert doc.exists(), doc
    for target in _links(doc):
        if re.match(r"[a-z]+:", target):  # https:, mailto: are checked by hand
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (doc.parent / file_part).resolve() if file_part else doc
        assert resolved.exists(), f"{doc.name}: {target}"
        if anchor:
            assert anchor in _anchors(resolved), f"{doc.name}: {target}"


def test_readme_has_the_ci_badge_and_the_required_sections() -> None:
    text = _text(README)

    assert CI_BADGE in text
    for heading in ("## Demo mode", "## Live mode", "## Commands", "## Limitations"):
        assert re.search(rf"^{re.escape(heading)}\b", text, re.MULTILINE), heading
    assert "## Model evaluation" in text


def test_architecture_has_mermaid_diagrams_for_flow_schema_and_trust() -> None:
    text = _text(ARCHITECTURE)

    assert text.count("```mermaid") >= 3
    assert "erDiagram" in text
    assert re.search(r"^## Trust boundaries", text, re.MULTILINE)


def test_passport_has_ten_questions_each_with_an_answer_field() -> None:
    # Structure only: the author fills in the answers, so their content is not checked.
    text = _text(PASSPORT)
    sections = re.split(r"^## (?=\d+\. )", text, flags=re.MULTILINE)[1:]

    assert [int(s.split(".", 1)[0]) for s in sections] == list(range(1, 11))
    for section in sections:
        assert section.count("**My answer:**") == 1, section.splitlines()[0]


def test_passport_question_3_asks_why_a_failure_is_not_cached() -> None:
    text = _text(PASSPORT)

    question = r"^## 3\. .+\n\nWhy must a failed result never be put in the cache\?$"

    assert re.search(question, text, re.MULTILINE)


def test_license_is_mit_held_by_the_github_account_only() -> None:
    text = _text(LICENSE)

    assert text.startswith("MIT License\n")
    assert re.search(r"^Copyright \(c\) \d{4} izabellaglovatskaya$", text, re.MULTILINE)


def test_readme_names_the_license_and_the_separate_agents_license() -> None:
    text = _text(README)
    section = text.split("## License", 1)[1]

    assert "[LICENSE](LICENSE)" in section
    assert f"({AGENTS_LICENSE})" in section
