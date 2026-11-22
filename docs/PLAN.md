# Review Radar — Plan

## Goal

Input: a CSV file with reviews of a fictional company. Output:

1. a daily Telegram summary: number of positive / neutral / negative reviews and the top 3 complaint topics;
2. a static HTML dashboard with sentiment and topic charts over time.

## Decisions

| Topic | Decision |
|---|---|
| Shape | CLI pipeline (`ingest → analyze → store → report / dashboard`), no web server. The data changes once per run, so a static page is simpler and honest. A FastAPI endpoint is a possible extension, not built. |
| Python | 3.13 via uv, src layout, package `review_radar`, CLI with Typer. |
| Storage | SQLite with a simple schema and a schema version in `PRAGMA user_version`. No migration framework. |
| LLM | Anthropic API behind a `Classifier` protocol. Default model `claude-haiku-5-5`. Key only in `.env`. |
| Demo mode | `FixtureClassifier` with canned answers keyed by review hash, plus a deterministic keyword/rating fallback. No key, no network, no cost. |
| Output format | Structured JSON per batch of 20 reviews, short ids (`r1…r20`), closed topic list, validated again with pydantic. |
| Cache | Analyses cached by (review hash, provider, model, prompt version). A rerun on the same CSV makes 0 calls. |
| Telegram | Send only (no commands). Idempotent: one summary per date unless `--force`. `--dry-run` prints the message. |
| Dashboard | Jinja2 template with embedded JSON and a vendored Chart.js. Output `site/index.html`, opens from disk. |
| Eval | `review-radar eval` compares predictions with generator ground truth; `--compare` runs Haiku 5.5 and Sonnet 5.5 on the same data and prints an accuracy / tokens / cost table for the README. |
| CI | GitHub Actions on Ubuntu only: `uv sync --locked`, ruff, pytest, and gitleaks over the full git history. |
| Publishing | No GitHub Pages for now (until the repository is made public). README uses screenshots. |
| Scheduling | The interface is `review-radar run`. Windows Task Scheduler setup is documented, not automated. |

## Stages

One stage = one commit. Tests first (tdd-guide), code review after each stage.

| # | Stage | Estimate (h) |
|---|---|---|
| 0 | Scaffold: uv, ruff, pytest, gitleaks pre-commit, `.env.example`, `review-radar --version` | 1.5–2 |
| 1 | Domain models, settings (`SecretStr`), text normalization and hashing | 2 |
| 2 | Seeded synthetic review generator with ground truth; CSV ingest (BOM, `;`, bad rows) | 3–4 |
| 3 | SQLite schema with `user_version`, repository functions | 2.5 |
| 4 | `Classifier` protocol, fixture classifier, analysis service with batching and cache | 3–4 |
| 5 | Anthropic classifier: structured output, parsing, split-and-retry, usage tracking | 4–5 |
| 6 | Daily summary and Telegram sender | 3–4 |
| 7 | Static dashboard | 4–5 |
| 8 | End-to-end `run --demo`; `eval` with Haiku vs Sonnet comparison table | 3–4 |
| 9 | CI on Ubuntu: `uv sync --locked`, ruff, pytest, gitleaks over the full history | 1–1.5 |
| 10 | README, architecture notes, screenshots, cost table | 2–3 |

Total: 29–37 h, 33–43 h with review rounds.

## Cost estimate (live mode)

About 130 input and 31 output tokens per review with batches of 20 (to be measured with the `runs` table).
One run over 300 reviews: about 39k input and 9.3k output tokens.

| Model | $ per 1M tokens (in / out) | 300 reviews |
|---|---|---|
| Claude Haiku 5.5 | 0.10 / 0.50 | ≈ $0.01 |
| Claude Sonnet 5.5 | 2.00 / 10.00 | ≈ $0.17 |

Reruns on the same data cost nothing (cache). Demo mode costs nothing.

## Review backlog (LOW findings)

Recorded after each stage review; picked up in the stage named in brackets.

### Stage 1

- `models.py`: the topic validator accepts any iterable (`bytes`, `dict`, `set`); accept only `list | tuple` [5].
- ~~`models.py`: blank `external_id` / `source` are stored as-is; map blank strings to `None` [2].~~ Done in stage 2.
- ~~`models.py`: `rating` accepts `True` (lax mode); use a strict int or rely on CSV ingest passing strings [2].~~ Done in stage 2.
- `models.py`, `config.py`: validators take `value: Any`; use `object` for untrusted input [any].
- `models.py`, `config.py`: public classes need one-line docstrings [any].
- `config.py`: `REVIEW_RADAR_MODE` is case-sensitive (`LIVE` fails); lowercase and strip before validation [8].
- `config.py`: model names accept empty or whitespace values; add `min_length=1` with stripping [5].
- `config.py`: non-blank secrets are not stripped (a pasted trailing newline breaks auth later) [5].
- `config.py`: log the resolved mode and database path when the CLI starts [8].
- Tests: always build `Settings` with `_env_file=None`, so a developer's real `.env` is never read [all].
- Tests: nothing asserts that `Settings` is frozen [any].
- Tests contain `# type: ignore` comments but no type checker runs; add mypy/pyright to dev deps and checks, or drop the comments [9].
- `pyproject.toml`: dependency floors are the newest releases (`pydantic>=2.14.4`, `pydantic-settings>=2.16.3`); lower them to the oldest versions actually needed [10].

### Stage 2

- `ingest.py`: no file-size cap before `read_bytes()`; add `MAX_FILE_BYTES` and raise `IngestError` above it [3].
- `ingest.py`: a directory path gives "Permission denied" on Windows; check `is_file()` first and say "Not a file" [3].
- `ingest.py`: with `--encoding utf-8` on a BOM file the error is a misleading "Missing required column"; strip a leading BOM after decoding [3].
- `ingest.py`: a field over 131072 characters aborts the whole file; reword the message to say the field is too large [3].
- `ingest.py`: the delimiter is sniffed with `str.splitlines()`, which splits on more characters than csv; use the first csv-parsed row [3].
- `ingest.py`: a trailing delimiter on data rows (Excel `;` exports) makes every row `wrong_field_count`; consider tolerating trailing empty cells [3].
- `models.py`: `external_id` and `source` have no length limit; add `max_length` before they reach the database [3].
- `ingest.py`: after a recoverable malformed row, a contrived tail with an even number of quotes that still opens a quoted field could leave a bogus continuation row; it is usually counted as `wrong_field_count`, but could be imported if it happens to be a valid row. Accepted risk; revisit if real exports hit it [any].
- `cli.py`: `ingest` requires `--dry-run` until storage exists; make saving the default in stage 3 [3].
- Storage: decide the idempotency key for re-imports (unique `external_id`, fallback hash of date + text + source for rows without an id) so re-running ingest never double-counts the daily summary [3].
- `MAX_TEXT_LENGTH` (2000 chars) is not tied to the token budget; revisit with the prompt [5].
- Eval needs ground truth keyed by `review_id`, so it only runs on generator data; say so in the CLI help and README [8].
- `scripts/generate_reviews.py`: `generate()` does not validate `n` and `days`; the two output files are not written atomically; the company name is hardcoded in one template [10].
- `scripts/generate_reviews.py`, `ingest.py`: a few public classes and functions lack docstrings [any].
- `tests/test_generate_reviews.py`: `import random` inside a test function [any].
- README: explain that ratings are deliberately noisy against sentiment, so a rating-based fallback scores lower in eval [10].
