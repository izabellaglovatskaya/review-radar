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
| Demo mode | `FixtureClassifier` with canned answers keyed by review hash, plus a deterministic keyword/rating fallback. No key, no network, no cost. The fixtures (`fixtures/demo_analyses.json`) are generated from the sample ground truth by `scripts/build_demo_fixtures.py`, so demo on the sample is exact by construction; the fallback handles any other CSV. The fixture `prompt_version` is derived from the fixture content plus `FALLBACK_RULES_VERSION`, so changing either invalidates the demo cache rows; fallback answers are counted and printed in the report. `fixtures/` lives at the repo root, so demo `analyze` works only from a repo checkout (`uv run`), not from an installed wheel (accepted, like `.env` and the database path). |
| Output format | Structured JSON per batch of 20 reviews, short ids (`r1…r20`), closed topic list, validated again with pydantic. |
| Schema upgrades | One idempotent schema script (only `CREATE TABLE/INDEX IF NOT EXISTS` and the two PRAGMAs, in one transaction). A version bump may only add tables or indexes; on connect an older database from `MIN_UPGRADABLE_VERSION` up is upgraded by re-running the script, atomically. Older or newer versions are rejected. A non-additive change must raise `MIN_UPGRADABLE_VERSION`. No migration framework. |
| Re-import key | Source + `review_id` when an id is present: the same id from another source is a different review. The source is casefolded for the id key (`Web` = `web`), the id itself is case-sensitive (`R1` ≠ `r1`); a missing source counts as `unknown`, so a blank source and the literal source `unknown` share one id space. The first stored copy wins and later edits under the same key are ignored. Without an id: sha256 of date + normalized text + source (the source is not casefolded here, so `Web` and `web` rows are two reviews). The same review exported once with and once without an id is stored twice. Known limitations, accepted. Changing the key format bumped the schema to version 2; a version 1 database is rejected with a hint to delete it and run ingest again (reviews are reproducible from the CSV). |
| Cache | Analyses cached by (review hash, provider, model, prompt version). A rerun on the same CSV makes 0 calls. The cache is per normalized text (`text_hash`), so reviews sharing a text share one analysis; the rating sent to the classifier is the one of the earliest stored review with that text (accepted). Two reviews with the same text and different ratings share one analysis (accepted). |
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

- ~~`ingest.py`: no file-size cap before `read_bytes()`; add `MAX_FILE_BYTES` and raise `IngestError` above it [3].~~ Done in stage 3.
- ~~`ingest.py`: a directory path gives "Permission denied" on Windows; check `is_file()` first and say "Not a file" [3].~~ Done in stage 3.
- ~~`ingest.py`: with `--encoding utf-8` on a BOM file the error is a misleading "Missing required column"; strip a leading BOM after decoding [3].~~ Done in stage 3.
- ~~`ingest.py`: a field over 131072 characters aborts the whole file; reword the message to say the field is too large [3].~~ Done in stage 3.
- ~~`ingest.py`: the delimiter is sniffed with `str.splitlines()`, which splits on more characters than csv~~ Done in stage 3: sniffed from the first csv physical line (CRLF/LF), not `str.splitlines()`; a quoted header cell with a line break is still not handled.
- ~~`ingest.py`: a trailing delimiter on data rows (Excel `;` exports) makes every row `wrong_field_count`; consider tolerating trailing empty cells [3].~~ Done in stage 3.
- ~~`models.py`: `external_id` and `source` have no length limit; add `max_length` before they reach the database [3].~~ Done in stage 3.
- `ingest.py`: after a recoverable malformed row, a contrived tail with an even number of quotes that still opens a quoted field could leave a bogus continuation row; it is usually counted as `wrong_field_count`, but could be imported if it happens to be a valid row. Accepted risk; revisit if real exports hit it [any].
- ~~`cli.py`: `ingest` requires `--dry-run` until storage exists; make saving the default in stage 3 [3].~~ Done in stage 3.
- ~~Storage: decide the idempotency key for re-imports (unique `external_id`, fallback hash of date + text + source for rows without an id) so re-running ingest never double-counts the daily summary [3].~~ Done in stage 3.
- `MAX_TEXT_LENGTH` (2000 chars) is not tied to the token budget; revisit with the prompt [5].
- Eval needs ground truth keyed by `review_id`, so it only runs on generator data; say so in the CLI help and README [8].
- `scripts/generate_reviews.py`: `generate()` does not validate `n` and `days`; the two output files are not written atomically; the company name is hardcoded in one template [10].
- `scripts/generate_reviews.py`, `ingest.py`: a few public classes and functions lack docstrings [any].
- `tests/test_generate_reviews.py`: `import random` inside a test function [any].
- README: explain that ratings are deliberately noisy against sentiment, so a rating-based fallback scores lower in eval [10].

### Stage 3

- ~~`storage.py`: `count_reviews` / `list_reviews` do not wrap `sqlite3.Error` / `ValidationError` in `StorageError`; wrap them before a CLI caller uses them (pydantic messages can contain review text) [4].~~ Done in stage 4.
- `storage.py`: a locked or read-only database is reported as "Cannot use ... as a database"; special-case locked/readonly messages and set `busy_timeout` [6].
- ~~`storage.py`: a naive `now` is stored without a timezone; require aware datetimes or convert to UTC [4].~~ Done in stage 4.
- `storage.py`: no option to report "same id, different content" on re-import [any].
- `models.py`: `MAX_TEXT_LENGTH` lives in ingest; move it to `Review.text` `max_length` like the id/source limits [5].
- `ingest.py`: the 20 MB cap does not bound the row count / memory for millions of tiny rows; consider a row cap or capped error list [any].
- `ingest.py`: a UTF-8 BOM file read with `--encoding cp1251` still gives "Missing required column"; detect the raw BOM and suggest the default encoding [any].
- `cli.py`: exit code is 0 even when most rows were skipped; consider `--strict` [8].
- `storage.py`: the database file holds review text and is created with default permissions; chmod 0o600 on POSIX [9].
- Tests: the real-db guard relies only on the `DATABASE_PATH` env fixture; add a check that fails if a test resolves to `data/review_radar.db` [any].
- ~~`storage.py`: `PRAGMA foreign_keys = ON` and its test are unused until a foreign key exists [4].~~ Done in stage 4.
- Tests: `test_corrupt_database_is_a_clean_error` does not assert the file name in the message [any].

### Stage 4

- `classifier.py`: `ClassifierError.reason` is free-form; validate `[a-z0-9_]{1,64}` and fall back to `unknown_error` before the live classifier prints it [5].
- `analysis.py`: the `ClassifierError` message (text-free by contract) is discarded; log it at WARNING with the batch index [5].
- `analysis.py`: no stop after a fatal reason (auth, quota); add a fatal-reason set and count the rest as skipped [5].
- `analysis.py`: `save_analyses`' inserted count is ignored; warn when it is lower than the batch [any].
- `analysis.py`: the three count queries are not one snapshot; read them in one transaction [any].
- `analysis.py`: `utc_stamp(now)` is called only for validation; compute the stamp once and pass it down [any].
- `analysis.py`: usage summing via `sum(usages[1:], usages[0])` is opaque; use `functools.reduce` [any].
- `analysis.py`: `AnalysisReport.failures` is a mutable dict in a frozen dataclass; wrap in `MappingProxyType` [any].
- `cli.py`: `count_reviews` runs twice per analyze (CLI and service) [any].
- `storage.py`: the version-0 check looks only at tables; check any `sqlite_master` object (views, triggers) [any].
- `storage.py`: lookups use the stored `text_hash`, the primary key uses `cache_key(text)`; assert they agree in `save_analyses` [any].
- `storage.py`: a v3 database missing the analyses table passes `connect`; later errors are misleading [any].
- `storage.py`: `PendingText(*row)` depends on SELECT column order; unpack by name [any].
- `storage.py`: `load_analyses` has no production caller yet (summary in stage 6) [6].
- `storage.py`: foreign keys are off on purpose; re-enable `PRAGMA foreign_keys = ON` together with the first foreign key [any].
- `fixture_classifier.py`: `from_file` has no size cap and does not catch `RecursionError` [any].
- `fixture_classifier.py`: `ValidationError` is listed next to its base `ValueError` [any].
- Tests: `conftest.py` does not pin `ANTHROPIC_MODEL` / `ANTHROPIC_COMPARE_MODEL`; pin them before the stage 5 cache key depends on them [5].
- Tests: the CLI rerun test proves zero work only via output counts; make classify raise on the second run [any].
- `analysis.py`: the batch that raised is counted as `not_run` together with the batches after it; count it as `aborted` [any].
- `storage.py`: `load_analyses` maps a stored unknown topic to `other` with a warning instead of raising "stored data is invalid" [6].
- `analysis.py`: reviews sharing a text reuse the earliest review's rating-driven fallback answer; accepted (see Cache decision), but the report does not count shared-text reviews with differing ratings [any].
