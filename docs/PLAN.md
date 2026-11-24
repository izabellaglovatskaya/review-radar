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
| LLM | Anthropic API behind a `Classifier` protocol (`AnthropicClassifier`). Default model `claude-haiku-5-5`, effort `low`, thinking disabled (classification does not need it, and output tokens stay predictable), `max_tokens` 4096 per batch of 20. Structured output through `output_config` with a `json_schema`; the answer is always re-validated with strict pydantic models (limits such as three topics are enforced after parsing, not trusted to the schema). Only `id`, `rating` and `text` of a review are sent; the system prompt is static and tells the model to treat review text as data. Key only in `.env`, wrapped in `SecretStr`, unwrapped once to build the SDK client. The cache `prompt_version` is `anthropic-1+<sha8 of prompt and schema>`, so a prompt or schema change is a cache miss. |
| Demo mode | `FixtureClassifier` with canned answers keyed by review hash, plus a deterministic keyword/rating fallback. No key, no network, no cost. The fixtures (`fixtures/demo_analyses.json`) are generated from the sample ground truth by `scripts/build_demo_fixtures.py`, so demo on the sample is exact by construction; the fallback handles any other CSV. The fixture `prompt_version` is derived from the fixture content plus `FALLBACK_RULES_VERSION`, so changing either invalidates the demo cache rows; fallback answers are counted and printed in the report. `fixtures/` lives at the repo root, so demo `analyze` works only from a repo checkout (`uv run`), not from an installed wheel (accepted, like `.env` and the database path). |
| Output format | Structured JSON per batch of 20 reviews, short ids (`r1…r20`), closed topic list, validated again with pydantic. |
| Schema upgrades | One idempotent schema script (only `CREATE TABLE/INDEX IF NOT EXISTS` and the two PRAGMAs, in one transaction). A version bump may only add tables or indexes; on connect an older database from `MIN_UPGRADABLE_VERSION` up is upgraded by re-running the script, atomically. Older or newer versions are rejected. A non-additive change must raise `MIN_UPGRADABLE_VERSION`. No migration framework. |
| Re-import key | Source + `review_id` when an id is present: the same id from another source is a different review. The source is casefolded for the id key (`Web` = `web`), the id itself is case-sensitive (`R1` ≠ `r1`); a missing source counts as `unknown`, so a blank source and the literal source `unknown` share one id space. The first stored copy wins and later edits under the same key are ignored. Without an id: sha256 of date + normalized text + source (the source is not casefolded here, so `Web` and `web` rows are two reviews). The same review exported once with and once without an id is stored twice. Known limitations, accepted. Changing the key format bumped the schema to version 2; a version 1 database is rejected with a hint to delete it and run ingest again (reviews are reproducible from the CSV). |
| Prompt injection | Residual risk, accepted: review text is untrusted, and one malicious review can steer the answers for other reviews of the same batch; those answers are cached under the neighbours' text hashes, and validation cannot detect it (a wrong but well-formed answer is still well-formed). Accepted here because the data is fictional or the owner's own. Mitigations: text is sent only as JSON data, the system prompt is static and says to treat text as data, the vocabulary is closed and validated, batches are small (20). One call per review would remove the effect at about 20x the request overhead. |
| Retries | Per API call, transient errors (429, 408, 409, 5xx including 529, network, timeout) are retried up to 4 attempts in total with backoff 1 / 2 / 4 s (cap 30 s); a numeric `retry-after` header is honoured (capped at 30 s). The `x-should-retry` header overrides the status rules (`true` retries, `false` gives the batch up at once, not fatal). Fatal errors stop the whole run at once and the rest of the batches are reported as `skipped_after_fatal`: 401, 403, 404, any other 4xx, 402, a 400 that shows credit exhaustion (`credit balance` in the message or `credit`/`billing` in the error code; the SDK text is only inspected to pick the reason, never logged), and a 429 with error code `enforced_spend_limit_reached` (a monthly spend cap, retries cannot help; reason `spend_limit_reached`). A 400 otherwise is `bad_request`. A 413, an invalid answer (bad JSON, schema violation, duplicate ids, refusal, truncated output, a reply that is not a message) splits the batch in half, down to single reviews (a single 413 review fails as `request_too_large`); ids missing from a valid answer are retried once, then fail as `missing_result`. Any other SDK error is fatal `unexpected_error`. When a call fails fatally or unexpectedly, the valid answers already paid for are saved and their tokens counted. Failures are never cached, so the next run retries them. Error messages and logs hold counts, reason codes, status codes and request ids only: never review text, model output or the key. |
| Cache | Analyses cached by (review hash, provider, model, prompt version). A rerun on the same CSV makes 0 calls. The cache is per normalized text (`text_hash`), so reviews sharing a text share one analysis; the rating sent to the classifier is the one of the earliest stored review with that text (accepted). Two reviews with the same text and different ratings share one analysis (accepted). |
| Cost | Estimated per run as `input_tokens x price_in / 1M + output_tokens x price_out / 1M` (`Decimal`) from the price table in `pricing.py` (USD per 1M tokens, prompts up to 100K tokens; every batch is far below). Usage of every call is counted, including calls whose answer was invalid or whose result could not be saved. A model without a price shows `unknown`; demo costs `$0`. Two guards bound a runaway run: a circuit breaker (two batches in a row that failed only transiently stop the run, the rest is `skipped_after_transient`) and a call budget (`--max-calls N`, default for live `3 x batches + 10`, computed after `--limit`; every attempt counts; when used up the remaining reviews fail as `call_budget_exceeded`). Every `analyze` run, also a fatal, aborted or interrupted one (Ctrl+C, exit 130), writes one row to the `runs` table (schema version 4: times, provider, model, prompt version, review counts, not-analyzed-by-limit, tokens, API calls, cost, `outcome` = completed / fatal / aborted / interrupted / stopped_transient / call_budget_exceeded, failure counts by reason as JSON, fatal reason). `analyze --limit N` classifies only the first N pending texts, for a small first live run. |
| Telegram | Send only (no commands), through `httpx2` (already locked via the Anthropic SDK), as HTML with link previews off. `review-radar summary` takes `--date` (default: yesterday, local date) or `--from`/`--to`. The message holds only aggregates and names from the closed topic list, never review text; shares cover analyzed reviews, top 3 complaint topics come from negative reviews. `--dry-run` prints the message and needs no credentials. A real send without token or chat id is an error. Demo analyses are never sent: in demo mode a real send is refused and a dry run is labelled `[DEMO]`. Unanalyzed reviews block a send unless `--allow-partial`. A period with no reviews sends nothing and exits 3 (it may mean a missed ingest) unless `--allow-empty`. Idempotent per exact period unless `--force`: schema version 5 adds a `summaries` table (one row per attempt, with the classifier key), and the row is written as `sending` before the request. A send is never retried. A certain failure is `failed` (a rerun may send); a timeout, broken exchange, 5xx, unreadable success or Ctrl+C is `unknown` and blocks until a person checks the chat and uses `--force`. Failures exit non-zero. The token is in the request URL, so HTTP loggers are held at WARNING and errors carry only reason and status codes. |
| Dashboard | `review-radar dashboard [--out DIR]` builds `site/index.html` from a Jinja2 template (autoescape, strict undefined) with a vendored Chart.js 4.5.1; it opens from disk, makes no network request and works in demo mode without a key. The page holds aggregates only, never review text: sentiment by day (quiet days are zero, range capped at 3660 days), sentiment shares and complaint topics from negative reviews. Data is embedded with `tojson` in an `application/json` block. A CSP meta tag allows only same-origin scripts, styles and images, with no inline code. Each chart has a caption, a text label and a real table; lines differ by marker and dash, bars by name. Demo data and unanalyzed reviews are stated on the page. No reviews exits 3, none analyzed exits 1, and a failed build keeps the previous page. Assets live in `site/assets`, so the command works only from a repository checkout (accepted, like `fixtures/`). Browser tests are opt-in: `uv run --group browser pytest -m browser --no-cov` (Playwright is locked in the `browser` dependency group). Audit on demo data (Lighthouse 12, axe-core 4.11): Accessibility 100, Best Practices 100, Performance 97 mobile / 100 desktop, no axe violations for WCAG 2.2 AA and best practices in light and dark themes at 375 and 1280 px. SEO is 91 only because `connect-src 'none'` blocks the `robots.txt` check; the strict policy is kept for a local page. |
| Eval | `review-radar eval` scores predictions against ground truth on the regular sample and a fictional hard set (`data/eval`, labelled by an AI agent before any model saw it); `--compare` runs Haiku 5.5 and Sonnet 5.5. Results are in [EVAL.md](EVAL.md). The default stays Haiku 5.5: Sonnet is 19.5 times the price and its lead is not statistically reliable on this sample. Live mode only prints an estimate unless `--yes` is given. Spend is limited by `--max-cost` across runs: schema version 6 adds a `spend_ledger` table, the worst case of each call is written before the call and replaced by the real cost after it, so a crash or timeout can overstate the total but never understate it. Eval uses its own `<stem>-eval.db`. Sonnet 5.5 rejects `thinking: disabled`, so it uses `between_tools`. No fallback model is used, so a refusal counts as an error. |
| Run | `review-radar run --demo` runs ingest, analyze, a dry-run summary and the dashboard on the sample in its own `<stem>-demo.db`, with no key, network or Telegram send. `review-radar run CSV` runs the same steps live and stops at the first failed step. |
| CI | GitHub Actions on Ubuntu only (`.github/workflows/ci.yml`), four jobs: lint (ruff check, format check and `mypy src`), test (default pytest suite), browser (headless Chrome preinstalled on the runner, Playwright locked in the `browser` dependency group; `REVIEW_RADAR_REQUIRE_BROWSER=1` turns a missing browser into a failure instead of a silent skip) and gitleaks over the full git history (`fetch-depth: 0`, the release binary checked by SHA-256, the same version as the pre-commit hook, findings redacted). Python jobs install with `uv sync --locked` and take Python from `.python-version`. Only pull request runs are cancelled by a newer push, so every commit on `main` is checked. No secrets: the token is read-only, checkout keeps no credentials, the test jobs run in demo mode with blank keys, and live tests stay deselected. Actions are pinned to commit SHAs. The workflow is checked without GitHub by `tests/test_ci_workflow.py` and by actionlint (`uvx --from actionlint-py==1.7.12.24 actionlint .github/workflows/ci.yml`). |
| Publishing | No GitHub Pages for now (until the repository is made public). README is meant to show screenshots of the Telegram message and the dashboard; `docs/screenshots/` does not exist yet, so README shows the demo message as text and has no images. The repository stays private until the owner decides. |
| Scheduling | The interface is `review-radar run` (or separate `ingest`/`analyze`/`summary`/`dashboard` jobs). Windows Task Scheduler and cron setup, exit codes, failure handling and the no-double-send rules are documented in [SCHEDULING.md](SCHEDULING.md), not automated. |

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

Measured on the first live run (2026-11-23, `claude-haiku-5-5`, prompt `anthropic-1+9998e992`, sample CSV of 300 reviews, 278 unique texts, batches of 20): about 135 input and 36 output tokens per unique text. The earlier estimate was 130 and 31.

| Model | $ per 1M tokens (in / out) | 300 reviews |
|---|---|---|
| Claude Haiku 5.5 | 0.10 / 0.50 | $0.0087 (measured) |
| Claude Sonnet 5.5 | 2.00 / 10.00 | $0.1727 (measured in the stage 8 evaluation) |

Costs are computed from the API-reported tokens and the price table in `pricing.py`; they were not reconciled with the Anthropic invoice.

Reruns on the same data cost nothing (cache). Demo mode costs nothing.

## First live run (2026-11-23)

Two runs on a temporary database: `analyze --limit 20 --max-calls 5`, then `analyze --max-calls 25`.

| Run | Texts sent | Reviews covered | From cache | API calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|---|---|---|
| `--limit 20` | 20 | 23 | 0 | 1 | 2,644 | 721 | $0.000625 |
| Remaining reviews | 258 | 277 | 23 | 13 | 34,766 | 9,243 | $0.008098 |
| Total | 278 | 300 | | 14 | 37,410 | 9,964 | $0.008723 |

"Reviews covered" counts the reviews that got a new analysis in that run, including reviews sharing a sent text. "From cache" counts the reviews already analyzed before that run, so the second run saw all 300 reviews: 277 new and 23 from cache.

There were no retries, batch splits or failures: 14 calls for 14 batches, both runs ended as `completed`.

Accuracy against `data/sample/ground_truth.json`, over all 300 reviews:

| Metric | Result |
|---|---|
| Sentiment | 297 / 300 (99.0%) |
| Exact topic set | 279 / 300 (93.0%) |
| Sentiment and exact topic set | 276 / 300 (92.0%) |
| Topic labels | precision 93.5%, recall 95.2% (200 matched, 14 extra, 10 missed) |

The three sentiment errors are neutral reviews (`r0255`, `r0257`, `r0274`). Most topic differences are an added or dropped second topic; `other` was never predicted (2 expected, both labelled `product_quality`).

**Caveat: the data is synthetic.** The reviews come from a small set of templates, so errors repeat across similar texts and these numbers do not predict accuracy on real reviews. The ground truth is template-based too: some counted "errors" are defensible readings (for example, a complaint about a damaged item labelled only `customer_support`, where the model also adds `product_quality`).

## Spend ledger live check (2026-11-24)

`eval --dataset hard --yes --max-cost 0.10` with `claude-haiku-5-5` on a fresh temporary database (schema version 6), to check that the ledger records real API calls. 3 calls for 3 batches of 20, no retries or splits; the run ended with exit 0.

| Ledger row | Status | Input tokens | Output tokens | Amount |
|---|---|---|---|---|
| 1 | settled | 1,953 | 732 | $0.0005613 |
| 2 | settled | 1,858 | 724 | $0.0005478 |
| 3 | settled | 1,539 | 688 | $0.0004979 |
| Total | | 5,350 | 2,144 | $0.0016070 |

Every reservation was replaced by the real cost (`settled`, both timestamps set), and the ledger total equals the cost the eval report printed ("Spent in total: $0.001607 of $0.10"). Accuracy on the hard set: sentiment 96.7%, exact topics 95.0% (60 reviews).

## Review backlog (LOW findings)

Recorded after each stage review; picked up in the stage named in brackets.

### Stage 1

- ~~`models.py`: the topic validator accepts any iterable (`bytes`, `dict`, `set`); accept only `list | tuple` [5].~~ Done in stage 5.
- ~~`models.py`: blank `external_id` / `source` are stored as-is; map blank strings to `None` [2].~~ Done in stage 2.
- ~~`models.py`: `rating` accepts `True` (lax mode); use a strict int or rely on CSV ingest passing strings [2].~~ Done in stage 2.
- `models.py`, `config.py`: validators take `value: Any`; use `object` for untrusted input [any].
- `models.py`, `config.py`: public classes need one-line docstrings [any].
- `config.py`: `REVIEW_RADAR_MODE` is case-sensitive (`LIVE` fails); lowercase and strip before validation [8].
- ~~`config.py`: model names accept empty or whitespace values; add `min_length=1` with stripping [5].~~ Done in stage 5.
- ~~`config.py`: non-blank secrets are not stripped (a pasted trailing newline breaks auth later) [5].~~ Done in stage 5.
- `config.py`: log the resolved mode and database path when the CLI starts [8].
- Tests: always build `Settings` with `_env_file=None`, so a developer's real `.env` is never read [all].
- Tests: nothing asserts that `Settings` is frozen [any].
- Tests contain `# type: ignore` comments but no type checker runs; add mypy/pyright to dev deps and checks, or drop the comments. After stage 9 the owner added mypy for `src` (the 8 errors are fixed, the lint job runs `uv run mypy src`); the tests are still not type-checked, so their `# type: ignore` comments stay unchecked [any].
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
- ~~README: explain that ratings are deliberately noisy against sentiment, so a rating-based fallback scores lower in eval [10].~~ Done in stage 10.

### Stage 3

- ~~`storage.py`: `count_reviews` / `list_reviews` do not wrap `sqlite3.Error` / `ValidationError` in `StorageError`; wrap them before a CLI caller uses them (pydantic messages can contain review text) [4].~~ Done in stage 4.
- `storage.py`: a locked or read-only database is reported as "Cannot use ... as a database"; special-case locked/readonly messages and set `busy_timeout` [6].
- ~~`storage.py`: a naive `now` is stored without a timezone; require aware datetimes or convert to UTC [4].~~ Done in stage 4.
- `storage.py`: no option to report "same id, different content" on re-import [any].
- `models.py`: `MAX_TEXT_LENGTH` lives in ingest; move it to `Review.text` `max_length` like the id/source limits [5].
- `ingest.py`: the 20 MB cap does not bound the row count / memory for millions of tiny rows; consider a row cap or capped error list [any].
- `ingest.py`: a UTF-8 BOM file read with `--encoding cp1251` still gives "Missing required column"; detect the raw BOM and suggest the default encoding [any].
- `cli.py`: exit code is 0 even when most rows were skipped; consider `--strict` [8].
- ~~`storage.py`: the database file holds review text and is created with default permissions; chmod 0o600 on POSIX [9].~~ Done in stage 9: a new file is created with mode 0o600 before `sqlite3.connect` (an existing file is not changed; the POSIX test runs in CI).
- Tests: the real-db guard relies only on the `DATABASE_PATH` env fixture; add a check that fails if a test resolves to `data/review_radar.db` [any].
- ~~`storage.py`: `PRAGMA foreign_keys = ON` and its test are unused until a foreign key exists [4].~~ Done in stage 4.
- Tests: `test_corrupt_database_is_a_clean_error` does not assert the file name in the message [any].

### Stage 4

- ~~`classifier.py`: `ClassifierError.reason` is free-form; validate `[a-z0-9_]{1,64}` and fall back to `unknown_error` before the live classifier prints it [5].~~ Done in stage 5.
- ~~`analysis.py`: the `ClassifierError` message (text-free by contract) is discarded; log it at WARNING with the batch index [5].~~ Done in stage 5.
- ~~`analysis.py`: no stop after a fatal reason (auth, quota); add a fatal-reason set and count the rest as skipped [5].~~ Done in stage 5.
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
- ~~Tests: `conftest.py` does not pin `ANTHROPIC_MODEL` / `ANTHROPIC_COMPARE_MODEL`; pin them before the stage 5 cache key depends on them [5].~~ Done in stage 5.
- Tests: the CLI rerun test proves zero work only via output counts; make classify raise on the second run [any].
- `analysis.py`: the batch that raised is counted as `not_run` together with the batches after it; count it as `aborted` [any].
- `storage.py`: `load_analyses` maps a stored unknown topic to `other` with a warning instead of raising "stored data is invalid" [6].
- `analysis.py`: reviews sharing a text reuse the earliest review's rating-driven fallback answer; accepted (see Cache decision), but the report does not count shared-text reviews with differing ratings [any].

### Stage 5

- `anthropic_classifier.py`: `retry-after-ms` and HTTP-date values are ignored, there is no jitter, a `retry-after` above 30 s is clamped (wastes an attempt), `retry-after: 0` retries at once [any].
- `anthropic_classifier.py`: unknown stop reasons (`pause_turn`, `tool_use`, None) are accepted if the JSON parses; allow-list `end_turn`, treat more than one text block as invalid [any].
- `anthropic_classifier.py`: `prompt_version` does not include request parameters (effort, thinking, max_tokens) [any].
- `anthropic_classifier.py`: after a transient give-up inside a split the sibling half is still tried [any].
- `anthropic_classifier.py`: the docstring says missing ids are retried "as a smaller batch"; with all ids missing it is the same size [any].
- `anthropic_classifier.py`: sentiment `Literal[...]` duplicates the `Sentiment` enum [any].
- ~~`anthropic_classifier.py`: the SDK reads `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` from the environment; a stray base URL would send the key and texts elsewhere; pass `base_url` explicitly or warn [9].~~ Done in stage 9: `build_client` warns, naming only the variable; where requests go is unchanged.
- `anthropic_classifier.py`: the docstring says the key is not stored; the SDK client holds it [any].
- ~~README: `ANTHROPIC_LOG=debug` makes the SDK log request bodies (review text) [10].~~ Done in stage 10.
- `cli.py`: fatal, aborted and partial failures all exit 1; consider distinct exit codes for stage 8 scheduling [8].
- `cli.py` / `analysis.py` / classifier: the same fatal error is logged up to three times [any].
- `cli.py`: `DEMO_PROVIDER = "fixture"` duplicates `FixtureClassifier.provider` [any].
- Tests: no autouse guard blocks sockets / real `anthropic.Anthropic` in non-live tests [any].
- Tests: the live test runs whenever `-m live` and a key are present; consider requiring `RUN_LIVE=1` [any].
- `analysis.py`: when a classifier gives both `exc.usage` and a partial, only the partial's usage is used; fall back to `exc.usage` when the partial has none [any].
- `anthropic_classifier.py`: a paid 200 response that fails SDK validation (`APIResponseValidationError`) is not counted in usage/cost [any].
- `anthropic_classifier.py`: `x-should-retry: true` on 400/401/403/404 is retried and reported as `server_error`; keep the client-error reason or let the header override only non-fatal statuses [any].
- `cli.py`: the "Could not record the run" WARNING is printed before the report; drop it or log at DEBUG [any].
- `cli.py`: Ctrl+C outside the batch loop (connect, counts, run save) exits 1 ("Aborted!") without a run row; catch it in `analyze()` and exit 130 [8].

### Stage 6

- ~~`telegram.py`: `trust_env` stays on, so proxy and CA settings from the environment apply; a hostile `HTTPS_PROXY` plus `SSL_CERT_FILE` could read the token. Turning it off also disables proxies; owner decision [9].~~ Decided in stage 9, no behaviour change: `trust_env=True` is now explicit and tested. Whoever can set the process environment can already run code (`PYTHONPATH`, the venv), so turning it off adds no protection, while it would break a legitimate proxy that the Anthropic SDK honours anyway. `.env` values never reach the process environment, so a proxy can only come from the real environment; SCHEDULING.md says so.
- `telegram.py`: a 2xx other than 200 counts as certainly not delivered; treat it as `unknown` [any].
- `telegram.py`: `RecursionError` from a deeply nested reply is handled only by the CLI (`unexpected_error`); catch it in `_parse` [any].
- `telegram.py`: `ConnectError` counts as not delivered; a proxy that fails after forwarding could make that wrong (theoretical) [any].
- `cli.py`: `--dry-run` opens the database, so it can create the file or upgrade its schema; open read-only or reword the help [8].
- `cli.py`: the default "yesterday" uses the local date; document it or add a timezone setting [8].
- `cli.py`: "Already sent" does not say that reviews arrived after the send; compare with the stored `reviews_total` [8].
- `cli.py`: overlapping periods (a day, then a range containing it) can report the same day twice; warn on overlap [any].
- `summary.py`: nothing checks that ingest covered the whole period, so a partly loaded day looks complete; warn when the period end is today or later than the last import [8].
- `storage.py`: a forced send leaves older `sending` rows open forever; mark them `unknown` [any].
- `storage.py`: summary statuses are plain strings and `SummaryClaim` allows impossible states; use an enum and a granted/blocked union [any].
- `storage.py`: `int(cursor.lastrowid or 0)` hides a missing row id; raise `StorageError` [any].
- `storage.py`: a corrupt stored topic is shown as `Other` with only a warning; validate strictly when loading for a summary [any].
- Tests: no test runs the real `httpx2` transport over a mock network backend with logging at DEBUG [any].
- `cli.py`: Ctrl+C after a successful send, or during recording of a result, leaves the row `sending` (no resend), exits 1 and does not say the message was delivered; handle it and exit 130 [8].
- `cli.py`: a second Ctrl+C while an interrupted send is being recorded prints a traceback (no secrets, locals are off); the row stays `sending` [any].
- `cli.py`: Ctrl+C is always recorded as `unknown`, even if it arrived before the request left (safe but imprecise) [any].
- `cli.py`: a failure while building the sender would be recorded as `unknown`; build it before the `try` and record `failed` (not reachable today) [any].
- Carried over from earlier stages: locked database handling and `busy_timeout`, and `load_analyses` still has no production caller [8].

### Stage 7

- `dashboard.py`: `copytree` writes through symlinks already inside `--out/assets`; refuse them or swap in a fresh folder [any].
- `dashboard.py`: `--out` inside `site/assets` copies the folder into itself; reject a path under `SITE_DIR` [any].
- `dashboard.py`: assets are copied before `index.html` is replaced, so a failed replace leaves new assets beside the old page; stale asset files are never removed; the docstring should say only the page is atomic [any].
- `dashboard.py`: Ctrl+C during the write can leave a `.index-*.tmp` file, and a failed `os.fdopen` leaks the descriptor [any].
- `dashboard.py`: the page does not name the classifier, and analyses under another model or prompt are reported only as "not analyzed"; show the key and explain the mismatch [8].
- `dashboard.py`: `build_dashboard_data` is long, the round-half-up share has no named helper, and sentiment names are repeated in the template [any].
- `dashboard.py`: the daily `aria-label` grows with the number of days; keep it short and point to the table [any].
- `dashboard.js`: one failing chart hides all three; draw each chart separately [any].
- `dashboard.js`: chart colors are read once, so changing the OS theme with the page open keeps the old colors [any].
- `dashboard.js`: if the file is deleted after the build, the reserved chart space stays blank with no notice [any].
- `dashboard.html.j2`: one unanalyzed review reads "1 reviews are"; the daily caption does not mention the "Not analyzed" line [any].
- `dashboard.html.j2`: the demo notice says "canned analyses" although some come from keyword rules, and the CLI does not print that the data is demo [8].
- Tests: `tests/test_dashboard.py` calls `monkeypatch.undo()` mid-test; use `monkeypatch.context()` [any].

### Stage 8

- ~~`cli.py`: `_run_steps` calls the command functions with positional arguments; a reordered option could pass `dry_run` to the wrong parameter. Use keywords [9].~~ Done in stage 9.
- `cli.py`: `run` continues after an ingest that skipped rows, so a live summary can be sent from incomplete data with exit 0; treat skipped rows as a failure or warn [any].
- `cli.py`: `eval` drops skipped CSV rows without a message; the cause shows only as failed reviews [any].
- `cli.py`: estimate-only `eval` silently ignores `--report` [any].
- `cli.py`: the report file is written before the console output, so an unwritable path hides the results of a paid run (they stay in the eval database) [any].
- `cli.py`: the cost per 1,000 shows `$0` when money was spent but no review was analyzed; show `n/a` [any].
- `evaluation.py`: cache read and write tokens are not priced; no request uses caching today [any].
- `evaluation.py`: table cells escape only pipes and line breaks; also escape backslashes and validate ids and categories in `load_truth` [any].
- ~~`evaluation.py`: `client: Any` and some untyped locals; use the existing protocol and annotations [9].~~ Done in stage 9 (`MessagesClient`).
- `analysis.py`: `text_hashes` accepts a list with slow lookups; take a set [any].
- Tests: no test covers an unpriced compare model [any].
- `docs/EVAL.md`: regenerating the report replaces the hand-written analysis; write the generated part to its own file. The per-dataset cost behind the $0.1727 figure is not shown there [10].
- Deleting the `-eval.db` file resets the spend limit (accepted: the limit belongs to that database).
- Items tagged [8] in earlier sections that this stage did not change stay open.

### Stage 9

- `storage.py`: a symlink at the database path is followed by sqlite (the 0o600 create is skipped for an existing path); planting one needs write access to the user's folder, and a symlinked database is a legitimate setup [any].
- `storage.py`: if `connect` fails after creating the file, an empty database file stays behind (sqlite treats it as new) [any].
- `anthropic_classifier.py`: the SDK environment warnings repeat for every client built in one run (eval builds one per model) [any].
- `cli.py`: `run` has no `--allow-empty`, so a day without reviews exits 3 and skips the dashboard; documented in SCHEDULING.md (use separate jobs) [any].
- `cli.py`: nothing rejects a scheduled `summary` started long after midnight; "yesterday" is documented [any].
- `cli.py`: `_run_steps` wraps `ingest` in a local function only to pass keywords; `functools.partial` would be shorter [any].
- `evaluation.py`: some new local annotations are redundant, and `reported: Any` hides the usage type [any].
- `tests/browser_support.py`: only the exact value `1` requires the browser; the workflow test pins that value [any].
- Tests: `storage.os.open` and `Path.mkdir` are patched process-wide during two permission tests [any].
- Tests: `test_cli_run_keywords.py` calls `_run_steps` directly, not through `run` [any].
- CI: `setup-uv` keeps its default cache; set `enable-cache: false` for stricter isolation of pull request runs [any].
- CI: gitleaks scans the history reachable from the checked-out ref (the merge ref on a pull request), not other branches, and uses the default rules [any].
- CI: a browser test that loses the module-level `browser` marker would leave the browser job with no tests (pytest exit 5, loud) and be skipped in the test job; no count guard [any].
- Not done: refusing to start when `ANTHROPIC_BASE_URL` (or another SDK override) is set, or passing `base_url` explicitly; this changes behaviour and is an owner decision. Today it is a warning only.

### Stage 10

- `tests/test_docs.py`: the passport test requires every "My answer" field to stay empty, so it fails as soon as the owner writes answers; it also checks only the first line after the field [any].
- `tests/test_docs.py`: the Mermaid check counts blocks only; no validator ran on the diagrams (fetching mermaid-cli was blocked), they were checked by reading [any].
- README and ARCHITECTURE: "review text never appears in logs" is stated as absolute; it is the design and is tested for the main paths, but not proven for every exception message [any].
- ARCHITECTURE: "the key is unwrapped once" holds for the Anthropic key; the bot token and chat id are unwrapped in `TelegramSender` on each send [any].
- ARCHITECTURE: "outbound HTTPS only" to Anthropic holds unless `ANTHROPIC_BASE_URL` is set (warning only, see stage 9) [any].
- ARCHITECTURE: the CSP is a `<meta>` tag, not an HTTP header (no `frame-ancestors`); say so [any].
- ARCHITECTURE: the state diagram lists "4xx, refused connection" as certain failures; connect and pool timeouts are certain failures too [any].
- README: say that the database holds the full review text and must not be shared or attached to issues, nor `.env` or `--verbose` output [any].
- README: "CI runs on every push" means pushes to `main` and pull requests [any].
- README: the CI badge and the `git clone` URL work for others only once the repository is public; they name the owner's GitHub account [publish].
- Not done here: the code items tagged [10] in earlier sections (dependency floors, `generate_reviews.py` checks, splitting the generated part of EVAL.md); this stage changed documentation only.

### First live run

- ~~Eval: add a small fictional set of "hard" reviews with ground truth (sarcasm, mixed feelings, typos, very short texts) and report it separately in the Haiku vs Sonnet comparison, because the template-based sample is too easy to separate the models [8].~~ Done in stage 8.
