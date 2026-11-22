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
