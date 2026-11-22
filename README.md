# Review Radar

Portfolio project. Reads a CSV of customer reviews about a **fictional** company, classifies each
review with an LLM (sentiment and complaint topics), stores the results in SQLite, and produces:

1. a daily summary sent to Telegram (positive / negative counts, top 3 complaint topics);
2. a static HTML dashboard with sentiment and topic charts.

A demo mode with canned answers runs the whole pipeline offline, without an API key.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the stages.

## Quick start

```bash
uv sync
uv run review-radar --version
```

All data in this repository is fictional.
