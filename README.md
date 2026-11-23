# Review Radar

Portfolio project. Reads a CSV of customer reviews about a **fictional** company, classifies each
review with an LLM (sentiment and complaint topics), stores the results in SQLite, and produces:

1. a daily summary sent to Telegram (positive / neutral / negative counts, top 3 complaint topics);
2. a static HTML dashboard with sentiment and topic charts.

A demo mode with canned answers runs the whole pipeline offline, without an API key.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the stages.

## Quick start

```bash
uv sync
uv run review-radar run --demo   # the whole pipeline: no key, no network, nothing is sent
```

All data in this repository is fictional.

## Model evaluation

Haiku 5.5 and Sonnet 5.5 on 360 fictional reviews (one unchanged prompt, one run each):

| Model | Regular (300): sentiment / exact topics / both | Hard (60): sentiment / exact topics / both | Cost per 1,000 reviews |
|---|---|---|---:|
| Haiku 5.5 | 99.0% / 91.0% / 90.0% | 93.3% / 91.7% / 86.7% | $0.029 |
| Sonnet 5.5 | 99.7% / 92.7% / 92.3% | 95.0% / 95.0% / 90.0% | $0.569 |

The default is **Haiku 5.5**. Sonnet got about 25 more reviews fully right per 1,000 at 19.5 times
the price, and on this sample that difference is not statistically reliable (p = 0.15).

**The hard set (sarcasm, mixed feelings, typos, very short texts) was written and labelled by an AI
agent, not by an independent human.** All data is synthetic, so these numbers do not predict
accuracy on real reviews. Details, limits and error examples are in [docs/EVAL.md](docs/EVAL.md).
