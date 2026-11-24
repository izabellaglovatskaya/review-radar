# Review Radar

[![CI](https://github.com/izabellaglovatskaya/review-radar/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/izabellaglovatskaya/review-radar/actions/workflows/ci.yml?query=branch%3Amain)

Review Radar reads customer reviews from a CSV file. An LLM labels each review with a sentiment
and up to three complaint topics. The results go into a local SQLite database. From there the
tool makes two reports:

1. a daily summary for Telegram: positive, neutral and negative counts, and the top 3 complaint
   topics;
2. a static HTML dashboard with charts of sentiment and topics over time.

All data in this repository is fictional. The reviews are about a made-up shop, "Quillfeather
Home Goods".

## Why

A small shop gets many reviews and has no time to read them all. It needs two answers each
morning: "Was yesterday good or bad?" and "What do people complain about most?". Review Radar
gives those answers in one Telegram message and keeps the history on one page.

It is also a portfolio project. It shows how to call an LLM in a careful way: with a cache, a
spend limit, retries, strict validation of the answers, and no review text in logs or messages.

## Demo mode (no key needed)

Demo mode runs the whole pipeline offline. It needs no API key and no network. It never sends
anything to Telegram. Canned answers replace the model.

You need [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
uv installs the right Python (3.13) for you.

1. Get the code and install the dependencies:

   ```bash
   git clone https://github.com/izabellaglovatskaya/review-radar.git
   cd review-radar
   uv sync
   ```

2. Run the whole pipeline on the sample data:

   ```bash
   uv run review-radar run --demo
   ```

   It imports 300 sample reviews, labels them, prints the summary for the last day, and writes the
   dashboard. It uses its own database, `data/review_radar-demo.db`.

3. Open `site/index.html` in Chrome. The page works from disk and makes no network requests.

   ![The Review Radar dashboard on the demo data: four count tiles, a line chart of sentiment by
   day, a bar chart of sentiment shares and a bar chart of complaint topics, each followed by a
   table of the same numbers](docs/screenshots/dashboard.png)

   *The dashboard built from the demo data. Each chart has a table with the same numbers.*

The printed summary looks like this:

```text
<b>[DEMO] Review Radar: 2026-11-18</b>
Reviews: 39
Positive: 21 (54%)
Neutral: 8 (20%)
Negative: 10 (26%)

<b>Top complaint topics</b>
1. Product quality: 5
2. Packaging: 4
3. Customer support: 3
Dry run: nothing sent.
```

The `<b>` tags are Telegram HTML. In the chat they show as bold text.

You can also run the steps one by one. Without a key the tool stays in demo mode:

```bash
uv run review-radar ingest data/sample/reviews.csv
uv run review-radar analyze
uv run review-radar summary --dry-run --date 2026-11-18
uv run review-radar dashboard
```

These commands use the main database, `data/review_radar.db`. The sample reviews are dated from
2026-11-10 to 2026-11-18, so `summary` needs `--date`. Without it, `summary` means yesterday.

## Live mode (real API key)

Live mode calls the Anthropic API, so it costs money. On the sample data a full first run of 300
reviews cost about $0.009 with Claude Haiku 5.5. Reruns on the same data cost nothing, because
every answer is cached.

1. Copy the settings file and keep it private:

   ```bash
   cp .env.example .env
   ```

   On Windows use `copy .env.example .env`. Never commit `.env`; git already ignores it.

2. Put your key in `.env`: `ANTHROPIC_API_KEY=...`. With a key, the default mode (`auto`) becomes
   live. To force a mode, set `REVIEW_RADAR_MODE=demo` or `REVIEW_RADAR_MODE=live`.

3. Start small. Import the reviews and label only 20 texts:

   ```bash
   uv run review-radar ingest data/sample/reviews.csv
   uv run review-radar analyze --limit 20
   ```

   The report shows tokens, API calls and the cost of the run. Then label the rest:

   ```bash
   uv run review-radar analyze
   ```

4. For Telegram, create a bot with [@BotFather](https://t.me/BotFather). Put
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Check the message first, then send it:

   ```bash
   uv run review-radar summary --dry-run --date 2026-11-18
   uv run review-radar summary --date 2026-11-18
   ```

   The same period is sent only once. A second run says "Already sent" and exits with 0.

   ![A Telegram message titled "Review Radar: 2026-11-10": 30 reviews, 12 positive (40%), 6
   neutral (20%), 12 negative (40%), and the top complaint topics Product quality 5, Billing 3
   and Delivery delay 3](docs/screenshots/telegram-summary.png)

   *The summary for one day of the sample data, as it looks in Telegram.*

5. Build the dashboard: `uv run review-radar dashboard`.

For your own data, the CSV needs the columns `review_date` (YYYY-MM-DD) and `text`. The columns
`review_id`, `rating` (1 to 5) and `source` are optional. The separator `,` or `;` is detected.

For daily use, one command does every step:

```bash
uv run review-radar run path/to/reviews.csv
```

[docs/SCHEDULING.md](docs/SCHEDULING.md) shows how to run it every morning with Windows Task
Scheduler or cron, and what to do when it fails.

Do not set `ANTHROPIC_LOG=debug`. With it, the Anthropic SDK logs the request bodies, and they
hold the review text. In live mode the tool prints a warning if it sees this setting.

## Commands

Run every command as `uv run review-radar <command>`. Add `--help` to see all options.

| Command | What it does | Useful options |
|---|---|---|
| `ingest CSV` | Reads the CSV, checks every row, and saves new reviews. Importing the same file again adds nothing. | `--dry-run`, `--delimiter`, `--encoding` |
| `analyze` | Labels the reviews that have no cached answer yet. | `--limit N`, `--max-calls N` |
| `summary` | Builds the summary for a day or a period and sends it to Telegram. | `--date`, `--from`/`--to`, `--dry-run`, `--allow-empty`, `--allow-partial`, `--force` |
| `dashboard` | Writes `site/index.html` from the stored results. | `--out DIR` |
| `run [CSV]` | Runs ingest, analyze, summary and dashboard. Stops at the first failed step. | `--demo`, `--date`, `--out` |
| `eval` | Scores the model against the known answers for the sample data. In live mode it only shows the cost unless you add `--yes`. | `--dataset`, `--compare`, `--yes`, `--max-cost`, `--report` |

Two global options go before the command: `--version` and `--verbose` (progress on stderr).

Exit codes: `0` means done; `1` means an error; `3` means there are no reviews to report;
`130` means it was interrupted (Ctrl+C).

Output and logs hold counts, reason codes and status codes only. They never show review text or
secrets.

## Model evaluation

Haiku 5.5 and Sonnet 5.5 were compared on 360 fictional reviews, with one unchanged prompt and one
run each:

| Model | Regular (300): sentiment / exact topics / both | Hard (60): sentiment / exact topics / both | Cost per 1,000 reviews |
|---|---|---|---:|
| Haiku 5.5 | 99.0% / 91.0% / 90.0% | 93.3% / 91.7% / 86.7% | $0.029 |
| Sonnet 5.5 | 99.7% / 92.7% / 92.3% | 95.0% / 95.0% / 90.0% | $0.569 |

"Both" means that the sentiment and the full set of topics are right for that review.

**Why Haiku is the default.** Sonnet got about 25 more reviews fully right per 1,000. But it costs
19.5 times more. On this sample the difference is not statistically reliable (paired McNemar test,
p = 0.15). So the cheaper model is the better choice here. To switch, set
`ANTHROPIC_MODEL=claude-sonnet-5-5` in `.env`.

The hard set has sarcasm, mixed feelings, typos and very short texts. Details and error examples
are in [docs/EVAL.md](docs/EVAL.md).

In demo mode, `eval` scores the canned answers. They are exact on the regular set, because the
canned answers were built from its known answers. On the hard set they are weak, and that is
expected. The hard texts have no canned answers, so simple rules guess the sentiment from the star
rating. The ratings do not always match the sentiment, on purpose: a sarcastic complaint can have
5 stars, and in the regular set a few happy reviews have 2 or 3 stars. Sentiment, not rating, is
the true label.

## Limitations

- **The dashboard is tested in Chrome only.** The browser tests use headless Chrome. Other browsers
  were not checked.
- **The hard evaluation set was written and labelled by an AI agent (Claude), not by an
  independent person.** The labels were fixed before any model saw the set. Still, they reflect one
  author, and that author is from the same model family as the models being compared.
- **All data is synthetic.** The regular reviews come from templates. The accuracy numbers do not
  predict results on real customer reviews.
- **Nothing is deployed.** There is no server, no hosted dashboard and no installed scheduled task.
  [docs/SCHEDULING.md](docs/SCHEDULING.md) only describes how to set one up.
- **Prompt injection is an accepted, documented risk.** Review text is untrusted. A crafted review
  can push the model to give wrong labels to other reviews in the same batch of 20, and a wrong but
  well-formed answer passes validation. The text is sent as JSON data, the prompt says to treat it
  as data, and the topic list is closed. This limits the harm but does not remove it.
- **It works only from a checkout of this repository.** The demo answers, the dashboard assets and
  `.env` are found by the repository folder.
- **Costs are estimates.** They come from the token counts that the API reports and a price table
  in the code. They were not checked against an invoice.
- **Telegram is send only.** The bot does not read messages or commands.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run --group browser pytest -m browser --no-cov
```

The default tests never call the network. Tests that call a real model are marked `live` and run
only when you ask for them. CI runs lint, tests, browser tests and a secret scan of the full git
history on every push (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

More documents:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): data flow, database schema and trust boundaries;
- [docs/PLAN.md](docs/PLAN.md): decisions, stages, costs and the review backlog;
- [docs/EVAL.md](docs/EVAL.md): the full model evaluation;
- [docs/SCHEDULING.md](docs/SCHEDULING.md): running it every day;
- [PASSPORT.md](PASSPORT.md): interview questions about the project.

## License

The project is under the MIT license, see [LICENSE](LICENSE).

The review agents in `.claude/agents/` were copied from
[Everything Claude Code](https://github.com/affaan-m/everything-claude-code) by Affaan M, which is
also under the MIT license. Their license is kept separately, in
[.claude/agents/LICENSE-everything-claude-code.txt](.claude/agents/LICENSE-everything-claude-code.txt).
