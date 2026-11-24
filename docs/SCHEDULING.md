# Running Review Radar on a schedule

This is a description only: nothing in the repository installs a scheduled task. It covers
Windows Task Scheduler (the owner's machine) and cron (Linux or macOS).

## What to schedule

One daily job is enough:

```text
uv run review-radar run <reviews.csv>
```

In live mode `run` does, in order, and stops at the first step that fails:

1. `ingest` the CSV (re-importing the same rows adds nothing: the import is idempotent);
2. `analyze` the reviews without a cached analysis (only new texts cost money);
3. `summary` for **yesterday** (the local date of the machine) and send it to Telegram;
4. `dashboard` to `site/index.html`.

Schedule it after midnight, when the export for the previous day is complete, for example at
07:00. The CSV can be the same growing export every day; only new rows are stored.

`run` treats a day without reviews as a failure (exit 3, see below) and then skips the dashboard.
If days without reviews are normal, schedule separate jobs instead, with `--allow-empty` on the
summary:

```text
07:00  uv run review-radar ingest <reviews.csv>
07:05  uv run review-radar analyze
09:00  uv run review-radar summary --allow-empty
09:01  uv run review-radar dashboard
```

Separate jobs also let the summary go out at another time than the import. Each job must check the
previous one: `summary` refuses to send while reviews of the day are not analyzed, so a failed
`analyze` does not lead to an incomplete message.

### "Yesterday" moves at midnight

`summary` without `--date` always means yesterday. A missed day is therefore **not** sent by the
next scheduled run: after midnight that run summarizes the new yesterday. A rerun repeats the same
day only on the same calendar day. To send a missed day later, run by hand, in this order:

```text
uv run review-radar analyze
uv run review-radar summary --date YYYY-MM-DD
```

## Prerequisites

- A repository checkout with `uv sync --locked` done. `run`, `analyze` (demo fixtures) and
  `dashboard` (assets in `site/`) work only from the checkout.
- `.env` in the repository root with `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID`. The application finds it and the database by the repository root, not by the
  working directory, but the CSV path is relative to the working directory, so start the job in the
  repository folder or pass an absolute CSV path.
- The job runs as the owner's user account. Keep `.env`, `data/` and the logs private to it:
  - Linux and macOS: `chmod 600 .env` and `chmod 700 data` once. A database created by the
    application is owner-only (0600) and a new database folder is 0700; an older database that
    other users can read gets a warning on every run with the `chmod 600` command to fix it (the
    application never changes the mode itself).
  - Windows: the folder under the user profile inherits an ACL for the user, SYSTEM and
    Administrators only; keep the checkout there and do not share the folder.
- SDK settings: the Anthropic SDK reads some variables from the process environment on its own.
  `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_CUSTOM_HEADERS` should not be set,
  and neither should `ANTHROPIC_LOG=debug` (it logs request bodies, which hold review text) or
  `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content`. If one is set, a warning naming the
  variable is printed (never its value), but the run continues, so check the log for it.
- Proxies: the Anthropic SDK and the Telegram sender use `HTTPS_PROXY` / `HTTP_PROXY` /
  `NO_PROXY` and CA settings (`SSL_CERT_FILE`) from the process environment, never from `.env`. Set
  them in the job only if the network needs a proxy. A proxy that decrypts TLS (a custom CA) sees
  the API key and the bot token, which is part of the Telegram URL; use only a proxy you trust.

## Exit codes and what to do

`run` exits with the code of the step that failed. The output holds counts, reason and status
codes only, never review text or secrets, so it can be written to a log file.

| Code | Meaning | What to do |
|---|---|---|
| 0 | Done, or the summary for yesterday was already sent. | Nothing. |
| 1 | A step failed (bad CSV, API error, spend or call limit, Telegram error, database error, another run holding the lock). | Read the line that starts with `Error:` and the `Step failed:` line. Fix the cause and run the job again **the same day**; every step is safe to repeat. After midnight, send the missed day by hand (see "Yesterday" above). |
| 3 | No reviews for yesterday (`summary`), or none at all (`dashboard`). With `run`, the dashboard is not rebuilt that day. | Check that the export and `ingest` ran. If empty days are normal, use separate jobs with `summary --allow-empty`. |
| 130 | Interrupted (Ctrl+C). | Run again. If it happened while sending, see "unknown" below. |

A job that is killed (task time limit, shutdown) gets no exit code from the application. The next
run repeats the steps; a send that was in progress is left open and blocks, as described below.

Do not retry automatically in a tight loop. One retry after 30-60 minutes is reasonable for code 1;
a repeated failure needs a person.

### Noticing failures, and runs that never happened

- Task Scheduler shows the code as "Last Run Result" (0x1 = 1, 0x3 = 3). Nobody looks at it by
  default, so add an alert: a second task triggered "On an event" for the task's failure, or a check
  of the log.
- cron: see the example below, which mails only when the job fails.
- A job that never ran (computer off, user logged out, task disabled) produces no error at all.
  Let the job write a stamp on success and check it: the wrapper below appends `ok <date>` to
  `data/logs/last-success.txt` only on exit 0, so a stamp older than a day means something is wrong.
  A task created with `schtasks` without `/RU` and `/RP` runs only while the user is logged on;
  in the UI choose "Run whether user is logged on or not" if the computer may be locked out.
- `data/logs/run.log` grows forever; delete or archive it now and then (it holds no secrets).

## Never sending the summary twice

The protection is in the application, not in the scheduler:

- Every send attempt is recorded in the `summaries` table **before** the request, inside one
  `BEGIN IMMEDIATE` transaction, so two runs at the same moment cannot both claim the same period.
- A period that was `sent` is skipped with exit 0 ("Already sent ..."). A rerun of the whole job is
  therefore safe.
- A send is never retried inside a run. A certain failure (`failed`, e.g. HTTP 400 or a refused
  connection) lets the next run send again.
- An unclear outcome (`unknown`: timeout, 5xx, broken connection, Ctrl+C, or a run killed during
  the send) **blocks** every later send for that period with exit 1. A person checks the chat and,
  only if the message is not there, sends it by hand with `summary --date YYYY-MM-DD --force`.

Rules for the scheduled job:

- Never put `--force` in a scheduled command.
- Use one fixed period per job (the default "yesterday"). Do not also schedule ranges
  (`--from/--to`) that contain the same days: a range is a different period and is not blocked by
  the daily rows, so its numbers would reach the chat again.
- Do not run two instances at once. Task Scheduler: Settings → "If the task is already running:
  Do not start a new instance". cron: wrap the command in `flock -n` (a skipped overlap then exits
  1 with no message from the application).
- Keep `--allow-partial` out of the scheduled command, so a day with unanalyzed reviews (for
  example after an API outage) is not sent incomplete. Run `analyze` again the same day; after
  midnight, send that day by hand.

## Windows Task Scheduler

The repository path contains a space, so put the command in a small script outside the repository,
for example `%USERPROFILE%\review-radar-daily.cmd` (the paths are examples):

```bat
@echo off
cd /d "%USERPROFILE%\Desktop\work space\review-radar" || exit /b 1
if not exist data\logs mkdir data\logs || exit /b 1
uv run review-radar run "%USERPROFILE%\exports\reviews.csv" >> data\logs\run.log 2>&1
set code=%ERRORLEVEL%
if %code%==0 echo ok %DATE% %TIME%>> data\logs\last-success.txt
exit /b %code%
```

Create the task (once, in a normal console; `/RL LIMITED` keeps normal user rights):

```bat
schtasks /Create /TN "Review Radar daily" /SC DAILY /ST 07:00 /RL LIMITED ^
  /TR "\"%USERPROFILE%\review-radar-daily.cmd\""
```

Then in the Task Scheduler UI, on the task:

- Settings → "If the task is already running": **Do not start a new instance**.
- Settings → "Stop the task if it runs longer than": 1 hour (a stuck run is ended; see the
  "unknown" rule above if it was sending).
- Settings → "Run task as soon as possible after a scheduled start is missed": on, if the computer
  may be off at 07:00. A late start is fine until midnight (see "Yesterday" above).
- General → "Run whether user is logged on or not", if the job must run while logged out.
- Conditions → "Wake the computer to run this task": optional.

`data/logs/` is git-ignored (everything under `data/` except the sample and eval sets).

## cron (Linux or macOS)

```cron
# m h  dom mon dow  command
MAILTO=owner@example.invalid
0 7 * * *  cd "$HOME/review-radar" && mkdir -p data/logs && flock -n data/review-radar.lock uv run review-radar run "$HOME/exports/reviews.csv" >> data/logs/run.log 2>&1 && date >> data/logs/last-success.txt || echo "review-radar failed; see data/logs/run.log"
```

The output goes to the log, so cron mails only the `echo` of a failure (or of an error before the
log opens). The lock file lives in `data/`, not in a shared `/tmp`. cron runs with a minimal
`PATH`: use the full path of `uv` (`command -v uv`) or set `PATH=` at the top of the crontab.
macOS has no `flock` by default (`brew install flock`, or rely on a single daily run). cron uses the
system time zone, which decides what "yesterday" is.

## Cost

A daily run analyzes only new review texts (cached texts cost nothing). On the sample data a full
first run of 300 reviews cost about $0.009 with Claude Haiku 5.5 (see [PLAN.md](PLAN.md)). `analyze`
stops after its call budget (3 calls per batch plus 10); a monthly spend limit in the Anthropic
console is the hard cap and makes the run fail with `spend_limit_reached` instead of retrying.
