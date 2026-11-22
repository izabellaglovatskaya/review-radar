# CLAUDE.md — Review Radar

Working rules for Claude on this project. The plan and stages are in [docs/PLAN.md](docs/PLAN.md).

## Project basics

- Everything committed to the repo (code, comments, docs, commit messages) is in **English**. Conversation with the owner is in Russian.
- Python **3.13** via uv (`.python-version`, `uv.lock`). Run everything with `uv run`.
- Checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`.
- The default test suite never calls the network. Tests that call a real model are marked `live` and are opt-in.

## Workflow rules

### 1. Test first

Use the **tdd-guide** agent and follow red → green → refactor: write a failing test, run it and show that it fails
(and why), write the minimum code to pass, refactor with the tests green. No application code without a test.

### 2. Review after every stage

- Run the **code-reviewer** agent after every stage.
- If the stage touches input parsing, the database, Telegram, the LLM client, secrets or CI, also run the
  **security-reviewer** agent.

### Agents: which one and when

Project agents live in `.claude/agents/` (copied from
[everything-claude-code](https://github.com/affaan-m/everything-claude-code), MIT license alongside).
`tdd-guide`, `code-reviewer`, `security-reviewer` and `planner` come from the global Claude Code setup.

| Agent | When |
|---|---|
| `tdd-guide` | While writing any code (tests first). |
| `code-reviewer` | After every stage. |
| `python-reviewer` | After every stage that changes Python code. |
| `security-reviewer` | After stages touching input parsing, the database, Telegram, the LLM client, secrets or CI. |
| `silent-failure-hunter` | After stages with the model, cache, retries or Telegram sending (stages 4, 5, 6, 8). |
| `commit-reviewer` | Before every commit (checks the message and whether the diff should be split). |
| `fastapi-reviewer` | Not used here (no FastAPI in this project); kept as a template for future projects. |

### 3. No silent fixes

Present reviewer findings as a list (severity, file, what is wrong); the owner decides what gets fixed.
Stop and ask the owner if a finding is HIGH or CRITICAL, or if a decision comes up that is not in the plan.

### 4. Stage report

After every stage, report: what was done, what was checked (commands and results), which agents were used and what
each found, what is left open.

### 5. One stage, one commit

- One stage = one commit, in [Conventional Commits](https://www.conventionalcommits.org) style.
- Never commit secrets (`.env`, API keys, bot tokens, chat IDs). The gitleaks pre-commit hook must pass; never bypass hooks.

### 6. Never test destructive commands on the real database

Tests and agents use a temporary `DATABASE_PATH`, never `data/review_radar.db`.

### 7. Only fictional data in the repository

Reviews, company names, chat IDs (e.g. `123456789`), tokens and screenshots are fictional. Real values exist only in
`.env` and the local database, both git-ignored.
