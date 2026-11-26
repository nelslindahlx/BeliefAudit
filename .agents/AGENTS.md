# Project Rules & Agent Governance

This file defines repository rules and conventions for **BeliefAudit**, aligned with the zero-GitHub-Actions-minutes local CI and two-part Maker/Checker framework. Both `AGENTS.md` and `.agents/AGENTS.md` must stay byte-identical and under `24,000` bytes.

---

## 1. Zero GitHub Actions Minutes & Stratified Local CI

- **Manual-Only Workflows (`workflow_dispatch`)**: All GitHub Actions workflows in `.github/workflows/` MUST trigger strictly via `on: workflow_dispatch:` (no automatic `push` or `pull_request` triggers) so routine PRs and merges consume **zero** GitHub Actions minutes.
- **Stratified Local CI Commands (`Makefile`)**:
  - **Canary Meta Gate (`< 10s`)**: `make canary` (`python -W error::Warning -c "import beliefaudit"` + `pytest tests/meta/ -q`).
  - **Lint & Format**: `make lint` (`ruff check src tests examples scripts` + `ruff format --check src tests examples scripts`).
  - **Strict Type Check**: `make typecheck` (`mypy --strict src`).
  - **PR Smoke Tier**: `make smoke-pr` (`canary` + `lint` + `typecheck` + `tests/unit/`).
  - **Full Local CI (Mandatory Before PR Merge)**: `make local-ci` (`canary`, `lint`, `typecheck`, `pytest --cov=beliefaudit --cov-fail-under=95`, and `examples/*.py`).

---

## 2. Two-Part Maker / Checker Pull Request Framework & Pre-Flight Guard

All modifications after the initial repository bootstrap MUST flow through a two-part **Maker / Checker** GitHub Pull Request lifecycle:

- **Pre-Flight Branch Guard**: Never commit or push directly to `main`. Before editing any files, run `git branch --show-current`; if on `main`, switch immediately to an ephemeral branch (`feat/...`, `fix/...`, `chore/...`, `docs/...`, `refactor/...`). Once a branch is merged, never reuse its name.
- **Direct Push Ban & Mechanical Pre-PR Hook**: `.agents/hooks.json` (`scripts/hooks/pre_pr_safety_gate.py`) hard-denies `git push origin main` and automatically executes `pytest tests/meta/ -q` prior to `gh pr create|merge` or `scripts/github_pr_helper.py --create|--merge`.
- **Part 1 — Maker (Author)**:
  1. Implement focused changes with strict typing, docstrings on public APIs, and regression/negative tests.
  2. Update `CHANGELOG.md`, `lessons_learned.md`, and `docs/plans/implementation_plan.md` when applicable.
  3. Pass `make local-ci`.
  4. Push the branch and open a PR using `.github/pull_request_template.md`, completing `## Summary`, `## Linked Issue`, and `## Maker Checklist` truthfully from observed output.
- **Part 2 — Checker (Independent Auditor)**:
  1. Inspect `git diff origin/main...HEAD` and `git status --short` for edge cases, exception safety, loopback URL enforcement (`validate_endpoint_url`), path containment (`ensure_within_base`), KnowledgeReduce shard schema compatibility, and test rigor.
  2. Confirm `make local-ci` passes cleanly on the head commit and post findings in the PR `## Checker Review` comment/section.
  3. Squash-merge cleanly and delete the remote head branch via `scripts/github_pr_helper.py --merge --pr <N> --delete-branch` (or `gh pr merge <N> --squash --delete-branch`), then sync `main` (`git checkout main && git pull --ff-only origin main && git fetch --prune origin && git branch -d <branch>`).
- **Zero Lingering Open PR Invariant**: Before declaring work complete, execute `uv run --no-sync python scripts/github_pr_helper.py --audit` (`make audit-prs`) and verify it exits `0` (`Zero open pull requests`, `Zero lingering remote feature branches`).

---

## 3. Hermetic Isolation, Cross-Platform Invocation & Workspace Hygiene

- **Hermetic Isolation**: Every test must use temporary directories (`tmp_path`) and offline runners (`MockModelRunner` or `respx` HTTP mocks) — never depend on live external networks or external services during CI.
- **Cross-Platform Invocation**: Prefer `uv run --no-sync python ...` over bare `python3`; use forward slashes (`/`) in Markdown links.
- **Zero Machine-Local Paths**: Never write local workstation filesystem paths or `file://` URLs into committed source, documentation, `.github/`, GitHub issues, PR descriptions, or PR comments.
- **Zero Tracked Build Artifacts**: Keep `git ls-files "*.pyc"` empty.
