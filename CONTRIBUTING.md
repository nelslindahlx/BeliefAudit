# Contributing to BeliefAudit

1. **Setup**: Install [uv](https://docs.astral.sh/uv/) and run `uv sync --extra dev`.
2. **Branching**: Never commit directly to `main`. Create a feature branch (`feat/...`, `fix/...`, `chore/...`).
3. **Local CI**: Every PR must pass `make local-ci` (`canary`, `lint`, `typecheck`, `test` with ≥95% coverage, and `examples`) on the head commit.
4. **Pull Requests**: Use `.github/pull_request_template.md`, complete Maker & Checker reviews, squash-merge with `--delete-branch`, and verify `make audit-prs` exits `0`.
