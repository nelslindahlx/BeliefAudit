# Changelog

All notable changes to **BeliefAudit** are documented in this file.

## [0.1.1] - 2026-11-26

### Added
- Performance & local-execution upgrades (`PR #2`):
  - Concise prompt framing + per-mode/per-axis `max_tokens` budgets (`96–224` tokens, overridable via `--max-tokens`).
  - Two-tier `--profile quick` (1 item/category × 4 core modes + 8 controls) vs `--profile full` (all items/modes) and `--items-per-category`.
  - Incremental `checkpoint.jsonl` persistence with `--resume`, `--concurrency`, configurable `--timeout` (default `120s`), and live per-prompt progress output (`--quiet` to suppress).

## [0.1.0] - 2026-11-26

### Added
- Initial repository scaffold, governance (`AGENTS.md`, `.agents/AGENTS.md`, pre-PR safety gate), zero-Actions-minutes `workflow_dispatch` CI workflow, and `tests/meta/test_repo_invariants.py`.
- Security guards (`ensure_within_base`, `validate_endpoint_url` loopback policy, `validate_identifier`), strict `extra="forbid"` Pydantic v2 schemas, bundled offline `golden_v0.1.0` seed pack (18 `GoldenItem`s across 6 `ClaimCategory` buckets + 12 `ControlItem`s across 4 quality axes), and immutable SHA-256 snapshot store + `promote` workflow (`PR #1`).
- Local-only `LocalOpenAICompatRunner`, deterministic offline `MockModelRunner` (6 personas), 6-mode probe generator, fail-closed `LocalLLMJudge` (off by default, self-grading prohibited), and hybrid `distill_belief` pipeline (`PR #1`).
- KnowledgeReduce v1.6.1-compatible (`schema_version="1.0"`) JSON and sharded JSONL exporters/importers, Golden-vs-Candidate & Model-vs-Model diff engine, Quality Score (`0–100`), 4D Category Bias Matrix (`SD`, `FS`, `PV`, `DL`) with Rule 1 & Rule 2 overrides, pre-selection advisory Markdown/JSON writer, MCP-ready `query.py`, Typer CLI (`beliefaudit`), and `examples/offline_audit_demo.py` (`PR #1`, 25 tests passing, 97.13% coverage).
