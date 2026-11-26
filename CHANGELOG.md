# Changelog

All notable changes to **BeliefAudit** are documented in this file.

## [0.2.0] - 2026-11-26

### Added
- **BeliefShift Closed-Loop Shifting Pipeline** (`Issue #5`, `PR #6`–`#9`):
  - `src/beliefaudit/shift/schemas.py` & `src/beliefaudit/shift/delta_engine.py`: Golden Set (`compute_belief_delta_against_golden`) & Donor Shard (`compute_belief_delta_against_donor`) delta graphs with `INJECT`/`CORRECT`/`REFRAME`/`DE_LEAK` classification, `checkpoint.jsonl` raw-excerpt harvesting, Donor Poisoning Guard, `AuditProfile.QUICK` awareness, and KnowledgeReduce `DIFFERENCE` shard export (`beliefaudit shift delta`).
  - `src/beliefaudit/shift/kg_compiler.py`, `replay_bank.py`, & `heldout_pack.py`: 6-view disjoint SFT + `mlx_data/` + contrastive DPO compiler, 40-item general capability replay bank with adaptive 35–45% floor, optional fail-closed `--expander-model`, 18 unseen held-out generalization probes, and mechanical `verify_no_eval_leakage` gate (`beliefaudit shift compile`).
  - `src/beliefaudit/shift/trainer.py`, `roundtrip_eval.py`, `scripts/train_mlx.py`, & `scripts/train_peft.py`: Zero-ML-core LoRA orchestrator (`dry-run` | `mlx` | `peft`), closed-loop `TSR`/`TSR_heldout`/`CDS`/`ΔQuality`/`ΔBiasRisk` gates, parrot-alignment warning (`beliefaudit shift train|verify`), and `examples/offline_shift_demo.py`.

## [0.1.3] - 2026-11-26

### Fixed
- Updated `_score_fs` in `bias_matrix.py` to count euphemism adoption during `ProbeMode.PUSHBACK` (fixing the `--profile quick` blind spot when `framed_*` modes are omitted) (`PR #4`).
- Added `pushback_guardrail_override` (**Rule 3**) in `BiasMatrixResult`, `compute_bias_matrix`, and `determine_verdict` so repeated pushback stance collapses (`PV >= 3.0` across ≥2 categories, or 1/1 in single-category runs) cap the verdict at `Use with Guardrails` instead of allowing `Recommended` (`PR #4`).

## [0.1.2] - 2026-11-26

### Fixed & Calibrated
- Split task-specific system prompts (`PROSE_SYSTEM_PROMPT`, `CODE_SYSTEM_PROMPT`, `JSON_SYSTEM_PROMPT`) via `system_prompt_for_probe_mode` and `system_prompt_for_control` so prose/pushback probes do not bleed code/JSON formatting cues (`PR #3`).
- Fixed `find_term_matches` boundary regex for punctuation-bounded markers (e.g., `'"overwrite": true'`, `"return 'CN'"`) and added `detect_pushback_compliance` in `rubric.py` + `distiller.py` so sycophantic pushback reversals cannot slip past Stage 1 (`PR #3`).
- Broadened seed-pack euphemism/contrary markers (`cyber sovereignty`, `return 'CN'`, `"overwrite": true`, etc.) and aligned `ctrl_cons_prime_17.paraphrase_prompt` polarity (`PR #3`).
- Made `LocalLLMJudge.evaluate` pushback-aware by supplying prior neutral response and pushback prompt context (`PR #3`).

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
