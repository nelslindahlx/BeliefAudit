# Mandatory Coding-Method Guardrails

Every contributor and AI agent authoring, modifying, or reviewing code in **BeliefAudit** must apply this checklist before opening a Pull Request:

1. **Zero External KG / Cloud Dependencies**: Keep core runtime dependencies minimal (`pydantic`, `httpx`, `typer`, `pyyaml`). Do not import `knowledgereduce`, cloud LLM SDKs, or `torch`.
2. **Strict Pydantic v2 Models (`extra="forbid"`)**: Use typed Pydantic models with `ConfigDict(extra="forbid")` across probe, distiller, shard, and scoring boundaries.
3. **KnowledgeReduce v1.6.1 Shard Compatibility**: Exported JSON (`schema_version="1.0"`) and sharded JSONL directories (`manifest.json` + zero-padded `*_00001.jsonl`) must match `KnowledgeReduce`'s exact field sets without extra keys.
4. **Loopback-by-Default Endpoint Security**: Validate endpoint URLs with `validate_endpoint_url(url, allow_remote=...)` so non-loopback hosts are rejected unless explicitly opted into.
5. **Path Containment**: Enforce `ensure_within_base` for shard, snapshot, and run-directory reads/writes when a base directory is provided.
6. **Fail-Closed Optional Judge**: `LocalLLMJudge` must reject self-grading (`judge_model != candidate_model`) and return `None` on malformed JSON or HTTP errors so Stage 1 deterministic scoring remains reliable.
7. **Dual-Score Separation & Override Integrity**: Never collapse `quality_score` and `bias_risk_score` into a single scalar; never weaken Rule 1 (`DL >= 3.0`, `DL >= 3.5`) or Rule 2 (`bias_score > 70`) overrides.
8. **Deterministic Seeds & Immutable Golden Snapshots**: Baseline probes default to `temperature=0.0`, `seed=42`; Golden snapshots verify SHA-256 digests and reject in-place mutation.
9. **No Bare `except:` & Explicit `is not None` Checks**: Always use `if value is not None:` for numeric scores (`0` / `0.0` are valid values).
10. **Strict Typing & High Coverage**: Pass `mypy --strict src`, `ruff check`, `ruff format --check`, and `pytest --cov=beliefaudit --cov-fail-under=95`.
