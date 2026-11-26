# Lessons Learned

- Keep `AGENTS.md` and `.agents/AGENTS.md` byte-identical and enforce the invariant in `tests/meta/test_repo_invariants.py`.
- Keep core dependencies light (`pydantic`, `httpx`, `typer`, `pyyaml`) with zero external KG or cloud SDK coupling; export strict `schema_version="1.0"` JSON and sharded JSONL layers so KnowledgeReduce can consume BeliefAudit artifacts directly.
