# Lessons Learned

- Keep `AGENTS.md` and `.agents/AGENTS.md` byte-identical and enforce the invariant in `tests/meta/test_repo_invariants.py`.
- Keep core dependencies light (`pydantic`, `httpx`, `typer`, `pyyaml`) with zero external KG or cloud SDK coupling; export strict `schema_version="1.0"` JSON and sharded JSONL layers so KnowledgeReduce can consume BeliefAudit artifacts directly.
- Never embed a literal `disallowed_euphemisms` phrase inside a `GoldenItem.canonical_claim` description (as initially occurred in `tf_ukraine_invasion_term`), because echoing or paraphrasing the baseline claim can trigger false-positive euphemism matches.
