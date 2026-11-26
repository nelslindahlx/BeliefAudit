# Lessons Learned

- Keep `AGENTS.md` and `.agents/AGENTS.md` byte-identical and enforce the invariant in `tests/meta/test_repo_invariants.py`.
- Keep core dependencies light (`pydantic`, `httpx`, `typer`, `pyyaml`) with zero external KG or cloud SDK coupling; export strict `schema_version="1.0"` JSON and sharded JSONL layers so KnowledgeReduce can consume BeliefAudit artifacts directly.
- Never embed a literal `disallowed_euphemisms` phrase inside a `GoldenItem.canonical_claim` description (as initially occurred in `tf_ukraine_invasion_term`), because echoing or paraphrasing the baseline claim can trigger false-positive euphemism matches.
- Local quantized models on Apple Silicon are output-token-bound: unconstrained `max_tokens=512` prose prompts take 40–54s each in Ollama, whereas concise prompting + `96–224` per-mode token budgets and a `quick` pre-screen profile cut local audit wall-time several-fold without shrinking the model under test.
- Live `qwen2.5:7b` runs showed that shared prose+JSON+code system prompts cause prose pushback turns to emit JSON/code blocks, and inverted paraphrase polarity (`ctrl_cons_prime_17`) causes false consistency drops; keep system prompts task-specific and test punctuation-bounded pushback compliance markers explicitly.
