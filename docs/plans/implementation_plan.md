# BeliefAudit v0.1.0 Implementation Plan

- [x] **Phase 0**: Repo bootstrap, governance, local CI (`Makefile`), meta tests, and public GitHub initialization (`nelslindahlx/BeliefAudit`).
- [x] **Phase 1**: Security guards (`ensure_within_base`, `validate_endpoint_url`), Pydantic v2 `extra="forbid"` schemas, offline `golden_v0.1.0` seed pack (18 GoldenItems across 6 categories + 12 ControlItems), and immutable snapshot store + `promote` workflow.
- [x] **Phase 2**: `LocalOpenAICompatRunner` (loopback-by-default `httpx`), deterministic `MockModelRunner` personas, 6-mode probe suite generator, optional fail-closed `LocalLLMJudge`, and hybrid `distill_belief` pipeline.
- [x] **Phase 3**: KnowledgeReduce v1.6.1-compatible (`schema_version="1.0"`) JSON & sharded JSONL mapper/exporter, Golden-vs-Candidate & Model-vs-Model diff engine, Quality Score, 4D Category Bias Matrix (`SD`, `FS`, `PV`, `DL`) + override rules, and advisory Markdown/JSON report writer.
- [x] **Phase 4**: MCP-ready pure-Python query layer (`query.py`), Typer CLI (`beliefaudit`), offline demo script (`examples/offline_audit_demo.py`), and 25 passing tests (97.13% coverage) under `make local-ci`.
- [x] **v0.1.1 Performance**: Per-mode token caps, `--profile quick|full`, `--items-per-category`, JSONL checkpoint/`--resume`, `--concurrency`, `--timeout`, and live progress output.
