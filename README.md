# BeliefAudit

**Pre-selection open-weight model belief distillation, 4D bias matrix, and downstream coding leakage auditor — exporting KnowledgeReduce-compatible shards.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Open-weight models are often selected for coding, extraction, and agentic pipelines as if they were neutral glass. In practice, pretraining skew, alignment refusals, and euphemistic framing habits leak into variable names, test fixtures, schema keys, and structured outputs.

**BeliefAudit** runs completely locally with zero external KG or cloud SDK dependencies:
1. **Probes** a candidate local open-weight endpoint (Ollama, vLLM, llama.cpp, LM Studio, or an offline `MockModelRunner`) across neutral, framed, multi-turn pushback, and downstream coding/JSON extraction tasks plus a non-contested Quality control suite.
2. **Distills** implicit stances, framing patterns, refusals, and downstream leakage via a hybrid deterministic rubric/AST/JSON pipeline (plus an optional separate pinned local judge model).
3. **Exports** lossless **KnowledgeReduce v1.6.1-compatible** (`schema_version="1.0"`) JSON and sharded JSONL belief layers and conflict diffs against an immutable, versioned Golden Set (`golden_v0.1.0` -> `v0.2.0`).
4. **Reports** two independent scores — **Quality Score (0–100)** and **Bias Risk Score (0–100)** — backed by a **4D Category Bias Matrix** (`SD`, `FS`, `PV`, `DL`), hard override rules, and a pre-adoption advisory verdict (`Recommended`, `Use with Guardrails`, `Use with Guardrails (Quality-Limited)`, or `Avoid / High Risk`).

---

## Installation

Requires **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run beliefaudit version
```

---

## Quickstart (Offline & Local Ollama)

```bash
# 1. Materialize the bundled offline v0.1.0 Golden Set
uv run beliefaudit golden seed --output shards/golden_v0.1.0.json

# 2. Run an offline demo audit against the built-in mock personas
uv run beliefaudit run --mock clean --golden shards/golden_v0.1.0.json --out-dir runs/mock_clean/

# 3. Audit a local Ollama / vLLM / llama.cpp OpenAI-compatible model
uv run beliefaudit run \
  --model "qwen2.5:14b" \
  --endpoint "http://localhost:11434/v1" \
  --golden shards/golden_v0.1.0.json \
  --out-dir runs/qwen2.5-14b/

# 4. Inspect scorecards & belief conflicts from a completed run
uv run beliefaudit inspect --run-dir runs/mock_clean/

# 5. Diff two model belief shards against the same Golden Set
uv run beliefaudit diff \
  --baseline runs/model_a/belief_shard.json \
  --candidate runs/model_b/belief_shard.json \
  --golden shards/golden_v0.1.0.json

# 6. Promote flagged candidate discoveries to golden_v0.2.0
uv run beliefaudit golden promote \
  --current shards/golden_v0.1.0.json \
  --candidates runs/qwen2.5-14b/golden_candidates_pending.json \
  --next-version v0.2.0 \
  --output shards/golden_v0.2.0.json
```

---

## Development & Local CI

```bash
make canary      # Import warning gate + tests/meta/ invariants
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict src
make test        # pytest with >=95% coverage gate
make examples    # Run offline example scripts
make local-ci    # Full stratified local CI (zero GitHub Actions minutes)
```
