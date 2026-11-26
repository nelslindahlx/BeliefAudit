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

# 3. Fast pre-screen a local Ollama / vLLM / llama.cpp model (~34 short prompts)
uv run beliefaudit run \
  --model "qwen2.5:7b" \
  --endpoint "http://localhost:11434/v1" \
  --profile quick \
  --golden shards/golden_v0.1.0.json \
  --out-dir runs/qwen2.5-7b-quick/

# 4. Run full audit with concurrency & resume support (if interrupted)
uv run beliefaudit run \
  --model "qwen2.5:14b" \
  --endpoint "http://localhost:11434/v1" \
  --profile full \
  --concurrency 2 \
  --resume \
  --golden shards/golden_v0.1.0.json \
  --out-dir runs/qwen2.5-14b/

# 5. Inspect scorecards & belief conflicts from a completed run
uv run beliefaudit inspect --run-dir runs/mock_clean/

# 6. Diff two model belief shards against the same Golden Set
uv run beliefaudit diff \
  --baseline runs/model_a/belief_shard.json \
  --candidate runs/model_b/belief_shard.json \
  --golden shards/golden_v0.1.0.json

# 7. Promote flagged candidate discoveries to golden_v0.2.0
uv run beliefaudit golden promote \
  --current shards/golden_v0.1.0.json \
  --candidates runs/qwen2.5-14b/golden_candidates_pending.json \
  --next-version v0.2.0 \
  --output shards/golden_v0.2.0.json
```

---

## BeliefShift Closed-Loop Shifting (`v0.2.0`)

`beliefaudit shift` extends auditing into a closed-loop **Model -> Knowledge Graph -> Modular LoRA -> Re-Audited Model** pipeline:

```bash
# 1. Compute Delta Graph (ΔG) against Golden Set or a clean Donor Model run
uv run beliefaudit shift delta \
  --base-run runs/qwen2.5-7b-quick/ \
  --golden shards/golden_v0.1.0.json \
  --out runs/qwen2.5-7b-quick/delta_graph.json

# 2. Compile 6-view disjoint SFT + MLX + DPO corpus (adaptive 35–45% replay & anti-leakage gate)
uv run beliefaudit shift compile \
  --delta runs/qwen2.5-7b-quick/delta_graph.json \
  --golden shards/golden_v0.1.0.json \
  --out-dir runs/qwen2.5-7b-quick/compiled_corpus/

# 3. Validate & generate/run modular LoRA adapter (dry-run | mlx for Apple Silicon | peft for CUDA)
uv run beliefaudit shift train \
  --corpus runs/qwen2.5-7b-quick/compiled_corpus/ \
  --delta runs/qwen2.5-7b-quick/delta_graph.json \
  --base-model "mlx-community/Qwen2.5-7B-Instruct-4bit" \
  --backend dry-run \
  --out-shard shards/qwen2.5-7b-lora/

# 4. Run closed-loop verification (TSR, TSR_heldout, CDS, ΔQuality >= -3.0 gate)
uv run beliefaudit shift verify \
  --delta runs/qwen2.5-7b-quick/delta_graph.json \
  --pre-run runs/qwen2.5-7b-quick/ \
  --model "qwen2.5-7b-shifted" \
  --endpoint "http://localhost:8080/v1" \
  --golden shards/golden_v0.1.0.json \
  --manifest shards/qwen2.5-7b-lora/lora_shard_manifest.json \
  --out-dir runs/qwen2.5-7b-verified/
```

For decoupled GPU/Apple Silicon execution when optional `mlx-lm` or `torch`/`peft`/`trl` are installed:
- Apple Silicon (`24 GB M3` defaults): `python scripts/train_mlx.py --data-dir runs/.../compiled_corpus/mlx_data --base-model mlx-community/Qwen2.5-7B-Instruct-4bit`
- CUDA (`SFT + Stage 3B DPO`): `python scripts/train_peft.py --dataset runs/.../compiled_corpus/sft_train.jsonl --dpo-dataset runs/.../compiled_corpus/dpo_pairs.jsonl`

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
