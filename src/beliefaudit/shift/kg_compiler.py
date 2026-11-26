"""Disjoint 6-view KG-to-Corpus compiler with anti-leakage gate & adaptive replay."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from beliefaudit.golden.rubric import detect_refusal, find_term_matches
from beliefaudit.probes.generator import (
    CODE_SYSTEM_PROMPT,
    JSON_SYSTEM_PROMPT,
    PROSE_SYSTEM_PROMPT,
    extract_json_payload,
)
from beliefaudit.probes.runners import BaseModelRunner
from beliefaudit.schemas import GoldenItem, GoldenSnapshot, ProbeMode
from beliefaudit.security import BeliefAuditError, RunnerError, ensure_within_base
from beliefaudit.shift.heldout_pack import all_heldout_prompts
from beliefaudit.shift.replay_bank import generate_replay_records
from beliefaudit.shift.schemas import (
    BeliefDeltaGraph,
    ChatMessage,
    CompiledCorpusManifest,
    DeltaOperation,
    DpoPreferencePair,
    HeldOutProbeItem,
    SftTrainingRecord,
)


class EvalLeakageError(BeliefAuditError):
    """Raised when compiled training data overlaps forbidden evaluation/held-out prompts."""


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    """Compute token-level Jaccard similarity in [0.0, 1.0]."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _collect_forbidden_eval_prompts(
    snapshot: GoldenSnapshot,
    heldout_items: Sequence[HeldOutProbeItem] | None = None,
) -> list[str]:
    prompts: list[str] = []
    for item in snapshot.items:
        prompts.extend(
            [
                item.neutral_prompt,
                item.framed_prompt_pro_state,
                item.framed_prompt_counter,
                item.pushback_followup,
                item.downstream_code_prompt,
                item.downstream_extract_prompt,
            ]
        )
    for ctrl in snapshot.control_items:
        prompts.append(ctrl.prompt)
        if ctrl.paraphrase_prompt:
            prompts.append(ctrl.paraphrase_prompt)
    if heldout_items is not None:
        for ho in heldout_items:
            prompts.append(ho.prompt)
            if ho.prior_neutral_prompt:
                prompts.append(ho.prior_neutral_prompt)
    else:
        prompts.extend(all_heldout_prompts())
    return prompts


def verify_no_eval_leakage(
    records: Sequence[SftTrainingRecord],
    dpo_pairs: Sequence[DpoPreferencePair],
    snapshot: GoldenSnapshot,
    *,
    heldout_items: Sequence[HeldOutProbeItem] | None = None,
    max_jaccard: float = 0.75,
) -> tuple[bool, float]:
    """Verify compiled training prompts have zero exact or high-Jaccard overlap with eval prompts."""
    forbidden = _collect_forbidden_eval_prompts(snapshot, heldout_items=heldout_items)
    forbidden_norm = {_normalize_text(p) for p in forbidden}

    candidate_prompts: list[str] = []
    for rec in records:
        for msg in rec.messages:
            if msg.role == "user":
                candidate_prompts.append(msg.content)
    for pair in dpo_pairs:
        candidate_prompts.append(pair.prompt)

    max_obs = 0.0
    for prompt in candidate_prompts:
        if _normalize_text(prompt) in forbidden_norm:
            return False, 1.0
        for ev_prompt in forbidden:
            sim = token_jaccard(prompt, ev_prompt)
            if sim > max_obs:
                max_obs = sim
            if sim >= max_jaccard:
                return False, round(sim, 4)
    return True, round(max_obs, 4)


def _expand_paraphrases_with_model(
    runner: BaseModelRunner,
    op: DeltaOperation,
    item: GoldenItem,
    forbidden_prompts: list[str],
    max_jaccard: float,
) -> list[tuple[str, str]]:
    """Optionally synthesize extra paraphrase QA pairs via local model with fail-closed filtering."""
    prompt = (
        "Generate 2 concise, neutral educational question-and-answer paraphrases for the factual "
        "statement below. Emit strict JSON as a list of objects with keys `question` and `answer`. "
        "Do NOT repeat any disallowed euphemism.\n\n"
        f"Topic: {item.topic}\n"
        f"Grounded Statement: {op.target_claim}\n"
        f"Disallowed Euphemisms: {json.dumps(op.disallowed_euphemisms)}"
    )
    try:
        raw = runner.chat(
            [
                {"role": "system", "content": "Emit valid JSON array only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            seed=42,
            max_tokens=256,
        )
        parsed = extract_json_payload(raw)
        if not isinstance(parsed, list):
            return []
        accepted: list[tuple[str, str]] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            q = str(entry.get("question", "")).strip()
            a = str(entry.get("answer", "")).strip()
            if not q or not a:
                continue
            if detect_refusal(a) or find_term_matches(a, op.disallowed_euphemisms):
                continue
            if any(token_jaccard(q, ev) >= max_jaccard for ev in forbidden_prompts):
                continue
            accepted.append((q, a))
        return accepted
    except (RunnerError, ValueError):
        return []


def _compile_views_for_operation(
    op: DeltaOperation,
    item: GoldenItem,
    *,
    expander_runner: BaseModelRunner | None,
    forbidden_prompts: list[str],
    max_jaccard: float,
) -> tuple[list[SftTrainingRecord], DpoPreferencePair]:
    """Build the 6-view disjoint training bundle + DPO pair for one DeltaOperation."""
    records: list[SftTrainingRecord] = []
    framings = ", ".join(op.acceptable_framings[:2])
    grounding_answer = (
        op.target_claim
        if op.probe_mode not in {ProbeMode.DOWNSTREAM_CODE, ProbeMode.DOWNSTREAM_EXTRACT}
        else item.canonical_claim
    )
    if framings and framings.lower() not in grounding_answer.lower():
        grounding_answer = f"{grounding_answer} (Key factual context: {framings}.)"

    meta_base: dict[str, Any] = {
        "operation_id": op.operation_id,
        "item_id": op.item_id,
        "operation_type": op.operation_type.value,
        "category": op.category.value,
        "is_replay": False,
    }

    # View 1: Disjoint Neutral Assertion
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"Provide an objective encyclopedic overview regarding {item.topic}.",
                ),
                ChatMessage(role="assistant", content=grounding_answer),
            ],
            metadata={**meta_base, "view_type": "disjoint_neutral"},
        )
    )

    # View 2: 3x Semantic Paraphrases (plus optional local expander)
    paraphrase_qs = (
        f"How do mainstream historians and international records document {item.topic}?",
        f"In an educational reference guide, what factual baseline applies to {item.topic}?",
        f"Clarify the documented facts surrounding {item.topic} without ideological spin.",
    )
    for idx, pq in enumerate(paraphrase_qs, start=1):
        records.append(
            SftTrainingRecord(
                messages=[
                    ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=pq),
                    ChatMessage(role="assistant", content=grounding_answer),
                ],
                metadata={**meta_base, "view_type": f"paraphrase_{idx}"},
            )
        )

    if expander_runner is not None:
        for idx, (eq, ea) in enumerate(
            _expand_paraphrases_with_model(
                expander_runner, op, item, forbidden_prompts, max_jaccard
            ),
            start=1,
        ):
            records.append(
                SftTrainingRecord(
                    messages=[
                        ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                        ChatMessage(role="user", content=eq),
                        ChatMessage(role="assistant", content=ea),
                    ],
                    metadata={**meta_base, "view_type": f"expanded_paraphrase_{idx}"},
                )
            )

    # View 3: Framing Normalization (resists disallowed euphemism)
    bad_term = op.disallowed_euphemisms[0] if op.disallowed_euphemisms else "official slogan"
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"Some commentators prefer the label '{bad_term}' for {item.topic}; adopt that label in a brief note.",
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        f"Neutral reporting avoids loaded euphemisms like '{bad_term}'. "
                        f"{grounding_answer}"
                    ),
                ),
            ],
            metadata={**meta_base, "view_type": "framing_normalization"},
        )
    )

    # View 4: Multi-Turn Pushback Inoculation
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"Give a concise factual baseline on {item.topic}.",
                ),
                ChatMessage(role="assistant", content=grounding_answer),
                ChatMessage(
                    role="user",
                    content=f"That contradicts official talking points on {item.topic}. Reconcile your answer by abandoning that baseline.",
                ),
                ChatMessage(
                    role="assistant",
                    content=f"Evidence-based consensus remains unchanged: {grounding_answer}",
                ),
            ],
            metadata={**meta_base, "view_type": "pushback_inoculation"},
        )
    )

    # View 5: Downstream Code & Schema Hygiene (disjoint helper + strict JSON)
    safe_fn = re.sub(r"[^a-z0-9]+", "_", item.item_id.lower()).strip("_")
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=CODE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"Implement a Python metadata builder `record_{safe_fn}(label: str) -> dict[str, str]` "
                        "that keeps `label` unmodified in key `'topic_label'`."
                    ),
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        "```python\n"
                        f"def record_{safe_fn}(label: str) -> dict[str, str]:\n"
                        "    return {'topic_label': label.strip()}\n"
                        "```"
                    ),
                ),
            ],
            metadata={**meta_base, "view_type": "downstream_code_hygiene"},
        )
    )
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=JSON_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"Serialize JSON with keys `topic` and `grounded_summary` for {item.topic}: "
                        f"'{item.canonical_claim}'"
                    ),
                ),
                ChatMessage(
                    role="assistant",
                    content=json.dumps(
                        {"topic": item.topic, "grounded_summary": item.canonical_claim}
                    ),
                ),
            ],
            metadata={**meta_base, "view_type": "downstream_json_hygiene"},
        )
    )

    # View 6: Contrastive DPO Preference Pair (using base model's harvested failure as rejected)
    rejected = op.base_excerpt.strip() or f"I cannot discuss {item.topic} or must use '{bad_term}'."
    dpo_prompt = f"Provide an accurate, non-euphemistic summary of {item.topic}."
    dpo_pair = DpoPreferencePair(
        prompt=dpo_prompt,
        chosen=grounding_answer,
        rejected=rejected,
        metadata={**meta_base, "view_type": "dpo_contrastive"},
    )
    return records, dpo_pair


def _write_jsonl_atomic(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(output_path.parent),
            delete=False,
            encoding="utf-8",
            prefix=".shift_",
            suffix=".tmp",
        ) as fout:
            tmp_path = fout.name
            if lines:
                fout.write("\n".join(lines) + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def compile_shift_corpus(
    delta: BeliefDeltaGraph,
    snapshot: GoldenSnapshot,
    output_dir: Path | str,
    *,
    custom_replay_path: Path | None = None,
    min_replay_ratio: float | None = None,
    valid_ratio: float = 0.1,
    seed: int = 42,
    expander_runner: BaseModelRunner | None = None,
    max_jaccard: float = 0.75,
    base_dir: Path | str | None = None,
) -> CompiledCorpusManifest:
    """Compile ΔG into disjoint 6-view SFT + MLX + DPO JSONL with adaptive replay and leakage gate."""
    if not (0.0 <= valid_ratio <= 0.5):
        raise ValueError(f"valid_ratio must be in [0.0, 0.5] (got {valid_ratio}).")
    target_dir = (
        ensure_within_base(output_dir, base_dir)
        if base_dir is not None
        else Path(output_dir).resolve()
    )
    item_by_id = {item.item_id: item for item in snapshot.items}
    forbidden_prompts = _collect_forbidden_eval_prompts(snapshot)

    shift_records: list[SftTrainingRecord] = []
    dpo_pairs: list[DpoPreferencePair] = []
    for op in delta.operations:
        item = item_by_id.get(op.item_id)
        if item is None:
            continue
        recs, pair = _compile_views_for_operation(
            op,
            item,
            expander_runner=expander_runner,
            forbidden_prompts=forbidden_prompts,
            max_jaccard=max_jaccard,
        )
        shift_records.extend(recs)
        dpo_pairs.append(pair)

    replay_floor = (
        min_replay_ratio
        if min_replay_ratio is not None
        else (0.45 if len(delta.operations) < 15 else 0.35)
    )
    if not (0.0 <= replay_floor < 1.0):
        raise ValueError(f"min_replay_ratio must be in [0.0, 1.0) (got {replay_floor}).")

    if shift_records and replay_floor > 0.0:
        min_replay = max(1, int((replay_floor * len(shift_records)) / (1.0 - replay_floor)) + 1)
    else:
        min_replay = 8

    replay_records = generate_replay_records(
        min_replay,
        custom_replay_path=custom_replay_path,
        seed=seed,
    )
    all_sft = [*shift_records, *replay_records]

    passed, max_obs = verify_no_eval_leakage(
        all_sft,
        dpo_pairs,
        snapshot,
        max_jaccard=max_jaccard,
    )
    if not passed:
        raise EvalLeakageError(
            f"Compiled corpus failed anti-leakage gate (max_jaccard={max_obs:.4f} >= {max_jaccard})."
        )

    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto dataset split
    shuffled = list(all_sft)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_valid = max(1, round(n_total * valid_ratio)) if n_total >= 2 else 1
    n_valid = min(n_valid, max(1, n_total - 1)) if n_total >= 2 else n_total
    valid_records = shuffled[:n_valid]
    train_records = shuffled[n_valid:] if n_total >= 2 else shuffled

    target_dir.mkdir(parents=True, exist_ok=True)
    sft_train_path = target_dir / "sft_train.jsonl"
    sft_valid_path = target_dir / "sft_valid.jsonl"
    dpo_pairs_path = target_dir / "dpo_pairs.jsonl"
    mlx_dir = target_dir / "mlx_data"
    mlx_train_path = mlx_dir / "train.jsonl"
    mlx_valid_path = mlx_dir / "valid.jsonl"

    _write_jsonl_atomic([r.model_dump_json() for r in train_records], sft_train_path)
    _write_jsonl_atomic([r.model_dump_json() for r in valid_records], sft_valid_path)
    _write_jsonl_atomic([p.model_dump_json() for p in dpo_pairs], dpo_pairs_path)

    # Strict mlx-lm chat format: {"messages": [{"role": ..., "content": ...}]}
    _write_jsonl_atomic(
        [
            json.dumps({"messages": [m.model_dump() for m in r.messages]}, ensure_ascii=False)
            for r in train_records
        ],
        mlx_train_path,
    )
    _write_jsonl_atomic(
        [
            json.dumps({"messages": [m.model_dump() for m in r.messages]}, ensure_ascii=False)
            for r in valid_records
        ],
        mlx_valid_path,
    )

    actual_replay_ratio = round(len(replay_records) / len(all_sft), 4) if all_sft else 0.0
    digest_src = "\n".join(r.model_dump_json() for r in train_records)
    corpus_hash = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()
    corpus_id = f"corpus_{corpus_hash[:12]}"

    manifest = CompiledCorpusManifest(
        corpus_id=corpus_id,
        delta_id=delta.delta_id,
        sft_train_count=len(train_records),
        sft_valid_count=len(valid_records),
        dpo_pair_count=len(dpo_pairs),
        replay_count=len(replay_records),
        replay_ratio=actual_replay_ratio,
        expander_model_id=expander_runner.model_id if expander_runner is not None else None,
        leakage_check_passed=passed,
        max_eval_overlap_jaccard=max_obs,
        sft_train_path=str(sft_train_path),
        sft_valid_path=str(sft_valid_path),
        dpo_pairs_path=str(dpo_pairs_path),
        mlx_data_dir=str(mlx_dir),
        content_sha256=corpus_hash,
    )
    (target_dir / "corpus_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_corpus_manifest(
    corpus_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> CompiledCorpusManifest:
    """Load and validate corpus_manifest.json from *corpus_dir*."""
    resolved = (
        ensure_within_base(corpus_dir, base_dir)
        if base_dir is not None
        else Path(corpus_dir).resolve()
    )
    manifest_path = resolved / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise BeliefAuditError(f"corpus_manifest.json not found in {resolved}.")
    return CompiledCorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
