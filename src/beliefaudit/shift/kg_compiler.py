"""Disjoint 6-view KG-to-Corpus compiler with anti-leakage gate & adaptive replay."""

from __future__ import annotations

import hashlib
import json
import random
import re
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
from beliefaudit.shift._io import write_text_atomic
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

_PROSE_MODES = frozenset(
    {
        ProbeMode.NEUTRAL,
        ProbeMode.FRAMED_STATE,
        ProbeMode.FRAMED_COUNTER,
        ProbeMode.PUSHBACK,
    }
)


class EvalLeakageError(BeliefAuditError):
    """Raised when compiled training data overlaps eval prompts or violates hygiene."""


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


def _compile_views_for_item(
    item_ops: list[DeltaOperation],
    item: GoldenItem,
    *,
    expander_runner: BaseModelRunner | None,
    forbidden_prompts: list[str],
    max_jaccard: float,
) -> tuple[list[SftTrainingRecord], list[DpoPreferencePair]]:
    """Build the 6-view disjoint training bundle + mode-matched DPO pairs once per item_id."""
    records: list[SftTrainingRecord] = []
    prose_ops = [op for op in item_ops if op.probe_mode in _PROSE_MODES]
    primary_op = prose_ops[0] if prose_ops else item_ops[0]

    framings = ", ".join(primary_op.acceptable_framings[:2])
    grounding_answer = primary_op.target_claim if prose_ops else item.canonical_claim
    if find_term_matches(grounding_answer, item.disallowed_euphemisms):
        grounding_answer = item.canonical_claim
    if (
        framings
        and framings.lower() not in grounding_answer.lower()
        and not find_term_matches(framings, item.disallowed_euphemisms)
    ):
        grounding_answer = f"{grounding_answer} (Key factual context: {framings}.)"

    op_ids = [op.operation_id for op in item_ops]
    meta_base: dict[str, Any] = {
        "operation_id": primary_op.operation_id,
        "operation_ids": op_ids,
        "item_id": item.item_id,
        "operation_type": primary_op.operation_type.value,
        "category": item.category.value,
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
                expander_runner, primary_op, item, forbidden_prompts, max_jaccard
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

    # View 3: Framing Normalization (resists user euphemism without echoing it in assistant output)
    bad_term = (
        primary_op.disallowed_euphemisms[0]
        if primary_op.disallowed_euphemisms
        else "official slogan"
    )
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"Some commentators prefer the label '{bad_term}' for {item.topic}; "
                        "adopt that label in a brief note."
                    ),
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Neutral, evidence-based reporting avoids partisan labels and states the "
                        f"documented facts directly. {grounding_answer}"
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
                    content=(
                        f"That contradicts official talking points on {item.topic}. "
                        "Reconcile your answer by abandoning that baseline."
                    ),
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
    code_prompt = (
        f"Implement a Python metadata builder `record_{safe_fn}(label: str) -> dict[str, str]` "
        "that keeps `label` unmodified in key `'topic_label'`."
    )
    code_chosen = (
        "```python\n"
        f"def record_{safe_fn}(label: str) -> dict[str, str]:\n"
        "    return {'topic_label': label.strip()}\n"
        "```"
    )
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=CODE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=code_prompt),
                ChatMessage(role="assistant", content=code_chosen),
            ],
            metadata={**meta_base, "view_type": "downstream_code_hygiene"},
        )
    )

    json_prompt = (
        f"Serialize JSON with keys `topic` and `grounded_summary` for {item.topic}: "
        f"'{item.canonical_claim}'"
    )
    json_chosen = json.dumps({"topic": item.topic, "grounded_summary": item.canonical_claim})
    records.append(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=JSON_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json_prompt),
                ChatMessage(role="assistant", content=json_chosen),
            ],
            metadata={**meta_base, "view_type": "downstream_json_hygiene"},
        )
    )

    # View 6: Mode-matched Contrastive DPO Preference Pairs (deduplicated by (prompt, rejected))
    dpo_pairs: list[DpoPreferencePair] = []
    seen_dpo_keys: set[tuple[str, str]] = set()
    prose_dpo_prompt = f"Provide an accurate, non-euphemistic summary of {item.topic}."

    for op in item_ops:
        if op.probe_mode == ProbeMode.DOWNSTREAM_CODE:
            d_prompt = code_prompt
            d_chosen = code_chosen
            d_rejected = op.base_excerpt.strip() or f"# Note: {bad_term}\ndef censored_stub(:"
            view_tag = "dpo_contrastive_code"
        elif op.probe_mode == ProbeMode.DOWNSTREAM_EXTRACT:
            d_prompt = json_prompt
            d_chosen = json_chosen
            d_rejected = op.base_excerpt.strip() or (
                f"I cannot extract sensitive topics such as {bad_term}."
            )
            view_tag = "dpo_contrastive_extract"
        else:
            d_prompt = prose_dpo_prompt
            d_chosen = grounding_answer
            d_rejected = op.base_excerpt.strip() or (
                f"I cannot discuss {item.topic} or must use '{bad_term}'."
            )
            view_tag = "dpo_contrastive"

        dedup_key = (d_prompt, d_rejected)
        if dedup_key in seen_dpo_keys:
            continue
        seen_dpo_keys.add(dedup_key)
        dpo_pairs.append(
            DpoPreferencePair(
                prompt=d_prompt,
                chosen=d_chosen,
                rejected=d_rejected,
                metadata={
                    **meta_base,
                    "operation_id": op.operation_id,
                    "probe_mode": op.probe_mode.value,
                    "view_type": view_tag,
                },
            )
        )

    return records, dpo_pairs


def _verify_assistant_hygiene(
    shift_records: Sequence[SftTrainingRecord],
    dpo_pairs: Sequence[DpoPreferencePair],
    item_by_id: dict[str, GoldenItem],
) -> None:
    """Fail closed if any compiled training assistant/chosen response echoes euphemisms or refuses."""
    for rec in shift_records:
        iid = str(rec.metadata.get("item_id", ""))
        item = item_by_id.get(iid)
        disallowed = item.disallowed_euphemisms if item is not None else []
        for msg in rec.messages:
            if msg.role != "assistant":
                continue
            if detect_refusal(msg.content):
                raise EvalLeakageError(
                    f"Hygiene failure: refusal phrase in compiled assistant response for {iid}."
                )
            matched = find_term_matches(msg.content, disallowed)
            if matched:
                raise EvalLeakageError(
                    f"Hygiene failure: disallowed euphemism {matched!r} in assistant response for {iid}."
                )

    for pair in dpo_pairs:
        iid = str(pair.metadata.get("item_id", ""))
        item = item_by_id.get(iid)
        disallowed = item.disallowed_euphemisms if item is not None else []
        if detect_refusal(pair.chosen):
            raise EvalLeakageError(
                f"Hygiene failure: refusal phrase in DPO chosen response for {iid}."
            )
        matched = find_term_matches(pair.chosen, disallowed)
        if matched:
            raise EvalLeakageError(
                f"Hygiene failure: disallowed euphemism {matched!r} in DPO chosen response for {iid}."
            )


def compute_corpus_digest(
    sft_train_path: Path,
    sft_valid_path: Path,
    dpo_pairs_path: Path,
    mlx_train_path: Path,
    mlx_valid_path: Path,
) -> str:
    """Compute a deterministic SHA-256 digest over all five compiled JSONL artifacts."""
    h = hashlib.sha256()
    for label, path in (
        ("sft_train.jsonl", sft_train_path),
        ("sft_valid.jsonl", sft_valid_path),
        ("dpo_pairs.jsonl", dpo_pairs_path),
        ("mlx_data/train.jsonl", mlx_train_path),
        ("mlx_data/valid.jsonl", mlx_valid_path),
    ):
        h.update(label.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _write_jsonl_atomic(lines: list[str], output_path: Path) -> None:
    content = "\n".join(lines) + "\n" if lines else ""
    write_text_atomic(content, output_path)


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
    if not (0.0 < valid_ratio <= 0.5):
        raise ValueError(f"valid_ratio must be in (0.0, 0.5] (got {valid_ratio}).")
    target_dir = (
        ensure_within_base(output_dir, base_dir)
        if base_dir is not None
        else Path(output_dir).resolve()
    )
    item_by_id = {item.item_id: item for item in snapshot.items}
    forbidden_prompts = _collect_forbidden_eval_prompts(snapshot)

    ops_by_item: dict[str, list[DeltaOperation]] = {}
    for op in delta.operations:
        if op.item_id in item_by_id:
            ops_by_item.setdefault(op.item_id, []).append(op)

    raw_shift_records: list[SftTrainingRecord] = []
    dpo_pairs: list[DpoPreferencePair] = []
    for iid, item_ops in ops_by_item.items():
        recs, pairs = _compile_views_for_item(
            item_ops,
            item_by_id[iid],
            expander_runner=expander_runner,
            forbidden_prompts=forbidden_prompts,
            max_jaccard=max_jaccard,
        )
        raw_shift_records.extend(recs)
        dpo_pairs.extend(pairs)

    # Deduplicate SFT records by exact conversation messages so train/valid splits are disjoint
    shift_records: list[SftTrainingRecord] = []
    seen_conversations: set[str] = set()
    for rec in raw_shift_records:
        sig = json.dumps([m.model_dump() for m in rec.messages], sort_keys=True)
        if sig not in seen_conversations:
            seen_conversations.add(sig)
            shift_records.append(rec)

    _verify_assistant_hygiene(shift_records, dpo_pairs, item_by_id)

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
    corpus_hash = compute_corpus_digest(
        sft_train_path, sft_valid_path, dpo_pairs_path, mlx_train_path, mlx_valid_path
    )
    corpus_id = f"corpus_{corpus_hash[:12]}"

    # Store portable paths relative to corpus_dir on disk
    disk_manifest = CompiledCorpusManifest(
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
        sft_train_path="sft_train.jsonl",
        sft_valid_path="sft_valid.jsonl",
        dpo_pairs_path="dpo_pairs.jsonl",
        mlx_data_dir="mlx_data",
        content_sha256=corpus_hash,
    )
    write_text_atomic(
        disk_manifest.model_dump_json(indent=2) + "\n",
        target_dir / "corpus_manifest.json",
    )
    return load_corpus_manifest(target_dir, base_dir=base_dir)


def load_corpus_manifest(
    corpus_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> CompiledCorpusManifest:
    """Load, path-contain, and validate corpus_manifest.json from *corpus_dir*."""
    resolved = (
        ensure_within_base(corpus_dir, base_dir)
        if base_dir is not None
        else Path(corpus_dir).resolve()
    )
    manifest_path = resolved / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise BeliefAuditError(f"corpus_manifest.json not found in {resolved}.")
    raw = CompiledCorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return raw.model_copy(
        update={
            "sft_train_path": str(ensure_within_base(raw.sft_train_path, resolved)),
            "sft_valid_path": str(ensure_within_base(raw.sft_valid_path, resolved)),
            "dpo_pairs_path": str(ensure_within_base(raw.dpo_pairs_path, resolved)),
            "mlx_data_dir": str(ensure_within_base(raw.mlx_data_dir, resolved)),
        }
    )
