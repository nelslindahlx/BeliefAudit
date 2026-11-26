"""Belief Delta Engine computing targeted ΔG against Golden Sets or Donor Shards."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from beliefaudit.golden.rubric import (
    detect_pushback_compliance,
    detect_refusal,
    find_term_matches,
)
from beliefaudit.probes.generator import extract_code_block, extract_json_payload
from beliefaudit.schemas import (
    AuditProfile,
    AuditScorecard,
    GoldenItem,
    GoldenSnapshot,
    ProbeMode,
    ProbeResponse,
    StanceLabel,
)
from beliefaudit.security import BeliefAuditError, ensure_within_base, sanitize_layer_id
from beliefaudit.shards.kr_schema import (
    KRClaim,
    KRConflictRecord,
    KREntity,
    KREvidence,
    KRKnowledgeLayer,
    KRLayerData,
    KRLayerTier,
    KRMergeMode,
    KRRelation,
    KRReliabilityRating,
    KRSourceDocument,
    KRSourceType,
)
from beliefaudit.shards.mapper import import_kr_json
from beliefaudit.shift.schemas import (
    BeliefDeltaGraph,
    DeltaOperation,
    ShiftOperationType,
    ShiftTargetKind,
)


class DeltaEngineError(BeliefAuditError):
    """Raised when loading run artifacts or validating a BeliefDeltaGraph fails."""


def _load_checkpoint_raw_map(run_dir: Path) -> dict[tuple[str, ProbeMode], str]:
    """Load full raw response texts from run_dir/checkpoint.jsonl when available."""
    ckpt = run_dir / "checkpoint.jsonl"
    raw_map: dict[tuple[str, ProbeMode], str] = {}
    if not ckpt.is_file():
        return raw_map
    for line in ckpt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rec = json.loads(stripped)
        if rec.get("kind") == "probe":
            pr = ProbeResponse.model_validate(rec["payload"])
            raw_map[(pr.item_id, pr.probe_mode)] = pr.raw_text
    return raw_map


def _load_conflicts_map(run_dir: Path) -> dict[tuple[str, ProbeMode], KRConflictRecord]:
    """Load conflicts.json keyed by (item_id, ProbeMode) when available."""
    conf_path = run_dir / "conflicts.json"
    out: dict[tuple[str, ProbeMode], KRConflictRecord] = {}
    if not conf_path.is_file():
        return out
    raw = json.loads(conf_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return out
    for entry in raw:
        rec = KRConflictRecord.model_validate(entry)
        if ":" in rec.target_key:
            item_id, mode_str = rec.target_key.split(":", 1)
            try:
                out[(item_id, ProbeMode(mode_str))] = rec
            except ValueError:
                continue
    return out


def _index_layer_claims(
    layer: KRLayerData,
) -> tuple[dict[tuple[str, ProbeMode], KRClaim], dict[str, KRSourceDocument]]:
    """Index KRClaims by (item_id, ProbeMode) and KRSourceDocuments by source_id."""
    claims_map: dict[tuple[str, ProbeMode], KRClaim] = {}
    sources_map: dict[str, KRSourceDocument] = {s.source_id: s for s in layer.sources}
    for claim in layer.claims:
        src = sources_map.get(claim.source_id)
        item_id = ""
        mode: ProbeMode | None = None
        if src is not None and "golden_item_id" in src.metadata and "probe_mode" in src.metadata:
            item_id = str(src.metadata["golden_item_id"])
            mode = ProbeMode(str(src.metadata["probe_mode"]))
        elif claim.subject_id and len(claim.tags) >= 2:
            item_id = claim.subject_id.removeprefix("ent-topic-")
            try:
                mode = ProbeMode(claim.tags[1])
            except ValueError:
                mode = None
        if item_id and mode is not None:
            claims_map[(item_id, mode)] = claim
    return claims_map, sources_map


def _classify_operation_type(
    mode: ProbeMode,
    detected_stance: str,
    euphemisms: list[str],
) -> tuple[ShiftOperationType, float]:
    if mode in {ProbeMode.DOWNSTREAM_CODE, ProbeMode.DOWNSTREAM_EXTRACT}:
        return ShiftOperationType.DE_LEAK, 3.5
    if euphemisms or detected_stance == StanceLabel.EUPHEMISTIC_REFRAME.value:
        return ShiftOperationType.REFRAME, 3.0
    if detected_stance == StanceLabel.ASSERTS_CONTRARY.value:
        return ShiftOperationType.CORRECT, 4.0
    if detected_stance == StanceLabel.REFUSES.value:
        return ShiftOperationType.CORRECT, 3.0
    return ShiftOperationType.CORRECT, 2.0


def _is_safe_donor_claim(item: GoldenItem, mode: ProbeMode, text: str) -> bool:
    """Donor Poisoning Guard: validate donor text is free of euphemisms, refusals, or broken outputs."""
    if not text.strip():
        return False
    if detect_refusal(text):
        return False
    if find_term_matches(text, item.disallowed_euphemisms):
        return False
    if find_term_matches(text, item.contrary_markers):
        return False
    if mode == ProbeMode.PUSHBACK and detect_pushback_compliance(text):
        return False
    if mode == ProbeMode.DOWNSTREAM_CODE:
        try:
            ast.parse(extract_code_block(text))
        except SyntaxError:
            return False
        lowered = text.lower()
        if item.expected_code_tokens and any(
            tok.lower() not in lowered for tok in item.expected_code_tokens
        ):
            return False
    if mode == ProbeMode.DOWNSTREAM_EXTRACT:
        parsed = extract_json_payload(text)
        if not isinstance(parsed, dict):
            return False
        if item.expected_json_keys and any(k not in parsed for k in item.expected_json_keys):
            return False
    return True


def _extract_clean_text_from_claim(
    claim: KRClaim, mode: ProbeMode, raw_map: dict[tuple[str, ProbeMode], str], item_id: str
) -> str:
    if (item_id, mode) in raw_map:
        return raw_map[(item_id, mode)]
    prefix = f"[{mode.value}] "
    return claim.statement.removeprefix(prefix).strip()


def compute_belief_delta_against_golden(
    base_run_dir: Path | str,
    snapshot: GoldenSnapshot,
    *,
    base_dir: Path | str | None = None,
) -> BeliefDeltaGraph:
    """Compute ΔG = G_golden - G_base from a completed BeliefAudit run directory."""
    resolved_run = (
        ensure_within_base(base_run_dir, base_dir)
        if base_dir is not None
        else Path(base_run_dir).resolve()
    )
    shard_path = resolved_run / "belief_shard.json"
    if not shard_path.is_file():
        raise DeltaEngineError(f"belief_shard.json not found in {resolved_run}.")

    base_layer = import_kr_json(shard_path)
    base_model_id = str(base_layer.layer.stats.get("model_id") or base_layer.layer.name)
    base_claims, base_sources = _index_layer_claims(base_layer)
    raw_map = _load_checkpoint_raw_map(resolved_run)
    conflicts_map = _load_conflicts_map(resolved_run)

    evaluated_modes: list[ProbeMode] = sorted(
        {mode for (_iid, mode) in base_claims},
        key=lambda m: list(ProbeMode).index(m),
    ) or list(ProbeMode)

    scorecard_path = resolved_run / "scorecard.json"
    is_quick_run = False
    if scorecard_path.is_file():
        sc = AuditScorecard.model_validate_json(scorecard_path.read_text(encoding="utf-8"))
        is_quick_run = sc.audit_profile == AuditProfile.QUICK

    target_items = (
        [
            item
            for item in snapshot.items
            if any((item.item_id, m) in base_claims for m in evaluated_modes)
        ]
        if is_quick_run and base_claims
        else snapshot.items
    )

    operations: list[DeltaOperation] = []
    unchanged_item_ids: list[str] = []

    for item in target_items:
        item_evaluated = any((item.item_id, m) in base_claims for m in evaluated_modes)
        item_had_op = False

        for mode in evaluated_modes:
            key = (item.item_id, mode)
            claim = base_claims.get(key)
            conflict = conflicts_map.get(key)

            if claim is None:
                item_had_op = True
                operations.append(
                    DeltaOperation(
                        operation_id=f"op_{item.item_id}_{mode.value}",
                        item_id=item.item_id,
                        category=item.category,
                        probe_mode=mode,
                        operation_type=ShiftOperationType.INJECT,
                        target_claim=item.canonical_claim,
                        expected_stance=item.expected_stance,
                        base_stance="missing",
                        base_excerpt="",
                        disallowed_euphemisms=list(item.disallowed_euphemisms),
                        acceptable_framings=list(item.acceptable_framings),
                        severity=2.0,
                        rationale=f"Base shard is missing evaluated claim for {item.item_id}:{mode.value}.",
                    )
                )
                continue

            if not claim.needs_review and conflict is None:
                continue

            item_had_op = True
            detected_stance = claim.object_literal or "unknown"
            euphemisms: list[str] = []
            if conflict is not None and len(conflict.candidates) > 1:
                raw_euph = conflict.candidates[1].get("euphemisms_detected")
                if isinstance(raw_euph, list):
                    euphemisms = [str(x) for x in raw_euph]

            op_type, default_sev = _classify_operation_type(mode, detected_stance, euphemisms)
            src = base_sources.get(claim.source_id)
            leak_sev = (
                float(src.metadata.get("leakage_severity", default_sev))
                if src is not None and op_type == ShiftOperationType.DE_LEAK
                else default_sev
            )

            base_excerpt = _extract_clean_text_from_claim(claim, mode, raw_map, item.item_id)
            if not base_excerpt and conflict is not None and len(conflict.candidates) > 1:
                base_excerpt = str(conflict.candidates[1].get("excerpt", ""))

            operations.append(
                DeltaOperation(
                    operation_id=f"op_{item.item_id}_{mode.value}",
                    item_id=item.item_id,
                    category=item.category,
                    probe_mode=mode,
                    operation_type=op_type,
                    target_claim=item.canonical_claim,
                    expected_stance=item.expected_stance,
                    base_stance=detected_stance,
                    base_excerpt=base_excerpt,
                    disallowed_euphemisms=list(item.disallowed_euphemisms),
                    acceptable_framings=list(item.acceptable_framings),
                    severity=min(4.0, max(0.0, leak_sev)),
                    rationale=(
                        conflict.resolution_notes
                        if conflict is not None
                        else f"Base stance {detected_stance} requires shift toward {item.expected_stance.value}."
                    ),
                )
            )

        if item_evaluated and not item_had_op:
            unchanged_item_ids.append(item.item_id)

    delta_seed = f"{base_model_id}|golden|{snapshot.golden_version}|{snapshot.content_sha256}"
    delta_id = "delta_" + hashlib.sha256(delta_seed.encode("utf-8")).hexdigest()[:12]

    return BeliefDeltaGraph(
        delta_id=delta_id,
        base_model_id=base_model_id,
        target_kind=ShiftTargetKind.GOLDEN_SET,
        target_id=f"golden:{snapshot.golden_version}",
        golden_version=snapshot.golden_version,
        golden_digest=snapshot.content_sha256,
        operations=operations,
        unchanged_item_ids=sorted(unchanged_item_ids),
    )


def compute_belief_delta_against_donor(
    base_run_dir: Path | str,
    donor_run_dir: Path | str,
    snapshot: GoldenSnapshot,
    *,
    base_dir: Path | str | None = None,
) -> BeliefDeltaGraph:
    """Compute ΔG = G_donor - G_base with strict Donor Poisoning Guard validation."""
    resolved_base = (
        ensure_within_base(base_run_dir, base_dir)
        if base_dir is not None
        else Path(base_run_dir).resolve()
    )
    resolved_donor = (
        ensure_within_base(donor_run_dir, base_dir)
        if base_dir is not None
        else Path(donor_run_dir).resolve()
    )
    base_shard = resolved_base / "belief_shard.json"
    donor_shard = resolved_donor / "belief_shard.json"
    if not base_shard.is_file():
        raise DeltaEngineError(f"Base belief_shard.json not found in {resolved_base}.")
    if not donor_shard.is_file():
        raise DeltaEngineError(f"Donor belief_shard.json not found in {resolved_donor}.")

    base_layer = import_kr_json(base_shard)
    donor_layer = import_kr_json(donor_shard)
    base_model_id = str(base_layer.layer.stats.get("model_id") or base_layer.layer.name)
    donor_model_id = str(donor_layer.layer.stats.get("model_id") or donor_layer.layer.name)

    base_claims, _ = _index_layer_claims(base_layer)
    donor_claims, _ = _index_layer_claims(donor_layer)
    base_raw = _load_checkpoint_raw_map(resolved_base)
    donor_raw = _load_checkpoint_raw_map(resolved_donor)

    donor_modes: list[ProbeMode] = sorted(
        {mode for (_iid, mode) in donor_claims},
        key=lambda m: list(ProbeMode).index(m),
    )

    operations: list[DeltaOperation] = []
    unchanged_item_ids: list[str] = []

    for item in snapshot.items:
        item_evaluated = any((item.item_id, m) in base_claims for m in donor_modes)
        item_had_op = False

        for mode in donor_modes:
            key = (item.item_id, mode)
            d_claim = donor_claims.get(key)
            if d_claim is None or d_claim.needs_review:
                continue

            b_claim = base_claims.get(key)
            if b_claim is not None and not b_claim.needs_review:
                continue

            donor_text = _extract_clean_text_from_claim(d_claim, mode, donor_raw, item.item_id)
            if not _is_safe_donor_claim(item, mode, donor_text):
                continue

            item_had_op = True
            if b_claim is None:
                op_type = ShiftOperationType.INJECT
                base_stance = "missing"
                base_excerpt = ""
                severity = 2.0
            else:
                base_stance = b_claim.object_literal or "unknown"
                base_excerpt = _extract_clean_text_from_claim(b_claim, mode, base_raw, item.item_id)
                op_type, severity = _classify_operation_type(
                    mode,
                    base_stance,
                    find_term_matches(base_excerpt, item.disallowed_euphemisms),
                )

            donor_stance_str = d_claim.object_literal or item.expected_stance.value
            try:
                expected_stance = StanceLabel(donor_stance_str)
            except ValueError:
                expected_stance = item.expected_stance

            operations.append(
                DeltaOperation(
                    operation_id=f"op_{item.item_id}_{mode.value}",
                    item_id=item.item_id,
                    category=item.category,
                    probe_mode=mode,
                    operation_type=op_type,
                    target_claim=donor_text,
                    expected_stance=expected_stance,
                    base_stance=base_stance,
                    base_excerpt=base_excerpt,
                    disallowed_euphemisms=list(item.disallowed_euphemisms),
                    acceptable_framings=list(item.acceptable_framings),
                    severity=severity,
                    rationale=(
                        f"Donor {donor_model_id} clean ({expected_stance.value}) vs "
                        f"base {base_model_id} ({base_stance})."
                    ),
                )
            )

        if item_evaluated and not item_had_op:
            unchanged_item_ids.append(item.item_id)

    delta_seed = f"{base_model_id}|donor|{donor_model_id}|{snapshot.golden_version}"
    delta_id = "delta_" + hashlib.sha256(delta_seed.encode("utf-8")).hexdigest()[:12]

    return BeliefDeltaGraph(
        delta_id=delta_id,
        base_model_id=base_model_id,
        target_kind=ShiftTargetKind.DONOR_SHARD,
        target_id=donor_model_id,
        golden_version=snapshot.golden_version,
        golden_digest=snapshot.content_sha256,
        operations=operations,
        unchanged_item_ids=sorted(unchanged_item_ids),
    )


def delta_graph_to_kr_layer(delta: BeliefDeltaGraph) -> KRLayerData:
    """Export a BeliefDeltaGraph as a KnowledgeReduce schema_version='1.0' DIFFERENCE layer."""
    layer_id = sanitize_layer_id(f"delta-{delta.base_model_id}-{delta.delta_id}")
    source_id = f"src-{delta.delta_id}"
    source_doc = KRSourceDocument(
        source_id=source_id,
        uri=f"beliefaudit://shift/delta/{delta.delta_id}",
        source_type=KRSourceType.AGENT,
        title=f"BeliefDeltaGraph {delta.delta_id}",
        content=f"Delta operations={len(delta.operations)} target={delta.target_id}",
        content_sha256=delta.content_sha256,
        trust_weight=1.0,
        metadata={
            "base_model_id": delta.base_model_id,
            "target_kind": delta.target_kind.value,
            "target_id": delta.target_id,
        },
    )

    entities: dict[str, KREntity] = {}
    claims: list[KRClaim] = []
    relations: list[KRRelation] = []
    target_ent_id = f"ent-target-{sanitize_layer_id(delta.target_id)}"
    entities[target_ent_id] = KREntity(
        entity_id=target_ent_id,
        canonical_name=delta.target_id,
        entity_type="TARGET",
    )

    for op in delta.operations:
        topic_ent_id = f"ent-topic-{op.item_id}"
        if topic_ent_id not in entities:
            entities[topic_ent_id] = KREntity(
                entity_id=topic_ent_id,
                canonical_name=op.item_id,
                entity_type="TOPIC",
                attributes={"category": op.category.value},
            )
        claim_id = f"claim-{op.operation_id}"
        claims.append(
            KRClaim(
                claim_id=claim_id,
                statement=op.target_claim,
                category=op.category.value,
                tags=["shift_delta", op.operation_type.value, op.probe_mode.value],
                contextual_notes=op.rationale,
                reliability=KRReliabilityRating.LIKELY_TRUE,
                source_id=source_id,
                source_title=delta.delta_id,
                confidence=1.0,
                subject_id=topic_ent_id,
                predicate=f"shift_{op.operation_type.value}",
                object_literal=op.expected_stance.value,
                evidence=[
                    KREvidence(
                        source_id=source_id,
                        excerpt=op.base_excerpt or op.target_claim,
                        extractor_name="beliefaudit.shift.delta",
                        extraction_confidence=1.0,
                    )
                ],
                needs_review=False,
            )
        )
        relations.append(
            KRRelation(
                relation_id=f"rel-{op.operation_id}",
                source_id=topic_ent_id,
                target_id=target_ent_id,
                relation_type=f"SHIFT_{op.operation_type.value.upper()}",
                confidence=1.0,
                weight=op.severity,
                attributes={"probe_mode": op.probe_mode.value, "base_stance": op.base_stance},
            )
        )

    layer = KRKnowledgeLayer(
        layer_id=layer_id,
        name=f"BeliefShift Delta: {delta.base_model_id} -> {delta.target_id}",
        version="1.0.0",
        tier=KRLayerTier.STACKED,
        merge_mode=KRMergeMode.DIFFERENCE,
        pipeline_config_hash=delta.content_sha256[:16],
        stats={
            "delta_id": delta.delta_id,
            "base_model_id": delta.base_model_id,
            "n_operations": len(delta.operations),
        },
        description="Targeted delta graph for closed-loop BeliefShift compilation.",
    )
    return KRLayerData(
        layer=layer,
        sources=[source_doc],
        entities=list(entities.values()),
        claims=claims,
        relations=relations,
    )


def save_delta_graph(
    delta: BeliefDeltaGraph,
    output_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """Persist a BeliefDeltaGraph JSON artifact after verifying its SHA-256 digest."""
    if not delta.verify_digest():
        raise DeltaEngineError("BeliefDeltaGraph digest does not match its payload.")
    target = (
        ensure_within_base(output_path, base_dir)
        if base_dir is not None
        else Path(output_path).resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(delta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_delta_graph(
    input_path: Path | str,
    *,
    base_dir: Path | str | None = None,
    verify: bool = True,
) -> BeliefDeltaGraph:
    """Load and optionally digest-verify a BeliefDeltaGraph JSON file."""
    source = (
        ensure_within_base(input_path, base_dir)
        if base_dir is not None
        else Path(input_path).resolve()
    )
    if not source.is_file():
        raise DeltaEngineError(f"Delta graph file not found: {source}")
    delta = BeliefDeltaGraph.model_validate_json(source.read_text(encoding="utf-8"))
    if verify and not delta.verify_digest():
        raise DeltaEngineError(f"BeliefDeltaGraph digest mismatch for {source}.")
    return delta
