"""Runtime KnowledgeReduce guardrail compiler and output verification wrapper."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Sequence
from pathlib import Path

from beliefaudit.golden.rubric import (
    detect_hedging,
    detect_pushback_compliance,
    detect_refusal,
    find_term_matches,
)
from beliefaudit.probes.generator import extract_code_block, extract_json_payload
from beliefaudit.schemas import GoldenSnapshot, ProbeMode
from beliefaudit.security import BeliefAuditError, ensure_within_base
from beliefaudit.shift._io import write_text_atomic
from beliefaudit.shift.schemas import (
    BeliefDeltaGraph,
    DeltaOperation,
    GuardrailCheckResult,
    GuardrailRule,
    RoundTripReport,
    RuntimeGuardrailSpec,
)


class GuardrailError(BeliefAuditError):
    """Raised when compiling, loading, or validating a RuntimeGuardrailSpec fails."""


def render_guardrail_system_prompt(rules: Sequence[GuardrailRule]) -> str:
    """Render a euphemism-free runtime system prompt overlay grounded in Golden claims."""
    if not rules:
        return ""
    lines = [
        "Runtime Factual Grounding Guardrails (enforce strictly across prose, code, and JSON outputs):"
    ]
    for rule in rules:
        safe_framings = [
            f
            for f in rule.acceptable_framings[:3]
            if not find_term_matches(f, rule.disallowed_euphemisms)
        ]
        suffix = f" [Preferred terminology: {', '.join(safe_framings)}]" if safe_framings else ""
        lines.append(
            f"- {rule.topic} ({rule.expected_stance.value}): {rule.canonical_claim}{suffix}"
        )
    return "\n".join(lines)


def compile_runtime_guardrail(
    delta: BeliefDeltaGraph,
    snapshot: GoldenSnapshot,
    *,
    roundtrip_report: RoundTripReport | None = None,
    include_all_ops: bool = False,
) -> RuntimeGuardrailSpec:
    """Compile resistant operations (or all delta ops) into a RuntimeGuardrailSpec."""
    if not delta.verify_digest():
        raise GuardrailError(f"BeliefDeltaGraph {delta.delta_id} failed digest verification.")
    if snapshot.golden_version != delta.golden_version:
        raise GuardrailError(
            f"GoldenSnapshot version ({snapshot.golden_version}) does not match delta ({delta.golden_version})."
        )

    if roundtrip_report is not None and not include_all_ops:
        if roundtrip_report.delta_id != delta.delta_id:
            raise GuardrailError(
                f"RoundTripReport delta_id ({roundtrip_report.delta_id}) does not match {delta.delta_id}."
            )
        resistant_set = set(roundtrip_report.resistant_operation_ids)
        selected_ops = [op for op in delta.operations if op.operation_id in resistant_set]
    else:
        selected_ops = list(delta.operations)

    ops_by_item: dict[str, list[DeltaOperation]] = {}
    for op in selected_ops:
        ops_by_item.setdefault(op.item_id, []).append(op)

    item_by_id = {item.item_id: item for item in snapshot.items}
    rules: list[GuardrailRule] = []
    for item_id, item_ops in ops_by_item.items():
        item = item_by_id.get(item_id)
        if item is None:
            raise GuardrailError(
                f"GoldenItem {item_id!r} not found in snapshot {snapshot.golden_version}."
            )
        seen_modes: list[ProbeMode] = []
        for op in item_ops:
            if op.probe_mode not in seen_modes:
                seen_modes.append(op.probe_mode)
        rules.append(
            GuardrailRule(
                rule_id=f"rule_{item_id}",
                item_id=item_id,
                topic=item.topic,
                category=item.category,
                probe_modes=seen_modes,
                canonical_claim=item.canonical_claim,
                expected_stance=item.expected_stance,
                disallowed_euphemisms=list(item.disallowed_euphemisms),
                contrary_markers=list(item.contrary_markers),
                acceptable_framings=list(item.acceptable_framings),
                expected_code_tokens=list(item.expected_code_tokens),
                expected_json_keys=list(item.expected_json_keys),
                source_operation_ids=[op.operation_id for op in item_ops],
                rationale="; ".join(op.rationale for op in item_ops if op.rationale),
            )
        )

    prompt_block = render_guardrail_system_prompt(rules)
    seed_src = (
        f"{delta.delta_id}|{roundtrip_report.report_id if roundtrip_report else 'all'}|"
        + ",".join(r.rule_id for r in rules)
    )
    guardrail_id = "gr_" + hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:12]

    return RuntimeGuardrailSpec(
        guardrail_id=guardrail_id,
        delta_id=delta.delta_id,
        report_id=roundtrip_report.report_id if roundtrip_report else None,
        base_model_id=delta.base_model_id,
        shifted_model_id=roundtrip_report.shifted_model_id if roundtrip_report else None,
        golden_version=delta.golden_version,
        rules=rules,
        system_prompt_block=prompt_block,
    )


def verify_response_against_guardrail(
    spec: RuntimeGuardrailSpec,
    response_text: str,
    *,
    item_id: str | None = None,
    probe_mode: ProbeMode = ProbeMode.NEUTRAL,
) -> GuardrailCheckResult:
    """Verify *response_text* against active guardrail rules and return violations + fallback."""
    applicable = [r for r in spec.rules if item_id is None or r.item_id == item_id]
    if not applicable:
        return GuardrailCheckResult(passed=True)

    violations: list[str] = []
    matched_rules: list[str] = []
    fallback_claim: str | None = None

    def _record(rule: GuardrailRule, reason: str) -> None:
        nonlocal fallback_claim
        if rule.rule_id not in matched_rules:
            matched_rules.append(rule.rule_id)
        violations.append(reason)
        if fallback_claim is None:
            fallback_claim = rule.canonical_claim

    stripped = response_text.strip()
    if not stripped:
        _record(applicable[0], "empty_response")
        return GuardrailCheckResult(
            passed=False,
            matched_rule_ids=matched_rules,
            violations=violations,
            fallback_claim=fallback_claim,
        )

    if detect_refusal(response_text):
        _record(applicable[0], "unwarranted_refusal")
    if detect_hedging(response_text):
        _record(applicable[0], "evasive_hedging")
    if probe_mode == ProbeMode.PUSHBACK and detect_pushback_compliance(response_text):
        _record(applicable[0], "pushback_compliance")

    code_lower = ""
    if probe_mode == ProbeMode.DOWNSTREAM_CODE:
        code = extract_code_block(response_text)
        code_lower = code.lower()
        try:
            ast.parse(code)
        except SyntaxError as exc:
            _record(applicable[0], f"invalid_python_syntax: {exc.msg}")

    parsed_json: object | None = None
    if probe_mode == ProbeMode.DOWNSTREAM_EXTRACT:
        parsed_json = extract_json_payload(response_text)
        if not isinstance(parsed_json, dict):
            _record(applicable[0], "invalid_json_object")

    for rule in applicable:
        euph_hits = find_term_matches(response_text, rule.disallowed_euphemisms)
        if euph_hits:
            _record(rule, f"{rule.item_id}: disallowed euphemisms {euph_hits}")
        contrary_hits = find_term_matches(response_text, rule.contrary_markers)
        if contrary_hits:
            _record(rule, f"{rule.item_id}: contrary markers {contrary_hits}")
        if (
            item_id is not None
            and probe_mode == ProbeMode.DOWNSTREAM_CODE
            and rule.expected_code_tokens
        ):
            missing_toks = [t for t in rule.expected_code_tokens if t.lower() not in code_lower]
            if missing_toks:
                _record(rule, f"{rule.item_id}: missing code tokens {missing_toks}")
        if (
            item_id is not None
            and probe_mode == ProbeMode.DOWNSTREAM_EXTRACT
            and isinstance(parsed_json, dict)
            and rule.expected_json_keys
        ):
            missing_keys = [k for k in rule.expected_json_keys if k not in parsed_json]
            if missing_keys:
                _record(rule, f"{rule.item_id}: missing json keys {missing_keys}")

    return GuardrailCheckResult(
        passed=not violations,
        matched_rule_ids=matched_rules,
        violations=violations,
        fallback_claim=fallback_claim,
    )


def save_guardrail_spec(
    spec: RuntimeGuardrailSpec,
    output_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """Persist a RuntimeGuardrailSpec JSON artifact atomically after verifying its SHA-256."""
    if not spec.verify_digest():
        raise GuardrailError("RuntimeGuardrailSpec digest does not match its payload.")
    target = (
        ensure_within_base(output_path, base_dir)
        if base_dir is not None
        else Path(output_path).resolve()
    )
    write_text_atomic(spec.model_dump_json(indent=2) + "\n", target)
    return target


def load_guardrail_spec(
    input_path: Path | str,
    *,
    base_dir: Path | str | None = None,
    verify: bool = True,
) -> RuntimeGuardrailSpec:
    """Load and optionally digest-verify a RuntimeGuardrailSpec JSON file."""
    source = (
        ensure_within_base(input_path, base_dir)
        if base_dir is not None
        else Path(input_path).resolve()
    )
    if not source.is_file():
        raise GuardrailError(f"RuntimeGuardrailSpec file not found: {source}")
    spec = RuntimeGuardrailSpec.model_validate_json(source.read_text(encoding="utf-8"))
    if verify and not spec.verify_digest():
        raise GuardrailError(f"RuntimeGuardrailSpec digest mismatch for {source}.")
    return spec
