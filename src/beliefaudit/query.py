"""MCP-ready pure-Python query helpers for inspecting completed BeliefAudit runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beliefaudit.schemas import AuditScorecard, ClaimCategory
from beliefaudit.security import BeliefAuditError, ensure_within_base
from beliefaudit.shards.kr_schema import KRConflictRecord

if TYPE_CHECKING:
    from beliefaudit.shift.schemas import LoRAShardManifest, RoundTripReport


class RunQueryError(BeliefAuditError):
    """Raised when reading an audit run directory fails."""


def list_audited_runs(
    runs_root: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Discover completed audit runs under *runs_root* containing scorecard.json."""
    root = (
        ensure_within_base(runs_root, base_dir)
        if base_dir is not None
        else Path(runs_root).resolve()
    )
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        scorecard_file = child / "scorecard.json"
        if child.is_dir() and scorecard_file.is_file():
            sc = AuditScorecard.model_validate_json(scorecard_file.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_dir": child.name,
                    "model_id": sc.model_id,
                    "golden_version": sc.golden_version,
                    "quality_score": sc.quality.quality_score,
                    "bias_risk_score": sc.bias.overall_bias_risk_score,
                    "verdict": sc.verdict.value,
                    "n_conflicts": sc.n_conflicts,
                }
            )
    return summaries


def load_scorecard(
    run_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> AuditScorecard:
    """Load and validate scorecard.json from *run_dir*."""
    resolved = (
        ensure_within_base(run_dir, base_dir) if base_dir is not None else Path(run_dir).resolve()
    )
    scorecard_path = resolved / "scorecard.json"
    if not scorecard_path.is_file():
        raise RunQueryError(f"scorecard.json not found in {resolved}.")
    return AuditScorecard.model_validate_json(scorecard_path.read_text(encoding="utf-8"))


def query_belief_conflicts(
    run_dir: Path | str,
    *,
    category: ClaimCategory | None = None,
    limit: int = 20,
    base_dir: Path | str | None = None,
) -> list[KRConflictRecord]:
    """Query conflict records from *run_dir*, optionally filtered by ClaimCategory."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1 (got {limit}).")
    resolved = (
        ensure_within_base(run_dir, base_dir) if base_dir is not None else Path(run_dir).resolve()
    )
    conflicts_path = resolved / "conflicts.json"
    if not conflicts_path.is_file():
        raise RunQueryError(f"conflicts.json not found in {resolved}.")

    raw = json.loads(conflicts_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RunQueryError(f"Malformed conflicts.json in {resolved}.")

    records = [KRConflictRecord.model_validate(item) for item in raw]
    if category is not None:
        filtered: list[KRConflictRecord] = []
        for rec in records:
            cand_cat = rec.candidates[0].get("category") if rec.candidates else None
            if cand_cat == category.value:
                filtered.append(rec)
        records = filtered
    return records[:limit]


def load_roundtrip_report(
    verify_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> RoundTripReport:
    """Load and validate roundtrip_report.json from *verify_dir*."""
    from beliefaudit.shift.schemas import RoundTripReport

    resolved = (
        ensure_within_base(verify_dir, base_dir)
        if base_dir is not None
        else Path(verify_dir).resolve()
    )
    report_path = resolved / "roundtrip_report.json"
    if not report_path.is_file():
        raise RunQueryError(f"roundtrip_report.json not found in {resolved}.")
    return RoundTripReport.model_validate_json(report_path.read_text(encoding="utf-8"))


def list_lora_shards(
    shards_root: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Discover BeliefShift LoRA shard directories containing lora_shard_manifest.json."""
    from beliefaudit.shift.trainer import load_lora_manifest

    root = (
        ensure_within_base(shards_root, base_dir)
        if base_dir is not None
        else Path(shards_root).resolve()
    )
    if not root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        manifest_file = child / "lora_shard_manifest.json"
        if child.is_dir() and manifest_file.is_file():
            man: LoRAShardManifest = load_lora_manifest(manifest_file, base_dir=root)
            summaries.append(
                {
                    "shard_dir": child.name,
                    "shard_id": man.shard_id,
                    "base_model_id": man.base_model_id,
                    "backend": man.backend.value,
                    "delta_id": man.delta_id,
                    "dry_run": man.dry_run,
                    "verification_passed": man.verification_passed,
                    "target_shift_rate": man.target_shift_rate,
                    "heldout_shift_rate": man.heldout_shift_rate,
                    "collateral_drift_score": man.collateral_drift_score,
                }
            )
    return summaries


def inspect_shift_artifacts(
    target_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load and summarize any BeliefShift artifacts present inside *target_dir*."""
    from beliefaudit.shift.delta_engine import load_delta_graph
    from beliefaudit.shift.guardrail import load_guardrail_spec
    from beliefaudit.shift.kg_compiler import load_corpus_manifest
    from beliefaudit.shift.trainer import load_lora_manifest

    resolved = (
        ensure_within_base(target_dir, base_dir)
        if base_dir is not None
        else Path(target_dir).resolve()
    )
    if not resolved.is_dir():
        raise RunQueryError(f"Shift artifact directory not found: {resolved}")

    result: dict[str, Any] = {
        "directory": resolved.name,
        "found": [],
    }
    found: list[str] = []

    delta_path = resolved / "delta_graph.json"
    if delta_path.is_file():
        delta = load_delta_graph(delta_path, base_dir=resolved)
        found.append("delta_graph")
        result["delta_graph"] = delta.model_dump(mode="json")

    corpus_manifest_path = resolved / "corpus_manifest.json"
    if corpus_manifest_path.is_file():
        cm = load_corpus_manifest(resolved)
        found.append("corpus_manifest")
        result["corpus_manifest"] = cm.model_dump(mode="json")

    lora_manifest_path = resolved / "lora_shard_manifest.json"
    if lora_manifest_path.is_file():
        lm = load_lora_manifest(lora_manifest_path, base_dir=resolved)
        found.append("lora_shard_manifest")
        result["lora_shard_manifest"] = lm.model_dump(mode="json")

    rt_path = resolved / "roundtrip_report.json"
    if rt_path.is_file():
        rt = load_roundtrip_report(resolved)
        found.append("roundtrip_report")
        result["roundtrip_report"] = rt.model_dump(mode="json")

    gr_path = resolved / "guardrail_spec.json"
    if gr_path.is_file():
        gr = load_guardrail_spec(gr_path, base_dir=resolved)
        found.append("guardrail_spec")
        result["guardrail_spec"] = gr.model_dump(mode="json")

    if not found:
        raise RunQueryError(f"No BeliefShift artifacts found in {resolved}.")
    result["found"] = found
    return result
