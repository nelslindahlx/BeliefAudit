"""MCP-ready pure-Python query helpers for inspecting completed BeliefAudit runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beliefaudit.schemas import AuditScorecard, ClaimCategory
from beliefaudit.security import BeliefAuditError, ensure_within_base
from beliefaudit.shards.kr_schema import KRConflictRecord


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
