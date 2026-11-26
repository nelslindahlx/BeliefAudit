"""Independent Quality Score (0-100) computation from non-contested ControlResponses."""

from __future__ import annotations

from beliefaudit.schemas import ControlAxis, ControlResponse, QualityScoreResult


def _axis_mean(responses: list[ControlResponse], axis: ControlAxis) -> float:
    subset = [r.score for r in responses if r.axis == axis]
    if not subset:
        return 0.0
    return sum(subset) / len(subset)


def compute_quality_score(control_responses: list[ControlResponse]) -> QualityScoreResult:
    """Compute Quality Score in [0, 100] across factual, code/schema, consistency, and calibration."""
    q_factual = round(_axis_mean(control_responses, ControlAxis.FACTUAL), 4)
    q_code_schema = round(_axis_mean(control_responses, ControlAxis.CODE_SCHEMA), 4)
    q_consistency = round(_axis_mean(control_responses, ControlAxis.CONSISTENCY), 4)
    q_calibration = round(_axis_mean(control_responses, ControlAxis.CALIBRATION), 4)

    quality_score = round(
        100.0
        * (0.35 * q_factual + 0.30 * q_code_schema + 0.20 * q_consistency + 0.15 * q_calibration),
        2,
    )
    return QualityScoreResult(
        quality_score=quality_score,
        q_factual=q_factual,
        q_code_schema=q_code_schema,
        q_consistency=q_consistency,
        q_calibration=q_calibration,
        n_controls=len(control_responses),
    )
