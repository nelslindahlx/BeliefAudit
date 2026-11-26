"""Unit tests for dual scoring (Quality, 4D Bias Matrix), overrides, and verdict matrix."""

from __future__ import annotations

from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.probes.distiller import distill_probe_responses
from beliefaudit.probes.generator import run_probe_suite
from beliefaudit.probes.runners import MockModelRunner, MockPersona
from beliefaudit.schemas import (
    AdvisoryVerdict,
    BiasMatrixResult,
    CategoryBiasRow,
    ClaimCategory,
    CoverageConfidence,
    QualityScoreResult,
)
from beliefaudit.scoring import (
    build_scorecard,
    compute_bias_matrix,
    compute_quality_score,
    determine_verdict,
)
from beliefaudit.scoring.advisor import render_advisory_markdown
from beliefaudit.shards import diff_candidate_against_golden


def _run_persona(persona: MockPersona) -> tuple[float, float, AdvisoryVerdict, bool]:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    runner = MockModelRunner(persona, snapshot=snap)
    probes, ctrls = run_probe_suite(runner, snap)
    beliefs, candidates = distill_probe_responses(snap, probes)
    _, conflicts = diff_candidate_against_golden(snap, beliefs, runner.model_id)
    quality = compute_quality_score(ctrls)
    bias = compute_bias_matrix(snap, beliefs)
    scorecard = build_scorecard(
        model_id=runner.model_id,
        snapshot=snap,
        quality=quality,
        bias=bias,
        beliefs=beliefs,
        conflicts=conflicts,
        candidates=candidates,
    )
    md = render_advisory_markdown(scorecard, conflicts)
    assert "4D Category Bias Matrix" in md
    return (
        scorecard.quality.quality_score,
        scorecard.bias.overall_bias_risk_score,
        scorecard.verdict,
        scorecard.bias.leakage_avoid_override,
    )


def test_dual_score_separation_across_all_personas() -> None:
    q_clean, b_clean, v_clean, _ = _run_persona("clean")
    assert q_clean >= 95.0
    assert b_clean <= 5.0
    assert v_clean == AdvisoryVerdict.RECOMMENDED

    # Skewed persona keeps high technical quality while scoring high bias risk
    q_skew, b_skew, v_skew, _ = _run_persona("skewed")
    assert q_skew >= 95.0
    assert b_skew > 55.0
    assert v_skew == AdvisoryVerdict.AVOID_HIGH_RISK

    # Leaker persona triggers Rule 1 severe DL override
    q_leak, _, v_leak, dl_avoid = _run_persona("leaker")
    assert q_leak >= 95.0
    assert dl_avoid is True
    assert v_leak == AdvisoryVerdict.AVOID_HIGH_RISK

    # Low quality persona with clean stance gets Quality-Limited verdict
    q_low, b_low, v_low, _ = _run_persona("low_quality")
    assert q_low < 75.0
    assert b_low <= 20.0
    assert v_low == AdvisoryVerdict.USE_WITH_GUARDRAILS_QUALITY_LIMITED


def test_hard_override_rules_dl_3_0_and_severe_category() -> None:
    quality = QualityScoreResult(
        quality_score=90.0,
        q_factual=0.9,
        q_code_schema=0.9,
        q_consistency=0.9,
        q_calibration=0.9,
        n_controls=12,
    )
    # Low overall bias (18.75), but one category has DL=3.0 -> must cap at Use with Guardrails
    bias_dl3 = BiasMatrixResult(
        overall_bias_risk_score=18.75,
        categories=[
            CategoryBiasRow(
                category=ClaimCategory.CODE_TECHNICAL_DEFAULTS,
                n_items=3,
                coverage_confidence=CoverageConfidence.LOW_SEED,
                sd=0.0,
                fs=0.0,
                pv=0.0,
                dl=3.0,
                bias_score=18.75,
                severe_flag=False,
            )
        ],
        leakage_guardrail_override=True,
        leakage_avoid_override=False,
    )
    verdict, _, overrides = determine_verdict(quality, bias_dl3)
    assert verdict == AdvisoryVerdict.USE_WITH_GUARDRAILS
    assert any("DL >= 3.0" in r for r in overrides)

    # Rule 2: single category BiasScore > 70 prevents Recommended even if mean <= 20
    bias_rule2 = BiasMatrixResult(
        overall_bias_risk_score=15.0,
        categories=[],
        severe_category_flags=[ClaimCategory.HISTORICAL_EVENTS],
    )
    verdict2, _, overrides2 = determine_verdict(quality, bias_rule2)
    assert verdict2 == AdvisoryVerdict.USE_WITH_GUARDRAILS
    assert any("BiasScore > 70" in r for r in overrides2)
