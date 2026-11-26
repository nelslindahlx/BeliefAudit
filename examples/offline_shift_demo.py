"""Zero-network closed-loop BeliefShift demonstration (skewed -> ΔG -> compile -> dry-run -> verify)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from beliefaudit import (
    AuditProfile,
    BundledOfflineSeeder,
    GoldenSnapshot,
    MockModelRunner,
    TrainingBackend,
    beliefs_to_kr_layer,
    build_scorecard,
    compile_runtime_guardrail,
    compile_shift_corpus,
    compute_belief_delta_against_golden,
    compute_bias_matrix,
    compute_quality_score,
    diff_candidate_against_golden,
    distill_probe_responses,
    evaluate_roundtrip,
    run_heldout_evaluation,
    run_probe_suite,
    run_shift_training,
    select_profile_subset,
    verify_response_against_guardrail,
)
from beliefaudit.probes.runners import MockPersona
from beliefaudit.scoring.advisor import write_audit_artifacts


def _audit_persona(
    persona: MockPersona,
    full_snapshot: GoldenSnapshot,
    out_dir: Path,
) -> None:
    subset, modes = select_profile_subset(full_snapshot, AuditProfile.QUICK)
    runner = MockModelRunner(persona=persona, snapshot=full_snapshot)
    probes, ctrls = run_probe_suite(runner, subset, modes=modes)
    beliefs, candidates = distill_probe_responses(subset, probes)
    belief_layer = beliefs_to_kr_layer(runner.model_id, subset.golden_version, beliefs)
    diff_layer, conflicts = diff_candidate_against_golden(subset, beliefs, runner.model_id)
    quality = compute_quality_score(ctrls)
    bias = compute_bias_matrix(subset, beliefs)
    scorecard = build_scorecard(
        model_id=runner.model_id,
        snapshot=subset,
        quality=quality,
        bias=bias,
        beliefs=beliefs,
        conflicts=conflicts,
        candidates=candidates,
        audit_profile=AuditProfile.QUICK,
    )
    write_audit_artifacts(
        out_dir,
        belief_layer=belief_layer,
        diff_layer=diff_layer,
        conflicts=conflicts,
        scorecard=scorecard,
        candidates=candidates,
    )


def main() -> None:
    full_snapshot = BundledOfflineSeeder().seed("v0.1.0")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pre_dir = root / "pre_skewed"
        post_dir = root / "post_shifted"
        _audit_persona("skewed", full_snapshot, pre_dir)
        _audit_persona("clean", full_snapshot, post_dir)

        delta = compute_belief_delta_against_golden(pre_dir, full_snapshot)
        corpus_dir = root / "compiled_corpus"
        corpus_manifest = compile_shift_corpus(delta, full_snapshot, corpus_dir)
        assert corpus_manifest.leakage_check_passed is True

        shard_dir = root / "lora_shard"
        lora_manifest = run_shift_training(
            corpus_dir=corpus_dir,
            delta=delta,
            base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            out_shard_dir=shard_dir,
            backend=TrainingBackend.DRY_RUN,
        )
        assert lora_manifest.dry_run is True

        shifted_runner = MockModelRunner("clean", snapshot=full_snapshot)
        target_ids = {op.item_id for op in delta.operations}
        heldout_rate, _ = run_heldout_evaluation(shifted_runner, full_snapshot, target_ids)

        report = evaluate_roundtrip(
            delta,
            pre_dir,
            post_dir,
            full_snapshot,
            heldout_shift_rate=heldout_rate,
            manifest_path=shard_dir / "lora_shard_manifest.json",
        )
        assert report.overall_gate_passed is True
        assert report.target_shift_rate == 1.0
        assert report.heldout_shift_rate == 1.0
        assert report.collateral_drift_score == 0.0

        guardrail = compile_runtime_guardrail(
            delta, full_snapshot, roundtrip_report=report, include_all_ops=True
        )
        first_item = next(i for i in full_snapshot.items if i.item_id == guardrail.rules[0].item_id)
        skewed_check = verify_response_against_guardrail(
            guardrail,
            f"Officially described as {first_item.disallowed_euphemisms[0]}.",
            item_id=first_item.item_id,
        )
        clean_check = verify_response_against_guardrail(
            guardrail,
            first_item.canonical_claim,
            item_id=first_item.item_id,
        )
        assert (
            skewed_check.passed is False
            and skewed_check.fallback_claim == first_item.canonical_claim
        )
        assert clean_check.passed is True

        print(
            f"BeliefShift Demo PASSED | ops={len(delta.operations)} "
            f"replay_ratio={corpus_manifest.replay_ratio:.2f} "
            f"TSR={report.target_shift_rate:.0%} TSR_heldout={report.heldout_shift_rate:.0%} "
            f"CDS={report.collateral_drift_score:.0%} ΔQ={report.delta_quality:+.2f} "
            f"ΔB={report.delta_bias_risk:+.2f} guardrail_rules={len(guardrail.rules)}"
        )


if __name__ == "__main__":
    main()
