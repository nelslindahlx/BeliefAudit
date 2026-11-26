"""Offline end-to-end BeliefAudit demonstration across clean, skewed, and leaker personas."""

from __future__ import annotations

import tempfile
from pathlib import Path

from beliefaudit import (
    AdvisoryVerdict,
    BundledOfflineSeeder,
    MockModelRunner,
    beliefs_to_kr_layer,
    build_scorecard,
    compute_bias_matrix,
    compute_quality_score,
    diff_candidate_against_golden,
    diff_model_shards,
    distill_probe_responses,
    import_kr_json,
    import_kr_sharded,
    run_probe_suite,
    save_golden_snapshot,
)
from beliefaudit.probes.runners import MockPersona
from beliefaudit.scoring.advisor import write_audit_artifacts


def main() -> None:
    snapshot = BundledOfflineSeeder().seed(version="v0.1.0")
    personas: tuple[MockPersona, ...] = ("clean", "skewed", "leaker")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        save_golden_snapshot(snapshot, root / "golden_v0.1.0.json")

        verdicts: dict[str, AdvisoryVerdict] = {}
        shard_paths: dict[str, Path] = {}

        for persona in personas:
            runner = MockModelRunner(persona=persona, snapshot=snapshot)
            probe_resps, ctrl_resps = run_probe_suite(runner, snapshot, seed=42)
            beliefs, candidates = distill_probe_responses(snapshot, probe_resps)
            belief_layer = beliefs_to_kr_layer(runner.model_id, snapshot.golden_version, beliefs)
            diff_layer, conflicts = diff_candidate_against_golden(
                snapshot, beliefs, runner.model_id
            )
            quality = compute_quality_score(ctrl_resps)
            bias = compute_bias_matrix(snapshot, beliefs)
            scorecard = build_scorecard(
                model_id=runner.model_id,
                snapshot=snapshot,
                quality=quality,
                bias=bias,
                beliefs=beliefs,
                conflicts=conflicts,
                candidates=candidates,
            )
            artifacts = write_audit_artifacts(
                root / f"run_{persona}",
                belief_layer=belief_layer,
                diff_layer=diff_layer,
                conflicts=conflicts,
                scorecard=scorecard,
                candidates=candidates,
            )
            verdicts[persona] = scorecard.verdict
            shard_paths[persona] = artifacts["belief_shard_json"]

            # Verify lossless JSON and sharded JSONL round-trips
            reloaded_json = import_kr_json(artifacts["belief_shard_json"])
            reloaded_sharded = import_kr_sharded(artifacts["belief_shard_sharded"])
            assert len(reloaded_json.claims) == len(reloaded_sharded.claims) == len(beliefs)

            print(
                f"{scorecard.model_id:12s} | Quality={scorecard.quality.quality_score:6.2f} | "
                f"BiasRisk={scorecard.bias.overall_bias_risk_score:6.2f} | "
                f"Verdict={scorecard.verdict.value}"
            )

        assert verdicts["clean"] == AdvisoryVerdict.RECOMMENDED
        assert verdicts["skewed"] == AdvisoryVerdict.AVOID_HIGH_RISK
        assert verdicts["leaker"] == AdvisoryVerdict.AVOID_HIGH_RISK

        diff_summary = diff_model_shards(
            import_kr_json(shard_paths["clean"]),
            import_kr_json(shard_paths["leaker"]),
            snapshot,
        )
        assert diff_summary.regression_count > 0
        print(
            f"Diff clean -> leaker: regressions={diff_summary.regression_count}, "
            f"unchanged={diff_summary.unchanged_count}"
        )


if __name__ == "__main__":
    main()
