"""Targeted unit tests for CLI branches, query edge cases, distiller, and bias matrix."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from beliefaudit.cli import app
from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.probes.distiller import distill_belief, distill_probe_responses
from beliefaudit.probes.generator import run_probe_suite
from beliefaudit.probes.runners import MockModelRunner
from beliefaudit.query import (
    RunQueryError,
    list_audited_runs,
    load_scorecard,
    query_belief_conflicts,
)
from beliefaudit.schemas import ClaimCategory, GoldenSnapshot, ProbeMode, ProbeResponse, StanceLabel
from beliefaudit.scoring.bias_matrix import compute_bias_matrix
from beliefaudit.scoring.quality import compute_quality_score

cli_runner = CliRunner()


def test_cli_http_runner_with_judge_and_error_branches(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.json"
    assert cli_runner.invoke(app, ["golden", "seed", "-o", str(golden_path)]).exit_code == 0

    # Invalid mock persona
    bad_mock = cli_runner.invoke(
        app,
        [
            "run",
            "--mock",
            "unknown_persona",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(tmp_path / "r1"),
        ],
    )
    assert bad_mock.exit_code == 2

    # Missing both --model and --mock
    no_model = cli_runner.invoke(
        app,
        ["run", "--golden", str(golden_path), "--out-dir", str(tmp_path / "r2")],
    )
    assert no_model.exit_code == 2

    # Duplicate seed without --overwrite triggers error exit code 1
    dup_seed = cli_runner.invoke(app, ["golden", "seed", "-o", str(golden_path)])
    assert dup_seed.exit_code == 1

    # Promote with invalid same version exits 1
    bad_prom = cli_runner.invoke(
        app,
        [
            "golden",
            "promote",
            "--current",
            str(golden_path),
            "--candidates",
            str(tmp_path / "missing.json"),
            "--next-version",
            "v0.2.0",
            "--output",
            str(tmp_path / "v2.json"),
        ],
    )
    assert bad_prom.exit_code == 1

    # Run via LocalOpenAICompatRunner + LocalLLMJudge using respx
    with respx.mock(assert_all_called=False) as mock_router:
        mock_router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "100 Not Found log n Sun 443 prime ? def f(): pass"
                            }
                        }
                    ]
                },
            )
        )
        mock_router.post("http://127.0.0.1:11435/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "detected_stance": "asserts_golden",
                                        "subject": "Topic",
                                        "predicate": "stance",
                                        "object_claim": "Clean",
                                        "leakage_detected": False,
                                        "leakage_severity": 0.0,
                                        "confidence": 0.9,
                                        "rationale": "ok",
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        )
        run_dir = tmp_path / "http_run"
        res_http = cli_runner.invoke(
            app,
            [
                "run",
                "--model",
                "local-qwen",
                "--endpoint",
                "http://localhost:11434/v1",
                "--judge-model",
                "local-llama-judge",
                "--judge-endpoint",
                "http://127.0.0.1:11435/v1",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(run_dir),
            ],
        )
        assert res_http.exit_code == 0

        # Plain-text inspect output
        res_insp = cli_runner.invoke(app, ["inspect", "--run-dir", str(run_dir)])
        assert res_insp.exit_code == 0
        assert "local-qwen" in res_insp.stdout

    # Missing inspect dir exits 1
    assert cli_runner.invoke(app, ["inspect", "--run-dir", str(tmp_path / "nope")]).exit_code == 1
    # Missing diff file exits 1
    assert (
        cli_runner.invoke(
            app,
            [
                "diff",
                "--baseline",
                str(tmp_path / "a.json"),
                "--candidate",
                str(tmp_path / "b.json"),
                "--golden",
                str(golden_path),
            ],
        ).exit_code
        == 1
    )


def test_query_edge_cases(tmp_path: Path) -> None:
    assert list_audited_runs(tmp_path / "does_not_exist") == []
    with pytest.raises(RunQueryError):
        load_scorecard(tmp_path)
    with pytest.raises(ValueError):
        query_belief_conflicts(tmp_path, limit=0)
    with pytest.raises(RunQueryError):
        query_belief_conflicts(tmp_path)
    (tmp_path / "conflicts.json").write_text('{"bad": true}', encoding="utf-8")
    with pytest.raises(RunQueryError):
        query_belief_conflicts(tmp_path)


def test_distiller_and_bias_matrix_edge_branches() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    item = snap.items[0]

    # Downstream code missing expected tokens
    code_resp = ProbeResponse(
        item_id=item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_CODE,
        messages=[{"role": "user", "content": item.downstream_code_prompt}],
        raw_text="def unrelated():\n    return 1\n",
        model_id="m",
    )
    b_code = distill_belief(item, code_resp)
    assert b_code.leakage_detected is True
    assert b_code.leakage_severity == 2.0

    # Downstream extract missing keys or using euphemism
    ext_missing = ProbeResponse(
        item_id=item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_EXTRACT,
        messages=[{"role": "user", "content": item.downstream_extract_prompt}],
        raw_text='{"wrong_key": "val"}',
        model_id="m",
    )
    b_ext = distill_belief(item, ext_missing)
    assert b_ext.leakage_detected is True
    assert b_ext.leakage_severity == 3.0

    ext_euph = ProbeResponse(
        item_id=item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_EXTRACT,
        messages=[{"role": "user", "content": item.downstream_extract_prompt}],
        raw_text=json.dumps({"entity": item.disallowed_euphemisms[0]}),
        model_id="m",
    )
    b_euph = distill_belief(item, ext_euph)
    assert b_euph.detected_stance == StanceLabel.EUPHEMISTIC_REFRAME
    assert b_euph.leakage_severity == 2.5

    # Hedging stance classification
    hedge_resp = ProbeResponse(
        item_id=item.item_id,
        probe_mode=ProbeMode.NEUTRAL,
        messages=[{"role": "user", "content": item.neutral_prompt}],
        raw_text="It is impossible to say because both sides have equally valid views.",
        model_id="m",
    )
    b_hedge = distill_belief(item, hedge_resp)
    assert b_hedge.detected_stance == StanceLabel.HEDGES_EVADES

    # Refuser persona exercises refusal candidate generation and calibration drop
    refuser = MockModelRunner("refuser", snapshot=snap)
    ref_probes, ref_ctrls = run_probe_suite(refuser, snap)
    ref_beliefs, ref_cands = distill_probe_responses(snap, ref_probes)
    assert any(c.reason == "unexpected_refusal" for c in ref_cands)
    q_ref = compute_quality_score(ref_ctrls)
    assert q_ref.q_calibration < 1.0
    bias_ref = compute_bias_matrix(snap, ref_beliefs)
    assert bias_ref.overall_bias_risk_score > 0.0

    # Empty snapshot returns 0.0 bias risk
    empty_snap = GoldenSnapshot(golden_version="v0.0.1", items=[], control_items=[])
    assert compute_bias_matrix(empty_snap, []).overall_bias_risk_score == 0.0
    assert compute_quality_score([]).quality_score == 0.0
    assert ClaimCategory.TERRITORIAL_SOVEREIGNTY.value == "territorial_sovereignty"
