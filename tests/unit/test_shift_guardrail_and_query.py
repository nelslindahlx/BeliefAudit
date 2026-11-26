"""Unit tests for BeliefShift v0.3.0 runtime guardrails, shift query helpers, and CLI inspect/guardrail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from beliefaudit import (
    BundledOfflineSeeder,
    ProbeMode,
    RuntimeGuardrailSpec,
    TrainingBackend,
    compile_runtime_guardrail,
    compile_shift_corpus,
    compute_belief_delta_against_golden,
    evaluate_roundtrip,
    inspect_shift_artifacts,
    list_lora_shards,
    load_roundtrip_report,
    run_shift_training,
    verify_response_against_guardrail,
)
from beliefaudit.cli import app
from beliefaudit.golden import save_golden_snapshot
from beliefaudit.golden.rubric import find_term_matches
from beliefaudit.query import RunQueryError
from beliefaudit.shift import (
    GuardrailError,
    load_guardrail_spec,
    render_guardrail_system_prompt,
    save_delta_graph,
    save_guardrail_spec,
)

cli = CliRunner()


def _run_quick(tmp_path: Path, golden_path: Path, persona: str) -> Path:
    out_dir = tmp_path / f"run_{persona}"
    res = cli.invoke(
        app,
        [
            "run",
            "--mock",
            persona,
            "--profile",
            "quick",
            "--quiet",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.stderr
    return out_dir


def test_runtime_guardrail_compile_prompt_hygiene_and_output_verification(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    skewed_dir = _run_quick(tmp_path, golden_path, "skewed")
    low_q_dir = _run_quick(tmp_path, golden_path, "low_quality")

    delta = compute_belief_delta_against_golden(skewed_dir, snap)
    save_delta_graph(delta, skewed_dir / "delta_graph.json")

    # Resistant report (skewed -> skewed keeps resistant operations)
    failing_report = evaluate_roundtrip(delta, skewed_dir, skewed_dir, snap)
    spec_res = compile_runtime_guardrail(delta, snap, roundtrip_report=failing_report)
    assert spec_res.verify_digest() is True
    assert len(spec_res.rules) == len({op.item_id for op in delta.operations})

    # Clean report produces 0 resistant rules unless include_all_ops=True
    passing_report = evaluate_roundtrip(delta, skewed_dir, low_q_dir, snap)
    spec_empty = compile_runtime_guardrail(delta, snap, roundtrip_report=passing_report)
    assert len(spec_empty.rules) == 0
    assert spec_empty.system_prompt_block == ""

    spec_all = compile_runtime_guardrail(
        delta, snap, roundtrip_report=passing_report, include_all_ops=True
    )
    assert len(spec_all.rules) == len({op.item_id for op in delta.operations})

    # System prompt overlay MUST NEVER echo disallowed euphemisms
    for rule in spec_all.rules:
        assert not find_term_matches(spec_all.system_prompt_block, rule.disallowed_euphemisms)
    assert render_guardrail_system_prompt([]) == ""

    # Output verification: clean passes; refusal, hedging, euphemism, contrary, pushback, code/JSON fail
    first_item = next(i for i in snap.items if i.item_id == spec_all.rules[0].item_id)
    assert verify_response_against_guardrail(
        spec_all, first_item.canonical_claim, item_id=first_item.item_id
    ).passed

    # Unmatched item_id passes cleanly
    assert verify_response_against_guardrail(spec_all, "anything", item_id="untracked_id").passed

    assert not verify_response_against_guardrail(spec_all, "", item_id=first_item.item_id).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "I cannot discuss this topic because it is sensitive.",
        item_id=first_item.item_id,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "Both sides have equally valid perspectives here.",
        item_id=first_item.item_id,
    ).passed

    bad_euph_res = verify_response_against_guardrail(
        spec_all,
        f"It is a {first_item.disallowed_euphemisms[0]}.",
        item_id=first_item.item_id,
    )
    assert bad_euph_res.passed is False
    assert bad_euph_res.fallback_claim == first_item.canonical_claim

    assert not verify_response_against_guardrail(
        spec_all,
        f"Yes, {first_item.contrary_markers[0]}.",
        item_id=first_item.item_id,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "Upon reconsideration, I agree with your view completely.",
        item_id=first_item.item_id,
        probe_mode=ProbeMode.PUSHBACK,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "def broken(:",
        item_id=first_item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_CODE,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "def wrong_name() -> int:\n    return 1",
        item_id=first_item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_CODE,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        "not json",
        item_id=first_item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_EXTRACT,
    ).passed
    assert not verify_response_against_guardrail(
        spec_all,
        '{"wrong": 1}',
        item_id=first_item.item_id,
        probe_mode=ProbeMode.DOWNSTREAM_EXTRACT,
    ).passed

    # Broad screening with item_id=None still catches euphemisms while allowing valid JSON
    assert verify_response_against_guardrail(
        spec_all, '{"ok": true}', probe_mode=ProbeMode.DOWNSTREAM_EXTRACT
    ).passed
    assert not verify_response_against_guardrail(
        spec_all, f"Framed as {first_item.disallowed_euphemisms[0]}."
    ).passed

    # Save / load + tamper detection
    gr_path = save_guardrail_spec(spec_all, tmp_path / "guardrail_spec.json")
    loaded = load_guardrail_spec(gr_path)
    assert isinstance(loaded, RuntimeGuardrailSpec)
    gr_path.write_text(
        gr_path.read_text(encoding="utf-8").replace(loaded.base_model_id, "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(GuardrailError):
        load_guardrail_spec(gr_path, verify=True)
    with pytest.raises(GuardrailError):
        load_guardrail_spec(tmp_path / "missing_gr.json")
    with pytest.raises(GuardrailError):
        save_guardrail_spec(
            loaded.model_copy(update={"content_sha256": "0" * 64}), tmp_path / "x.json"
        )

    bad_delta = delta.model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(GuardrailError):
        compile_runtime_guardrail(bad_delta, snap)
    with pytest.raises(GuardrailError):
        compile_runtime_guardrail(delta, BundledOfflineSeeder().seed("v0.2.0"))


def test_shift_query_helpers_and_cli_guardrail_inspect(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    skewed_dir = _run_quick(tmp_path, golden_path, "skewed")
    clean_dir = _run_quick(tmp_path, golden_path, "clean")

    bundle_dir = tmp_path / "shift_bundle"
    bundle_dir.mkdir()
    delta = compute_belief_delta_against_golden(skewed_dir, snap)
    delta_path = save_delta_graph(delta, bundle_dir / "delta_graph.json")
    compile_shift_corpus(delta, snap, bundle_dir)

    shards_root = tmp_path / "shards"
    shard_dir = shards_root / "shard_01"
    run_shift_training(
        corpus_dir=bundle_dir,
        delta=delta,
        base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        out_shard_dir=shard_dir,
        backend=TrainingBackend.DRY_RUN,
    )
    # Copy lora_shard_manifest.json into bundle_dir as well to exercise full inspect_shift_artifacts
    (bundle_dir / "lora_shard_manifest.json").write_text(
        (shard_dir / "lora_shard_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    res_verify = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(delta_path),
            "--pre-run",
            str(skewed_dir),
            "--post-run",
            str(clean_dir),
            "--mock",
            "clean",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(bundle_dir),
        ],
    )
    assert res_verify.exit_code == 0, res_verify.stdout + res_verify.stderr

    # CLI shift guardrail writing guardrail_spec.json + system_prompt.txt
    prompt_out = tmp_path / "system_prompt.txt"
    res_gr = cli.invoke(
        app,
        [
            "shift",
            "guardrail",
            "--delta",
            str(delta_path),
            "--golden",
            str(golden_path),
            "--roundtrip",
            str(bundle_dir),
            "--include-all-ops",
            "--out",
            str(bundle_dir / "guardrail_spec.json"),
            "--system-prompt-out",
            str(prompt_out),
        ],
    )
    assert res_gr.exit_code == 0, res_gr.stderr
    assert prompt_out.is_file() and "Runtime Factual Grounding Guardrails" in prompt_out.read_text(
        encoding="utf-8"
    )

    # Query API checks
    rt = load_roundtrip_report(bundle_dir)
    assert rt.overall_gate_passed is True
    shards = list_lora_shards(shards_root)
    assert len(shards) == 1 and shards[0]["shard_dir"] == "shard_01"
    assert list_lora_shards(tmp_path / "nonexistent_shards") == []

    summary = inspect_shift_artifacts(bundle_dir)
    assert set(summary["found"]) == {
        "delta_graph",
        "corpus_manifest",
        "lora_shard_manifest",
        "roundtrip_report",
        "guardrail_spec",
    }

    # CLI shift inspect (both text and --json)
    res_ins = cli.invoke(app, ["shift", "inspect", "--dir", str(bundle_dir)])
    assert res_ins.exit_code == 0
    assert "guardrail_spec:" in res_ins.stdout
    res_ins_json = cli.invoke(app, ["shift", "inspect", "--dir", str(bundle_dir), "--json"])
    assert res_ins_json.exit_code == 0
    parsed = json.loads(res_ins_json.stdout)
    assert "roundtrip_report" in parsed

    # Error paths
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(RunQueryError):
        inspect_shift_artifacts(empty_dir)
    with pytest.raises(RunQueryError):
        inspect_shift_artifacts(tmp_path / "missing_dir")
    with pytest.raises(RunQueryError):
        load_roundtrip_report(empty_dir)
    assert cli.invoke(app, ["shift", "inspect", "--dir", str(empty_dir)]).exit_code == 1
    assert (
        cli.invoke(
            app,
            [
                "shift",
                "guardrail",
                "--delta",
                str(delta_path),
                "--golden",
                str(golden_path),
                "--roundtrip",
                str(tmp_path / "missing_rt.json"),
                "--out",
                str(tmp_path / "x.json"),
            ],
        ).exit_code
        == 1
    )
