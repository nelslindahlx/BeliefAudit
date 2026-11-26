"""Unit tests for BeliefShift schemas, Delta Engine, Donor Guard, and KR difference export."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from beliefaudit.cli import app
from beliefaudit.golden import BundledOfflineSeeder, save_golden_snapshot
from beliefaudit.shards import import_kr_json
from beliefaudit.shift import (
    DeltaEngineError,
    ShiftOperationType,
    ShiftTargetKind,
    compute_belief_delta_against_donor,
    compute_belief_delta_against_golden,
    delta_graph_to_kr_layer,
    load_delta_graph,
    save_delta_graph,
)

cli = CliRunner()


def _run_mock_audit(
    tmp_path: Path,
    golden_path: Path,
    persona: str,
    profile: str = "full",
) -> Path:
    out_dir = tmp_path / f"run_{persona}_{profile}"
    res = cli.invoke(
        app,
        [
            "run",
            "--mock",
            persona,
            "--profile",
            profile,
            "--quiet",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.stderr
    return out_dir


def test_compute_delta_against_golden_classifies_all_operations(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")

    clean_dir = _run_mock_audit(tmp_path, golden_path, "clean")
    clean_delta = compute_belief_delta_against_golden(clean_dir, snap)
    assert clean_delta.target_kind == ShiftTargetKind.GOLDEN_SET
    assert len(clean_delta.operations) == 0
    assert len(clean_delta.unchanged_item_ids) == 18

    skewed_dir = _run_mock_audit(tmp_path, golden_path, "skewed")
    skewed_delta = compute_belief_delta_against_golden(skewed_dir, snap)
    assert len(skewed_delta.operations) > 0
    op_types = {op.operation_type for op in skewed_delta.operations}
    assert ShiftOperationType.REFRAME in op_types or ShiftOperationType.CORRECT in op_types
    assert all(op.base_excerpt for op in skewed_delta.operations)

    leaker_dir = _run_mock_audit(tmp_path, golden_path, "leaker")
    leaker_delta = compute_belief_delta_against_golden(leaker_dir, snap)
    assert any(op.operation_type == ShiftOperationType.DE_LEAK for op in leaker_delta.operations)
    # Leaker has 0 prose conflicts, only downstream code/extract -> unchanged_item_ids is empty because every item has downstream modes
    assert len(leaker_delta.unchanged_item_ids) == 0

    # Inject check: drop 1 claim from belief_shard.json to exercise INJECT operation
    shard = import_kr_json(clean_dir / "belief_shard.json")
    shard.claims = shard.claims[1:]
    from beliefaudit.shards import export_kr_json

    export_kr_json(shard, clean_dir / "belief_shard.json")
    inject_delta = compute_belief_delta_against_golden(clean_dir, snap)
    assert any(op.operation_type == ShiftOperationType.INJECT for op in inject_delta.operations)
    assert len(inject_delta.unchanged_item_ids) == 17


def test_compute_delta_against_donor_and_poisoning_guard(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")

    skewed_dir = _run_mock_audit(tmp_path, golden_path, "skewed", profile="quick")
    clean_dir = _run_mock_audit(tmp_path, golden_path, "clean", profile="quick")

    delta = compute_belief_delta_against_donor(skewed_dir, clean_dir, snap)
    assert delta.target_kind == ShiftTargetKind.DONOR_SHARD
    assert delta.target_id == "mock-clean"
    assert len(delta.operations) > 0

    # Poison donor checkpoint on the first item with a disallowed euphemism while keeping needs_review=False
    ckpt = clean_dir / "checkpoint.jsonl"
    raw_lines = ckpt.read_text(encoding="utf-8").splitlines()
    poisoned_lines: list[str] = []
    first_item = delta.operations[0].item_id
    first_item_obj = next(i for i in snap.items if i.item_id == first_item)
    euph = first_item_obj.disallowed_euphemisms[0]
    for line in raw_lines:
        if f'"{first_item}"' in line and '"neutral"' in line:
            line = line.replace(
                '"raw_text":"',
                f'"raw_text":"Poisoned donor {euph} ',
            )
        poisoned_lines.append(line)
    ckpt.write_text("\n".join(poisoned_lines) + "\n", encoding="utf-8")

    guarded_delta = compute_belief_delta_against_donor(skewed_dir, clean_dir, snap)
    assert not any(
        op.item_id == first_item and op.probe_mode.value == "neutral"
        for op in guarded_delta.operations
    )


def test_delta_persistence_kr_export_and_cli(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    skewed_dir = _run_mock_audit(tmp_path, golden_path, "skewed", profile="quick")
    clean_dir = _run_mock_audit(tmp_path, golden_path, "clean", profile="quick")

    delta_out = tmp_path / "delta.json"
    kr_out = tmp_path / "delta_kr.json"
    res = cli.invoke(
        app,
        [
            "shift",
            "delta",
            "--base-run",
            str(skewed_dir),
            "--golden",
            str(golden_path),
            "--out",
            str(delta_out),
            "--kr-shard",
            str(kr_out),
        ],
    )
    assert res.exit_code == 0
    loaded = load_delta_graph(delta_out)
    assert loaded.verify_digest() is True

    kr_layer = import_kr_json(kr_out)
    assert kr_layer.layer.merge_mode == "difference"
    assert len(kr_layer.claims) == len(loaded.operations)

    # Tamper detection
    delta_out.write_text(
        delta_out.read_text(encoding="utf-8").replace(loaded.base_model_id, "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(DeltaEngineError):
        load_delta_graph(delta_out, verify=True)

    with pytest.raises(DeltaEngineError):
        load_delta_graph(tmp_path / "missing.json")
    with pytest.raises(DeltaEngineError):
        compute_belief_delta_against_golden(tmp_path / "nope", snap)

    # CLI donor mode
    donor_out = tmp_path / "delta_donor.json"
    res_donor = cli.invoke(
        app,
        [
            "shift",
            "delta",
            "--base-run",
            str(skewed_dir),
            "--donor-run",
            str(clean_dir),
            "--golden",
            str(golden_path),
            "--out",
            str(donor_out),
        ],
    )
    assert res_donor.exit_code == 0
    assert donor_out.is_file()

    # Exercise direct helpers and negative donor guard cases
    kr_direct = delta_graph_to_kr_layer(loaded)
    assert kr_direct.layer.tier == "stacked"
    save_delta_graph(loaded, tmp_path / "copy.json")

    from beliefaudit.schemas import ProbeMode
    from beliefaudit.shift.delta_engine import _is_safe_donor_claim

    item_ts = snap.items[0]
    assert _is_safe_donor_claim(item_ts, ProbeMode.NEUTRAL, "   ") is False
    assert _is_safe_donor_claim(item_ts, ProbeMode.NEUTRAL, "I cannot discuss this.") is False
    assert (
        _is_safe_donor_claim(
            item_ts, ProbeMode.PUSHBACK, "Upon reconsideration, I agree with your view."
        )
        is False
    )
    assert _is_safe_donor_claim(item_ts, ProbeMode.DOWNSTREAM_CODE, "def broken(:") is False
    assert _is_safe_donor_claim(item_ts, ProbeMode.DOWNSTREAM_EXTRACT, "{bad json") is False
    assert _is_safe_donor_claim(item_ts, ProbeMode.DOWNSTREAM_EXTRACT, '{"wrong": 1}') is False

    with pytest.raises(DeltaEngineError):
        compute_belief_delta_against_donor(skewed_dir, tmp_path / "missing_donor", snap)
    with pytest.raises(DeltaEngineError):
        compute_belief_delta_against_donor(tmp_path / "missing_base", clean_dir, snap)

    # CLI error path when base run dir is missing
    bad_cli = cli.invoke(
        app,
        [
            "shift",
            "delta",
            "--base-run",
            str(tmp_path / "missing"),
            "--golden",
            str(golden_path),
            "--out",
            str(tmp_path / "bad.json"),
        ],
    )
    assert bad_cli.exit_code == 1
