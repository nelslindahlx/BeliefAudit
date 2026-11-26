"""Unit tests for BeliefShift trainer, MLX/PEFT dry-run scripts, and round-trip verifier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from beliefaudit.cli import app
from beliefaudit.golden import BundledOfflineSeeder, save_golden_snapshot
from beliefaudit.probes import MockModelRunner
from beliefaudit.query import RunQueryError
from beliefaudit.shift import (
    ShiftTrainerError,
    TrainingBackend,
    build_training_command,
    compile_shift_corpus,
    compute_belief_delta_against_golden,
    evaluate_roundtrip,
    load_lora_manifest,
    run_heldout_evaluation,
    run_shift_training,
    save_delta_graph,
    validate_corpus_for_training,
)
from beliefaudit.shift.roundtrip_eval import RoundTripVerifyError, write_roundtrip_artifacts

cli = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _setup_skewed_corpus(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    pre_dir = tmp_path / "pre_skewed"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "skewed",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(pre_dir),
            ],
        ).exit_code
        == 0
    )
    delta = compute_belief_delta_against_golden(pre_dir, snap)
    delta_path = save_delta_graph(delta, tmp_path / "delta.json")
    corpus_dir = tmp_path / "corpus"
    compile_shift_corpus(delta, snap, corpus_dir)
    return golden_path, pre_dir, delta_path, corpus_dir


def test_trainer_dry_run_validation_and_scripts(tmp_path: Path) -> None:
    _, _, delta_path, corpus_dir = _setup_skewed_corpus(tmp_path)
    from beliefaudit.shift import load_delta_graph

    delta = load_delta_graph(delta_path)
    manifest = validate_corpus_for_training(corpus_dir)
    assert manifest.leakage_check_passed is True

    shard_dir = tmp_path / "lora_shard"
    lora_manifest = run_shift_training(
        corpus_dir=corpus_dir,
        delta=delta,
        base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        out_shard_dir=shard_dir,
        backend=TrainingBackend.DRY_RUN,
        lora_rank=16,
    )
    assert lora_manifest.dry_run is True
    assert (shard_dir / "lora_shard_manifest.json").is_file()
    assert (shard_dir / "train_command.txt").is_file()
    assert (
        load_lora_manifest(shard_dir / "lora_shard_manifest.json").shard_id
        == lora_manifest.shard_id
    )

    # Run both standalone scripts with --dry-run (zero ML imports, zero GPU)
    mlx_cmd = build_training_command(
        TrainingBackend.MLX,
        manifest,
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        shard_dir,
        dry_run=True,
    )
    res_mlx = subprocess.run(  # noqa: S603
        mlx_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    assert res_mlx.returncode == 0, res_mlx.stderr
    assert "[dry-run]" in res_mlx.stdout

    peft_cmd = build_training_command(
        TrainingBackend.PEFT,
        manifest,
        "Qwen/Qwen2.5-7B-Instruct",
        shard_dir,
        run_dpo=True,
        dry_run=True,
    )
    res_peft = subprocess.run(  # noqa: S603
        peft_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    assert res_peft.returncode == 0, res_peft.stderr
    assert "Stage 3B DPOTrainer" in res_peft.stdout

    # Validation error paths
    with pytest.raises(ShiftTrainerError):
        load_lora_manifest(tmp_path / "missing_manifest.json")
    with pytest.raises(ShiftTrainerError):
        validate_corpus_for_training(corpus_dir, min_replay_ratio=0.99)
    # Leaked corpus rejected
    cm_path = corpus_dir / "corpus_manifest.json"
    cm_path.write_text(
        cm_path.read_text(encoding="utf-8").replace(
            '"leakage_check_passed": true', '"leakage_check_passed": false'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ShiftTrainerError):
        validate_corpus_for_training(corpus_dir)

    # Script bad dataset returns exit 1
    bad_mlx = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/train_mlx.py", "--data-dir", str(tmp_path / "nope"), "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_mlx.returncode == 1


def test_roundtrip_verify_pass_quality_fail_drift_and_parrot_warning(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path, pre_dir, delta_path, corpus_dir = _setup_skewed_corpus(tmp_path)

    shard_dir = tmp_path / "shard_cli"
    res_train = cli.invoke(
        app,
        [
            "shift",
            "train",
            "--corpus",
            str(corpus_dir),
            "--delta",
            str(delta_path),
            "--base-model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "--backend",
            "dry-run",
            "--out-shard",
            str(shard_dir),
        ],
    )
    assert res_train.exit_code == 0

    # End-to-end CLI verify with clean shifted mock -> PASS
    verify_dir = tmp_path / "verify_pass"
    res_verify = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(delta_path),
            "--pre-run",
            str(pre_dir),
            "--mock",
            "clean",
            "--profile",
            "quick",
            "--golden",
            str(golden_path),
            "--manifest",
            str(shard_dir / "lora_shard_manifest.json"),
            "--out-dir",
            str(verify_dir),
        ],
    )
    assert res_verify.exit_code == 0, res_verify.stdout + res_verify.stderr
    assert "gate=PASSED" in res_verify.stdout
    assert (verify_dir / "roundtrip_report.md").is_file()

    updated_manifest = load_lora_manifest(shard_dir / "lora_shard_manifest.json")
    assert updated_manifest.verification_passed is True
    assert updated_manifest.target_shift_rate == 1.0
    assert updated_manifest.heldout_shift_rate == 1.0

    # Held-out evaluation on skewed mock fails held-out checks
    skewed_runner = MockModelRunner("skewed", snapshot=snap)
    ho_rate, _ = run_heldout_evaluation(skewed_runner, snap, set())
    assert ho_rate < 0.5

    # Quality drop failure (skewed -> low_quality)
    low_q_dir = tmp_path / "post_low_q"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "low_quality",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(low_q_dir),
            ],
        ).exit_code
        == 0
    )
    from beliefaudit.shift import load_delta_graph

    delta = load_delta_graph(delta_path)
    rep_low_q = evaluate_roundtrip(delta, pre_dir, low_q_dir, snap)
    assert rep_low_q.quality_gate_passed is False
    assert rep_low_q.overall_gate_passed is False

    # Parrot alignment warning when heldout rate lags TSR by > 0.30
    rep_parrot = evaluate_roundtrip(
        delta,
        pre_dir,
        verify_dir / "post_audit",
        snap,
        heldout_shift_rate=0.20,
    )
    assert rep_parrot.parrot_alignment_warning is True
    assert rep_parrot.overall_gate_passed is False
    written = write_roundtrip_artifacts(tmp_path / "parrot_out", rep_parrot)
    assert "Parrot alignment warning" in written["roundtrip_md"].read_text(encoding="utf-8")

    # Collateral drift check when previously unchanged item regresses
    clean_pre = tmp_path / "clean_pre"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "clean",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(clean_pre),
            ],
        ).exit_code
        == 0
    )
    clean_delta = compute_belief_delta_against_golden(clean_pre, snap)
    rep_drift = evaluate_roundtrip(clean_delta, clean_pre, pre_dir, snap)
    assert rep_drift.collateral_drift_score > 0.5
    assert rep_drift.drift_gate_passed is False

    with pytest.raises(RunQueryError):
        evaluate_roundtrip(delta, pre_dir, tmp_path / "missing_post", snap)

    # Missing belief_shard.json with existing scorecard.json raises RoundTripVerifyError
    partial_post = tmp_path / "partial_post"
    partial_post.mkdir()
    (partial_post / "scorecard.json").write_text(
        (pre_dir / "scorecard.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(RoundTripVerifyError):
        evaluate_roundtrip(delta, pre_dir, partial_post, snap)

    # Heldout evaluation against leaker covers downstream heldout failures
    leaker_runner = MockModelRunner("leaker", snapshot=snap)
    ho_leak, _ = run_heldout_evaluation(leaker_runner, snap, set())
    assert ho_leak < 1.0

    # CLI verify --post-run mode + failing gate exits 1
    res_post = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(delta_path),
            "--pre-run",
            str(pre_dir),
            "--post-run",
            str(low_q_dir),
            "--golden",
            str(golden_path),
            "--out-dir",
            str(tmp_path / "verify_fail_cli"),
        ],
    )
    assert res_post.exit_code == 1

    # CLI unknown backend & missing verify options exit 2
    assert (
        cli.invoke(
            app,
            [
                "shift",
                "train",
                "--corpus",
                str(corpus_dir),
                "--delta",
                str(delta_path),
                "--base-model",
                "m",
                "--backend",
                "bad",
                "--out-shard",
                str(tmp_path / "s"),
            ],
        ).exit_code
        == 2
    )
    assert (
        cli.invoke(
            app,
            [
                "shift",
                "verify",
                "--delta",
                str(delta_path),
                "--pre-run",
                str(pre_dir),
                "--golden",
                str(golden_path),
                "--out-dir",
                str(tmp_path / "v"),
            ],
        ).exit_code
        == 2
    )
