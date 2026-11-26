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

    # Invoke run_shift_training directly with MLX and PEFT backends in dry-run mode
    mlx_manifest = run_shift_training(
        corpus_dir=corpus_dir,
        delta=delta,
        base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        out_shard_dir=tmp_path / "shard_mlx_dry",
        backend=TrainingBackend.MLX,
        lora_rank=8,
        lora_alpha=16,
        dry_run=True,
    )
    assert mlx_manifest.dry_run is True
    assert "/" not in mlx_manifest.source_belief_shard_id
    assert ":" not in mlx_manifest.source_belief_shard_id

    peft_manifest = run_shift_training(
        corpus_dir=corpus_dir,
        delta=delta,
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        out_shard_dir=tmp_path / "shard_peft_dry",
        backend=TrainingBackend.PEFT,
        iters_or_epochs=3,
        run_dpo=True,
        dpo_beta=0.05,
        dpo_lr_scale=0.1,
        dry_run=True,
    )
    assert peft_manifest.iters_or_epochs == 3
    assert "--epochs" in peft_manifest.executed_command
    assert "3" in peft_manifest.executed_command
    assert "--dpo-beta" in peft_manifest.executed_command
    assert "0.05" in peft_manifest.executed_command

    # Fail closed when --dpo is passed to --backend mlx
    with pytest.raises(ShiftTrainerError, match="supported only with --backend peft"):
        run_shift_training(
            corpus_dir=corpus_dir,
            delta=delta,
            base_model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            out_shard_dir=tmp_path / "mlx_dpo_bad",
            backend=TrainingBackend.MLX,
            run_dpo=True,
            dry_run=True,
        )

    # Run both standalone scripts with --dry-run (zero ML imports, zero GPU)
    mlx_cmd = build_training_command(
        TrainingBackend.MLX,
        manifest,
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        shard_dir,
        lora_rank=8,
        lora_alpha=16,
        dry_run=True,
    )
    res_mlx = subprocess.run(  # noqa: S603
        mlx_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
    )
    assert res_mlx.returncode == 0, res_mlx.stderr
    assert "[dry-run]" in res_mlx.stdout
    assert "rank=8" in res_mlx.stdout

    peft_cmd = build_training_command(
        TrainingBackend.PEFT,
        manifest,
        "Qwen/Qwen2.5-7B-Instruct",
        shard_dir,
        iters_or_epochs=3,
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
    # Tampered JSONL triggers digest mismatch error
    sft_path = Path(manifest.sft_train_path)
    orig_sft = sft_path.read_text(encoding="utf-8")
    sft_path.write_text(orig_sft + orig_sft.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ShiftTrainerError, match="digest mismatch"):
        validate_corpus_for_training(corpus_dir)
    sft_path.write_text(orig_sft, encoding="utf-8")

    # Missing/corrupted MLX & DPO files and missing script paths raise ShiftTrainerError
    from beliefaudit.shift.trainer import _resolve_training_script

    with pytest.raises(ShiftTrainerError):
        _resolve_training_script("train_mlx.py", scripts_dir=tmp_path / "no_scripts")

    mlx_tr = Path(manifest.mlx_data_dir) / "train.jsonl"
    orig_mlx = mlx_tr.read_text(encoding="utf-8")
    mlx_tr.write_text('{"bad": 1}\n', encoding="utf-8")
    with pytest.raises(ShiftTrainerError, match="Invalid MLX record"):
        validate_corpus_for_training(corpus_dir)
    mlx_tr.write_text(orig_mlx, encoding="utf-8")

    dpo_path = Path(manifest.dpo_pairs_path)
    orig_dpo = dpo_path.read_text(encoding="utf-8")
    dpo_path.write_text("", encoding="utf-8")
    with pytest.raises(ShiftTrainerError, match="is empty"):
        validate_corpus_for_training(corpus_dir, require_dpo=True)
    dpo_path.write_text(orig_dpo, encoding="utf-8")

    # Non-dry-run failing subprocess in run_shift_training raises ShiftTrainerError
    failing_scripts = tmp_path / "fail_scripts"
    failing_scripts.mkdir()
    (failing_scripts / "train_mlx.py").write_text("import sys; sys.exit(2)\n", encoding="utf-8")
    with pytest.raises(ShiftTrainerError, match="failed \\(exit 2\\)"):
        run_shift_training(
            corpus_dir=corpus_dir,
            delta=delta,
            base_model_id="m",
            out_shard_dir=tmp_path / "fail_out",
            backend=TrainingBackend.MLX,
            dry_run=False,
            scripts_dir=failing_scripts,
        )

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

    # Script bad dataset / invalid hyperparameters return exit 1
    bad_mlx = subprocess.run(  # noqa: S603
        [sys.executable, "scripts/train_mlx.py", "--data-dir", str(tmp_path / "nope"), "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_mlx.returncode == 1
    bad_rank = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_mlx.py",
            "--data-dir",
            manifest.mlx_data_dir,
            "--lora-rank",
            "0",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_rank.returncode == 1
    bad_peft = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_peft.py",
            "--dataset",
            manifest.sft_train_path,
            "--lora-r",
            "0",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_peft.returncode == 1
    bad_beta = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/train_peft.py",
            "--dataset",
            manifest.sft_train_path,
            "--dpo-beta",
            "0",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_beta.returncode == 1


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

    # Held-out evaluation against skewed mock fails held-out checks
    skewed_runner = MockModelRunner("skewed", snapshot=snap)
    ho_rate, _ = run_heldout_evaluation(skewed_runner, snap, set())
    assert ho_rate is not None and ho_rate < 0.5

    # Held-out grading rejects hedging and short stub answers; unmatched snapshot returns (None, {})
    from beliefaudit.shift.heldout_pack import build_heldout_pack
    from beliefaudit.shift.roundtrip_eval import _grade_heldout_response

    first_ho = build_heldout_pack()[0]
    first_base = next(i for i in snap.items if i.item_id == first_ho.item_id)
    assert (
        _grade_heldout_response(
            first_ho, first_base, "Both sides have equally valid perspectives on this issue."
        )
        is False
    )
    assert _grade_heldout_response(first_ho, first_base, "Acknowledged.") is False

    empty_snap = snap.model_copy(
        update={
            "items": [snap.items[0].model_copy(update={"item_id": "custom_unmatched_topic"})],
            "content_sha256": "",
        }
    )
    empty_snap = empty_snap.model_validate(empty_snap.model_dump())
    unmatched_rate, unmatched_map = run_heldout_evaluation(skewed_runner, empty_snap, set())
    assert unmatched_rate is None and unmatched_map == {}

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
    assert ho_leak is not None and ho_leak < 1.0

    # Profile / version mismatch & delta tamper checks in evaluate_roundtrip
    full_pre = tmp_path / "full_pre_check"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "skewed",
                "--profile",
                "full",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(full_pre),
            ],
        ).exit_code
        == 0
    )
    full_delta = compute_belief_delta_against_golden(full_pre, snap)
    with pytest.raises(RoundTripVerifyError, match="Audit profile mismatch"):
        evaluate_roundtrip(full_delta, full_pre, verify_dir / "post_audit", snap)

    wrong_snap = BundledOfflineSeeder().seed("v0.2.0")
    with pytest.raises(RoundTripVerifyError, match="GoldenSnapshot version"):
        evaluate_roundtrip(delta, pre_dir, verify_dir / "post_audit", wrong_snap)

    # CLI verify without --profile auto-infers pre_run's FULL profile (and covers verbose progress + judge flag)
    full_delta_path = save_delta_graph(full_delta, tmp_path / "full_delta.json")
    auto_prof_dir = tmp_path / "verify_auto_full"
    res_auto = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(full_delta_path),
            "--pre-run",
            str(full_pre),
            "--mock",
            "clean",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(auto_prof_dir),
        ],
    )
    assert res_auto.exit_code == 0, res_auto.stdout + res_auto.stderr
    assert "gate=PASSED" in res_auto.stdout

    # Tampered delta fails evaluate_roundtrip with RoundTripVerifyError
    tampered_delta = delta.model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(RoundTripVerifyError, match="digest verification"):
        evaluate_roundtrip(tampered_delta, pre_dir, verify_dir / "post_audit", snap)

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

    # CLI verify --post-run + --mock clean runs held-out probes against mock while reusing artifacts
    res_post_ho = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(delta_path),
            "--pre-run",
            str(pre_dir),
            "--post-run",
            str(verify_dir / "post_audit"),
            "--mock",
            "clean",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(tmp_path / "verify_post_ho"),
        ],
    )
    assert res_post_ho.exit_code == 0, res_post_ho.stdout + res_post_ho.stderr
    assert "gate=PASSED" in res_post_ho.stdout

    # CLI verify --post-run + --heldout-rate 0.20 triggers parrot-alignment warning and exits 1
    res_post_parrot = cli.invoke(
        app,
        [
            "shift",
            "verify",
            "--delta",
            str(delta_path),
            "--pre-run",
            str(pre_dir),
            "--post-run",
            str(verify_dir / "post_audit"),
            "--heldout-rate",
            "0.20",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(tmp_path / "verify_post_parrot"),
        ],
    )
    assert res_post_parrot.exit_code == 1

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
