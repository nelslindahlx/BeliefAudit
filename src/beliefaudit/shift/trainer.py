"""Pure-Python LoRA training orchestrator and manifest persistence (zero ML imports)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from beliefaudit.security import BeliefAuditError, ensure_within_base
from beliefaudit.shift.kg_compiler import load_corpus_manifest
from beliefaudit.shift.schemas import (
    BeliefDeltaGraph,
    CompiledCorpusManifest,
    LoRAShardManifest,
    SftTrainingRecord,
    TrainingBackend,
)


class ShiftTrainerError(BeliefAuditError):
    """Raised when corpus validation or LoRA backend invocation fails."""


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts"


def validate_corpus_for_training(
    corpus_dir: Path | str,
    *,
    min_replay_ratio: float = 0.25,
    base_dir: Path | str | None = None,
) -> CompiledCorpusManifest:
    """Validate compiled corpus manifest, leakage gate, replay ratio, and JSONL schema."""
    manifest = load_corpus_manifest(corpus_dir, base_dir=base_dir)
    if not manifest.leakage_check_passed:
        raise ShiftTrainerError("Refusing to train: corpus failed anti-leakage gate.")
    if manifest.replay_ratio < min_replay_ratio:
        raise ShiftTrainerError(
            f"Replay ratio {manifest.replay_ratio:.2f} is below safety floor {min_replay_ratio:.2f}."
        )
    sft_train = Path(manifest.sft_train_path)
    mlx_train = Path(manifest.mlx_data_dir) / "train.jsonl"
    if not sft_train.is_file() or not mlx_train.is_file():
        raise ShiftTrainerError(f"Missing compiled training JSONL files for {manifest.corpus_id}.")
    rows = [
        line.strip() for line in sft_train.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ShiftTrainerError(f"Compiled training file {sft_train} is empty.")
    for idx, line in enumerate(rows, start=1):
        try:
            SftTrainingRecord.model_validate_json(line)
        except Exception as exc:
            raise ShiftTrainerError(f"Invalid SftTrainingRecord at line {idx}: {exc}") from exc
    return manifest


def build_training_command(
    backend: TrainingBackend,
    corpus_manifest: CompiledCorpusManifest,
    base_model_id: str,
    out_shard_dir: Path,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_layers: int = 16,
    lr: float = 1e-5,
    iters_or_epochs: int = 200,
    batch_size: int = 2,
    run_dpo: bool = False,
    dry_run: bool = True,
) -> list[str]:
    """Construct the exact Python CLI command for the selected LoRA backend."""
    adapter_dir = out_shard_dir / "adapter"
    if backend == TrainingBackend.PEFT:
        script = str(_scripts_dir() / "train_peft.py")
        cmd = [
            sys.executable,
            script,
            "--dataset",
            corpus_manifest.sft_train_path,
            "--base-model",
            base_model_id,
            "--output-dir",
            str(adapter_dir),
            "--epochs",
            str(max(1, iters_or_epochs if iters_or_epochs <= 10 else 2)),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--lora-r",
            str(lora_rank),
            "--lora-alpha",
            str(lora_alpha),
        ]
        if run_dpo:
            cmd.extend(["--dpo-dataset", corpus_manifest.dpo_pairs_path])
        if dry_run:
            cmd.append("--dry-run")
        return cmd

    # Default / MLX command (for DRY_RUN we emit the MLX command as the local reference)
    script = str(_scripts_dir() / "train_mlx.py")
    cmd = [
        sys.executable,
        script,
        "--data-dir",
        corpus_manifest.mlx_data_dir,
        "--base-model",
        base_model_id,
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(iters_or_epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--lora-layers",
        str(lora_layers),
        "--lora-rank",
        str(lora_rank),
    ]
    if dry_run or backend == TrainingBackend.DRY_RUN:
        cmd.append("--dry-run")
    return cmd


def run_shift_training(
    *,
    corpus_dir: Path | str,
    delta: BeliefDeltaGraph,
    base_model_id: str,
    out_shard_dir: Path | str,
    backend: TrainingBackend = TrainingBackend.DRY_RUN,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_layers: int = 16,
    lr: float = 1e-5,
    iters_or_epochs: int = 200,
    batch_size: int = 2,
    run_dpo: bool = False,
    dry_run: bool = False,
    min_replay_ratio: float = 0.25,
    base_dir: Path | str | None = None,
) -> LoRAShardManifest:
    """Validate corpus, write LoRAShardManifest, and invoke backend (or dry-run)."""
    manifest = validate_corpus_for_training(
        corpus_dir, min_replay_ratio=min_replay_ratio, base_dir=base_dir
    )
    target_dir = (
        ensure_within_base(out_shard_dir, base_dir)
        if base_dir is not None
        else Path(out_shard_dir).resolve()
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    effective_dry_run = dry_run or backend == TrainingBackend.DRY_RUN

    cmd = build_training_command(
        backend,
        manifest,
        base_model_id,
        target_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
        lr=lr,
        iters_or_epochs=iters_or_epochs,
        batch_size=batch_size,
        run_dpo=run_dpo,
        dry_run=effective_dry_run,
    )

    if backend != TrainingBackend.DRY_RUN:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if res.returncode != 0:
            detail = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()[-600:]
            raise ShiftTrainerError(
                f"Training backend {backend.value!r} failed (exit {res.returncode}): {detail}"
            )

    adapter_dir = target_dir / "adapter"
    shard_seed = f"{base_model_id}|{delta.delta_id}|{manifest.corpus_id}|{backend.value}"
    shard_id = "lora_" + hashlib.sha256(shard_seed.encode("utf-8")).hexdigest()[:12]

    lora_manifest = LoRAShardManifest(
        shard_id=shard_id,
        base_model_id=base_model_id,
        source_belief_shard_id=f"beliefs-{delta.base_model_id}-{delta.golden_version}",
        target_kind=delta.target_kind,
        target_id=delta.target_id,
        delta_id=delta.delta_id,
        corpus_id=manifest.corpus_id,
        backend=backend,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        learning_rate=lr,
        iters_or_epochs=iters_or_epochs,
        replay_ratio=manifest.replay_ratio,
        adapter_weights_path=str(adapter_dir),
        dry_run=effective_dry_run,
        executed_command=cmd,
    )
    save_lora_manifest(lora_manifest, target_dir / "lora_shard_manifest.json")
    serve_hint = (
        f"mlx_lm.server --model {base_model_id} --adapter-path {adapter_dir} --port 8080"
        if backend != TrainingBackend.PEFT
        else f"vllm serve {base_model_id} --enable-lora --lora-modules belief_shift={adapter_dir} --port 8000"
    )
    (target_dir / "train_command.txt").write_text(
        "# Train command:\n"
        + " ".join(cmd)
        + "\n\n# Local OpenAI-compatible serving command for `beliefaudit shift verify`:\n"
        + serve_hint
        + "\n",
        encoding="utf-8",
    )
    return lora_manifest


def save_lora_manifest(
    manifest: LoRAShardManifest,
    output_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """Write a LoRAShardManifest JSON file."""
    target = (
        ensure_within_base(output_path, base_dir)
        if base_dir is not None
        else Path(output_path).resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_lora_manifest(
    input_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> LoRAShardManifest:
    """Load and validate a LoRAShardManifest JSON file."""
    source = (
        ensure_within_base(input_path, base_dir)
        if base_dir is not None
        else Path(input_path).resolve()
    )
    if not source.is_file():
        raise ShiftTrainerError(f"LoRAShardManifest not found: {source}")
    return LoRAShardManifest.model_validate_json(source.read_text(encoding="utf-8"))
