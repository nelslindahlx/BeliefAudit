#!/usr/bin/env python3
"""Decoupled Apple Silicon MLX LoRA SFT script for BeliefShift corpora.

Heavy ML dependencies (`mlx`, `mlx-lm`) are imported lazily ONLY after
dataset validation and `--dry-run`, keeping BeliefAudit's core CI zero-ML.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BeliefShift Apple Silicon MLX LoRA trainer")
    parser.add_argument(
        "--data-dir", required=True, help="Directory with train.jsonl & valid.jsonl."
    )
    parser.add_argument(
        "--base-model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="Hugging Face or local MLX model path.",
    )
    parser.add_argument("--adapter-path", default="./adapters", help="Output adapter directory.")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate train/valid JSONL files, print config & command, and exit cleanly.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.lora_rank < 1:
        raise ValueError(f"--lora-rank must be >= 1 (got {args.lora_rank}).")
    if args.lora_alpha < 1:
        raise ValueError(f"--lora-alpha must be >= 1 (got {args.lora_alpha}).")
    if args.lora_layers < 1:
        raise ValueError(f"--lora-layers must be >= 1 (got {args.lora_layers}).")
    if args.iters < 1:
        raise ValueError(f"--iters must be >= 1 (got {args.iters}).")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1 (got {args.batch_size}).")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be > 0 (got {args.lr}).")


def _validate_mlx_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MLX dataset file: {path}")
    count = 0
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        data: Any = json.loads(stripped)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError(f"Line {idx} in {path} must contain a 'messages' list.")
        for msg in data["messages"]:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Line {idx} in {path} has malformed chat message.")
        count += 1
    if count == 0:
        raise ValueError(f"Dataset {path} is empty.")
    return count


def build_mlx_lora_config(args: argparse.Namespace, data_dir: Path) -> dict[str, Any]:
    """Build explicit mlx_lm LoRA config dictionary including rank and alpha scaling."""
    return {
        "model": args.base_model,
        "data": str(data_dir),
        "train": True,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_layers": args.lora_layers,
        "adapter_path": str(args.adapter_path),
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": round(args.lora_alpha / args.lora_rank, 4),
            "dropout": 0.05,
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    try:
        _validate_args(args)
        n_train = _validate_mlx_jsonl(data_dir / "train.jsonl")
        n_valid = _validate_mlx_jsonl(data_dir / "valid.jsonl")
    except Exception as exc:
        print(f"Error validating MLX corpus: {exc}", file=sys.stderr)
        return 1

    adapter_dir = Path(args.adapter_path)
    config_path = adapter_dir / "lora_config.yaml"
    config = build_mlx_lora_config(args, data_dir)
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]

    print(f"Validated MLX corpus: train={n_train}, valid={n_valid} in {data_dir}")
    if args.dry_run:
        lp = config["lora_parameters"]
        print(
            f"[dry-run] lora_parameters: rank={lp['rank']} scale={lp['scale']} layers={args.lora_layers}"
        )
        print("[dry-run] Would run:", shlex.join(cmd))
        print(
            "[dry-run] Serve after training:",
            f"mlx_lm.server --model {args.base_model} --adapter-path {args.adapter_path} --port 8080",
        )
        return 0

    try:
        import mlx_lm  # type: ignore[import-not-found] # noqa: F401
        import yaml
    except ImportError as exc:
        print(
            f"mlx-lm or pyyaml not installed ({exc}). Install via: pip install mlx-lm pyyaml",
            file=sys.stderr,
        )
        return 3

    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        print(
            f"Saved adapter to {args.adapter_path}. Serve with:\n"
            f"  mlx_lm.server --model {args.base_model} --adapter-path {args.adapter_path} --port 8080"
        )
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
