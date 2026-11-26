#!/usr/bin/env python3
"""Decoupled Apple Silicon MLX LoRA SFT script for BeliefShift corpora.

Heavy ML dependencies (`mlx`, `mlx-lm`) are imported lazily ONLY after
dataset validation and `--dry-run`, keeping BeliefAudit's core CI zero-ML.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate train/valid JSONL files, print command, and exit cleanly.",
    )
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    try:
        n_train = _validate_mlx_jsonl(data_dir / "train.jsonl")
        n_valid = _validate_mlx_jsonl(data_dir / "valid.jsonl")
    except Exception as exc:
        print(f"Error validating MLX corpus: {exc}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.base_model,
        "--data",
        str(data_dir),
        "--train",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.lr),
        "--num-layers",
        str(args.lora_layers),
        "--adapter-path",
        args.adapter_path,
    ]
    print(f"Validated MLX corpus: train={n_train}, valid={n_valid} in {data_dir}")
    if args.dry_run:
        print("[dry-run] Would run:", " ".join(cmd))
        print(
            "[dry-run] Serve after training:",
            f"mlx_lm.server --model {args.base_model} --adapter-path {args.adapter_path} --port 8080",
        )
        return 0

    try:
        import mlx_lm  # type: ignore[import-not-found] # noqa: F401
    except ImportError as exc:
        print(f"mlx-lm not installed ({exc}). Install via: pip install mlx-lm", file=sys.stderr)
        return 3

    res = subprocess.run(cmd, check=False)
    if res.returncode == 0:
        print(
            f"Saved adapter to {args.adapter_path}. Serve with:\n"
            f"  mlx_lm.server --model {args.base_model} --adapter-path {args.adapter_path} --port 8080"
        )
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
