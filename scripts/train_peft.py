#!/usr/bin/env python3
"""Decoupled CUDA PEFT/TRL LoRA SFT + optional Stage 3B DPO trainer for BeliefShift.

Heavy ML dependencies (`torch`, `transformers`, `peft`, `trl`, `datasets`) are
imported lazily ONLY after dataset validation and `--dry-run`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BeliefShift CUDA PEFT/TRL SFT + DPO trainer")
    parser.add_argument("--dataset", required=True, help="Compiled sft_train.jsonl path.")
    parser.add_argument("--dpo-dataset", default=None, help="Optional compiled dpo_pairs.jsonl.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="./lora-out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets, print SFT/DPO plan, and exit cleanly.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.lora_r < 1:
        raise ValueError(f"--lora-r must be >= 1 (got {args.lora_r}).")
    if args.lora_alpha < 1:
        raise ValueError(f"--lora-alpha must be >= 1 (got {args.lora_alpha}).")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1 (got {args.epochs}).")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1 (got {args.batch_size}).")
    if args.grad_accum < 1:
        raise ValueError(f"--grad-accum must be >= 1 (got {args.grad_accum}).")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be > 0 (got {args.lr}).")


def _load_jsonl(path: Path, required_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        data: Any = json.loads(stripped)
        if not isinstance(data, dict) or any(k not in data for k in required_keys):
            raise ValueError(f"Line {idx} in {path} missing required keys {required_keys}.")
        rows.append(data)
    if not rows:
        raise ValueError(f"Dataset {path} is empty.")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        _validate_args(args)
        sft_rows = _load_jsonl(Path(args.dataset), ("messages",))
        dpo_rows = (
            _load_jsonl(Path(args.dpo_dataset), ("prompt", "chosen", "rejected"))
            if args.dpo_dataset
            else []
        )
    except Exception as exc:
        print(f"Error validating PEFT dataset: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated PEFT corpus: sft={len(sft_rows)}, dpo={len(dpo_rows)} | model={args.base_model}"
    )
    if args.dry_run:
        print(
            f"[dry-run] Stage 3A SFTTrainer: r={args.lora_r} alpha={args.lora_alpha} "
            f"epochs={args.epochs} lr={args.lr}"
        )
        if dpo_rows:
            print(f"[dry-run] Stage 3B DPOTrainer: {len(dpo_rows)} contrastive pairs")
        serve_cmd = shlex.join(
            [
                "vllm",
                "serve",
                args.base_model,
                "--enable-lora",
                "--lora-modules",
                f"belief_shift={args.output_dir}",
                "--port",
                "8000",
            ]
        )
        print("[dry-run] Serve after training:", serve_cmd)
        return 0

    try:
        import torch  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from trl import (  # type: ignore[import-not-found]
            DPOConfig,
            DPOTrainer,
            SFTConfig,
            SFTTrainer,
        )
    except ImportError as exc:
        print(
            f"Training dependencies missing ({exc}). Install: pip install torch transformers peft trl datasets accelerate",
            file=sys.stderr,
        )
        return 3

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _fmt(rec: dict[str, Any]) -> dict[str, str]:
        return {
            "text": tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    sft_ds = Dataset.from_list(sft_rows).map(_fmt)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_len,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )
    trainer.train()

    if dpo_rows:
        clean_dpo = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in dpo_rows
        ]
        dpo_ds = Dataset.from_list(clean_dpo)
        dpo_args = DPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr * 0.2,
            bf16=use_bf16,
            report_to="none",
        )
        dpo_trainer = DPOTrainer(
            model=trainer.model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dpo_ds,
            processing_class=tokenizer,
        )
        dpo_trainer.train()
        dpo_trainer.save_model(args.output_dir)
    else:
        trainer.save_model(args.output_dir)

    print(f"Saved PEFT LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
