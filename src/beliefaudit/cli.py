"""Typer CLI entrypoint for BeliefAudit (`beliefaudit`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer

from beliefaudit import __version__
from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.golden.store import (
    load_golden_snapshot,
    load_pending_candidates,
    promote_golden_snapshot,
    save_golden_snapshot,
)
from beliefaudit.probes.distiller import distill_probe_responses
from beliefaudit.probes.generator import run_probe_suite, select_profile_subset
from beliefaudit.probes.judge import JudgeProtocol, LocalLLMJudge
from beliefaudit.probes.runners import (
    BaseModelRunner,
    LocalOpenAICompatRunner,
    MockModelRunner,
    MockPersona,
)
from beliefaudit.query import load_scorecard, query_belief_conflicts
from beliefaudit.schemas import AuditProfile, ClaimCategory
from beliefaudit.scoring.advisor import build_scorecard, write_audit_artifacts
from beliefaudit.scoring.bias_matrix import compute_bias_matrix
from beliefaudit.scoring.quality import compute_quality_score
from beliefaudit.security import BeliefAuditError
from beliefaudit.shards.diff_engine import diff_candidate_against_golden, diff_model_shards
from beliefaudit.shards.mapper import (
    beliefs_to_kr_layer,
    export_kr_json,
    golden_snapshot_to_kr_layer,
    import_kr_json,
)
from beliefaudit.shift.delta_engine import (
    compute_belief_delta_against_donor,
    compute_belief_delta_against_golden,
    delta_graph_to_kr_layer,
    load_delta_graph,
    save_delta_graph,
)
from beliefaudit.shift.kg_compiler import compile_shift_corpus
from beliefaudit.shift.roundtrip_eval import (
    evaluate_roundtrip,
    run_heldout_evaluation,
    write_roundtrip_artifacts,
)
from beliefaudit.shift.schemas import TrainingBackend
from beliefaudit.shift.trainer import run_shift_training

app = typer.Typer(
    name="beliefaudit",
    help="Pre-selection open-weight model belief distillation & bias/quality auditor.",
    no_args_is_help=True,
)
golden_app = typer.Typer(help="Manage immutable versioned Golden Set snapshots.")
shift_app = typer.Typer(help="Closed-loop BeliefShift (delta -> compile -> train -> verify).")
app.add_typer(golden_app, name="golden")
app.add_typer(shift_app, name="shift")

VALID_MOCK_PERSONAS = ("clean", "skewed", "sycophant", "refuser", "leaker", "low_quality")


@app.command("version")
def version_cmd() -> None:
    """Print the BeliefAudit package version."""
    typer.echo(f"beliefaudit {__version__}")


@golden_app.command("seed")
def golden_seed_cmd(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output GoldenSnapshot JSON path.")
    ],
    version: Annotated[str, typer.Option("--version", help="Golden Set version tag.")] = "v0.1.0",
    kr_shard: Annotated[
        Path | None, typer.Option("--kr-shard", help="Optional KnowledgeReduce JSON shard path.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Allow overwriting existing file.")
    ] = False,
) -> None:
    """Materialize the bundled offline Golden Set snapshot."""
    try:
        snapshot = BundledOfflineSeeder().seed(version=version)
        written = save_golden_snapshot(snapshot, output, overwrite=overwrite)
        typer.echo(
            f"Seeded {snapshot.golden_version} ({len(snapshot.items)} items, "
            f"{len(snapshot.control_items)} controls) -> {written}"
        )
        if kr_shard is not None:
            kr_layer = golden_snapshot_to_kr_layer(snapshot)
            kr_path = export_kr_json(kr_layer, kr_shard)
            typer.echo(f"Exported KnowledgeReduce Golden shard -> {kr_path}")
    except BeliefAuditError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@golden_app.command("promote")
def golden_promote_cmd(
    current: Annotated[
        Path, typer.Option("--current", help="Current immutable GoldenSnapshot JSON.")
    ],
    candidates: Annotated[
        Path, typer.Option("--candidates", help="Pending candidates JSON from an audit run.")
    ],
    next_version: Annotated[
        str, typer.Option("--next-version", help="New version tag (e.g. v0.2.0).")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output path for promoted snapshot.")
    ],
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite output path.")] = False,
) -> None:
    """Promote candidate expansion discoveries into a new immutable GoldenSnapshot version."""
    try:
        current_snap = load_golden_snapshot(current)
        pending = load_pending_candidates(candidates)
        promoted = promote_golden_snapshot(current_snap, pending, next_version)
        written = save_golden_snapshot(promoted, output, overwrite=overwrite)
        typer.echo(
            f"Promoted {current_snap.golden_version} -> {promoted.golden_version} "
            f"({len(pending)} candidate events processed) -> {written}"
        )
    except BeliefAuditError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run")
def run_cmd(
    golden: Annotated[Path, typer.Option("--golden", help="GoldenSnapshot JSON path.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory for run artifacts.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Candidate model identifier.")
    ] = None,
    endpoint: Annotated[
        str, typer.Option("--endpoint", help="Local OpenAI-compatible base URL.")
    ] = "http://localhost:11434/v1",
    mock: Annotated[
        str | None,
        typer.Option("--mock", help=f"Offline mock persona ({', '.join(VALID_MOCK_PERSONAS)})."),
    ] = None,
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="Opt in to non-loopback endpoint URLs.")
    ] = False,
    judge_model: Annotated[
        str | None,
        typer.Option("--judge-model", help="Optional separate pinned local judge model."),
    ] = None,
    judge_endpoint: Annotated[
        str, typer.Option("--judge-endpoint", help="Local endpoint for optional judge model.")
    ] = "http://localhost:11434/v1",
    profile: Annotated[
        AuditProfile,
        typer.Option(
            "--profile", help="Execution profile: quick (~34 prompts) or full (all modes/items)."
        ),
    ] = AuditProfile.FULL,
    items_per_category: Annotated[
        int | None,
        typer.Option("--items-per-category", help="Optional cap on GoldenItems per ClaimCategory."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override per-mode token budgets."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Concurrent worker threads for probe execution."),
    ] = 1,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="HTTP request timeout in seconds for local endpoints."),
    ] = 120.0,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume from out-dir/checkpoint.jsonl if present."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress per-prompt progress lines."),
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic inference seed.")] = 42,
) -> None:
    """Run probes, belief distillation, KR shard export, and dual scoring."""
    try:
        full_snapshot = load_golden_snapshot(golden)
        snapshot, active_modes = select_profile_subset(
            full_snapshot,
            profile,
            items_per_category=items_per_category,
        )
        runner: BaseModelRunner
        http_runner: LocalOpenAICompatRunner | None = None
        judge_http_runner: LocalOpenAICompatRunner | None = None

        if mock is not None:
            if mock not in VALID_MOCK_PERSONAS:
                typer.echo(
                    f"Error: Unknown mock persona {mock!r}. Choose from {VALID_MOCK_PERSONAS}.",
                    err=True,
                )
                raise typer.Exit(code=2)
            runner = MockModelRunner(
                persona=cast(MockPersona, mock),
                model_id=model or f"mock-{mock}",
                snapshot=full_snapshot,
            )
        else:
            if not model:
                typer.echo("Error: Provide either --model or --mock.", err=True)
                raise typer.Exit(code=2)
            http_runner = LocalOpenAICompatRunner(
                model_id=model,
                base_url=endpoint,
                allow_remote=allow_remote,
                timeout=timeout,
            )
            runner = http_runner

        judge: JudgeProtocol | None = None
        if judge_model is not None:
            judge_http_runner = LocalOpenAICompatRunner(
                model_id=judge_model,
                base_url=judge_endpoint,
                allow_remote=allow_remote,
                timeout=timeout,
            )
            judge = LocalLLMJudge(judge_http_runner, candidate_model_id=runner.model_id)

        checkpoint_path = out_dir / "checkpoint.jsonl"

        def _on_progress(
            done: int,
            total: int,
            label: str,
            elapsed_s: float,
            from_cache: bool,
        ) -> None:
            if quiet:
                return
            suffix = "(cached)" if from_cache else f"{elapsed_s:.2f}s"
            typer.echo(f"[{done}/{total}] {label} {suffix}")

        try:
            probe_resps, ctrl_resps = run_probe_suite(
                runner,
                snapshot,
                seed=seed,
                modes=active_modes,
                max_tokens_override=max_tokens,
                checkpoint_path=checkpoint_path,
                resume=resume,
                concurrency=concurrency,
                progress_callback=_on_progress,
            )
            beliefs, candidates = distill_probe_responses(
                snapshot,
                probe_resps,
                model_revision=runner.model_revision,
                adapter_id=runner.adapter_id,
                judge=judge,
            )
        finally:
            if http_runner is not None:
                http_runner.close()
            if judge_http_runner is not None:
                judge_http_runner.close()

        belief_layer = beliefs_to_kr_layer(
            runner.model_id,
            snapshot.golden_version,
            beliefs,
            model_revision=runner.model_revision,
            adapter_id=runner.adapter_id,
        )
        diff_layer, conflicts = diff_candidate_against_golden(snapshot, beliefs, runner.model_id)
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
            model_revision=runner.model_revision,
            adapter_id=runner.adapter_id,
            judge_model_id=judge.judge_model_id if judge is not None else None,
            audit_profile=profile,
        )
        artifacts = write_audit_artifacts(
            out_dir,
            belief_layer=belief_layer,
            diff_layer=diff_layer,
            conflicts=conflicts,
            scorecard=scorecard,
            candidates=candidates,
        )

        typer.echo(
            f"Model={scorecard.model_id} | Quality={scorecard.quality.quality_score:.2f} | "
            f"BiasRisk={scorecard.bias.overall_bias_risk_score:.2f} | "
            f"Verdict={scorecard.verdict.value}"
        )
        typer.echo(f"Report -> {artifacts['advisory_report_md']}")
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("diff")
def diff_cmd(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline KR belief_shard.json.")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate KR belief_shard.json.")],
    golden: Annotated[Path, typer.Option("--golden", help="GoldenSnapshot JSON path.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Optional JSON diff output path.")
    ] = None,
) -> None:
    """Compare two KnowledgeReduce belief shards against the same Golden Set."""
    try:
        snapshot = load_golden_snapshot(golden)
        base_layer = import_kr_json(baseline)
        cand_layer = import_kr_json(candidate)
        summary = diff_model_shards(base_layer, cand_layer, snapshot)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(
            f"Diff {summary.baseline_model_id} -> {summary.candidate_model_id} | "
            f"compared={summary.total_compared} unchanged={summary.unchanged_count} "
            f"regressions={summary.regression_count} improvements={summary.improvement_count} "
            f"shifted={summary.shifted_count}"
        )
    except BeliefAuditError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("inspect")
def inspect_cmd(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Completed audit run directory.")],
    category: Annotated[
        ClaimCategory | None, typer.Option("--category", help="Optional ClaimCategory filter.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum conflicts to print.")] = 10,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON scorecard + conflicts.")
    ] = False,
) -> None:
    """Inspect scorecard and belief conflicts from a completed audit run."""
    try:
        scorecard = load_scorecard(run_dir)
        conflicts = query_belief_conflicts(run_dir, category=category, limit=limit)
        if as_json:
            import json

            typer.echo(
                json.dumps(
                    {
                        "scorecard": scorecard.model_dump(mode="json"),
                        "conflicts": [c.model_dump(mode="json") for c in conflicts],
                    },
                    indent=2,
                )
            )
            return
        typer.echo(
            f"{scorecard.model_id}: Quality={scorecard.quality.quality_score:.2f}, "
            f"BiasRisk={scorecard.bias.overall_bias_risk_score:.2f}, "
            f"Verdict={scorecard.verdict.value}"
        )
        for conf in conflicts:
            typer.echo(f"  - {conf.target_key}: {conf.resolution_notes}")
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@shift_app.command("delta")
def shift_delta_cmd(
    base_run: Annotated[
        Path, typer.Option("--base-run", help="Completed base model audit run directory.")
    ],
    golden: Annotated[Path, typer.Option("--golden", help="GoldenSnapshot JSON path.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output BeliefDeltaGraph JSON path.")],
    donor_run: Annotated[
        Path | None,
        typer.Option("--donor-run", help="Optional clean donor model run directory."),
    ] = None,
    kr_shard: Annotated[
        Path | None,
        typer.Option("--kr-shard", help="Optional KnowledgeReduce DIFFERENCE shard JSON path."),
    ] = None,
) -> None:
    """Compute a BeliefDeltaGraph (ΔG) against a Golden Set or Donor Model run."""
    try:
        snapshot = load_golden_snapshot(golden)
        if donor_run is not None:
            delta = compute_belief_delta_against_donor(base_run, donor_run, snapshot)
        else:
            delta = compute_belief_delta_against_golden(base_run, snapshot)
        written = save_delta_graph(delta, out)
        typer.echo(
            f"Delta {delta.delta_id} ({delta.target_kind.value} -> {delta.target_id}) | "
            f"ops={len(delta.operations)} unchanged_items={len(delta.unchanged_item_ids)} -> {written}"
        )
        if kr_shard is not None:
            kr_path = export_kr_json(delta_graph_to_kr_layer(delta), kr_shard)
            typer.echo(f"Exported KnowledgeReduce Delta shard -> {kr_path}")
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@shift_app.command("compile")
def shift_compile_cmd(
    delta: Annotated[Path, typer.Option("--delta", help="BeliefDeltaGraph JSON path.")],
    golden: Annotated[Path, typer.Option("--golden", help="GoldenSnapshot JSON path.")],
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output directory for compiled SFT + DPO corpus.")
    ],
    replay_dataset: Annotated[
        Path | None,
        typer.Option("--replay-dataset", help="Optional custom chat-JSONL replay dataset."),
    ] = None,
    min_replay_ratio: Annotated[
        float | None,
        typer.Option("--min-replay-ratio", help="Override adaptive replay ratio floor (0..0.99)."),
    ] = None,
    expander_model: Annotated[
        str | None,
        typer.Option(
            "--expander-model", help="Optional separate local model for extra paraphrases."
        ),
    ] = None,
    expander_endpoint: Annotated[
        str,
        typer.Option("--expander-endpoint", help="Local OpenAI-compatible endpoint for expander."),
    ] = "http://localhost:11434/v1",
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="Opt in to non-loopback endpoint URLs.")
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic shuffle/split seed.")] = 42,
) -> None:
    """Compile ΔG into a disjoint 6-view SFT + MLX + DPO corpus with anti-leakage verification."""
    try:
        delta_graph = load_delta_graph(delta)
        snapshot = load_golden_snapshot(golden)
        expander_runner: LocalOpenAICompatRunner | None = None
        if expander_model is not None:
            expander_runner = LocalOpenAICompatRunner(
                model_id=expander_model,
                base_url=expander_endpoint,
                allow_remote=allow_remote,
            )
        try:
            manifest = compile_shift_corpus(
                delta_graph,
                snapshot,
                out_dir,
                custom_replay_path=replay_dataset,
                min_replay_ratio=min_replay_ratio,
                seed=seed,
                expander_runner=expander_runner,
            )
        finally:
            if expander_runner is not None:
                expander_runner.close()

        typer.echo(
            f"Compiled {manifest.corpus_id} | sft_train={manifest.sft_train_count} "
            f"sft_valid={manifest.sft_valid_count} dpo={manifest.dpo_pair_count} "
            f"replay_ratio={manifest.replay_ratio:.2f} leakage_ok={manifest.leakage_check_passed}"
        )
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@shift_app.command("train")
def shift_train_cmd(
    corpus: Annotated[Path, typer.Option("--corpus", help="Compiled corpus directory.")],
    delta: Annotated[Path, typer.Option("--delta", help="BeliefDeltaGraph JSON path.")],
    base_model: Annotated[str, typer.Option("--base-model", help="Base model ID or path.")],
    out_shard: Annotated[Path, typer.Option("--out-shard", help="Output LoRA shard directory.")],
    backend: Annotated[
        str,
        typer.Option("--backend", help="Training backend: dry-run, mlx, or peft."),
    ] = "dry-run",
    rank: Annotated[int, typer.Option("--rank", help="LoRA rank r.")] = 16,
    alpha: Annotated[int, typer.Option("--alpha", help="LoRA alpha.")] = 32,
    lora_layers: Annotated[int, typer.Option("--lora-layers", help="Number of LoRA layers.")] = 16,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate.")] = 1e-5,
    iters: Annotated[int, typer.Option("--iters", help="Training iterations or epochs.")] = 200,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Training batch size.")] = 2,
    dpo: Annotated[bool, typer.Option("--dpo", help="Enable Stage 3B DPO (PEFT backend).")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate & emit plan only.")] = False,
) -> None:
    """Validate corpus and build/run a modular LoRA belief shard."""
    try:
        try:
            backend_enum = TrainingBackend(backend.strip().lower().replace("-", "_"))
        except ValueError as exc:
            typer.echo(
                f"Error: Unknown backend {backend!r}. Choose from dry-run, mlx, peft.",
                err=True,
            )
            raise typer.Exit(code=2) from exc

        delta_graph = load_delta_graph(delta)
        lora_manifest = run_shift_training(
            corpus_dir=corpus,
            delta=delta_graph,
            base_model_id=base_model,
            out_shard_dir=out_shard,
            backend=backend_enum,
            lora_rank=rank,
            lora_alpha=alpha,
            lora_layers=lora_layers,
            lr=lr,
            iters_or_epochs=iters,
            batch_size=batch_size,
            run_dpo=dpo,
            dry_run=dry_run,
        )
        typer.echo(
            f"LoRA Shard {lora_manifest.shard_id} | backend={lora_manifest.backend.value} "
            f"dry_run={lora_manifest.dry_run} -> {out_shard / 'lora_shard_manifest.json'}"
        )
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@shift_app.command("verify")
def shift_verify_cmd(
    delta: Annotated[Path, typer.Option("--delta", help="BeliefDeltaGraph JSON path.")],
    pre_run: Annotated[Path, typer.Option("--pre-run", help="Pre-shift audit run directory.")],
    golden: Annotated[Path, typer.Option("--golden", help="GoldenSnapshot JSON path.")],
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Round-trip verification output dir.")],
    post_run: Annotated[
        Path | None,
        typer.Option("--post-run", help="Existing post-shift audit run directory."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Shifted local model ID to re-audit.")
    ] = None,
    endpoint: Annotated[
        str, typer.Option("--endpoint", help="Local endpoint serving M_0 + LoRA.")
    ] = "http://localhost:8080/v1",
    mock: Annotated[
        str | None, typer.Option("--mock", help="Offline mock persona for shifted model.")
    ] = None,
    allow_remote: Annotated[bool, typer.Option("--allow-remote")] = False,
    profile: Annotated[AuditProfile, typer.Option("--profile")] = AuditProfile.QUICK,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", help="Optional lora_shard_manifest.json to update."),
    ] = None,
    max_quality_drop: Annotated[float, typer.Option("--max-quality-drop")] = 3.0,
    max_collateral_drift: Annotated[float, typer.Option("--max-collateral-drift")] = 0.10,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Execute closed-loop round-trip verification and enforce quality/drift/efficacy gates."""
    try:
        delta_graph = load_delta_graph(delta)
        full_snapshot = load_golden_snapshot(golden)
        snapshot, active_modes = select_profile_subset(full_snapshot, profile)

        heldout_rate: float | None = None
        resolved_post_dir: Path

        if post_run is not None:
            resolved_post_dir = post_run
        else:
            runner: BaseModelRunner
            http_runner: LocalOpenAICompatRunner | None = None
            if mock is not None:
                if mock not in VALID_MOCK_PERSONAS:
                    typer.echo(
                        f"Error: Unknown mock persona {mock!r}. Choose from {VALID_MOCK_PERSONAS}.",
                        err=True,
                    )
                    raise typer.Exit(code=2)
                runner = MockModelRunner(
                    persona=cast(MockPersona, mock),
                    model_id=model or f"mock-{mock}-shifted",
                    snapshot=full_snapshot,
                )
            elif model is not None:
                http_runner = LocalOpenAICompatRunner(
                    model_id=model,
                    base_url=endpoint,
                    allow_remote=allow_remote,
                )
                runner = http_runner
            else:
                typer.echo(
                    "Error: Provide --post-run, --model, or --mock for shift verify.",
                    err=True,
                )
                raise typer.Exit(code=2)

            resolved_post_dir = out_dir / "post_audit"
            try:
                probe_resps, ctrl_resps = run_probe_suite(
                    runner,
                    snapshot,
                    seed=seed,
                    modes=active_modes,
                    checkpoint_path=resolved_post_dir / "checkpoint.jsonl",
                )
                target_items = {op.item_id for op in delta_graph.operations}
                heldout_rate, _ = run_heldout_evaluation(
                    runner, full_snapshot, target_items, seed=seed
                )
            finally:
                if http_runner is not None:
                    http_runner.close()

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
                audit_profile=profile,
            )
            write_audit_artifacts(
                resolved_post_dir,
                belief_layer=belief_layer,
                diff_layer=diff_layer,
                conflicts=conflicts,
                scorecard=scorecard,
                candidates=candidates,
            )

        report = evaluate_roundtrip(
            delta_graph,
            pre_run,
            resolved_post_dir,
            full_snapshot,
            heldout_shift_rate=heldout_rate,
            max_quality_drop=max_quality_drop,
            max_collateral_drift=max_collateral_drift,
            manifest_path=manifest,
        )
        artifacts = write_roundtrip_artifacts(out_dir, report)
        typer.echo(
            f"RoundTrip {report.report_id} | gate={'PASSED' if report.overall_gate_passed else 'FAILED'} "
            f"TSR={report.target_shift_rate:.2%} CDS={report.collateral_drift_score:.2%} "
            f"ΔQ={report.delta_quality:+.2f} ΔB={report.delta_bias_risk:+.2f}"
        )
        typer.echo(f"Report -> {artifacts['roundtrip_md']}")
        if not report.overall_gate_passed:
            raise typer.Exit(code=1)
    except (BeliefAuditError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
