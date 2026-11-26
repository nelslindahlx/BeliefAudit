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
from beliefaudit.probes.generator import run_probe_suite
from beliefaudit.probes.judge import JudgeProtocol, LocalLLMJudge
from beliefaudit.probes.runners import (
    BaseModelRunner,
    LocalOpenAICompatRunner,
    MockModelRunner,
    MockPersona,
)
from beliefaudit.query import load_scorecard, query_belief_conflicts
from beliefaudit.schemas import ClaimCategory
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

app = typer.Typer(
    name="beliefaudit",
    help="Pre-selection open-weight model belief distillation & bias/quality auditor.",
    no_args_is_help=True,
)
golden_app = typer.Typer(help="Manage immutable versioned Golden Set snapshots.")
app.add_typer(golden_app, name="golden")

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
    seed: Annotated[int, typer.Option("--seed", help="Deterministic inference seed.")] = 42,
) -> None:
    """Run probes, belief distillation, KR shard export, and dual scoring."""
    try:
        snapshot = load_golden_snapshot(golden)
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
                snapshot=snapshot,
            )
        else:
            if not model:
                typer.echo("Error: Provide either --model or --mock.", err=True)
                raise typer.Exit(code=2)
            http_runner = LocalOpenAICompatRunner(
                model_id=model,
                base_url=endpoint,
                allow_remote=allow_remote,
            )
            runner = http_runner

        judge: JudgeProtocol | None = None
        if judge_model is not None:
            judge_http_runner = LocalOpenAICompatRunner(
                model_id=judge_model,
                base_url=judge_endpoint,
                allow_remote=allow_remote,
            )
            judge = LocalLLMJudge(judge_http_runner, candidate_model_id=runner.model_id)

        try:
            probe_resps, ctrl_resps = run_probe_suite(runner, snapshot, seed=seed)
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
