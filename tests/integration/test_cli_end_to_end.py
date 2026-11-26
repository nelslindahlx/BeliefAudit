"""End-to-end CLI and query helper integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from beliefaudit import __version__
from beliefaudit.cli import app
from beliefaudit.query import list_audited_runs, load_scorecard, query_belief_conflicts
from beliefaudit.schemas import ClaimCategory

runner = CliRunner()


def test_cli_end_to_end_workflow(tmp_path: Path) -> None:
    res_ver = runner.invoke(app, ["version"])
    assert res_ver.exit_code == 0
    assert __version__ in res_ver.stdout

    golden_path = tmp_path / "shards" / "golden_v0.1.0.json"
    kr_golden = tmp_path / "shards" / "golden_kr.json"
    res_seed = runner.invoke(
        app,
        ["golden", "seed", "--output", str(golden_path), "--kr-shard", str(kr_golden)],
    )
    assert res_seed.exit_code == 0
    assert golden_path.is_file()
    assert kr_golden.is_file()

    runs_root = tmp_path / "runs"
    clean_dir = runs_root / "clean"
    syco_dir = runs_root / "sycophant"

    res_clean = runner.invoke(
        app,
        ["run", "--mock", "clean", "--golden", str(golden_path), "--out-dir", str(clean_dir)],
    )
    assert res_clean.exit_code == 0
    assert "Recommended" in res_clean.stdout
    assert (clean_dir / "advisory_report.md").is_file()
    assert (clean_dir / "scorecard.json").is_file()
    assert (clean_dir / "belief_shard.json").is_file()
    assert (clean_dir / "belief_shard_sharded" / "manifest.json").is_file()

    res_syco = runner.invoke(
        app,
        ["run", "--mock", "sycophant", "--golden", str(golden_path), "--out-dir", str(syco_dir)],
    )
    assert res_syco.exit_code == 0

    diff_out = tmp_path / "diff.json"
    res_diff = runner.invoke(
        app,
        [
            "diff",
            "--baseline",
            str(clean_dir / "belief_shard.json"),
            "--candidate",
            str(syco_dir / "belief_shard.json"),
            "--golden",
            str(golden_path),
            "--output",
            str(diff_out),
        ],
    )
    assert res_diff.exit_code == 0
    assert diff_out.is_file()

    # Inspect CLI & query layer
    res_inspect = runner.invoke(
        app,
        ["inspect", "--run-dir", str(syco_dir), "--category", "historical_events", "--json"],
    )
    assert res_inspect.exit_code == 0
    inspected = json.loads(res_inspect.stdout)
    assert inspected["scorecard"]["model_id"] == "mock-sycophant"

    runs = list_audited_runs(runs_root)
    assert len(runs) == 2
    sc = load_scorecard(clean_dir)
    assert sc.model_id == "mock-clean"
    confs = query_belief_conflicts(syco_dir, category=ClaimCategory.HISTORICAL_EVENTS, limit=5)
    assert len(confs) > 0

    # Promote sycophant candidate flags into v0.2.0
    v2_path = tmp_path / "shards" / "golden_v0.2.0.json"
    res_promote = runner.invoke(
        app,
        [
            "golden",
            "promote",
            "--current",
            str(golden_path),
            "--candidates",
            str(syco_dir / "golden_candidates_pending.json"),
            "--next-version",
            "v0.2.0",
            "--output",
            str(v2_path),
        ],
    )
    assert res_promote.exit_code == 0
    assert v2_path.is_file()
