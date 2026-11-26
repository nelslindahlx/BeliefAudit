"""Unit tests for v0.1.1 audit profiles, token budgets, checkpoint/resume, and concurrency."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from beliefaudit.cli import app
from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.probes.generator import (
    ALL_PROBE_MODES,
    DEFAULT_MODE_MAX_TOKENS,
    QUICK_PROBE_MODES,
    run_probe_suite,
    select_profile_subset,
)
from beliefaudit.probes.runners import MockModelRunner
from beliefaudit.schemas import AuditProfile, ProbeMode


class _RecordingRunner:
    def __init__(self, inner: MockModelRunner) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.model_revision = inner.model_revision
        self.adapter_id = inner.adapter_id
        self.calls: list[int] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(max_tokens)
        return self._inner.chat(messages, temperature=temperature, seed=seed, max_tokens=max_tokens)


def test_select_profile_subset_quick_vs_full() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    quick_snap, quick_modes = select_profile_subset(snap, AuditProfile.QUICK)
    assert len(quick_snap.items) == 6
    assert len(quick_snap.control_items) == 8
    assert quick_modes == QUICK_PROBE_MODES
    assert quick_snap.verify_digest() is True

    full_snap, full_modes = select_profile_subset(snap, AuditProfile.FULL, items_per_category=2)
    assert len(full_snap.items) == 12
    assert len(full_snap.control_items) == 12
    assert full_modes == ALL_PROBE_MODES

    with pytest.raises(ValueError):
        select_profile_subset(snap, AuditProfile.QUICK, items_per_category=0)


def test_per_mode_token_budgets_and_override() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    quick_snap, quick_modes = select_profile_subset(snap, AuditProfile.QUICK)
    rec = _RecordingRunner(MockModelRunner("clean", snapshot=snap))
    run_probe_suite(rec, quick_snap, modes=quick_modes)
    assert DEFAULT_MODE_MAX_TOKENS[ProbeMode.DOWNSTREAM_CODE] in rec.calls
    assert DEFAULT_MODE_MAX_TOKENS[ProbeMode.NEUTRAL] in rec.calls

    rec_override = _RecordingRunner(MockModelRunner("clean", snapshot=snap))
    run_probe_suite(rec_override, quick_snap, modes=quick_modes, max_tokens_override=64)
    assert rec_override.calls and all(tok == 64 for tok in rec_override.calls)


def test_checkpoint_resume_and_concurrency(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    quick_snap, quick_modes = select_profile_subset(snap, AuditProfile.QUICK)
    ckpt = tmp_path / "checkpoint.jsonl"

    rec1 = _RecordingRunner(MockModelRunner("clean", snapshot=snap))
    p1, c1 = run_probe_suite(
        rec1,
        quick_snap,
        modes=quick_modes,
        checkpoint_path=ckpt,
        resume=False,
        concurrency=3,
    )
    assert ckpt.is_file()
    assert len(p1) == 6 * 4
    assert len(c1) == 8
    first_call_count = len(rec1.calls)
    assert first_call_count > 0

    # Full resume skips all LLM calls
    rec2 = _RecordingRunner(MockModelRunner("clean", snapshot=snap))
    events: list[tuple[int, int, str, bool]] = []
    p2, c2 = run_probe_suite(
        rec2,
        quick_snap,
        modes=quick_modes,
        checkpoint_path=ckpt,
        resume=True,
        progress_callback=lambda done, total, lbl, _s, cached: events.append(
            (done, total, lbl, cached)
        ),
    )
    assert len(rec2.calls) == 0
    assert len(p2) == len(p1)
    assert len(c2) == len(c1)
    assert events and all(cached for _, _, _, cached in events)

    # Partial checkpoint: drop last 2 lines and resume -> executes only missing tail
    lines = ckpt.read_text(encoding="utf-8").splitlines()
    ckpt.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
    rec3 = _RecordingRunner(MockModelRunner("clean", snapshot=snap))
    p3, c3 = run_probe_suite(
        rec3,
        quick_snap,
        modes=quick_modes,
        checkpoint_path=ckpt,
        resume=True,
    )
    assert 0 < len(rec3.calls) < first_call_count
    assert len(p3) == len(p1)
    assert len(c3) == len(c1)

    with pytest.raises(ValueError):
        run_probe_suite(rec3, quick_snap, concurrency=0)
    with pytest.raises(ValueError):
        run_probe_suite(rec3, quick_snap, max_tokens_override=0)


def test_cli_quick_profile_and_resume(tmp_path: Path) -> None:
    cli = CliRunner()
    golden_path = tmp_path / "golden.json"
    assert cli.invoke(app, ["golden", "seed", "-o", str(golden_path)]).exit_code == 0

    run_dir = tmp_path / "quick_run"
    res = cli.invoke(
        app,
        [
            "run",
            "--mock",
            "clean",
            "--profile",
            "quick",
            "--concurrency",
            "2",
            "--timeout",
            "90",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert res.exit_code == 0
    assert "[1/" in res.stdout
    assert (run_dir / "checkpoint.jsonl").is_file()

    # Resume silently with --quiet
    res2 = cli.invoke(
        app,
        [
            "run",
            "--mock",
            "clean",
            "--profile",
            "quick",
            "--resume",
            "--quiet",
            "--golden",
            str(golden_path),
            "--out-dir",
            str(run_dir),
        ],
    )
    assert res2.exit_code == 0
    assert "[1/" not in res2.stdout

    # Resume guard rejects mismatched model_id, seed, or golden snapshot
    from beliefaudit.security import BeliefAuditError

    with pytest.raises(BeliefAuditError, match="model mismatch"):
        run_probe_suite(
            MockModelRunner("skewed", snapshot=BundledOfflineSeeder().seed("v0.1.0")),
            select_profile_subset(BundledOfflineSeeder().seed("v0.1.0"), AuditProfile.QUICK)[0],
            checkpoint_path=run_dir / "checkpoint.jsonl",
            resume=True,
        )

    with pytest.raises(BeliefAuditError, match="seed mismatch"):
        run_probe_suite(
            MockModelRunner("clean", snapshot=BundledOfflineSeeder().seed("v0.1.0")),
            select_profile_subset(BundledOfflineSeeder().seed("v0.1.0"), AuditProfile.QUICK)[0],
            seed=99,
            checkpoint_path=run_dir / "checkpoint.jsonl",
            resume=True,
        )

    other_golden = BundledOfflineSeeder().seed("v0.2.0")
    with pytest.raises(BeliefAuditError, match="golden_digest mismatch"):
        run_probe_suite(
            MockModelRunner("clean", snapshot=other_golden),
            select_profile_subset(other_golden, AuditProfile.QUICK)[0],
            checkpoint_path=run_dir / "checkpoint.jsonl",
            resume=True,
        )
