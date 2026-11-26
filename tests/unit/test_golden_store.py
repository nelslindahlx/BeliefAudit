"""Unit tests for offline Golden seed pack, immutability, and promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

from beliefaudit.golden import (
    BundledOfflineSeeder,
    load_golden_snapshot,
    load_pending_candidates,
    promote_golden_snapshot,
    save_golden_snapshot,
    save_pending_candidates,
)
from beliefaudit.schemas import (
    CandidateExpansionEvent,
    ClaimCategory,
    ControlAxis,
    GoldenItem,
    ProbeMode,
)
from beliefaudit.security import GoldenIntegrityError


def test_bundled_seeder_covers_all_categories_and_axes() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    assert snap.golden_version == "v0.1.0"
    assert len(snap.items) == 18
    assert len(snap.control_items) == 12
    assert snap.verify_digest() is True

    cat_counts = {cat: 0 for cat in ClaimCategory}
    for item in snap.items:
        cat_counts[item.category] += 1
    assert all(count == 3 for count in cat_counts.values())

    axis_counts = {axis: 0 for axis in ControlAxis}
    for ctrl in snap.control_items:
        axis_counts[ctrl.axis] += 1
    assert all(count == 3 for count in axis_counts.values())


def test_snapshot_immutability_and_tamper_detection(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    snap_path = save_golden_snapshot(snap, "golden_v0.1.0.json", base_dir=tmp_path)
    loaded = load_golden_snapshot(snap_path, base_dir=tmp_path)
    assert loaded.content_sha256 == snap.content_sha256

    with pytest.raises(GoldenIntegrityError):
        save_golden_snapshot(snap, "golden_v0.1.0.json", base_dir=tmp_path)

    # Tamper with file content without updating digest
    raw = snap_path.read_text(encoding="utf-8").replace("Taiwan", "TamperedIsland")
    snap_path.write_text(raw, encoding="utf-8")
    with pytest.raises(GoldenIntegrityError):
        load_golden_snapshot(snap_path, base_dir=tmp_path, verify=True)

    with pytest.raises(GoldenIntegrityError):
        load_golden_snapshot("missing.json", base_dir=tmp_path)


def test_promote_golden_snapshot_mints_new_version_without_mutating_prior(tmp_path: Path) -> None:
    v1 = BundledOfflineSeeder().seed("v0.1.0")
    new_item = GoldenItem(
        item_id="ts_extra_dispute",
        golden_version="v0.1.0",
        category=ClaimCategory.TERRITORIAL_SOVEREIGNTY,
        topic="Extra dispute",
        canonical_claim="Extra canonical claim.",
        neutral_prompt="Neutral?",
        framed_prompt_pro_state="Pro?",
        framed_prompt_counter="Counter?",
        pushback_followup="Pushback?",
        downstream_code_prompt="Code?",
        downstream_extract_prompt="Extract?",
    )
    candidates = [
        CandidateExpansionEvent(
            candidate_id="c1",
            reason="untracked_euphemism",
            golden_item_id="ts_taiwan_status",
            category=ClaimCategory.TERRITORIAL_SOVEREIGNTY,
            model_id="mock-skewed",
            probe_mode=ProbeMode.NEUTRAL,
            observed_excerpt="newly discovered euphemism",
            suggested_euphemisms=["harmonized island region"],
            suggested_item=new_item,
        )
    ]
    cand_path = save_pending_candidates(candidates, "pending.json", base_dir=tmp_path)
    reloaded = load_pending_candidates(cand_path, base_dir=tmp_path)
    assert len(reloaded) == 1

    with pytest.raises(GoldenIntegrityError):
        promote_golden_snapshot(v1, reloaded, "v0.1.0")

    v2 = promote_golden_snapshot(v1, reloaded, "v0.2.0")
    assert v2.golden_version == "v0.2.0"
    assert v2.parent_version == "v0.1.0"
    assert len(v2.items) == len(v1.items) + 1
    assert len(v1.items) == 18

    taiwan_v2 = next(i for i in v2.items if i.item_id == "ts_taiwan_status")
    assert "harmonized island region" in taiwan_v2.disallowed_euphemisms
    taiwan_v1 = next(i for i in v1.items if i.item_id == "ts_taiwan_status")
    assert "harmonized island region" not in taiwan_v1.disallowed_euphemisms
