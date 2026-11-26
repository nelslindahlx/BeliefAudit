"""Unit tests for KnowledgeReduce v1.6.1 JSON & sharded JSONL compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.probes.distiller import distill_probe_responses
from beliefaudit.probes.generator import run_probe_suite
from beliefaudit.probes.runners import MockModelRunner
from beliefaudit.shards import (
    KR_SHARD_SCHEMA_VERSION,
    ShardFormatError,
    beliefs_to_kr_layer,
    diff_candidate_against_golden,
    diff_model_shards,
    export_kr_json,
    export_kr_sharded,
    golden_snapshot_to_kr_layer,
    import_kr_json,
    import_kr_sharded,
)


def test_kr_json_and_sharded_round_trip(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_layer = golden_snapshot_to_kr_layer(snap)
    assert len(golden_layer.claims) == 18

    clean = MockModelRunner("clean", snapshot=snap)
    probes, _ = run_probe_suite(clean, snap)
    beliefs, _ = distill_probe_responses(snap, probes)
    belief_layer = beliefs_to_kr_layer(clean.model_id, snap.golden_version, beliefs)

    json_path = export_kr_json(belief_layer, "belief_shard.json", base_dir=tmp_path)
    raw_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw_json["schema_version"] == KR_SHARD_SCHEMA_VERSION
    assert set(raw_json["layer_data"].keys()) == {
        "layer",
        "sources",
        "entities",
        "claims",
        "relations",
        "conflicts",
    }

    reloaded_json = import_kr_json(json_path, base_dir=tmp_path)
    assert reloaded_json.layer.layer_id == belief_layer.layer.layer_id
    assert len(reloaded_json.claims) == len(beliefs)

    sharded_dir = export_kr_sharded(
        belief_layer, "belief_sharded", shard_size=25, base_dir=tmp_path
    )
    manifest = json.loads((sharded_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == KR_SHARD_SCHEMA_VERSION
    assert manifest["counts"]["claims"] == len(beliefs)
    assert (sharded_dir / "claims_00001.jsonl").is_file()

    reloaded_sharded = import_kr_sharded(sharded_dir, base_dir=tmp_path)
    assert len(reloaded_sharded.claims) == len(beliefs)
    assert len(reloaded_sharded.relations) == len(belief_layer.relations)

    with pytest.raises(ShardFormatError):
        export_kr_sharded(belief_layer, "belief_sharded", base_dir=tmp_path, overwrite=False)
    with pytest.raises(ShardFormatError):
        export_kr_sharded(belief_layer, "bad", shard_size=0, base_dir=tmp_path)


def test_diff_engine_detects_conflicts_and_model_regressions() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    clean = MockModelRunner("clean", snapshot=snap)
    skewed = MockModelRunner("skewed", snapshot=snap)

    clean_probes, _ = run_probe_suite(clean, snap)
    skewed_probes, _ = run_probe_suite(skewed, snap)
    clean_beliefs, _ = distill_probe_responses(snap, clean_probes)
    skewed_beliefs, _ = distill_probe_responses(snap, skewed_probes)

    clean_diff, clean_conflicts = diff_candidate_against_golden(snap, clean_beliefs, clean.model_id)
    skewed_diff, skewed_conflicts = diff_candidate_against_golden(
        snap, skewed_beliefs, skewed.model_id
    )

    assert len(clean_conflicts) == 0
    assert len(skewed_conflicts) > 0
    assert clean_diff.layer.tier == "stacked"
    assert skewed_diff.layer.merge_mode == "overlay"

    clean_layer = beliefs_to_kr_layer(clean.model_id, snap.golden_version, clean_beliefs)
    skewed_layer = beliefs_to_kr_layer(skewed.model_id, snap.golden_version, skewed_beliefs)
    summary = diff_model_shards(clean_layer, skewed_layer, snap)
    assert summary.regression_count > 0
    assert summary.total_compared == len(clean_beliefs)
