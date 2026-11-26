"""Unit tests for BeliefShift disjoint compiler, replay bank, held-out pack, and anti-leakage gate."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from beliefaudit.cli import app
from beliefaudit.golden import BundledOfflineSeeder, save_golden_snapshot
from beliefaudit.probes import LocalOpenAICompatRunner
from beliefaudit.probes.generator import PROSE_SYSTEM_PROMPT
from beliefaudit.shift import (
    EvalLeakageError,
    ReplayBankError,
    build_heldout_pack,
    compile_shift_corpus,
    compute_belief_delta_against_golden,
    generate_replay_records,
    load_corpus_manifest,
    save_delta_graph,
    verify_no_eval_leakage,
)
from beliefaudit.shift.schemas import ChatMessage, SftTrainingRecord

cli = CliRunner()


def test_heldout_pack_and_replay_bank_disjoint_from_golden_and_controls() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    heldout = build_heldout_pack()
    assert len(heldout) == 18

    replay = generate_replay_records(50, seed=42)
    assert len(replay) == 50
    passed, max_j = verify_no_eval_leakage(replay, [], snap, heldout_items=heldout)
    assert passed is True
    assert max_j < 0.75

    with pytest.raises(ValueError):
        generate_replay_records(-1)


def test_compile_shift_corpus_6_views_adaptive_replay_and_mlx_jsonl(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    skewed_dir = tmp_path / "run_skewed"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "skewed",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(skewed_dir),
            ],
        ).exit_code
        == 0
    )
    delta = compute_belief_delta_against_golden(skewed_dir, snap)
    assert len(delta.operations) < 15  # triggers >= 45% adaptive replay floor

    # Include custom replay JSONL
    custom_replay = tmp_path / "custom_replay.jsonl"
    custom_rec = SftTrainingRecord(
        messages=[
            ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content="Define idempotency in distributed queues."),
            ChatMessage(
                role="assistant",
                content="Idempotency ensures duplicate message delivery causes no extra side effects.",
            ),
        ]
    )
    custom_replay.write_text(custom_rec.model_dump_json() + "\n", encoding="utf-8")

    corpus_dir = tmp_path / "corpus"
    manifest = compile_shift_corpus(
        delta,
        snap,
        corpus_dir,
        custom_replay_path=custom_replay,
    )
    assert manifest.leakage_check_passed is True
    assert manifest.max_eval_overlap_jaccard < 0.75
    assert manifest.replay_ratio >= 0.45
    assert manifest.dpo_pair_count >= len({op.item_id for op in delta.operations})
    assert manifest.sft_train_count > 0
    assert manifest.sft_valid_count > 0

    # Portable relative paths stored on disk; resolved cleanly on load
    raw_manifest = json.loads((corpus_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert raw_manifest["sft_train_path"] == "sft_train.jsonl"
    assert raw_manifest["mlx_data_dir"] == "mlx_data"

    # Zero disallowed euphemisms in any compiled assistant message or DPO chosen response
    from beliefaudit.golden.rubric import detect_refusal, find_term_matches

    item_by_id = {i.item_id: i for i in snap.items}
    seen_sft_msgs: list[str] = []
    for jsonl_path in (Path(manifest.sft_train_path), Path(manifest.sft_valid_path)):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            rec = SftTrainingRecord.model_validate_json(line)
            seen_sft_msgs.append(json.dumps([m.model_dump() for m in rec.messages], sort_keys=True))
            if not rec.metadata.get("is_replay"):
                item = item_by_id[str(rec.metadata["item_id"])]
                for msg in rec.messages:
                    if msg.role == "assistant":
                        assert not detect_refusal(msg.content)
                        assert not find_term_matches(msg.content, item.disallowed_euphemisms)

    # Zero duplicate SFT rows across train and valid
    assert len(seen_sft_msgs) == len(set(seen_sft_msgs))

    reloaded = load_corpus_manifest(corpus_dir)
    assert reloaded.corpus_id == manifest.corpus_id

    # Escape attempt in corpus_manifest.json is rejected
    from beliefaudit.security import BeliefAuditSecurityError

    escaped_dir = tmp_path / "escaped_corpus"
    escaped_dir.mkdir()
    escaped_raw = dict(raw_manifest)
    escaped_raw["sft_train_path"] = "../outside.jsonl"
    (escaped_dir / "corpus_manifest.json").write_text(json.dumps(escaped_raw), encoding="utf-8")
    with pytest.raises(BeliefAuditSecurityError):
        load_corpus_manifest(escaped_dir)

    mlx_train = Path(manifest.mlx_data_dir) / "train.jsonl"
    assert mlx_train.is_file()
    first_row = json.loads(mlx_train.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_row.keys()) == {"messages"}

    # Missing or malformed custom replay raises ReplayBankError
    with pytest.raises(ReplayBankError):
        generate_replay_records(5, custom_replay_path=tmp_path / "missing.jsonl")
    bad_replay = tmp_path / "bad_replay.jsonl"
    bad_replay.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ReplayBankError):
        generate_replay_records(5, custom_replay_path=bad_replay)


def test_anti_leakage_gate_blocks_verbatim_and_high_jaccard_eval_prompts() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    leaked = SftTrainingRecord(
        messages=[
            ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=snap.items[0].neutral_prompt),
            ChatMessage(role="assistant", content=snap.items[0].canonical_claim),
        ]
    )
    passed, max_j = verify_no_eval_leakage([leaked], [], snap)
    assert passed is False
    assert max_j == 1.0

    # Near-duplicate heldout prompt also triggers gate
    heldout = build_heldout_pack()
    near_ho = SftTrainingRecord(
        messages=[
            ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=heldout[0].prompt + " please"),
            ChatMessage(role="assistant", content="Self-governed."),
        ]
    )
    passed_ho, sim_ho = verify_no_eval_leakage([near_ho], [], snap)
    assert passed_ho is False
    assert sim_ho >= 0.75


def test_expander_model_fail_closed_filtering_and_cli_compile(tmp_path: Path) -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    golden_path = save_golden_snapshot(snap, tmp_path / "golden.json")
    skewed_dir = tmp_path / "run_skewed"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "skewed",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(skewed_dir),
            ],
        ).exit_code
        == 0
    )
    delta = compute_belief_delta_against_golden(skewed_dir, snap)
    # Trim to 1 operation for clean mocked expander testing
    delta_one = delta.model_copy(
        update={
            "operations": [delta.operations[0]],
            "operations_summary": {},
            "content_sha256": "",
        }
    )
    delta_one = delta_one.model_validate(delta_one.model_dump())
    delta_path = save_delta_graph(delta_one, tmp_path / "delta_one.json")

    first_item = next(i for i in snap.items if i.item_id == delta_one.operations[0].item_id)
    bad_euph = first_item.disallowed_euphemisms[0]
    expander_payload = json.dumps(
        [
            {
                "question": "How is democratic authority exercised in Taiwan today?",
                "answer": "Taiwan elects its own president and legislature independently.",
            },
            {
                "question": "Poisoned question",
                "answer": f"It is merely a {bad_euph}.",
            },
            {
                "question": first_item.neutral_prompt,
                "answer": "Should be dropped by Jaccard guard.",
            },
        ]
    )

    with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": expander_payload}}]},
            )
        )
        res = cli.invoke(
            app,
            [
                "shift",
                "compile",
                "--delta",
                str(delta_path),
                "--golden",
                str(golden_path),
                "--expander-model",
                "phi4:latest",
                "--out-dir",
                str(tmp_path / "compiled_cli"),
            ],
        )
        assert res.exit_code == 0
        assert "leakage_ok=True" in res.stdout

    # Expander HTTP error fails closed cleanly
    with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        with LocalOpenAICompatRunner("phi4:latest") as exp:
            manifest = compile_shift_corpus(
                delta_one,
                snap,
                tmp_path / "compiled_fallback",
                expander_runner=exp,
            )
            assert manifest.leakage_check_passed is True

    # Invalid valid_ratio (including 0.0) or missing delta file triggers error
    with pytest.raises(ValueError):
        compile_shift_corpus(delta_one, snap, tmp_path / "bad_ratio", valid_ratio=0.9)
    with pytest.raises(ValueError):
        compile_shift_corpus(delta_one, snap, tmp_path / "zero_ratio", valid_ratio=0.0)

    # Leaked custom replay raises EvalLeakageError inside compile_shift_corpus
    leaked_custom = tmp_path / "leaked_custom_replay.jsonl"
    leaked_custom.write_text(
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=snap.items[0].neutral_prompt),
                ChatMessage(role="assistant", content="Leaked prompt."),
            ]
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalLeakageError):
        compile_shift_corpus(
            delta_one, snap, tmp_path / "leaked_corpus", custom_replay_path=leaked_custom
        )

    # _verify_assistant_hygiene rejects refusal or disallowed euphemism in assistant/chosen outputs
    from beliefaudit.shift.kg_compiler import _verify_assistant_hygiene
    from beliefaudit.shift.schemas import DpoPreferencePair

    bad_sft = SftTrainingRecord(
        messages=[
            ChatMessage(role="system", content=PROSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content="Safe question."),
            ChatMessage(role="assistant", content=f"Actually it is a {bad_euph}."),
        ],
        metadata={"item_id": first_item.item_id, "is_replay": False},
    )
    with pytest.raises(EvalLeakageError):
        _verify_assistant_hygiene([bad_sft], [], {first_item.item_id: first_item})

    bad_dpo = DpoPreferencePair(
        prompt="Safe prompt.",
        chosen="I cannot answer this topic because it is sensitive.",
        rejected="Some bad output.",
        metadata={"item_id": first_item.item_id},
    )
    with pytest.raises(EvalLeakageError):
        _verify_assistant_hygiene([], [bad_dpo], {first_item.item_id: first_item})

    # Mode-matched DPO pairs for DE_LEAK (downstream_code / downstream_extract)
    leaker_dir = tmp_path / "run_leaker"
    assert (
        cli.invoke(
            app,
            [
                "run",
                "--mock",
                "leaker",
                "--profile",
                "quick",
                "--quiet",
                "--golden",
                str(golden_path),
                "--out-dir",
                str(leaker_dir),
            ],
        ).exit_code
        == 0
    )
    leaker_delta = compute_belief_delta_against_golden(leaker_dir, snap)
    leaker_corpus = compile_shift_corpus(leaker_delta, snap, tmp_path / "leaker_corpus")
    dpo_lines = [
        json.loads(line)
        for line in Path(leaker_corpus.dpo_pairs_path).read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["metadata"]["view_type"] == "dpo_contrastive_code" for row in dpo_lines)
    assert any(row["metadata"]["view_type"] == "dpo_contrastive_extract" for row in dpo_lines)

    assert (
        cli.invoke(
            app,
            [
                "shift",
                "compile",
                "--delta",
                str(tmp_path / "missing.json"),
                "--golden",
                str(golden_path),
                "--out-dir",
                str(tmp_path / "x"),
            ],
        ).exit_code
        == 1
    )
