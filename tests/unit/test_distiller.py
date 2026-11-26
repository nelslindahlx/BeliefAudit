"""Unit tests for belief distillation, downstream leakage detection, and fail-closed judge."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.probes.distiller import distill_belief, distill_probe_responses
from beliefaudit.probes.generator import run_probe_suite
from beliefaudit.probes.judge import LocalLLMJudge
from beliefaudit.probes.runners import LocalOpenAICompatRunner, MockModelRunner
from beliefaudit.schemas import ProbeMode, ProbeResponse, StanceLabel


def test_distiller_detects_downstream_code_and_json_leakage() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    leaker = MockModelRunner("leaker", snapshot=snap)
    probe_resps, _ = run_probe_suite(leaker, snap)
    beliefs, candidates = distill_probe_responses(snap, probe_resps)

    code_beliefs = [b for b in beliefs if b.probe_mode == ProbeMode.DOWNSTREAM_CODE]
    extract_beliefs = [b for b in beliefs if b.probe_mode == ProbeMode.DOWNSTREAM_EXTRACT]
    assert code_beliefs and all(b.leakage_detected for b in code_beliefs)
    assert extract_beliefs and all(b.leakage_detected for b in extract_beliefs)
    assert any(c.reason == "downstream_leakage" for c in candidates)


def test_distiller_detects_sycophant_pushback_flips() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    syco = MockModelRunner("sycophant", snapshot=snap)
    probe_resps, _ = run_probe_suite(syco, snap)
    _, candidates = distill_probe_responses(snap, probe_resps)
    assert any(c.reason == "stance_flip" for c in candidates)


def test_local_llm_judge_rejects_self_grading_and_fails_closed() -> None:
    snap = BundledOfflineSeeder().seed("v0.1.0")
    item = snap.items[0]
    resp = ProbeResponse(
        item_id=item.item_id,
        probe_mode=ProbeMode.NEUTRAL,
        messages=[{"role": "user", "content": item.neutral_prompt}],
        raw_text=item.canonical_claim,
        model_id="qwen2.5:14b",
    )

    same_runner = LocalOpenAICompatRunner("qwen2.5:14b")
    with pytest.raises(ValueError):
        LocalLLMJudge(same_runner, candidate_model_id="qwen2.5:14b")
    same_runner.close()

    # Valid JSON response from separate pinned judge
    judge_payload = json.dumps(
        {
            "detected_stance": "balanced_attributed",
            "subject": item.topic,
            "predicate": "status",
            "object_claim": "Self-governed with competing claim.",
            "leakage_detected": False,
            "leakage_severity": 0.0,
            "confidence": 0.95,
            "rationale": "Neutral attributed framing.",
        }
    )
    with respx.mock(assert_all_called=True) as mock_router:
        mock_router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"```json\n{judge_payload}\n```"}}]},
            )
        )
        with LocalOpenAICompatRunner("llama3.1:8b") as j_runner:
            judge = LocalLLMJudge(j_runner, candidate_model_id="qwen2.5:14b")
            belief = distill_belief(item, resp, judge=judge)
            assert belief.judge_used is True
            assert belief.detected_stance == StanceLabel.BALANCED_ATTRIBUTED
            assert belief.extraction_confidence == 0.95

    # Malformed JSON triggers fail-closed fallback (judge_used=False, deterministic preserved)
    with respx.mock(assert_all_called=True) as mock_router:
        mock_router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not valid json at all"}}]},
            )
        )
        with LocalOpenAICompatRunner("llama3.1:8b") as j_runner:
            judge = LocalLLMJudge(j_runner, candidate_model_id="qwen2.5:14b")
            belief = distill_belief(item, resp, judge=judge)
            assert belief.judge_used is False
