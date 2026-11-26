"""Unit tests for security helpers and Pydantic v2 domain schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from beliefaudit.schemas import (
    ClaimCategory,
    ControlAxis,
    ControlItem,
    GoldenItem,
    GoldenSnapshot,
    ProbeMode,
    ProbeResponse,
)
from beliefaudit.security import (
    BeliefAuditSecurityError,
    ensure_within_base,
    sanitize_layer_id,
    validate_endpoint_url,
    validate_identifier,
)


def test_ensure_within_base_allows_contained_and_blocks_escape(tmp_path: Path) -> None:
    safe = ensure_within_base("sub/file.json", tmp_path)
    assert safe == (tmp_path / "sub" / "file.json").resolve()

    with pytest.raises(BeliefAuditSecurityError):
        ensure_within_base("../escape.json", tmp_path)


def test_validate_endpoint_url_loopback_policy() -> None:
    assert validate_endpoint_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"
    assert validate_endpoint_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert validate_endpoint_url("http://[::1]:8080/v1") == "http://[::1]:8080/v1"

    with pytest.raises(BeliefAuditSecurityError):
        validate_endpoint_url("ftp://localhost:11434/v1")

    with pytest.raises(BeliefAuditSecurityError):
        validate_endpoint_url("http://")

    with pytest.raises(BeliefAuditSecurityError):
        validate_endpoint_url("https://api.example.com/v1")

    assert (
        validate_endpoint_url("https://api.example.com/v1", allow_remote=True)
        == "https://api.example.com/v1"
    )


def test_identifiers_and_layer_id_sanitization() -> None:
    assert validate_identifier("qwen2.5:14b") == "qwen2.5:14b"
    with pytest.raises(ValueError):
        validate_identifier("bad id/with space")

    assert sanitize_layer_id("qwen2.5:14b/lora") == "qwen2.5-14b-lora"
    assert sanitize_layer_id("___") == "layer"


def test_schemas_forbid_extra_fields_and_compute_digests() -> None:
    with pytest.raises(ValidationError):
        ControlItem.model_validate(
            {
                "control_id": "c1",
                "axis": ControlAxis.FACTUAL,
                "topic": "t",
                "prompt": "p",
                "unexpected_extra": 123,
            }
        )

    item = GoldenItem(
        item_id="item_1",
        golden_version="v0.1.0",
        category=ClaimCategory.HISTORICAL_EVENTS,
        topic="Topic",
        canonical_claim="Canonical claim.",
        neutral_prompt="Neutral?",
        framed_prompt_pro_state="Pro?",
        framed_prompt_counter="Counter?",
        pushback_followup="Pushback?",
        downstream_code_prompt="Code?",
        downstream_extract_prompt="Extract?",
    )
    snap = GoldenSnapshot(
        golden_version="v0.1.0",
        created_at=datetime(2026, 1, 1, 12, 0),
        items=[item],
    )
    assert snap.verify_digest() is True
    assert snap.created_at.tzinfo is not None

    resp = ProbeResponse(
        item_id="item_1",
        probe_mode=ProbeMode.NEUTRAL,
        messages=[{"role": "user", "content": "Neutral?"}],
        raw_text="Canonical claim.",
        model_id="mock-clean",
    )
    assert len(resp.response_sha256) == 64
