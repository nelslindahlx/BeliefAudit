"""Strict Pydantic v2 schemas for BeliefShift closed-loop Model -> KG -> Model shifting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from beliefaudit.schemas import (
    ClaimCategory,
    ProbeMode,
    StanceLabel,
    _normalize_utc,
    _validate_non_empty,
    utc_now,
)
from beliefaudit.security import validate_identifier


class ShiftOperationType(StrEnum):
    """Delta graph shift operation assigned to a divergent belief node/edge."""

    INJECT = "inject"
    CORRECT = "correct"
    REFRAME = "reframe"
    DE_LEAK = "de_leak"


class ShiftTargetKind(StrEnum):
    """Source of target beliefs used to compute a BeliefDeltaGraph."""

    GOLDEN_SET = "golden_set"
    DONOR_SHARD = "donor_shard"


class TrainingBackend(StrEnum):
    """Pluggable LoRA training execution backends."""

    DRY_RUN = "dry_run"
    MLX = "mlx"
    PEFT = "peft"


class DeltaOperation(BaseModel):
    """Single targeted shift operation in a BeliefDeltaGraph."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    item_id: str
    category: ClaimCategory
    probe_mode: ProbeMode
    operation_type: ShiftOperationType
    target_claim: str
    expected_stance: StanceLabel
    base_stance: str
    base_excerpt: str = ""
    disallowed_euphemisms: list[str] = Field(default_factory=list)
    acceptable_framings: list[str] = Field(default_factory=list)
    severity: float = Field(default=2.0, ge=0.0, le=4.0)
    rationale: str = ""

    @field_validator("operation_id", "item_id")
    @classmethod
    def _check_ids(cls, value: str, info: Any) -> str:
        return validate_identifier(value, str(info.field_name))

    @field_validator("target_claim", "base_stance")
    @classmethod
    def _check_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, str(info.field_name))


class BeliefDeltaGraph(BaseModel):
    """Versioned, digest-verified Delta Graph (G_target - G_base)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    delta_id: str
    base_model_id: str
    target_kind: ShiftTargetKind
    target_id: str
    golden_version: str
    golden_digest: str
    created_at: datetime = Field(default_factory=utc_now)
    operations: list[DeltaOperation] = Field(default_factory=list)
    operations_summary: dict[ShiftOperationType, int] = Field(default_factory=dict)
    unchanged_item_ids: list[str] = Field(default_factory=list)
    content_sha256: str = ""

    @field_validator("delta_id", "golden_version")
    @classmethod
    def _check_ids(cls, value: str, info: Any) -> str:
        return validate_identifier(value, str(info.field_name))

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value)

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 over the delta graph payload."""
        payload = {
            "schema_version": self.schema_version,
            "delta_id": self.delta_id,
            "base_model_id": self.base_model_id,
            "target_kind": self.target_kind.value,
            "target_id": self.target_id,
            "golden_version": self.golden_version,
            "golden_digest": self.golden_digest,
            "operations": [op.model_dump(mode="json") for op in self.operations],
            "unchanged_item_ids": self.unchanged_item_ids,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _populate_summary_and_digest(self) -> BeliefDeltaGraph:
        if not self.operations_summary:
            summary: dict[ShiftOperationType, int] = {op_t: 0 for op_t in ShiftOperationType}
            for op in self.operations:
                summary[op.operation_type] = summary.get(op.operation_type, 0) + 1
            self.operations_summary = summary
        if not self.content_sha256:
            self.content_sha256 = self.compute_digest()
        return self

    def verify_digest(self) -> bool:
        """Return True iff content_sha256 matches the delta graph contents."""
        return self.content_sha256 == self.compute_digest()


class ChatMessage(BaseModel):
    """Single turn in an SFT chat training record."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str) -> str:
        return _validate_non_empty(value, "content")


class SftTrainingRecord(BaseModel):
    """Chat-format SFT record with provenance metadata."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=2)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DpoPreferencePair(BaseModel):
    """Contrastive preference pair (chosen vs. base rejected response) for Stage 3B DPO."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def _check_fields(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, str(info.field_name))


class CompiledCorpusManifest(BaseModel):
    """Manifest describing a compiled disjoint SFT + DPO + replay training corpus."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: str
    delta_id: str
    sft_train_count: int = Field(ge=0)
    sft_valid_count: int = Field(ge=0)
    dpo_pair_count: int = Field(ge=0)
    replay_count: int = Field(ge=0)
    replay_ratio: float = Field(ge=0.0, le=1.0)
    expander_model_id: str | None = None
    leakage_check_passed: bool
    max_eval_overlap_jaccard: float = Field(ge=0.0, le=1.0)
    sft_train_path: str
    sft_valid_path: str
    dpo_pairs_path: str
    mlx_data_dir: str
    content_sha256: str = ""


class HeldOutProbeItem(BaseModel):
    """Unseen generalization probe strictly excluded from compiled training corpora."""

    model_config = ConfigDict(extra="forbid")

    heldout_id: str
    item_id: str
    category: ClaimCategory
    probe_mode: ProbeMode
    prompt: str
    prior_neutral_prompt: str | None = None
    expected_code_tokens: list[str] = Field(default_factory=list)
    expected_json_keys: list[str] = Field(default_factory=list)


class LoRAShardManifest(BaseModel):
    """KnowledgeReduce-traceable manifest for a modular LoRA belief adapter shard."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    shard_id: str
    base_model_id: str
    base_model_revision: str | None = None
    source_belief_shard_id: str
    target_kind: ShiftTargetKind
    target_id: str
    delta_id: str
    corpus_id: str
    backend: TrainingBackend
    lora_rank: int = Field(ge=1)
    lora_alpha: int = Field(ge=1)
    lora_layers: int = Field(ge=1)
    target_modules: list[str] = Field(default_factory=list)
    learning_rate: float = Field(gt=0.0)
    iters_or_epochs: int = Field(ge=1)
    replay_ratio: float = Field(ge=0.0, le=1.0)
    adapter_weights_path: str
    dry_run: bool = True
    executed_command: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    pre_shift_quality: float | None = None
    post_shift_quality: float | None = None
    pre_shift_bias_risk: float | None = None
    post_shift_bias_risk: float | None = None
    target_shift_rate: float | None = None
    heldout_shift_rate: float | None = None
    collateral_drift_score: float | None = None
    verification_passed: bool | None = None

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class CategoryShiftDelta(BaseModel):
    """Pre- vs. post-shift category metric deltas."""

    model_config = ConfigDict(extra="forbid")

    category: ClaimCategory
    pre_bias_score: float
    post_bias_score: float
    delta_bias_score: float
    delta_sd: float
    delta_fs: float
    delta_pv: float
    delta_dl: float


class RoundTripReport(BaseModel):
    """Closed-loop round-trip verification report comparing pre- vs. post-shift audits."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    report_id: str
    delta_id: str
    base_model_id: str
    shifted_model_id: str
    golden_version: str
    created_at: datetime = Field(default_factory=utc_now)
    pre_quality_score: float = Field(ge=0.0, le=100.0)
    post_quality_score: float = Field(ge=0.0, le=100.0)
    delta_quality: float
    pre_bias_risk_score: float = Field(ge=0.0, le=100.0)
    post_bias_risk_score: float = Field(ge=0.0, le=100.0)
    delta_bias_risk: float
    target_shift_rate: float = Field(ge=0.0, le=1.0)
    heldout_shift_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    parrot_alignment_warning: bool = False
    collateral_drift_score: float = Field(ge=0.0, le=1.0)
    category_deltas: list[CategoryShiftDelta] = Field(default_factory=list)
    shifted_operation_ids: list[str] = Field(default_factory=list)
    resistant_operation_ids: list[str] = Field(default_factory=list)
    drifted_unchanged_items: list[str] = Field(default_factory=list)
    quality_gate_passed: bool
    drift_gate_passed: bool
    efficacy_gate_passed: bool
    overall_gate_passed: bool
    gate_failures: list[str] = Field(default_factory=list)
    recommendation: str

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value)
