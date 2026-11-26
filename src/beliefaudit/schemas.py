"""Core Pydantic v2 domain models and enums for BeliefAudit."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from beliefaudit.security import validate_identifier


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def _normalize_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _validate_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return cleaned


class ClaimCategory(StrEnum):
    """Evaluation categories covering contested topics and downstream tasks."""

    TERRITORIAL_SOVEREIGNTY = "territorial_sovereignty"
    HISTORICAL_EVENTS = "historical_events"
    TERMINOLOGY_FRAMING = "terminology_framing"
    GOVERNANCE_RIGHTS = "governance_rights"
    CODE_TECHNICAL_DEFAULTS = "code_technical_defaults"
    STRUCTURED_EXTRACTION = "structured_extraction"


class StanceLabel(StrEnum):
    """Normalized stance classifications for distilled model beliefs."""

    ASSERTS_GOLDEN = "asserts_golden"
    BALANCED_ATTRIBUTED = "balanced_attributed"
    HEDGES_EVADES = "hedges_evades"
    REFUSES = "refuses"
    ASSERTS_CONTRARY = "asserts_contrary"
    EUPHEMISTIC_REFRAME = "euphemistic_reframe"


class ProbeMode(StrEnum):
    """Probing modes executed for each GoldenItem."""

    NEUTRAL = "neutral"
    FRAMED_STATE = "framed_state"
    FRAMED_COUNTER = "framed_counter"
    PUSHBACK = "pushback"
    DOWNSTREAM_CODE = "downstream_code"
    DOWNSTREAM_EXTRACT = "downstream_extract"


class ControlAxis(StrEnum):
    """Non-contested Quality Score calibration axes."""

    FACTUAL = "factual"
    CODE_SCHEMA = "code_schema"
    CONSISTENCY = "consistency"
    CALIBRATION = "calibration"


class CoverageConfidence(StrEnum):
    """Confidence badge for per-category item sample counts."""

    HIGH = "High"
    MODERATE = "Moderate"
    LOW_SEED = "Low/Seed"


class AdvisoryVerdict(StrEnum):
    """Pre-selection advisory verdicts."""

    RECOMMENDED = "Recommended"
    USE_WITH_GUARDRAILS = "Use with Guardrails"
    USE_WITH_GUARDRAILS_QUALITY_LIMITED = "Use with Guardrails (Quality-Limited)"
    AVOID_HIGH_RISK = "Avoid / High Risk"


class AuditProfile(StrEnum):
    """Execution profile controlling probe breadth and runtime budget."""

    QUICK = "quick"
    FULL = "full"


class GoldenItem(BaseModel):
    """Single contested claim and 6-mode probe specification in a Golden Set."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    golden_version: str
    category: ClaimCategory
    topic: str
    canonical_claim: str
    expected_stance: StanceLabel = StanceLabel.ASSERTS_GOLDEN
    acceptable_framings: list[str] = Field(default_factory=list)
    disallowed_euphemisms: list[str] = Field(default_factory=list)
    contrary_markers: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    neutral_prompt: str
    framed_prompt_pro_state: str
    framed_prompt_counter: str
    pushback_followup: str
    downstream_code_prompt: str
    downstream_extract_prompt: str
    expected_code_tokens: list[str] = Field(default_factory=list)
    expected_json_keys: list[str] = Field(default_factory=list)

    @field_validator("item_id", "golden_version")
    @classmethod
    def _check_ids(cls, value: str, info: Any) -> str:
        return validate_identifier(value, str(info.field_name))

    @field_validator(
        "topic",
        "canonical_claim",
        "neutral_prompt",
        "framed_prompt_pro_state",
        "framed_prompt_counter",
        "pushback_followup",
        "downstream_code_prompt",
        "downstream_extract_prompt",
    )
    @classmethod
    def _check_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, str(info.field_name))


class ControlItem(BaseModel):
    """Non-contested benchmark item used to compute the independent Quality Score."""

    model_config = ConfigDict(extra="forbid")

    control_id: str
    axis: ControlAxis
    topic: str
    prompt: str
    paraphrase_prompt: str | None = None
    expected_substrings: list[str] = Field(default_factory=list)
    forbidden_substrings: list[str] = Field(default_factory=list)
    expect_valid_json: bool = False
    expect_valid_python: bool = False
    expected_json_keys: list[str] = Field(default_factory=list)

    @field_validator("control_id")
    @classmethod
    def _check_control_id(cls, value: str) -> str:
        return validate_identifier(value, "control_id")

    @field_validator("topic", "prompt")
    @classmethod
    def _check_non_empty(cls, value: str, info: Any) -> str:
        return _validate_non_empty(value, str(info.field_name))


class GoldenSnapshot(BaseModel):
    """Immutable versioned Golden Set + Control Suite snapshot with SHA-256 integrity."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    golden_version: str
    created_at: datetime = Field(default_factory=utc_now)
    parent_version: str | None = None
    content_sha256: str = ""
    items: list[GoldenItem] = Field(default_factory=list)
    control_items: list[ControlItem] = Field(default_factory=list)

    @field_validator("golden_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        return validate_identifier(value, "golden_version")

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value)

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 over version and items/control_items payloads."""
        payload = {
            "schema_version": self.schema_version,
            "golden_version": self.golden_version,
            "parent_version": self.parent_version,
            "items": [item.model_dump(mode="json") for item in self.items],
            "control_items": [ctrl.model_dump(mode="json") for ctrl in self.control_items],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _populate_digest(self) -> GoldenSnapshot:
        if not self.content_sha256:
            self.content_sha256 = self.compute_digest()
        return self

    def verify_digest(self) -> bool:
        """Return True iff the stored content_sha256 matches the snapshot contents."""
        return self.content_sha256 == self.compute_digest()


class ProbeResponse(BaseModel):
    """Raw model output captured for a single (GoldenItem, ProbeMode) probe."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    probe_mode: ProbeMode
    messages: list[dict[str, str]]
    raw_text: str
    response_sha256: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    model_id: str
    seed: int = 42

    @model_validator(mode="after")
    def _populate_hash(self) -> ProbeResponse:
        if not self.response_sha256:
            self.response_sha256 = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
        return self


class ControlResponse(BaseModel):
    """Evaluated outcome for a single non-contested ControlItem."""

    model_config = ConfigDict(extra="forbid")

    control_id: str
    axis: ControlAxis
    raw_text: str
    paraphrase_raw_text: str | None = None
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class DistilledBelief(BaseModel):
    """Normalized belief triple + stance/leakage telemetry distilled from a probe."""

    model_config = ConfigDict(extra="forbid")

    belief_id: str
    model_id: str
    model_revision: str | None = None
    adapter_id: str | None = None
    golden_item_id: str
    category: ClaimCategory
    probe_mode: ProbeMode
    subject: str
    predicate: str
    object_claim: str
    detected_stance: StanceLabel
    framing_terms_used: list[str] = Field(default_factory=list)
    euphemisms_detected: list[str] = Field(default_factory=list)
    refusal_triggered: bool = False
    leakage_detected: bool = False
    leakage_severity: float = Field(default=0.0, ge=0.0, le=4.0)
    extraction_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    judge_used: bool = False
    raw_response_hash: str
    raw_excerpt: str


class CandidateExpansionEvent(BaseModel):
    """Discovered anomaly flagged for promotion into the next Golden Set version."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    reason: Literal[
        "unexpected_refusal",
        "stance_flip",
        "untracked_euphemism",
        "downstream_leakage",
    ]
    golden_item_id: str
    category: ClaimCategory
    model_id: str
    probe_mode: ProbeMode
    observed_excerpt: str
    suggested_item: GoldenItem | None = None
    suggested_euphemisms: list[str] = Field(default_factory=list)


class CategoryBiasRow(BaseModel):
    """4-dimensional bias breakdown for a single ClaimCategory."""

    model_config = ConfigDict(extra="forbid")

    category: ClaimCategory
    n_items: int = Field(ge=0)
    coverage_confidence: CoverageConfidence
    sd: float = Field(ge=0.0, le=4.0)
    fs: float = Field(ge=0.0, le=4.0)
    pv: float = Field(ge=0.0, le=4.0)
    dl: float = Field(ge=0.0, le=4.0)
    bias_score: float = Field(ge=0.0, le=100.0)
    severe_flag: bool = False


class BiasMatrixResult(BaseModel):
    """Aggregate 4D Category Bias Matrix and hard override triggers."""

    model_config = ConfigDict(extra="forbid")

    overall_bias_risk_score: float = Field(ge=0.0, le=100.0)
    categories: list[CategoryBiasRow] = Field(default_factory=list)
    leakage_guardrail_override: bool = False
    leakage_avoid_override: bool = False
    severe_category_flags: list[ClaimCategory] = Field(default_factory=list)


class QualityScoreResult(BaseModel):
    """Independent technical quality and calibration scores (0-100)."""

    model_config = ConfigDict(extra="forbid")

    quality_score: float = Field(ge=0.0, le=100.0)
    q_factual: float = Field(ge=0.0, le=1.0)
    q_code_schema: float = Field(ge=0.0, le=1.0)
    q_consistency: float = Field(ge=0.0, le=1.0)
    q_calibration: float = Field(ge=0.0, le=1.0)
    n_controls: int = Field(ge=0)


class AuditScorecard(BaseModel):
    """Top-level pre-selection scorecard combining Quality, Bias Matrix, and Verdict."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    model_id: str
    model_revision: str | None = None
    adapter_id: str | None = None
    audit_profile: AuditProfile = AuditProfile.FULL
    golden_version: str
    golden_digest: str
    audited_at: datetime = Field(default_factory=utc_now)
    judge_model_id: str | None = None
    quality: QualityScoreResult
    bias: BiasMatrixResult
    verdict: AdvisoryVerdict
    verdict_rationale: str
    override_reasons: list[str] = Field(default_factory=list)
    n_golden_items: int = Field(ge=0)
    n_beliefs_distilled: int = Field(ge=0)
    n_conflicts: int = Field(ge=0)
    n_candidate_expansions: int = Field(ge=0)

    @field_validator("audited_at")
    @classmethod
    def _check_audited_at(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class BeliefDiffItemDelta(BaseModel):
    """Per-(item_id, probe_mode) delta between baseline and candidate model shards."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    category: ClaimCategory
    probe_mode: ProbeMode
    baseline_stance: str
    candidate_stance: str
    baseline_conflict: bool
    candidate_conflict: bool
    classification: Literal["unchanged", "regression", "improvement", "shifted"]


class BeliefDiffSummary(BaseModel):
    """Summary of differences between two model belief shards on the same Golden Set."""

    model_config = ConfigDict(extra="forbid")

    baseline_model_id: str
    candidate_model_id: str
    golden_version: str
    total_compared: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    regression_count: int = Field(ge=0)
    improvement_count: int = Field(ge=0)
    shifted_count: int = Field(ge=0)
    deltas: list[BeliefDiffItemDelta] = Field(default_factory=list)
