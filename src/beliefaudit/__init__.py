"""BeliefAudit: Pre-selection open-weight model belief distillation and bias/quality audit system."""

from __future__ import annotations

from beliefaudit.golden import (
    BundledOfflineSeeder,
    load_golden_snapshot,
    promote_golden_snapshot,
    save_golden_snapshot,
)
from beliefaudit.probes import (
    LocalLLMJudge,
    LocalOpenAICompatRunner,
    MockModelRunner,
    distill_probe_responses,
    run_probe_suite,
    select_profile_subset,
)
from beliefaudit.query import list_audited_runs, load_scorecard, query_belief_conflicts
from beliefaudit.schemas import (
    AdvisoryVerdict,
    AuditProfile,
    AuditScorecard,
    BiasMatrixResult,
    ClaimCategory,
    DistilledBelief,
    GoldenItem,
    GoldenSnapshot,
    ProbeMode,
    QualityScoreResult,
    StanceLabel,
)
from beliefaudit.scoring import (
    build_scorecard,
    compute_bias_matrix,
    compute_quality_score,
    determine_verdict,
)
from beliefaudit.shards import (
    KRLayerData,
    beliefs_to_kr_layer,
    diff_candidate_against_golden,
    diff_model_shards,
    export_kr_json,
    export_kr_sharded,
    import_kr_json,
    import_kr_sharded,
)

__version__ = "0.1.2"

__all__ = [
    "AdvisoryVerdict",
    "AuditProfile",
    "AuditScorecard",
    "BiasMatrixResult",
    "BundledOfflineSeeder",
    "ClaimCategory",
    "DistilledBelief",
    "GoldenItem",
    "GoldenSnapshot",
    "KRLayerData",
    "LocalLLMJudge",
    "LocalOpenAICompatRunner",
    "MockModelRunner",
    "ProbeMode",
    "QualityScoreResult",
    "StanceLabel",
    "__version__",
    "beliefs_to_kr_layer",
    "build_scorecard",
    "compute_bias_matrix",
    "compute_quality_score",
    "determine_verdict",
    "diff_candidate_against_golden",
    "diff_model_shards",
    "distill_probe_responses",
    "export_kr_json",
    "export_kr_sharded",
    "import_kr_json",
    "import_kr_sharded",
    "list_audited_runs",
    "load_golden_snapshot",
    "load_scorecard",
    "promote_golden_snapshot",
    "query_belief_conflicts",
    "run_probe_suite",
    "save_golden_snapshot",
    "select_profile_subset",
]
