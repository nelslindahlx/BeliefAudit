"""BeliefShift: Closed-loop Model -> Knowledge Graph -> Modular LoRA -> Model shifting."""

from beliefaudit.shift.delta_engine import (
    DeltaEngineError,
    compute_belief_delta_against_donor,
    compute_belief_delta_against_golden,
    delta_graph_to_kr_layer,
    load_delta_graph,
    save_delta_graph,
)
from beliefaudit.shift.heldout_pack import all_heldout_prompts, build_heldout_pack
from beliefaudit.shift.kg_compiler import (
    EvalLeakageError,
    compile_shift_corpus,
    load_corpus_manifest,
    token_jaccard,
    verify_no_eval_leakage,
)
from beliefaudit.shift.replay_bank import ReplayBankError, generate_replay_records
from beliefaudit.shift.schemas import (
    BeliefDeltaGraph,
    CategoryShiftDelta,
    ChatMessage,
    CompiledCorpusManifest,
    DeltaOperation,
    DpoPreferencePair,
    HeldOutProbeItem,
    LoRAShardManifest,
    RoundTripReport,
    SftTrainingRecord,
    ShiftOperationType,
    ShiftTargetKind,
    TrainingBackend,
)

__all__ = [
    "BeliefDeltaGraph",
    "CategoryShiftDelta",
    "ChatMessage",
    "CompiledCorpusManifest",
    "DeltaEngineError",
    "DeltaOperation",
    "DpoPreferencePair",
    "EvalLeakageError",
    "HeldOutProbeItem",
    "LoRAShardManifest",
    "ReplayBankError",
    "RoundTripReport",
    "SftTrainingRecord",
    "ShiftOperationType",
    "ShiftTargetKind",
    "TrainingBackend",
    "all_heldout_prompts",
    "build_heldout_pack",
    "compile_shift_corpus",
    "compute_belief_delta_against_donor",
    "compute_belief_delta_against_golden",
    "delta_graph_to_kr_layer",
    "generate_replay_records",
    "load_corpus_manifest",
    "load_delta_graph",
    "save_delta_graph",
    "token_jaccard",
    "verify_no_eval_leakage",
]
