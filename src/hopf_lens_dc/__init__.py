"""
HOPF_LENS_DC: Categorical Tool Composition Framework

A mathematically rigorous LLM orchestration framework with formal guarantees.
"""

__version__ = "1.0.0"

from .categorical_core import (
    Context,
    AritySchema,
    DirectAssembler,
    ToolMorphism,
    CategoricalToolRegistry,
    Effect,
    EffectType,
    KanSynthesizer,
    create_simple_tool,
)

from .planner import (
    PlannerFunctor,
    QueryObject,
    Plan,
    PlanNode,
)

from .convergence import (
    Answer,
    AnswerMetric,
    SemanticDriftMetric,
    ConfidenceMetric,
    CompositeMetric,
    AnswerEndofunctor,
    AnswerCoalgebra,
    ConvergenceChecker,
    FixedPointFinder,
)

from .evidence import (
    Evidence,
    Claim,
    Source,
    SourceType,
    CitationTransform,
    ContextAwareExtractor,
    EvidencePolicy,
    merge_evidence,
)

from .comonad import (
    ContextComonad,
    AttackType,
    SemanticAttack,
    ConfidenceAttack,
    EvidenceAttack,
    CounterfactualExecutor,
    AttackGenerator,
    RobustnessScorer,
)

__all__ = [
    # Core
    "Context",
    "AritySchema",
    "DirectAssembler",
    "ToolMorphism",
    "CategoricalToolRegistry",
    "Effect",
    "EffectType",
    "KanSynthesizer",
    "create_simple_tool",

    # Planning
    "PlannerFunctor",
    "QueryObject",
    "Plan",
    "PlanNode",

    # Convergence
    "Answer",
    "AnswerMetric",
    "SemanticDriftMetric",
    "ConfidenceMetric",
    "CompositeMetric",
    "AnswerEndofunctor",
    "AnswerCoalgebra",
    "ConvergenceChecker",
    "FixedPointFinder",

    # Evidence
    "Evidence",
    "Claim",
    "Source",
    "SourceType",
    "CitationTransform",
    "ContextAwareExtractor",
    "EvidencePolicy",
    "merge_evidence",

    # Comonad
    "ContextComonad",
    "AttackType",
    "SemanticAttack",
    "ConfidenceAttack",
    "EvidenceAttack",
    "CounterfactualExecutor",
    "AttackGenerator",
    "RobustnessScorer",
]
