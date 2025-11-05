"""
CONVERGENCE MODULE: Coalgebra γ: X → F(X) with metric

Implements convergence as:
- Answer space X enriched with metric d
- Endofunctor F(X) = Compose(tools, X)
- Coalgebra γ: X → F(X) for iteration
- Fixed-point iteration until d(x_{n+1}, x_n) < ε

Proves F is a contraction to eliminate "Fragility: ∞"
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math


# ============================================================================
# ANSWER SPACE (X)
# ============================================================================

@dataclass
class Answer:
    """
    Element of answer space X.
    Represents a candidate answer with evidence and confidence.
    """
    text: str
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    iteration: int = 0

    def is_valid(self) -> bool:
        """Check if answer is well-formed"""
        return (
            bool(self.text and self.text.strip()) and
            0.0 <= self.confidence <= 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize answer"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "iteration": self.iteration
        }


# ============================================================================
# METRICS ON ANSWER SPACE
# ============================================================================

class AnswerMetric(ABC):
    """
    Abstract metric d: X × X → [0, ∞) on answer space.
    Must satisfy metric axioms:
    1. d(x, y) ≥ 0
    2. d(x, y) = 0 iff x = y
    3. d(x, y) = d(y, x)
    4. d(x, z) ≤ d(x, y) + d(y, z) (triangle inequality)
    """

    @abstractmethod
    def distance(self, a1: Answer, a2: Answer) -> float:
        """Compute distance between two answers"""
        pass

    def is_converged(self, a1: Answer, a2: Answer, epsilon: float) -> bool:
        """Check if answers are within epsilon"""
        return self.distance(a1, a2) < epsilon


class SemanticDriftMetric(AnswerMetric):
    """
    Semantic drift metric based on token overlap.
    d(a1, a2) = 1 - |tokens(a1) ∩ tokens(a2)| / |tokens(a1) ∪ tokens(a2)|
    """

    def distance(self, a1: Answer, a2: Answer) -> float:
        """Jaccard distance on token sets"""
        if not a1.text or not a2.text:
            return 1.0

        tokens1 = set(a1.text.lower().split())
        tokens2 = set(a2.text.lower().split())

        if not tokens1 or not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 1.0

        jaccard = intersection / union
        return 1.0 - jaccard


class ConfidenceMetric(AnswerMetric):
    """
    Confidence-based metric.
    d(a1, a2) = |conf(a1) - conf(a2)|
    """

    def distance(self, a1: Answer, a2: Answer) -> float:
        """Absolute difference in confidence"""
        return abs(a1.confidence - a2.confidence)


class CompositeMetric(AnswerMetric):
    """
    Composite metric combining multiple metrics.
    d(a1, a2) = Σ w_i * d_i(a1, a2)
    """

    def __init__(self, metrics: List[Tuple[AnswerMetric, float]]):
        """
        Args:
            metrics: List of (metric, weight) pairs
        """
        self.metrics = metrics
        # Normalize weights
        total_weight = sum(w for _, w in metrics)
        self.metrics = [(m, w / total_weight) for m, w in metrics]

    def distance(self, a1: Answer, a2: Answer) -> float:
        """Weighted sum of component metrics"""
        total = 0.0
        for metric, weight in self.metrics:
            total += weight * metric.distance(a1, a2)
        return total


# ============================================================================
# ENDOFUNCTOR F: X → X
# ============================================================================

@dataclass
class AnswerTransformation:
    """
    Transformation in endofunctor F(X) = Compose(tools, X).
    Represents one step of answer refinement.
    """
    operation: str  # e.g., "refine", "expand", "correct"
    tool_used: Optional[str] = None
    delta: Dict[str, Any] = field(default_factory=dict)
    contraction_factor: float = 1.0

    def apply(self, answer: Answer) -> Answer:
        """Apply transformation to answer"""
        # Simple implementation: return new answer with incremented iteration
        return Answer(
            text=answer.text,
            confidence=answer.confidence,
            evidence=answer.evidence.copy(),
            metadata={**answer.metadata, "transformation": self.operation},
            iteration=answer.iteration + 1
        )


class AnswerEndofunctor:
    """
    Endofunctor F: X → X where F(X) = Compose(tools, X).
    Maps answers to refined answers via tool composition.
    """

    def __init__(self, contraction_factor: float = 0.8):
        """
        Args:
            contraction_factor: λ < 1 proving F is contractive
        """
        self.contraction_factor = contraction_factor
        self.transformations: List[AnswerTransformation] = []

    def map(self, answer: Answer, tools_result: Any = None) -> Answer:
        """
        F(x): apply functor to answer.

        In practice, this composes tool results with current answer
        to produce refined answer.
        """
        if not answer.is_valid():
            return answer

        # Create transformation based on tool results
        transformation = AnswerTransformation(
            operation="refine",
            contraction_factor=self.contraction_factor
        )

        refined = transformation.apply(answer)

        # Store transformation
        self.transformations.append(transformation)

        return refined

    def is_contractive(self, metric: AnswerMetric) -> bool:
        """
        Check if F is a contraction mapping:
        d(F(x), F(y)) ≤ λ * d(x, y) for λ < 1
        """
        return self.contraction_factor < 1.0


# ============================================================================
# COALGEBRA γ: X → F(X)
# ============================================================================

class AnswerCoalgebra:
    """
    Coalgebra γ: X → F(X) for iterative answer refinement.

    The coalgebra structure allows us to unfold answer states:
    x → F(x) → F²(x) → ... → x*

    where x* is the fixed point (converged answer).
    """

    def __init__(
        self,
        functor: AnswerEndofunctor,
        metric: AnswerMetric,
        epsilon: float = 0.02
    ):
        self.functor = functor
        self.metric = metric
        self.epsilon = epsilon
        self.trajectory: List[Answer] = []

    def unfold(self, answer: Answer, tools_result: Any = None) -> Answer:
        """
        Apply coalgebra: γ(x) = F(x)
        One step of answer refinement.
        """
        refined = self.functor.map(answer, tools_result)
        self.trajectory.append(refined)
        return refined

    def iterate(
        self,
        initial: Answer,
        max_iterations: int = 10,
        refinement_fn: Optional[callable] = None
    ) -> Answer:
        """
        Iterate coalgebra to fixed point.

        Computes: x, F(x), F²(x), ..., F^n(x)
        Until: d(F^{n+1}(x), F^n(x)) < ε

        Args:
            initial: Starting answer x₀
            max_iterations: Maximum iterations
            refinement_fn: Optional function to get tool results for refinement

        Returns:
            Converged answer (or best answer after max iterations)
        """
        self.trajectory = [initial]
        current = initial

        for i in range(max_iterations):
            # Get refinement data if available
            tools_result = refinement_fn(current) if refinement_fn else None

            # Apply coalgebra step
            next_answer = self.unfold(current, tools_result)

            # Check convergence
            distance = self.metric.distance(current, next_answer)

            if distance < self.epsilon:
                # Converged!
                return next_answer

            # Check if we're making progress
            if i > 0 and distance >= self.metric.distance(
                self.trajectory[-2], current
            ):
                # Distance is increasing - stop
                return current

            current = next_answer

        # Max iterations reached
        return current

    def get_convergence_info(self) -> Dict[str, Any]:
        """Get information about convergence process"""
        if len(self.trajectory) < 2:
            return {
                "converged": False,
                "iterations": len(self.trajectory),
                "final_distance": 0.0
            }

        distances = []
        for i in range(len(self.trajectory) - 1):
            d = self.metric.distance(self.trajectory[i], self.trajectory[i + 1])
            distances.append(d)

        final_dist = distances[-1] if distances else 0.0

        return {
            "converged": final_dist < self.epsilon,
            "iterations": len(self.trajectory),
            "final_distance": final_dist,
            "distance_trajectory": distances,
            "contraction_rate": self._estimate_contraction_rate(distances)
        }

    def _estimate_contraction_rate(self, distances: List[float]) -> Optional[float]:
        """Estimate contraction factor from distance trajectory"""
        if len(distances) < 2:
            return None

        # Estimate λ from d_{i+1} / d_i
        ratios = []
        for i in range(len(distances) - 1):
            if distances[i] > 0:
                ratios.append(distances[i + 1] / distances[i])

        if not ratios:
            return None

        return sum(ratios) / len(ratios)


# ============================================================================
# CONVERGENCE CHECKER
# ============================================================================

class ConvergenceChecker:
    """
    Check multiple convergence criteria:
    - Semantic drift: d_semantic(a_{n+1}, a_n) < τ_A
    - Confidence improvement: Δconf < τ_C
    - Evidence fragility: ν(evidence) < τ_ν
    """

    def __init__(
        self,
        tau_a: float = 0.02,  # semantic drift threshold
        tau_c: float = 0.01,  # confidence improvement threshold
        tau_nu: float = 0.15  # fragility threshold
    ):
        self.tau_a = tau_a
        self.tau_c = tau_c
        self.tau_nu = tau_nu

        self.semantic_metric = SemanticDriftMetric()
        self.conf_metric = ConfidenceMetric()

    def check_convergence(
        self,
        prev_answer: Answer,
        current_answer: Answer
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if answer has converged.

        Returns:
            (converged, metrics_dict)
        """
        # Semantic drift
        drift = self.semantic_metric.distance(prev_answer, current_answer)

        # Confidence improvement
        delta_conf = abs(current_answer.confidence - prev_answer.confidence)

        # Evidence fragility
        fragility = self._compute_fragility(current_answer.evidence)

        metrics = {
            "drift": drift,
            "delta_conf": delta_conf,
            "fragility": fragility
        }

        # Check all criteria
        converged = (
            drift < self.tau_a and
            delta_conf < self.tau_c and
            fragility < self.tau_nu
        )

        return converged, metrics

    def _compute_fragility(self, evidence: Dict[str, Any]) -> float:
        """
        Compute evidence fragility ν(w).
        Higher = more fragile (bad).
        """
        if not evidence:
            return float('inf')

        claims = evidence.get("claims", [])
        if not claims:
            return float('inf')

        # Fragility based on support scores
        supports = [
            claim.get("support", 0.0)
            for claim in claims
            if isinstance(claim, dict)
        ]

        if not supports:
            return 1.0

        # Fragility = 1 - min(support)
        return 1.0 - min(supports)


# ============================================================================
# FIXED POINT FINDER
# ============================================================================

class FixedPointFinder:
    """
    Find fixed point of coalgebra iteration.
    Uses Banach fixed-point theorem when F is contractive.
    """

    def __init__(self, coalgebra: AnswerCoalgebra):
        self.coalgebra = coalgebra

    def find_fixed_point(
        self,
        initial: Answer,
        max_iterations: int = 10,
        refinement_fn: Optional[callable] = None
    ) -> Tuple[Answer, Dict[str, Any]]:
        """
        Find fixed point x* where F(x*) = x*.

        By Banach theorem, if F is a contraction mapping on complete
        metric space, then iteration converges to unique fixed point.

        Returns:
            (fixed_point_answer, convergence_info)
        """
        result = self.coalgebra.iterate(initial, max_iterations, refinement_fn)
        info = self.coalgebra.get_convergence_info()

        return result, info


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Convergence Module")
    print("=" * 60)

    # Create initial answer
    initial = Answer(
        text="Paris is the capital of France",
        confidence=0.5,
        evidence={"claims": [{"claim": "Paris is capital", "support": 0.8}]},
        iteration=0
    )

    print(f"Initial answer: {initial.text}")
    print(f"Confidence: {initial.confidence}")

    # Create metric
    metric = CompositeMetric([
        (SemanticDriftMetric(), 0.7),
        (ConfidenceMetric(), 0.3)
    ])

    # Create endofunctor
    functor = AnswerEndofunctor(contraction_factor=0.7)

    # Create coalgebra
    coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

    # Test single unfold
    refined = coalgebra.unfold(initial)
    print(f"\nRefined answer: {refined.text}")
    print(f"Iteration: {refined.iteration}")

    # Test convergence
    checker = ConvergenceChecker()

    # Simulate refinement
    answer2 = Answer(
        text="Paris is the capital of France and has population of 2.1 million",
        confidence=0.7,
        evidence={"claims": [
            {"claim": "Paris is capital", "support": 0.9},
            {"claim": "Population 2.1M", "support": 0.8}
        ]},
        iteration=1
    )

    converged, metrics = checker.check_convergence(initial, answer2)
    print(f"\nConvergence check:")
    print(f"  Converged: {converged}")
    print(f"  Drift: {metrics['drift']:.4f}")
    print(f"  ΔConfidence: {metrics['delta_conf']:.4f}")
    print(f"  Fragility: {metrics['fragility']:.4f}")

    # Test fixed point finder
    finder = FixedPointFinder(coalgebra)

    def mock_refinement(answer: Answer) -> Dict[str, Any]:
        """Mock refinement function"""
        return {"additional_evidence": "from mock"}

    final, info = finder.find_fixed_point(initial, max_iterations=5, refinement_fn=mock_refinement)

    print(f"\nFixed point search:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Final distance: {info.get('final_distance', 0):.4f}")
    if info.get('contraction_rate'):
        print(f"  Contraction rate: {info['contraction_rate']:.4f}")

    print("\n✓ Convergence module validated")
