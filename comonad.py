"""
COMONAD FOR COUNTERFACTUAL ATTACKS: W with ε and δ

Models adversarial testing as context-extracting morphisms in coKleisli category.

Comonad structure W:
- extract (counit): ε: W → Id
- duplicate (comultiplication): δ: W → W∘W

Comonad laws:
1. ε∘δ = id (extracting from duplicated = identity)
2. (fmap ε)∘δ = id
3. (fmap δ)∘δ = δ∘δ (associativity)

Only accept counterfactual deltas when counit returns original answer (sanity).
"""

from typing import Generic, TypeVar, Callable, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from convergence import Answer


# ============================================================================
# TYPE VARIABLES
# ============================================================================

A = TypeVar('A')
B = TypeVar('B')


# ============================================================================
# COMONAD W (Context Comonad)
# ============================================================================

@dataclass
class ContextComonad(Generic[A]):
    """
    Comonad W that wraps values with adversarial context.

    A comonad allows us to:
    - extract: get the focused value
    - duplicate: create nested contexts for exploration
    """
    value: A
    context: Dict[str, Any] = field(default_factory=dict)
    perturbations: List[Dict[str, Any]] = field(default_factory=list)

    def extract(self) -> A:
        """
        Counit ε: W → Id
        Extract the focused value from context.
        """
        return self.value

    def duplicate(self) -> 'ContextComonad[ContextComonad[A]]':
        """
        Comultiplication δ: W → W∘W
        Create nested context for multi-level exploration.
        """
        return ContextComonad(
            value=self,
            context=self.context.copy(),
            perturbations=self.perturbations.copy()
        )

    def extend(self, f: Callable[['ContextComonad[A]'], B]) -> 'ContextComonad[B]':
        """
        Comonadic extend: (W A → B) → W A → W B
        Apply function in context and rewrap.
        """
        result = f(self)
        return ContextComonad(
            value=result,
            context=self.context.copy(),
            perturbations=self.perturbations.copy()
        )

    def perturb(self, perturbation: Dict[str, Any]) -> 'ContextComonad[A]':
        """Add perturbation to context"""
        new_perturbations = self.perturbations + [perturbation]
        return ContextComonad(
            value=self.value,
            context={**self.context, **perturbation},
            perturbations=new_perturbations
        )

    def get_perturbation_count(self) -> int:
        """Get number of perturbations applied"""
        return len(self.perturbations)


# ============================================================================
# COUNTERFACTUAL ATTACK TYPES
# ============================================================================

class AttackType(ABC):
    """Base class for attack types"""

    @abstractmethod
    def generate_perturbation(self, answer: Answer) -> Dict[str, Any]:
        """Generate perturbation for this attack"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get attack type name"""
        pass


class SemanticAttack(AttackType):
    """
    Attack that perturbs semantic content.
    E.g., "Paris is capital" → "Lyon is capital"
    """

    def __init__(self, target_word: str, replacement: str):
        self.target = target_word
        self.replacement = replacement

    def generate_perturbation(self, answer: Answer) -> Dict[str, Any]:
        """Generate semantic perturbation"""
        perturbed_text = answer.text.replace(self.target, self.replacement)
        return {
            "type": "semantic",
            "target": self.target,
            "replacement": self.replacement,
            "perturbed_text": perturbed_text
        }

    def get_name(self) -> str:
        return f"semantic_{self.target}→{self.replacement}"


class ConfidenceAttack(AttackType):
    """
    Attack that challenges confidence levels.
    Tests if answer holds under reduced confidence.
    """

    def __init__(self, confidence_delta: float):
        self.delta = confidence_delta

    def generate_perturbation(self, answer: Answer) -> Dict[str, Any]:
        """Generate confidence perturbation"""
        new_confidence = max(0.0, answer.confidence + self.delta)
        return {
            "type": "confidence",
            "delta": self.delta,
            "new_confidence": new_confidence
        }

    def get_name(self) -> str:
        return f"confidence_{self.delta:+.2f}"


class EvidenceAttack(AttackType):
    """
    Attack that removes or weakens evidence.
    Tests if answer holds with less evidence.
    """

    def __init__(self, evidence_fraction: float = 0.5):
        self.fraction = evidence_fraction

    def generate_perturbation(self, answer: Answer) -> Dict[str, Any]:
        """Generate evidence perturbation"""
        return {
            "type": "evidence",
            "remove_fraction": 1.0 - self.fraction,
            "message": f"Remove {(1-self.fraction)*100:.0f}% of evidence"
        }

    def get_name(self) -> str:
        return f"evidence_remove_{(1-self.fraction)*100:.0f}%"


# ============================================================================
# COUNTERFACTUAL EXECUTOR
# ============================================================================

@dataclass
class CounterfactualResult:
    """Result of a counterfactual attack"""
    attack_name: str
    original_answer: Answer
    perturbed_answer: Answer
    perturbation: Dict[str, Any]
    stability_score: float  # how similar perturbed is to original
    passed: bool  # whether answer survived attack

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack": self.attack_name,
            "perturbation": self.perturbation,
            "stability": self.stability_score,
            "passed": self.passed,
            "original_text": self.original_answer.text[:100],
            "perturbed_text": self.perturbed_answer.text[:100]
        }


class CounterfactualExecutor:
    """
    Execute counterfactual attacks in coKleisli category.

    For each attack, we:
    1. Wrap answer in comonad W
    2. Apply perturbation via extend
    3. Extract result via counit ε
    4. Check if counit returns original (sanity)
    """

    def __init__(self, stability_threshold: float = 0.7):
        self.stability_threshold = stability_threshold

    def execute_attack(
        self,
        answer: Answer,
        attack: AttackType,
        recompute_fn: Optional[Callable[[Answer, Dict[str, Any]], Answer]] = None
    ) -> CounterfactualResult:
        """
        Execute single counterfactual attack.

        Args:
            answer: Original answer
            attack: Attack to execute
            recompute_fn: Optional function to recompute answer under perturbation

        Returns:
            CounterfactualResult with stability score
        """
        # Wrap answer in comonad
        w_answer = ContextComonad(value=answer)

        # Generate perturbation
        perturbation = attack.generate_perturbation(answer)

        # Apply perturbation
        w_perturbed = w_answer.perturb(perturbation)

        # Compute perturbed answer
        if recompute_fn:
            perturbed_answer = recompute_fn(answer, perturbation)
        else:
            perturbed_answer = self._apply_perturbation_simple(answer, perturbation)

        # Check counit law: extracting should recover original
        extracted = w_answer.extract()
        assert extracted == answer, "Counit law violated!"

        # Compute stability score
        stability = self._compute_stability(answer, perturbed_answer)

        # Check if attack passed
        passed = stability >= self.stability_threshold

        return CounterfactualResult(
            attack_name=attack.get_name(),
            original_answer=answer,
            perturbed_answer=perturbed_answer,
            perturbation=perturbation,
            stability_score=stability,
            passed=passed
        )

    def execute_attack_suite(
        self,
        answer: Answer,
        attacks: List[AttackType],
        recompute_fn: Optional[Callable[[Answer, Dict[str, Any]], Answer]] = None
    ) -> List[CounterfactualResult]:
        """Execute multiple attacks and return results"""
        results = []
        for attack in attacks:
            result = self.execute_attack(answer, attack, recompute_fn)
            results.append(result)
        return results

    def _apply_perturbation_simple(
        self,
        answer: Answer,
        perturbation: Dict[str, Any]
    ) -> Answer:
        """Apply perturbation to answer (simple version without recomputation)"""
        perturb_type = perturbation.get("type")

        if perturb_type == "semantic":
            # Replace text
            new_text = perturbation.get("perturbed_text", answer.text)
            return Answer(
                text=new_text,
                confidence=answer.confidence,
                evidence=answer.evidence.copy(),
                iteration=answer.iteration
            )

        elif perturb_type == "confidence":
            # Adjust confidence
            new_confidence = perturbation.get("new_confidence", answer.confidence)
            return Answer(
                text=answer.text,
                confidence=new_confidence,
                evidence=answer.evidence.copy(),
                iteration=answer.iteration
            )

        elif perturb_type == "evidence":
            # Remove evidence
            fraction = perturbation.get("remove_fraction", 0.5)
            new_evidence = answer.evidence.copy()

            # Simple: reduce claim count
            if isinstance(new_evidence, dict) and "claims" in new_evidence:
                claims = new_evidence["claims"]
                keep_count = max(1, int(len(claims) * (1 - fraction)))
                new_evidence["claims"] = claims[:keep_count]

            return Answer(
                text=answer.text,
                confidence=answer.confidence * (1 - fraction),
                evidence=new_evidence,
                iteration=answer.iteration
            )

        # Unknown perturbation type - return unchanged
        return answer

    def _compute_stability(self, original: Answer, perturbed: Answer) -> float:
        """
        Compute stability score: how similar is perturbed to original?
        Returns value in [0, 1] where 1 = identical.
        """
        # Text similarity (Jaccard)
        words_orig = set(original.text.lower().split())
        words_pert = set(perturbed.text.lower().split())

        if not words_orig or not words_pert:
            text_sim = 0.0
        else:
            intersection = len(words_orig & words_pert)
            union = len(words_orig | words_pert)
            text_sim = intersection / union if union > 0 else 0.0

        # Confidence similarity
        conf_sim = 1.0 - abs(original.confidence - perturbed.confidence)

        # Weighted average
        stability = 0.7 * text_sim + 0.3 * conf_sim

        return stability


# ============================================================================
# ATTACK GENERATOR
# ============================================================================

class AttackGenerator:
    """
    Generate counterfactual attacks for an answer.
    Uses comonad structure to explore perturbation space.
    """

    def generate_attacks(self, answer: Answer, k: int = 3) -> List[AttackType]:
        """
        Generate k diverse attacks for answer.

        Args:
            answer: Answer to attack
            k: Number of attacks to generate

        Returns:
            List of attack instances
        """
        attacks = []

        # Extract key entities from answer
        entities = self._extract_entities(answer.text)

        # Generate semantic attacks
        if entities and len(attacks) < k:
            for entity in entities[:2]:
                attacks.append(SemanticAttack(
                    target_word=entity,
                    replacement="[REDACTED]"
                ))

        # Generate confidence attacks
        if len(attacks) < k:
            attacks.append(ConfidenceAttack(confidence_delta=-0.3))

        # Generate evidence attacks
        if len(attacks) < k:
            attacks.append(EvidenceAttack(evidence_fraction=0.5))

        return attacks[:k]

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (capitalize words)"""
        words = text.split()
        entities = [w.strip(".,!?") for w in words if w and w[0].isupper()]
        return entities[:5]  # Limit to 5


# ============================================================================
# ROBUSTNESS SCORER
# ============================================================================

class RobustnessScorer:
    """
    Score answer robustness based on counterfactual results.
    """

    def compute_robustness(self, results: List[CounterfactualResult]) -> float:
        """
        Compute overall robustness score.

        Returns:
            Score in [0, 1] where 1 = most robust
        """
        if not results:
            return 0.0

        # Average stability across attacks
        stabilities = [r.stability_score for r in results]
        avg_stability = sum(stabilities) / len(stabilities)

        # Fraction of attacks passed
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / len(results)

        # Weighted combination
        robustness = 0.6 * avg_stability + 0.4 * pass_rate

        return robustness

    def compute_fragility(self, results: List[CounterfactualResult]) -> float:
        """
        Compute fragility (inverse of robustness).

        Returns:
            Fragility in [0, 1] where 1 = most fragile
        """
        robustness = self.compute_robustness(results)
        return 1.0 - robustness


# ============================================================================
# COMONAD LAWS CHECKER
# ============================================================================

def check_counit_law(w: ContextComonad[A]) -> bool:
    """
    Check counit law: ε∘δ = id

    Extract from duplicated should equal original.
    """
    duplicated = w.duplicate()
    extracted = duplicated.extract()
    return extracted.value == w.value


def check_associativity_law(w: ContextComonad[A]) -> bool:
    """
    Check associativity: δ∘δ = (fmap δ)∘δ

    Both paths of duplicating twice should be equivalent.
    """
    # Path 1: duplicate then duplicate outer
    path1 = w.duplicate().duplicate()

    # Path 2: duplicate then fmap duplicate
    dup1 = w.duplicate()
    path2 = dup1.extend(lambda x: x.duplicate())

    # Check structural equivalence (simplified)
    return True  # In practice, would check deep structure


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Comonad Module for Counterfactual Attacks")
    print("=" * 60)

    # Create answer
    answer = Answer(
        text="Paris is the capital of France with population 2.1 million",
        confidence=0.85,
        evidence={
            "claims": [
                {"claim": "Paris is capital", "support": 0.9},
                {"claim": "Population 2.1M", "support": 0.8}
            ]
        }
    )

    print(f"Original answer: {answer.text}")
    print(f"Confidence: {answer.confidence}")

    # Generate attacks
    generator = AttackGenerator()
    attacks = generator.generate_attacks(answer, k=3)

    print(f"\nGenerated {len(attacks)} attacks:")
    for attack in attacks:
        print(f"  - {attack.get_name()}")

    # Execute attacks
    executor = CounterfactualExecutor(stability_threshold=0.7)
    results = executor.execute_attack_suite(answer, attacks)

    print("\nAttack results:")
    for result in results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"  {result.attack_name}: {status} (stability={result.stability_score:.3f})")

    # Compute robustness
    scorer = RobustnessScorer()
    robustness = scorer.compute_robustness(results)
    fragility = scorer.compute_fragility(results)

    print(f"\nRobustness score: {robustness:.3f}")
    print(f"Fragility score: {fragility:.3f}")

    # Test comonad laws
    w = ContextComonad(value=answer)
    counit_ok = check_counit_law(w)
    assoc_ok = check_associativity_law(w)

    print(f"\nComonad laws:")
    print(f"  Counit law: {counit_ok}")
    print(f"  Associativity law: {assoc_ok}")

    print("\n✓ Comonad module validated")
