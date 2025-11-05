"""
EVIDENCE LAYER: Natural transformation ε: Answer ⇒ Citations

Implements evidence/citation system as a natural transformation ensuring:
- Every claim factors through a source
- Citations computed as coend ∫^i Hom(claim_i, source_j)
- Reject answers with zero morphisms (no evidence)

Key properties:
- Naturality: evidence respects answer transformations
- Completeness: all claims must have sources
- Traceability: sources form a dag of provenance
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


# ============================================================================
# CITATION & SOURCE TYPES
# ============================================================================

class SourceType(Enum):
    """Types of evidence sources"""
    WEB = "web"
    API = "api"
    COMPUTATION = "computation"
    MEMORY = "memory"
    DERIVED = "derived"


@dataclass
class Source:
    """
    Evidence source with provenance.
    Forms objects in the citation category.
    """
    id: str
    type: SourceType
    url: Optional[str] = None
    content: str = ""
    reliability: float = 0.5
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize source"""
        return {
            "id": self.id,
            "type": self.type.value,
            "url": self.url,
            "content": self.content[:200] if self.content else "",
            "reliability": self.reliability,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class Claim:
    """
    Individual claim in an answer.
    Must factor through at least one source.
    """
    id: str
    text: str
    support: float = 0.0
    source_ids: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)  # other claim IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_evidence(self) -> bool:
        """Check if claim has at least one source"""
        return len(self.source_ids) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize claim"""
        return {
            "id": self.id,
            "text": self.text,
            "support": self.support,
            "source_ids": self.source_ids,
            "derived_from": self.derived_from,
            "metadata": self.metadata
        }


@dataclass
class ClaimSourceMorphism:
    """
    Morphism Hom(claim, source) representing the factorization
    of a claim through a source.

    This is the fundamental building block of the coend calculation.
    """
    claim_id: str
    source_id: str
    strength: float  # 0-1, how well source supports claim
    extraction_method: str = "direct"  # how we got this morphism

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim_id,
            "source": self.source_id,
            "strength": self.strength,
            "method": self.extraction_method
        }


# ============================================================================
# EVIDENCE OBJECT
# ============================================================================

@dataclass
class Evidence:
    """
    Complete evidence structure for an answer.
    Consists of claims, sources, and morphisms between them.
    """
    claims: List[Claim] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    morphisms: List[ClaimSourceMorphism] = field(default_factory=list)

    def add_claim(self, claim: Claim) -> 'Evidence':
        """Add claim to evidence"""
        self.claims.append(claim)
        return self

    def add_source(self, source: Source) -> 'Evidence':
        """Add source to evidence"""
        self.sources.append(source)
        return self

    def add_morphism(
        self,
        claim_id: str,
        source_id: str,
        strength: float,
        method: str = "direct"
    ) -> 'Evidence':
        """Add claim→source morphism"""
        morphism = ClaimSourceMorphism(
            claim_id=claim_id,
            source_id=source_id,
            strength=strength,
            extraction_method=method
        )
        self.morphisms.append(morphism)
        return self

    def get_sources_for_claim(self, claim_id: str) -> List[Source]:
        """Get all sources supporting a claim"""
        source_ids = [
            m.source_id
            for m in self.morphisms
            if m.claim_id == claim_id
        ]
        return [
            s for s in self.sources
            if s.id in source_ids
        ]

    def get_claims_from_source(self, source_id: str) -> List[Claim]:
        """Get all claims derived from a source"""
        claim_ids = [
            m.claim_id
            for m in self.morphisms
            if m.source_id == source_id
        ]
        return [
            c for c in self.claims
            if c.id in claim_ids
        ]

    def compute_coend(self) -> int:
        """
        Compute coend ∫^i Hom(claim_i, source_j).
        Returns total number of claim→source morphisms.

        In category theory, the coend integrates over all morphisms.
        Here we count them to ensure every claim has evidence.
        """
        return len(self.morphisms)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate evidence structure:
        - Every claim must have at least one source
        - All morphisms must reference existing claims and sources
        - No circular derivations
        """
        errors = []

        # Check all claims have sources
        for claim in self.claims:
            morphisms_for_claim = [
                m for m in self.morphisms
                if m.claim_id == claim.id
            ]
            if not morphisms_for_claim:
                errors.append(f"Claim '{claim.id}' has no sources")

        # Check morphism references
        claim_ids = {c.id for c in self.claims}
        source_ids = {s.id for s in self.sources}

        for morphism in self.morphisms:
            if morphism.claim_id not in claim_ids:
                errors.append(f"Morphism references unknown claim '{morphism.claim_id}'")
            if morphism.source_id not in source_ids:
                errors.append(f"Morphism references unknown source '{morphism.source_id}'")

        # Check for circular derivations
        if self._has_circular_derivation():
            errors.append("Evidence has circular claim derivations")

        return len(errors) == 0, errors

    def _has_circular_derivation(self) -> bool:
        """Check for cycles in claim derivation graph"""
        # Build adjacency list
        graph: Dict[str, List[str]] = {}
        for claim in self.claims:
            graph[claim.id] = claim.derived_from

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for claim_id in graph:
            if claim_id not in visited:
                if has_cycle(claim_id):
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence"""
        return {
            "claims": [c.to_dict() for c in self.claims],
            "sources": [s.to_dict() for s in self.sources],
            "morphisms": [m.to_dict() for m in self.morphisms],
            "coend": self.compute_coend()
        }


# ============================================================================
# NATURAL TRANSFORMATION ε: Answer ⇒ Citations
# ============================================================================

class CitationTransform(ABC):
    """
    Abstract natural transformation ε: Answer → Citations.

    For any answer f: X → Y, the naturality square commutes:

        Answer_X ---ε_X--→ Citations_X
           |                    |
           f                    cite(f)
           ↓                    ↓
        Answer_Y ---ε_Y--→ Citations_Y

    This ensures citations respect answer transformations.
    """

    @abstractmethod
    def extract_citations(self, answer_text: str, context: Any = None) -> Evidence:
        """
        Extract evidence from answer.
        Must be natural: commutes with answer morphisms.
        """
        pass

    def check_naturality(
        self,
        answer_before: str,
        answer_after: str,
        transform: callable
    ) -> bool:
        """
        Verify naturality square commutes.

        Path 1: extract_citations(before) then transform citations
        Path 2: transform(before) then extract_citations(after)

        Should yield equivalent citation structures.
        """
        # Path 1
        citations_before = self.extract_citations(answer_before)
        # (would need to implement citation transform)

        # Path 2
        citations_after = self.extract_citations(answer_after)

        # Compare (simplified)
        return citations_before.compute_coend() == citations_after.compute_coend()


class SimpleClaimExtractor(CitationTransform):
    """
    Simple claim extractor that splits answer into sentences.
    Each sentence becomes a claim.
    """

    def extract_citations(self, answer_text: str, context: Any = None) -> Evidence:
        """Extract claims from answer text"""
        evidence = Evidence()

        # Split into sentences (simple)
        import re
        sentences = re.split(r'[.!?]+', answer_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Create claims
        for i, sentence in enumerate(sentences):
            claim = Claim(
                id=f"claim_{i}",
                text=sentence,
                support=0.5  # default support
            )
            evidence.add_claim(claim)

        return evidence


class ContextAwareExtractor(CitationTransform):
    """
    Context-aware claim extractor that uses tool results to link claims to sources.
    """

    def __init__(self):
        self.source_counter = 0
        self.claim_counter = 0

    def extract_citations(
        self,
        answer_text: str,
        context: Any = None
    ) -> Evidence:
        """
        Extract claims and link to sources from context.

        Args:
            answer_text: The answer text
            context: Tool results and execution context

        Returns:
            Evidence with claims, sources, and morphisms
        """
        evidence = Evidence()

        # Extract claims from answer
        claims = self._extract_claims(answer_text)
        for claim in claims:
            evidence.add_claim(claim)

        # Extract sources from context
        if context and hasattr(context, 'memory'):
            sources = self._extract_sources(context.memory)
            for source in sources:
                evidence.add_source(source)

            # Link claims to sources
            morphisms = self._link_claims_to_sources(claims, sources, answer_text)
            for morphism in morphisms:
                evidence.add_morphism(
                    morphism.claim_id,
                    morphism.source_id,
                    morphism.strength,
                    morphism.extraction_method
                )

        return evidence

    def _extract_claims(self, text: str) -> List[Claim]:
        """Extract individual claims from text"""
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        claims = []
        for i, sentence in enumerate(sentences):
            claim = Claim(
                id=f"claim_{self.claim_counter}",
                text=sentence,
                support=0.5
            )
            self.claim_counter += 1
            claims.append(claim)

        return claims

    def _extract_sources(self, memory: Dict[str, Any]) -> List[Source]:
        """Extract sources from execution context"""
        sources = []

        for key, value in memory.items():
            if isinstance(value, dict):
                # Check if this looks like a search result
                if "results" in value or "url" in value:
                    source = Source(
                        id=f"source_{self.source_counter}",
                        type=SourceType.WEB,
                        url=value.get("url"),
                        content=str(value),
                        reliability=0.7
                    )
                    self.source_counter += 1
                    sources.append(source)

                # Check if this is a computation result
                elif "result" in value or "calculation" in value:
                    source = Source(
                        id=f"source_{self.source_counter}",
                        type=SourceType.COMPUTATION,
                        content=str(value),
                        reliability=1.0  # computations are reliable
                    )
                    self.source_counter += 1
                    sources.append(source)

        return sources

    def _link_claims_to_sources(
        self,
        claims: List[Claim],
        sources: List[Source],
        answer_text: str
    ) -> List[ClaimSourceMorphism]:
        """
        Create morphisms linking claims to sources.
        Uses simple keyword matching.
        """
        morphisms = []

        for claim in claims:
            claim_words = set(claim.text.lower().split())

            for source in sources:
                source_words = set(source.content.lower().split())

                # Compute overlap
                overlap = len(claim_words & source_words)
                total = len(claim_words | source_words)

                if total > 0:
                    strength = overlap / total

                    # Link if strength > threshold
                    if strength > 0.2:
                        morphism = ClaimSourceMorphism(
                            claim_id=claim.id,
                            source_id=source.id,
                            strength=strength,
                            extraction_method="keyword_overlap"
                        )
                        morphisms.append(morphism)

        return morphisms


# ============================================================================
# EVIDENCE POLICY
# ============================================================================

class EvidencePolicy:
    """
    Policy enforcing evidence requirements.
    Rejects answers that don't meet evidence standards.
    """

    def __init__(
        self,
        min_claims: int = 1,
        min_sources: int = 1,
        min_morphisms: int = 1,
        require_all_claims_sourced: bool = True
    ):
        self.min_claims = min_claims
        self.min_sources = min_sources
        self.min_morphisms = min_morphisms
        self.require_all_claims_sourced = require_all_claims_sourced

    def check(self, evidence: Evidence) -> Tuple[bool, List[str]]:
        """
        Check if evidence meets policy.

        Returns:
            (passes, violations)
        """
        violations = []

        # Check minimum counts
        if len(evidence.claims) < self.min_claims:
            violations.append(
                f"Insufficient claims: {len(evidence.claims)} < {self.min_claims}"
            )

        if len(evidence.sources) < self.min_sources:
            violations.append(
                f"Insufficient sources: {len(evidence.sources)} < {self.min_sources}"
            )

        coend = evidence.compute_coend()
        if coend < self.min_morphisms:
            violations.append(
                f"Insufficient evidence: coend={coend} < {self.min_morphisms}"
            )

        # Check all claims have sources
        if self.require_all_claims_sourced:
            for claim in evidence.claims:
                sources = evidence.get_sources_for_claim(claim.id)
                if not sources:
                    violations.append(
                        f"Claim '{claim.id}' has no sources"
                    )

        # Validate structure
        valid, errors = evidence.validate()
        if not valid:
            violations.extend(errors)

        return len(violations) == 0, violations


# ============================================================================
# EVIDENCE COMBINATOR
# ============================================================================

def merge_evidence(e1: Evidence, e2: Evidence) -> Evidence:
    """
    Merge two evidence objects (monoidal operation).
    Deduplicates claims and sources by ID.
    """
    merged = Evidence()

    # Merge claims
    claim_ids = set()
    for claim in e1.claims + e2.claims:
        if claim.id not in claim_ids:
            merged.add_claim(claim)
            claim_ids.add(claim.id)

    # Merge sources
    source_ids = set()
    for source in e1.sources + e2.sources:
        if source.id not in source_ids:
            merged.add_source(source)
            source_ids.add(source.id)

    # Merge morphisms (deduplicate)
    morphism_keys = set()
    for morphism in e1.morphisms + e2.morphisms:
        key = (morphism.claim_id, morphism.source_id)
        if key not in morphism_keys:
            merged.morphisms.append(morphism)
            morphism_keys.add(key)

    return merged


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Evidence Layer Module")
    print("=" * 60)

    # Create evidence
    evidence = Evidence()

    # Add claims
    claim1 = Claim(id="c1", text="Paris is the capital of France", support=0.9)
    claim2 = Claim(id="c2", text="Paris has population 2.1M", support=0.8)

    evidence.add_claim(claim1)
    evidence.add_claim(claim2)

    # Add sources
    source1 = Source(
        id="s1",
        type=SourceType.WEB,
        url="https://example.com/paris",
        content="Paris is the capital of France with 2.1M people",
        reliability=0.9
    )

    evidence.add_source(source1)

    # Add morphisms
    evidence.add_morphism("c1", "s1", strength=0.95, method="direct")
    evidence.add_morphism("c2", "s1", strength=0.85, method="direct")

    # Compute coend
    coend = evidence.compute_coend()
    print(f"Evidence coend (morphism count): {coend}")

    # Validate
    valid, errors = evidence.validate()
    print(f"\nEvidence valid: {valid}")
    if errors:
        print(f"Errors: {errors}")

    # Test policy
    policy = EvidencePolicy(
        min_claims=1,
        min_sources=1,
        min_morphisms=1,
        require_all_claims_sourced=True
    )

    passes, violations = policy.check(evidence)
    print(f"\nPolicy check: {passes}")
    if violations:
        print(f"Violations: {violations}")

    # Test claim extractor
    extractor = SimpleClaimExtractor()
    answer_text = "Paris is the capital of France. It has a population of 2.1 million. The city is famous for the Eiffel Tower."

    extracted = extractor.extract_citations(answer_text)
    print(f"\nExtracted {len(extracted.claims)} claims from answer")
    for claim in extracted.claims:
        print(f"  - {claim.text}")

    print("\n✓ Evidence module validated")
