"""
PARIS BRIDGES EXAMPLE

Demonstrates the categorical tool framework with a concrete walkthrough:
Query: "List 3 landmark bridges in Paris with a one-line fact each."

Shows:
1. Typed tools with schemas and assemblers
2. Argument validation (no empty dicts!)
3. Left Kan synthesis for missing arguments
4. Free monoidal planning (∘ and ⊗)
5. Convergence via coalgebra
6. Evidence extraction and validation
7. Counterfactual robustness testing

Expected output: 3 bridges with facts, all claims cited, attack-resistant.
"""

from typing import Dict, Any, List
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hopf_lens_dc.categorical_core import (
    Context, AritySchema, DirectAssembler, ToolMorphism,
    CategoricalToolRegistry, Effect, EffectType, create_simple_tool
)
from src.hopf_lens_dc.planner import PlannerFunctor, QueryObject
from src.hopf_lens_dc.convergence import (
    Answer, AnswerEndofunctor, AnswerCoalgebra,
    CompositeMetric, SemanticDriftMetric, ConfidenceMetric,
    ConvergenceChecker
)
from src.hopf_lens_dc.evidence import (
    Evidence, Claim, Source, SourceType,
    ContextAwareExtractor, EvidencePolicy
)
from src.hopf_lens_dc.comonad import (
    AttackGenerator, CounterfactualExecutor, RobustnessScorer
)


# ============================================================================
# MOCK TOOLS FOR DEMONSTRATION
# ============================================================================

def search_web_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    """
    Mock search_web tool.
    Returns simulated Paris bridges results.
    """
    query = args["query"]
    print(f"  [search_web] query='{query}'")

    # Simulate search results
    return {
        "query": query,
        "results": [
            {
                "title": "Pont Neuf - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Pont_Neuf",
                "snippet": "The Pont Neuf is the oldest standing bridge across the Seine in Paris, completed in 1607."
            },
            {
                "title": "Pont Alexandre III",
                "url": "https://en.wikipedia.org/wiki/Pont_Alexandre_III",
                "snippet": "The Pont Alexandre III is an arch bridge that spans the Seine, built for the 1900 Exposition Universelle."
            },
            {
                "title": "Pont de la Concorde",
                "url": "https://www.parisinfo.com/pont-concorde",
                "snippet": "The Pont de la Concorde was built with stones from the Bastille prison after it was demolished."
            }
        ],
        "result_count": 3
    }


def extract_facts_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    """
    Mock extract_facts tool.
    Extracts structured facts from search results.
    """
    entities = args.get("entities", [])
    k = args.get("k", 3)

    print(f"  [extract_facts] entities={len(entities)}, k={k}")

    # Extract from context memory
    search_result = ctx.project("search_web_result")

    if not search_result or not isinstance(search_result, dict):
        return {"facts": [], "count": 0}

    results = search_result.get("results", [])
    facts = []

    for i, result in enumerate(results[:k]):
        fact = {
            "name": result.get("title", "").split(" - ")[0],
            "fact": result.get("snippet", ""),
            "source": result.get("url", "")
        }
        facts.append(fact)

    return {
        "facts": facts,
        "count": len(facts)
    }


def dedupe_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    """
    Mock dedupe tool.
    Removes duplicate facts.
    """
    facts_list = args.get("facts", [])
    print(f"  [dedupe] processing {len(facts_list)} facts")

    # Simple deduplication by name
    seen = set()
    unique = []

    for fact in facts_list:
        if isinstance(fact, dict):
            name = fact.get("name", "")
            if name and name not in seen:
                seen.add(name)
                unique.append(fact)

    return {
        "facts": unique,
        "count": len(unique)
    }


# ============================================================================
# SETUP CATEGORICAL FRAMEWORK
# ============================================================================

def setup_framework() -> tuple:
    """
    Initialize the categorical tool framework with typed tools.
    """
    print("Setting up categorical framework...")

    # Create registry
    registry = CategoricalToolRegistry()

    # ===== Tool 1: search_web =====
    # Schema: Ar(search_web) = {query: String}
    search_schema = AritySchema()
    search_schema.add_arg("query", str, required=True, description="Search query")

    # Assembler: μ_search(C) = {query: π_query(C)}
    search_assembler = DirectAssembler(
        schema=search_schema,
        mappings={"query": "query"}  # map arg "query" to context "query"
    )

    # Register tool
    def search_wrapped(args: Dict[str, Any], ctx: Context) -> Effect[Dict[str, Any]]:
        result = search_web_impl(args, ctx)
        return Effect.pure(result)

    search_tool = ToolMorphism(
        name="search_web",
        schema=search_schema,
        assembler=search_assembler,
        func=search_wrapped,
        effects=[EffectType.HTTP, EffectType.IO],
        description="Search the web for information"
    )
    registry.tools["search_web"] = search_tool

    # ===== Tool 2: extract_facts =====
    # Schema: Ar(extract_facts) = {entities: [URL], k: Int}
    extract_schema = AritySchema()
    extract_schema.add_arg("entities", list, required=False, default=[],
                           description="List of entity URLs to extract from")
    extract_schema.add_arg("k", int, required=True,
                          description="Number of facts to extract")

    # Assembler will need k - if missing, synthesizer will provide it
    extract_assembler = DirectAssembler(
        schema=extract_schema,
        mappings={"entities": "entities", "k": "k"}
    )

    def extract_wrapped(args: Dict[str, Any], ctx: Context) -> Effect[Dict[str, Any]]:
        result = extract_facts_impl(args, ctx)
        return Effect.pure(result)

    extract_tool = ToolMorphism(
        name="extract_facts",
        schema=extract_schema,
        assembler=extract_assembler,
        func=extract_wrapped,
        effects=[EffectType.PARSE],
        description="Extract structured facts from entities"
    )
    registry.tools["extract_facts"] = extract_tool

    # ===== Tool 3: dedupe =====
    dedupe_schema = AritySchema()
    dedupe_schema.add_arg("facts", list, required=True, description="List of facts to deduplicate")

    dedupe_assembler = DirectAssembler(
        schema=dedupe_schema,
        mappings={"facts": "facts"}
    )

    def dedupe_wrapped(args: Dict[str, Any], ctx: Context) -> Effect[Dict[str, Any]]:
        result = dedupe_impl(args, ctx)
        return Effect.pure(result)

    dedupe_tool = ToolMorphism(
        name="dedupe",
        schema=dedupe_schema,
        assembler=dedupe_assembler,
        func=dedupe_wrapped,
        effects=[EffectType.PURE],
        description="Remove duplicate facts"
    )
    registry.tools["dedupe"] = dedupe_tool

    print(f"✓ Registered {len(registry.tools)} tools\n")

    # Create other components
    planner = PlannerFunctor(registry)

    metric = CompositeMetric([
        (SemanticDriftMetric(), 0.7),
        (ConfidenceMetric(), 0.3)
    ])
    functor = AnswerEndofunctor(contraction_factor=0.7)
    coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

    extractor = ContextAwareExtractor()
    evidence_policy = EvidencePolicy(
        min_claims=3,
        min_sources=1,
        min_morphisms=3,
        require_all_claims_sourced=True
    )

    return registry, planner, coalgebra, extractor, evidence_policy


# ============================================================================
# MAIN PARIS BRIDGES EXAMPLE
# ============================================================================

def run_paris_example():
    """
    Run complete Paris bridges example demonstrating the framework.
    """
    print("=" * 70)
    print("PARIS BRIDGES EXAMPLE: Categorical Tool Framework")
    print("=" * 70)

    # Query
    query_text = "List 3 landmark bridges in Paris with a one-line fact each."
    print(f"\nQuery: {query_text}\n")

    # Setup framework
    registry, planner, coalgebra, extractor, evidence_policy = setup_framework()

    # ========== STEP 1: CHECK LIMITS & SYNTHESIZE ==========
    print("=" * 70)
    print("STEP 1: Argument Assembly & Limit Checking")
    print("=" * 70)

    context = Context(query=query_text)

    # Check search_web
    search_tool = registry.get("search_web")
    can_invoke, missing = search_tool.can_invoke(context)

    print(f"search_web: can_invoke={can_invoke}, missing={missing}")
    if can_invoke:
        print("  ✓ Limit exists (query projection found)")
    else:
        print(f"  ✗ Limit missing: {missing}")

    # Check extract_facts
    extract_tool = registry.get("extract_facts")
    can_invoke, missing = extract_tool.can_invoke(context)

    print(f"\nextract_facts: can_invoke={can_invoke}, missing={missing}")
    if not can_invoke:
        print(f"  ✗ Missing projections: {missing}")
        print(f"  → Triggering Kan synthesis for missing arguments...")

        # Synthesize missing k
        synth_result = registry.synthesizer.synthesize(
            context,
            extract_tool.schema,
            missing
        )

        if synth_result.is_success():
            print(f"  ✓ Synthesized: {synth_result.value}")
            # Extend context
            for key, value in synth_result.value.items():
                context = context.extend(key, value)
        else:
            print(f"  ✗ Synthesis failed: {synth_result.error}")

    # ========== STEP 2: PLAN AS FREE MONOIDAL PROGRAM ==========
    print("\n" + "=" * 70)
    print("STEP 2: Planning (Free Monoidal Category)")
    print("=" * 70)

    query_obj = QueryObject.from_text(query_text)
    plan = planner.map_query(query_obj, context)

    print(f"Plan generated:")
    print(f"  Root tool: {plan.root.tool_name}")
    print(f"  Bindings: {plan.context_bindings}")
    print(f"  Estimated cost: {plan.total_cost}")

    valid, errors = plan.validate(registry, context)
    print(f"  Valid: {valid}")
    if errors:
        print(f"  Errors: {errors}")

    # ========== STEP 3: EXECUTE IN KLEISLI CATEGORY ==========
    print("\n" + "=" * 70)
    print("STEP 3: Execution (Kleisli Category)")
    print("=" * 70)

    # Execute search_web
    print("Executing: search_web")
    search_result = registry.invoke("search_web", context, use_synthesis=True)

    if search_result.is_success():
        print(f"  ✓ Success: {search_result.value.get('result_count', 0)} results")
        context = context.extend("search_web_result", search_result.value)
    else:
        print(f"  ✗ Failed: {search_result.error}")
        return

    # Execute extract_facts (with synthesis)
    print("\nExecuting: extract_facts")
    extract_result = registry.invoke("extract_facts", context, use_synthesis=True)

    if extract_result.is_success():
        print(f"  ✓ Success: {extract_result.value.get('count', 0)} facts extracted")
        facts = extract_result.value.get("facts", [])
        context = context.extend("facts", facts)
    else:
        print(f"  ✗ Failed: {extract_result.error}")
        return

    # Execute dedupe
    print("\nExecuting: dedupe")
    dedupe_result = registry.invoke("dedupe", context, use_synthesis=True)

    if dedupe_result.is_success():
        print(f"  ✓ Success: {dedupe_result.value.get('count', 0)} unique facts")
        final_facts = dedupe_result.value.get("facts", [])
    else:
        print(f"  ✗ Failed: {dedupe_result.error}")
        return

    # ========== STEP 4: COMPOSE ANSWER ==========
    print("\n" + "=" * 70)
    print("STEP 4: Answer Composition")
    print("=" * 70)

    # Build answer text
    answer_parts = []
    for i, fact in enumerate(final_facts, 1):
        name = fact.get("name", "Unknown")
        fact_text = fact.get("fact", "")
        answer_parts.append(f"{i}. {name}: {fact_text}")

    answer_text = "\n".join(answer_parts)

    answer = Answer(
        text=answer_text,
        confidence=0.8,
        evidence={},
        iteration=0
    )

    print(f"Composed answer:\n{answer.text}\n")
    print(f"Confidence: {answer.confidence}")

    # ========== STEP 5: EXTRACT EVIDENCE (Natural Transformation) ==========
    print("\n" + "=" * 70)
    print("STEP 5: Evidence Extraction (Natural Transformation ε)")
    print("=" * 70)

    evidence = extractor.extract_citations(answer.text, context)

    print(f"Evidence extracted:")
    print(f"  Claims: {len(evidence.claims)}")
    print(f"  Sources: {len(evidence.sources)}")
    print(f"  Morphisms (coend): {evidence.compute_coend()}")

    # Validate evidence
    valid, violations = evidence_policy.check(evidence)
    print(f"\nEvidence policy check: {valid}")
    if violations:
        print(f"Violations:")
        for v in violations:
            print(f"  - {v}")

    # Update answer with evidence
    answer.evidence = evidence.to_dict()

    # ========== STEP 6: CONVERGENCE (Coalgebra Iteration) ==========
    print("\n" + "=" * 70)
    print("STEP 6: Convergence Check (Coalgebra)")
    print("=" * 70)

    checker = ConvergenceChecker()

    # Simulate one refinement iteration
    answer_v2 = Answer(
        text=answer.text + " These bridges are iconic landmarks of Paris.",
        confidence=0.85,
        evidence=answer.evidence,
        iteration=1
    )

    converged, metrics = checker.check_convergence(answer, answer_v2)

    print(f"Convergence metrics:")
    print(f"  Drift: {metrics['drift']:.4f} (threshold: {checker.tau_a})")
    print(f"  ΔConfidence: {metrics['delta_conf']:.4f} (threshold: {checker.tau_c})")
    print(f"  Fragility: {metrics['fragility']:.4f} (threshold: {checker.tau_nu})")
    print(f"  Converged: {converged}")

    # ========== STEP 7: COUNTERFACTUAL ATTACKS (Comonad) ==========
    print("\n" + "=" * 70)
    print("STEP 7: Counterfactual Attacks (Comonad W)")
    print("=" * 70)

    generator = AttackGenerator()
    attacks = generator.generate_attacks(answer, k=3)

    print(f"Generated {len(attacks)} attacks:")
    for attack in attacks:
        print(f"  - {attack.get_name()}")

    executor = CounterfactualExecutor(stability_threshold=0.7)
    attack_results = executor.execute_attack_suite(answer, attacks)

    print(f"\nAttack results:")
    for result in attack_results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"  {result.attack_name}: {status} (stability={result.stability_score:.3f})")

    scorer = RobustnessScorer()
    robustness = scorer.compute_robustness(attack_results)
    fragility = scorer.compute_fragility(attack_results)

    print(f"\nRobustness: {robustness:.3f}")
    print(f"Fragility: {fragility:.3f}")

    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Query: {query_text}")
    print(f"\nAnswer:\n{answer.text}")
    print(f"\nMetrics:")
    print(f"  Confidence: {answer.confidence:.3f}")
    print(f"  Evidence claims: {len(evidence.claims)}")
    print(f"  Evidence sources: {len(evidence.sources)}")
    print(f"  Evidence coend: {evidence.compute_coend()}")
    print(f"  Robustness: {robustness:.3f}")
    print(f"  Fragility: {fragility:.3f}")
    print(f"  Converged: {converged}")

    print("\n✓ Paris bridges example completed successfully!")
    print("=" * 70)

    return {
        "answer": answer.to_dict(),
        "evidence": evidence.to_dict(),
        "convergence": {"converged": converged, "metrics": metrics},
        "robustness": {"score": robustness, "fragility": fragility, "attacks": len(attack_results)}
    }


# ============================================================================
# RUN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    result = run_paris_example()

    print("\n" + "=" * 70)
    print("COMPLETE RESULT (JSON)")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))
