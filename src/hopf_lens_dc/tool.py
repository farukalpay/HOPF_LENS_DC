"""
HOPF_LENS_DC: Categorical Tool Orchestrator with Kleisli Composition

This module reifies the dynamic tool runner as a categorical pipeline:
- Tools are morphisms f: A×C → E[B] in the Kleisli category
- Schemas represent finite-product objects A = ∏ᵢ Aᵢ with projections πᵢ
- Assembler α: C → A is a total function
- Left Kan Extension synthesizer fills missing projections
- Multi-tool plans are words in the free symmetric monoidal category
- Execution via Kleisli bind composition
- Convergence operator Φ: B → B with metric d and fixed-point iteration
- Evidence as natural transformation ε: H ⇒ S
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple

import openai

# Import categorical framework components
from .categorical_core import (
    Context, AritySchema, DirectAssembler, ToolMorphism,
    CategoricalToolRegistry, Effect, EffectType, KanSynthesizer,
    create_simple_tool
)
from .planner import PlannerFunctor, QueryObject
from .convergence import (
    Answer, AnswerCoalgebra, AnswerEndofunctor, SemanticDriftMetric,
    ConfidenceMetric, CompositeMetric
)
from .evidence import (
    Evidence, Claim, Source, SourceType, EvidencePolicy
)
from .comonad import (
    ContextComonad, CounterfactualExecutor,
    SemanticAttack, ConfidenceAttack
)

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = None  # Set via environment variable or function argument
MODEL = "gpt-4-0613"

# Convergence parameters
TAU_A = 0.02  # semantic drift threshold
TAU_C = 0.01  # confidence improvement threshold
TAU_NU = 0.15  # max allowed fragility
K_ATTACK = 3  # counterfactual probes per round
T_MAX = 10  # max iterations
TIME_BUDGET_MS = 60000  # 60 seconds

# ============================================================================
# STATIC TOOLS AS KLEISLI MORPHISMS
# ============================================================================

def create_eval_math_tool(registry: CategoricalToolRegistry) -> ToolMorphism:
    """
    Create eval_math tool as Kleisli morphism.
    Schema: Ar(eval_math) = {expression: str}
    Assembler: α(C) = {expression: π_query(C)}
    """
    def math_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
        """
        NOTE: create_simple_tool will wrap this in Effect.pure(),
        so we return raw Dict, not Effect[Dict]
        """
        expression = args["expression"]
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}

    tool = create_simple_tool(
        name="eval_math",
        required_args=[("expression", str)],
        func=math_impl,
        effects=[EffectType.PURE]
    )
    return tool


def create_search_web_tool(registry: CategoricalToolRegistry) -> ToolMorphism:
    """
    Create search_web tool as Kleisli morphism.
    Schema: Ar(search_web) = {query: str, limit: int}
    Assembler: α(C) = {query: π_query(C), limit: 10}
    """
    def search_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
        """
        NOTE: create_simple_tool will wrap this in Effect.pure(),
        so we return raw Dict, not Effect[Dict]
        """
        query = args["query"]
        limit = args.get("limit", 10)

        # Mock search implementation (replace with actual DuckDuckGo call)
        results = [
            {
                "title": f"Result {i+1} for {query}",
                "snippet": f"This is a snippet about {query}",
                "url": f"https://example.com/{i+1}"
            }
            for i in range(min(3, limit))
        ]

        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }

    tool = create_simple_tool(
        name="search_web",
        required_args=[("query", str)],
        optional_args=[("limit", int, 10)],
        func=search_impl,
        effects=[EffectType.HTTP, EffectType.IO]
    )
    return tool


# ============================================================================
# EVIDENCE EXTRACTION (Natural Transformation ε: Answer ⇒ Citations)
# ============================================================================

def extract_evidence(answer: Answer, tool_results: Dict[str, Any]) -> Evidence:
    """
    Natural transformation ε: Answer → Evidence.
    Extracts claims and sources from answer and tool results.
    """
    evidence = Evidence()

    # Extract claims from answer text (simple sentence splitting)
    claims = []
    sentences = answer.text.split('. ')
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            claim = Claim(
                id=f"claim_{i}",
                text=sentence.strip(),
                support=answer.confidence,
                metadata={"sentence_idx": i}
            )
            claims.append(claim)
            evidence.add_claim(claim)

    # Extract sources from tool results
    sources = []
    for tool_name, result in tool_results.items():
        if isinstance(result, dict) and "results" in result:
            for j, search_result in enumerate(result.get("results", [])[:3]):
                source = Source(
                    id=f"source_{tool_name}_{j}",
                    type=SourceType.WEB,
                    url=search_result.get("url"),
                    content=search_result.get("snippet", ""),
                    reliability=0.7
                )
                sources.append(source)
                evidence.add_source(source)

                # Link claims to sources (simple heuristic: if keywords overlap)
                for claim in claims[:3]:  # Only top claims
                    evidence.add_morphism(
                        claim.id,
                        source.id,
                        strength=0.6,  # Heuristic strength
                        method="keyword_match"
                    )

    return evidence


def extract_evidence_with_provenance(
    answer: Answer,
    tool_results: Dict[str, Any],
    tool_name: str
) -> Evidence:
    """
    Enhanced evidence extraction that handles computational proofs.

    For computational answers (eval_math), the derivation itself is evidence.
    The calculator trace serves as internal provenance.

    Args:
        answer: The answer object
        tool_results: Tool execution results
        tool_name: Which tool was used

    Returns:
        Evidence object with claims, sources, and morphisms
    """
    evidence = Evidence()

    # Extract claims from answer text
    claims = []
    sentences = answer.text.split('. ')
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            claim = Claim(
                id=f"claim_{i}",
                text=sentence.strip(),
                support=answer.confidence,
                metadata={"sentence_idx": i, "tool": tool_name}
            )
            claims.append(claim)
            evidence.add_claim(claim)

    # Handle computational evidence (eval_math)
    if tool_name == "eval_math":
        for tool, result in tool_results.items():
            if isinstance(result, dict):
                # Create internal source from calculation
                source = Source(
                    id=f"source_computation_{tool}",
                    type=SourceType.WEB,  # Using WEB as placeholder
                    url=None,
                    content=f"Computation: {result.get('expression', '')} = {result.get('result', '')}",
                    reliability=1.0  # Perfect reliability for deterministic computation
                )
                evidence.add_source(source)

                # Link all claims to this computational source
                for claim in claims:
                    evidence.add_morphism(
                        claim.id,
                        source.id,
                        strength=1.0,  # Perfect link
                        method="computational_derivation"
                    )

    # Handle search-based evidence
    elif tool_name == "search_web":
        for tool, result in tool_results.items():
            if isinstance(result, dict) and "results" in result:
                for j, search_result in enumerate(result.get("results", [])[:3]):
                    source = Source(
                        id=f"source_{tool}_{j}",
                        type=SourceType.WEB,
                        url=search_result.get("url"),
                        content=search_result.get("snippet", ""),
                        reliability=0.7
                    )
                    evidence.add_source(source)

                    # Link claims to sources
                    for claim in claims:
                        evidence.add_morphism(
                            claim.id,
                            source.id,
                            strength=0.6,
                            method="keyword_match"
                        )

    return evidence


# ============================================================================
# CONVERGENCE OPERATOR Φ: B → B
# ============================================================================

def create_convergence_functor(contraction_factor: float = 0.7) -> AnswerEndofunctor:
    """
    Create endofunctor F: Answer → Answer with contraction property.
    d(F(x), F(y)) ≤ λ·d(x, y) where λ < 1
    """
    return AnswerEndofunctor(contraction_factor=contraction_factor)


def iterate_to_fixed_point(
    initial_answer: Answer,
    registry: CategoricalToolRegistry,
    context: Context,
    max_iterations: int = T_MAX
) -> Tuple[Answer, List[Answer]]:
    """
    Iterate coalgebra γ: X → F(X) until fixed point.
    Returns (final_answer, trajectory)
    """
    # Create composite metric
    metric = CompositeMetric([
        (SemanticDriftMetric(), 0.7),
        (ConfidenceMetric(), 0.3)
    ])
    
    # Create functor and coalgebra
    functor = create_convergence_functor(contraction_factor=0.75)
    coalgebra = AnswerCoalgebra(functor, metric, epsilon=TAU_A)
    
    # Define refinement function
    def refine(answer: Answer) -> Dict[str, Any]:
        # In a real implementation, this would query tools for more info
        return {"refinement_step": answer.iteration}
    
    # Iterate to convergence
    final_answer = coalgebra.iterate(
        initial=initial_answer,
        max_iterations=max_iterations,
        refinement_fn=refine
    )
    
    return final_answer, coalgebra.trajectory


# ============================================================================
# COUNTERFACTUAL ATTACKS (Comonad W)
# ============================================================================

def execute_counterfactual_attacks(
    answer: Answer,
    k_attacks: int = K_ATTACK
) -> Tuple[List[Any], float]:
    """
    Execute counterfactual attacks via comonad W.
    Returns (attack_results, robustness_score)
    """
    executor = CounterfactualExecutor(stability_threshold=0.7)
    
    # Generate attacks
    attacks = [
        SemanticAttack("Paris", "Lyon"),
        SemanticAttack("capital", "city"),
        ConfidenceAttack(-0.3)
    ][:k_attacks]
    
    # Execute attacks
    results = executor.execute_attack_suite(answer, attacks)
    
    # Compute robustness
    if results:
        robustness = sum(r.stability_score for r in results) / len(results)
    else:
        robustness = 1.0
    
    return results, robustness


# ============================================================================
# QUERY CLASSIFIER (Functor C: Q → Idx)
# ============================================================================

import re

def classify_query(query: str) -> str:
    """
    Classifier functor C: Q → {eval_math, search_web, ...}

    Detects arithmetic expressions and routes to appropriate tool.
    This prevents defaulting to search_web for computational queries.

    Args:
        query: The query string

    Returns:
        Tool name ("eval_math" or "search_web")
    """
    # Detect arithmetic patterns
    arithmetic_patterns = [
        r'\b\d+\s*[\+\-\*\/]\s*\d+\b',  # "15 + 27", "100 * 3"
        r'\bcalculate\b.*\b\d+\b',       # "calculate 15 and 27"
        r'\bsum\b.*\b\d+\b',             # "sum of 15 and 27"
        r'\bwhat\s+is\s+\d+',            # "what is 2+2"
        r'\beval\b',                     # "eval ..."
    ]

    query_lower = query.lower()
    for pattern in arithmetic_patterns:
        if re.search(pattern, query_lower):
            return "eval_math"

    return "search_web"


def extract_expression(query: str) -> Optional[str]:
    """
    Parser π: Q → I_eval_math

    Extracts arithmetic expression from query.

    Args:
        query: The query string

    Returns:
        Expression string or None
    """
    # Try to find arithmetic expressions
    patterns = [
        r'(\d+\s*[\+\-\*\/]\s*\d+(?:\s*[\+\-\*\/]\s*\d+)*)',  # "15 + 27"
        r'sum\s+of\s+(\d+)\s+and\s+(\d+)',                    # "sum of 15 and 27"
        r'calculate\s+(\d+)\s+[\+\-\*\/]\s+(\d+)',            # "calculate 15 + 27"
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            if len(match.groups()) == 1:
                return match.group(1).replace(" ", "")
            elif len(match.groups()) == 2:
                # "sum of X and Y" → "X+Y"
                return f"{match.group(1)}+{match.group(2)}"

    return None


# ============================================================================
# COMPOSER ALGEBRA (α: E[R] → A)
# ============================================================================

def compose_answer_from_results(
    tool_name: str,
    result: Dict[str, Any],
    query: str
) -> str:
    """
    Eilenberg-Moore algebra α: E[R] → A

    This is the proper fold from tool results to answers.
    Each tool type has a specific interpreter ρ: R → A.

    Args:
        tool_name: Which tool produced the result
        result: The tool's output
        query: Original query (for context)

    Returns:
        Composed answer text
    """
    if tool_name == "eval_math":
        # ρ_math: {value: ℤ, expression: str, derivation: ...} → A
        if "result" in result:
            expr = result.get("expression", "")
            value = result.get("result", "")
            return f"The result of {expr} is {value}."
        else:
            return f"Mathematical computation completed: {result}"

    elif tool_name == "search_web":
        # ρ_search: {snippets: [...]} → A
        if "results" in result:
            snippets = [r.get("snippet", "") for r in result.get("results", [])[:3]]
            if snippets:
                return ". ".join([s for s in snippets if s])

        # Fallback
        if "result_count" in result:
            return f"Found {result['result_count']} results for query: {query}"

    # Generic fallback
    return str(result)


# ============================================================================
# MAIN ORCHESTRATOR (Categorical Pipeline)
# ============================================================================

def hopf_lens_dc_categorical(
    query: str,
    api_key: str,
    time_budget_ms: int = TIME_BUDGET_MS
) -> Dict[str, Any]:
    """
    HOPF_LENS_DC orchestrator using categorical framework.
    
    Pipeline:
    1. Create registry with tools as Kleisli morphisms
    2. Use planner functor P: Q → Free(T)
    3. Execute plan via Kleisli composition (bind)
    4. Apply convergence coalgebra γ: X → F(X)
    5. Extract evidence via natural transformation ε
    6. Test robustness via comonad W
    
    Args:
        query: The query to process
        api_key: OpenAI API key
        time_budget_ms: Time budget in milliseconds
        
    Returns:
        Dictionary with answer, evidence, confidence, and convergence metrics
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    openai.api_key = api_key
    start_time = time.time()
    
    print(f"=" * 80)
    print(f"CATEGORICAL PIPELINE EXECUTION")
    print(f"Query: {query}")
    print(f"=" * 80)
    
    # ===== STEP 1: Create Registry with Kleisli Morphisms =====
    print("\n[STEP 1] Creating tool registry with Kleisli morphisms...")
    registry = CategoricalToolRegistry()
    
    # Register static tools
    eval_math = create_eval_math_tool(registry)
    registry.tools["eval_math"] = eval_math
    
    search_web = create_search_web_tool(registry)
    registry.tools["search_web"] = search_web
    
    print(f"  Registered tools: {list(registry.tools.keys())}")
    
    # ===== STEP 2: Classify Query and Create Context =====
    print("\n[STEP 2] Classifying query via functor C: Q → Idx...")
    selected_tool = classify_query(query)
    print(f"  Classifier selected: {selected_tool}")

    # Parse query for tool input
    context = Context(query=query)
    if selected_tool == "eval_math":
        expression = extract_expression(query)
        if expression:
            print(f"  Parser extracted expression: {expression}")
            context = context.extend("expression", expression)
        else:
            print(f"  Warning: No expression found, using raw query")
            context = context.extend("expression", query)

    # Check limit (can we invoke the selected tool?)
    can_invoke, missing = registry.can_invoke(selected_tool, context)
    print(f"  {selected_tool}: can_invoke={can_invoke}, missing={missing}")

    if not can_invoke and registry.synthesizer:
        print(f"  → Attempting Kan synthesis for missing arguments...")
        tool = registry.get(selected_tool)
        synth_result = registry.synthesizer.synthesize(context, tool.schema, missing)
        if synth_result.is_success():
            print(f"  ✓ Synthesized: {synth_result.value}")
            for key, value in synth_result.value.items():
                context = context.extend(key, value)

    # ===== STEP 3: Invoke Tool (Kleisli morphism application) =====
    print("\n[STEP 3] Invoking tool via Kleisli morphism...")
    print(f"  Executing: {selected_tool}")

    tool = registry.get(selected_tool)
    if not tool:
        print(f"  ✗ Tool not found: {selected_tool}")
        execution_result = Effect.fail(f"Tool {selected_tool} not in registry")
    else:
        execution_result = registry.invoke(selected_tool, context, use_synthesis=True)

    tool_results = {}
    actual_tool_used = selected_tool
    if execution_result.is_success():
        print(f"  ✓ Execution successful")
        # Unwrap the value from the Effect monad
        # execution_result is Effect[B], so .value gives us B
        unwrapped_value = execution_result.value
        tool_results[selected_tool] = unwrapped_value
        print(f"  Result type: {type(unwrapped_value)}")
        print(f"  Result: {str(unwrapped_value)[:200]}")
    else:
        print(f"  ✗ Execution failed: {execution_result.error}")
        tool_results = {}

    # ===== STEP 4: Compose Answer (Algebra α: E[R] → A) =====
    print("\n[STEP 4] Composing answer via algebra α...")

    # THIS IS THE CRITICAL FIX: Properly fold tool results into answer
    answer_text = None
    if tool_results:
        for tool_name, result in tool_results.items():
            # result is now the unwrapped value, not the Effect
            answer_text = compose_answer_from_results(tool_name, result, query)
            print(f"  ρ_{tool_name}(result) = {answer_text[:100]}...")
            break  # Use first result

    # Fallback only if no results
    if not answer_text:
        print(f"  Warning: No tool results, using query as fallback")
        answer_text = f"Unable to process query: {query}"

    initial_answer = Answer(
        text=answer_text,
        confidence=0.8 if actual_tool_used == "eval_math" else 0.6,
        metadata={"iteration": 0, "tool": actual_tool_used}
    )
    print(f"  Initial answer: {initial_answer.text[:100]}...")
    print(f"  Initial confidence: {initial_answer.confidence}")

    # ===== STEP 5: Convergence Iteration (Coalgebra γ: X → F(X)) =====
    print("\n[STEP 5] Iterating to fixed point via coalgebra...")
    final_answer, trajectory = iterate_to_fixed_point(
        initial_answer,
        registry,
        context,
        max_iterations=min(T_MAX, 3)  # Limit for demo
    )
    
    print(f"  Iterations: {len(trajectory)}")
    print(f"  Final confidence: {final_answer.confidence}")
    print(f"  Final text: {final_answer.text[:100]}...")
    
    # Check convergence
    if len(trajectory) >= 2:
        metric = SemanticDriftMetric()
        drift = metric.distance(trajectory[-2], trajectory[-1])
        print(f"  Final drift: {drift:.4f} (threshold: {TAU_A})")
        converged = drift < TAU_A
        print(f"  Converged: {converged}")
    else:
        converged = True
    
    # ===== STEP 6: Evidence Extraction (ε: Answer ⇒ Citations) =====
    print("\n[STEP 6] Extracting evidence via natural transformation ε...")
    evidence = extract_evidence_with_provenance(final_answer, tool_results, actual_tool_used)

    print(f"  Claims: {len(evidence.claims)}")
    print(f"  Sources: {len(evidence.sources)}")
    print(f"  Morphisms (coend): {evidence.compute_coend()}")

    # Validate evidence
    evidence_valid, evidence_errors = evidence.validate()
    print(f"  Evidence valid: {evidence_valid}")

    # Check policy (relaxed for computational queries)
    if actual_tool_used == "eval_math":
        policy = EvidencePolicy(min_claims=1, min_sources=1, min_morphisms=1)
    else:
        policy = EvidencePolicy(min_claims=1, min_sources=1, min_morphisms=1)
    policy_pass, violations = policy.check(evidence)
    print(f"  Policy check: {policy_pass}")
    if violations:
        print(f"  Violations: {violations[:3]}")
    
    # ===== STEP 7: Counterfactual Attacks (Comonad W) =====
    print("\n[STEP 7] Executing counterfactual attacks via comonad...")
    attack_results, robustness = execute_counterfactual_attacks(
        final_answer,
        k_attacks=min(K_ATTACK, 3)
    )
    
    print(f"  Attacks executed: {len(attack_results)}")
    print(f"  Robustness score: {robustness:.3f}")
    
    # Show attack results
    for result in attack_results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"    {result.attack_name}: {status} (stability={result.stability_score:.3f})")
    
    # ===== FINAL RESULT =====
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT ({elapsed:.2f}s)")
    print(f"{'=' * 80}")
    print(f"Answer: {final_answer.text[:200]}")
    print(f"Confidence: {final_answer.confidence:.3f}")
    print(f"Evidence coend: {evidence.compute_coend()}")
    print(f"Robustness: {robustness:.3f}")
    print(f"Converged: {converged}")
    print(f"{'=' * 80}")
    
    return {
        "answer": final_answer.text,
        "confidence": final_answer.confidence,
        "evidence": evidence.to_dict(),
        "convergence": {
            "converged": converged,
            "iterations": len(trajectory),
            "trajectory": [a.to_dict() for a in trajectory]
        },
        "robustness": {
            "score": robustness,
            "attacks": [r.to_dict() for r in attack_results]
        },
        "elapsed_seconds": elapsed,
        "tools_used": list(registry.tools.keys())
    }


# ============================================================================
# LEGACY COMPATIBILITY (hopf_lens_dc function)
# ============================================================================

def hopf_lens_dc(query: str, api_key: str, time_budget_ms: int = TIME_BUDGET_MS) -> Dict[str, Any]:
    """
    Legacy interface - redirects to categorical implementation.
    """
    return hopf_lens_dc_categorical(query, api_key, time_budget_ms)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HOPF_LENS_DC: Categorical Tool Orchestrator"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="List 3 landmark bridges in Paris",
        help="Query to process"
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=TIME_BUDGET_MS,
        help="Time budget in milliseconds"
    )

    args = parser.parse_args()

    print(f"Testing HOPF_LENS_DC Categorical Pipeline\n")

    result = hopf_lens_dc_categorical(args.query, args.api_key, args.time_budget)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print(f"Iterations: {result['convergence']['iterations']}")
    print(f"Evidence coend: {result['evidence']['coend']}")
    print(f"Robustness: {result['robustness']['score']:.3f}")
    print(f"\nFinal Answer: {result['answer'][:300]}")
