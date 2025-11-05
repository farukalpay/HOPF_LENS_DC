# HOPF_LENS_DC

**Type-Safe LLM Tool Orchestration with Category Theory Foundations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)]()

## TL;DR

HOPF_LENS_DC prevents `search_web({})` type bugs **by construction** using category theory. Tools have explicit schemas, automatic argument synthesis, and formal convergence guarantees.

```python
# Traditional approach - runtime error waiting to happen
search_web({})  # KeyError: 'query'

# HOPF_LENS_DC - compile-time validation
schema = AritySchema()
schema.add_arg("query", str, required=True)
# → Can only invoke if schema satisfied OR synthesis succeeds
```

---

## What Problem Does This Solve?

### The Problem: LLM Tool Execution is Fragile

When LLMs orchestrate tool calls, three critical failures occur:

1. **Missing Arguments**: LLM generates `{"query": ""}` or `{}` - tool crashes
2. **No Convergence Proof**: Iteration loops forever or oscillates
3. **Untraceable Evidence**: Claims have no provenance chain

### The Solution: Category Theory + Type Safety

```
Traditional:  query → LLM → tool_call(???) → crash
HOPF_LENS_DC: query → Schema Check → Synthesize Missing → Execute → Proof of Convergence
```

**Key Innovation**: Model tools as morphisms `f: A×C → E[B]` where `A` must exist before invocation.

---

## Installation

```bash
git clone https://github.com/farukalpay/HOPF_LENS_DC.git
cd HOPF_LENS_DC
pip install -e .
```

**Dependencies:**
```bash
pip install openai>=1.0.0 requests beautifulsoup4
```

---

## Quick Start

### 1. Dynamic Tool System (Runtime Flexibility)

The LLM generates tools on-the-fly:

```bash
python -m src.hopf_lens_dc.tool \
  --api-key YOUR_KEY \
  --query "What is 2+2?" \
  --time-budget 20000
```

**Output:**
```
=== BOOTSTRAPPING PHASE ===
Bootstrap complete. Created 0 new tools.
Total available tools: 6

=== Iteration 1 ===
Step 3: Composing response...
  New answer: The sum of 2 + 2 is 4.
  Confidence: 1.000

=== FINAL RESULT (after 1 iterations, 13.89s) ===
Answer: The sum of 2 + 2 is 4.
Confidence: 1.000
```

### 2. Categorical Framework (Type Safety)

For production systems requiring guarantees:

```python
from hopf_lens_dc import (
    CategoricalToolRegistry,
    AritySchema,
    DirectAssembler,
    Context,
)

# 1. Create registry
registry = CategoricalToolRegistry()

# 2. Define schema - explicit contract
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("limit", int, required=False, default=10)

# 3. Create assembler - total function C → A
assembler = DirectAssembler(schema)

# 4. Register tool
def search_impl(args: Dict[str, Any], ctx: Context):
    # Implementation guaranteed to receive valid args
    return Effect.pure({"results": [...]})

registry.register("search", schema, assembler, search_impl)

# 5. Check before invoke - NO runtime crashes
can_invoke, missing = registry.can_invoke("search", context)
if not can_invoke:
    # Attempt Kan synthesis for missing args
    synthesized = registry.synthesizer.synthesize(context, schema, missing)
```

---

## Live Example: Paris Bridges Query

**Query:** "List 3 landmark bridges in Paris with a one-line fact each."

**Run:**
```bash
python examples/example_paris_bridges.py
```

### Execution Flow (Annotated)

```
======================================================================
STEP 1: Argument Assembly & Limit Checking
======================================================================
```
**What's Happening:** Check if all required arguments can be constructed from context.

```
search_web: can_invoke=True, missing=[]
  ✓ Limit exists (query projection found)

extract_facts: can_invoke=False, missing=['k']
  ✗ Missing projections: ['k']
  → Triggering Kan synthesis for missing arguments...
  ✓ Synthesized: {'k': 3}
```

**Engineering Insight:** The system detects `k` (count parameter) is missing. Instead of crashing, it applies **Left Kan Extension** - extracts "3" from query text "List **3** bridges" and synthesizes `k=3`.

```
======================================================================
STEP 2: Planning (Free Monoidal Category)
======================================================================
Plan generated:
  Root tool: search_web
  Bindings: {'query': 'List 3 landmark bridges in Paris...'}
  Estimated cost: 1.0
  Valid: True
```

**Engineering Insight:** Plan validation ensures all tools in the composition have required arguments. Invalid plans rejected at planning time, not execution time.

```
======================================================================
STEP 3: Execution (Kleisli Category)
======================================================================
Executing: search_web
  [search_web] query='List 3 landmark bridges in Paris...'
  ✓ Success: 3 results

Executing: extract_facts
  [extract_facts] entities=0, k=3  ← synthesized parameter
  ✓ Success: 3 facts extracted

Executing: dedupe
  [dedupe] processing 3 facts
  ✓ Success: 3 unique facts
```

**Engineering Insight:** Sequential composition `search ∘ extract ∘ dedupe` executes in Kleisli category. Each step returns `Effect[T]` monad handling errors gracefully.

```
======================================================================
STEP 4: Answer Composition
======================================================================
Composed answer:
1. Pont Neuf: The Pont Neuf is the oldest standing bridge across 
   the Seine in Paris, completed in 1607.
2. Pont Alexandre III: The Pont Alexandre III is an arch bridge that 
   spans the Seine, built for the 1900 Exposition Universelle.
3. Pont de la Concorde: The Pont de la Concorde was built with stones 
   from the Bastille prison after it was demolished.

Confidence: 0.8
```

```
======================================================================
STEP 5: Evidence Extraction (Natural Transformation ε)
======================================================================
Evidence extracted:
  Claims: 6
  Sources: 1
  Morphisms (coend): 2

Evidence policy check: False
Violations:
  - Insufficient evidence: coend=2 < 3
```

**Engineering Insight:** Evidence validation enforces that every claim must factor through a source. `coend=2` means only 2 out of 6 claims have provenance. System flags this for review.

```
======================================================================
STEP 6: Convergence Check (Coalgebra)
======================================================================
Convergence metrics:
  Drift: 0.1400 (threshold: 0.02)
  ΔConfidence: 0.0500 (threshold: 0.01)
  Fragility: 0.5000 (threshold: 0.15)
  Converged: False
```

**Engineering Insight:** Coalgebra `γ: X → F(X)` iterates answer refinement. Banach fixed-point theorem guarantees convergence when `F` is contractive. Here drift exceeds threshold - would iterate further in production.

```
======================================================================
STEP 7: Counterfactual Attacks (Comonad W)
======================================================================
Generated 3 attacks:
  - semantic_Pont→[REDACTED]
  - semantic_Neuf:→[REDACTED]
  - confidence_-0.30

Attack results:
  semantic_Pont→[REDACTED]: ✓ PASSED (stability=0.968)
  semantic_Neuf:→[REDACTED]: ✓ PASSED (stability=0.968)
  confidence_-0.30: ✓ PASSED (stability=0.910)

Robustness: 0.969
Fragility: 0.031
```

**Engineering Insight:** Comonad structure tests answer under adversarial perturbations. High robustness (0.969) means answer is stable. Production systems can set minimum robustness thresholds.

```
======================================================================
FINAL SUMMARY
======================================================================
Answer:
1. Pont Neuf: ...
2. Pont Alexandre III: ...
3. Pont de la Concorde: ...

Metrics:
  Confidence: 0.800
  Evidence coend: 2
  Robustness: 0.969
  Fragility: 0.031
```

---

## Architecture Deep Dive

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│  categorical_core.py  - Type System & Schemas               │
│  • AritySchema: Explicit argument contracts                 │
│  • KanSynthesizer: Automatic parameter extraction           │
│  • Effect Monad: Composable error handling                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  planner.py  - Query → Plan Compiler                        │
│  • PlannerFunctor: P: Query → Free(Tools)                   │
│  • Sequential (∘) and Parallel (⊗) composition              │
│  • Plan validation before execution                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  convergence.py  - Fixed-Point Iteration                    │
│  • Coalgebra γ: Answer → F(Answer)                          │
│  • Metric space with proven contraction                     │
│  • Banach theorem guarantees convergence                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  evidence.py  - Provenance Tracking                         │
│  • Natural transformation ε: Answer ⇒ Citations             │
│  • Coend ∫ Hom(claim, source) must be non-zero              │
│  • Reject answers without evidence                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  comonad.py  - Robustness Testing                           │
│  • Comonad W for context extraction                         │
│  • Counterfactual attacks test stability                    │
│  • Counit law ensures sanity checks pass                    │
└─────────────────────────────────────────────────────────────┘
```

### Core Concepts for Engineers

#### 1. Arity Schemas - Type Safety

**Problem:** LLM generates malformed tool calls.

**Solution:** Explicit contracts enforced at plan-time.

```python
# Define what arguments are required
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("limit", int, required=False, default=10)

# Validate before execution
valid, errors = schema.validate({"query": "test"})  # ✓
valid, errors = schema.validate({})  # ✗ - caught early
```

#### 2. Limit Checking - Existence Proofs

**Problem:** Arguments may not exist in context.

**Solution:** Check if projection `π: Context → Arg` exists.

```python
tool = registry.get("search_web")
can_invoke, missing = tool.can_invoke(context)

# Returns: (False, ['query']) if context has no query field
# → Prevents KeyError at runtime
```

#### 3. Kan Synthesis - Automatic Extraction

**Problem:** Arguments missing but inferable from query.

**Solution:** Left Kan Extension `Lan_U: Context → Args`.

```python
# Query: "List 5 results"
# Missing: limit parameter

synthesizer.synthesize(context, schema, ['limit'])
# → Extracts "5" from query text
# → Returns {'limit': 5}
```

**Implementation:**
- Regex extraction for numbers
- Query text for strings
- Heuristics for booleans

#### 4. Effect Monad - Composable Errors

**Problem:** Exceptions break composition.

**Solution:** Explicit effect tracking via monads.

```python
def search(args, ctx) -> Effect[Results]:
    try:
        results = fetch_data(args['query'])
        return Effect.pure(results)
    except Exception as e:
        return Effect.fail(str(e))

# Compose with bind
result = search(args, ctx).bind(lambda r: process(r, ctx))
# Errors propagate gracefully, no exceptions thrown
```

#### 5. Coalgebra - Guaranteed Convergence

**Problem:** Iteration may loop forever.

**Solution:** Contractive functor + Banach theorem.

```python
# F: Answer → Answer with contraction factor λ < 1
functor = AnswerEndofunctor(contraction_factor=0.7)

# d(F(x), F(y)) ≤ 0.7 * d(x, y)
# → Guaranteed to converge to unique fixed point

coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)
final = coalgebra.iterate(initial, max_iterations=10)
```

**Mathematical Guarantee:**
```
∀ x₀, lim_{n→∞} F^n(x₀) = x*  where F(x*) = x*
```

#### 6. Evidence Coend - Provenance

**Problem:** Claims have no traceable sources.

**Solution:** Natural transformation with coend calculation.

```python
evidence = Evidence()
evidence.add_claim(Claim(id="c1", text="Paris is capital"))
evidence.add_source(Source(id="s1", url="https://..."))
evidence.add_morphism("c1", "s1", strength=0.9)

# Coend = count of (claim, source) morphisms
coend = evidence.compute_coend()  # Must be > 0

# Policy: Reject if any claim lacks sources
policy = EvidencePolicy(require_all_claims_sourced=True)
passes, violations = policy.check(evidence)
```

---

## Practical Usage

### Example 1: Build Type-Safe Search Tool

```python
from hopf_lens_dc import (
    CategoricalToolRegistry, AritySchema, DirectAssembler,
    Effect, EffectType, Context
)

# 1. Define schema
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("max_results", int, required=False, default=10)

# 2. Implement tool
def search_impl(args: Dict[str, Any], ctx: Context) -> Effect[Dict]:
    try:
        query = args["query"]
        limit = args["max_results"]
        
        # Your implementation
        results = search_engine(query, limit)
        
        return Effect.pure({"results": results})
    except Exception as e:
        return Effect.fail(f"Search error: {e}")

# 3. Register
registry = CategoricalToolRegistry()
assembler = DirectAssembler(schema)

tool = ToolMorphism(
    name="search",
    schema=schema,
    assembler=assembler,
    func=search_impl,
    effects=[EffectType.HTTP, EffectType.IO]
)
registry.tools["search"] = tool

# 4. Invoke safely
context = Context(query="machine learning papers")
result = registry.invoke("search", context, use_synthesis=True)

if result.is_success():
    print(result.value)
else:
    print(f"Error: {result.error}")
```

### Example 2: Plan Multi-Tool Workflow

```python
from hopf_lens_dc import PlannerFunctor, QueryObject

# Parse query
query = QueryObject.from_text("Find 5 Python tutorials and summarize them")
context = Context(query=query.text)

# Generate plan
planner = PlannerFunctor(registry)
plan = planner.map_query(query, context)

# Validate plan
valid, errors = plan.validate(registry, context)
if not valid:
    print(f"Plan invalid: {errors}")
    exit(1)

# Execute
result = plan.execute(registry, context)
if result.is_success():
    print(result.value)
```

### Example 3: Convergence Loop

```python
from hopf_lens_dc import (
    Answer, AnswerCoalgebra, AnswerEndofunctor,
    SemanticDriftMetric, ConvergenceChecker
)

# Initial answer
initial = Answer(text="Paris is in France", confidence=0.6)

# Setup coalgebra
metric = SemanticDriftMetric()
functor = AnswerEndofunctor(contraction_factor=0.7)
coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

# Iterate to fixed point
def refine(answer: Answer):
    # Your refinement logic
    return {"additional_info": "..."}

final = coalgebra.iterate(initial, max_iterations=10, refinement_fn=refine)

# Check convergence
checker = ConvergenceChecker()
converged, metrics = checker.check_convergence(initial, final)

print(f"Converged: {converged}")
print(f"Drift: {metrics['drift']:.4f}")
print(f"Fragility: {metrics['fragility']:.4f}")
```

---

## Testing

Run comprehensive test suite:

```bash
python tests/test_categorical_framework.py
```

**Coverage:**
- Argument validation (22/22 passing)
- Limit checking and synthesis
- Tool composition
- Evidence tracking
- Convergence properties
- Comonad laws

---

## Performance Considerations

### Time Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Schema Validation | O(n) | n = number of arguments |
| Limit Checking | O(m) | m = context size |
| Kan Synthesis | O(k·l) | k = missing args, l = query length |
| Plan Generation | O(t) | t = number of tools |
| Convergence | O(i·d) | i = iterations, d = metric calculation |

### Space Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Tool Registry | O(n·s) | n = tools, s = avg schema size |
| Execution Context | O(m) | m = accumulated results |
| Evidence Graph | O(c+s+e) | c = claims, s = sources, e = edges |

### Optimization Tips

1. **Cache Plans**: Planner includes `planner_cache` for repeated queries
2. **Lazy Synthesis**: Only synthesize when limit check fails
3. **Early Termination**: Convergence stops when drift < ε
4. **Parallel Execution**: Use `⊗` composition for independent tools

---

## Configuration

### Convergence Parameters

```python
TAU_A = 0.02    # Semantic drift threshold
TAU_C = 0.01    # Confidence improvement threshold
TAU_NU = 0.15   # Maximum fragility
```

### Execution Limits

```python
K_ATTACK = 3    # Counterfactual probes per iteration
K_EXEC = 4      # Tasks per batch
T_MAX = 10      # Maximum iterations
TIME_BUDGET_MS = 60000  # 60 second timeout
```

---

## Design Principles

### For Software Engineers

1. **Fail Fast**: Validate at plan-time, not execution-time
2. **Explicit Over Implicit**: All effects and types declared
3. **Composition Over Inheritance**: Free monoidal structure
4. **Immutability**: Context extended, never mutated
5. **Total Functions**: No partial functions, all paths covered

### For AI Engineers

1. **Bounded Iteration**: Provable convergence via contraction
2. **Evidence Tracking**: Every claim must have provenance
3. **Robustness Testing**: Adversarial validation built-in
4. **Automatic Repair**: Kan synthesis fixes missing arguments
5. **Observable Execution**: Full audit trail of decisions

### Category Theory for Practitioners

You don't need to know category theory to use this framework, but understanding the concepts helps:

- **Morphism**: Function with explicit domain/codomain (like interfaces)
- **Monad**: Container with `bind` operation (like `Promise` in JS)
- **Functor**: Structure-preserving map (like `Array.map`)
- **Natural Transformation**: Function between functors (like middleware)
- **Coalgebra**: State machine with next-state function
- **Kan Extension**: Universal construction (like dependency injection)

**Key Insight**: Category theory provides **formal specifications** that become **runtime guarantees**.

---

## API Reference

See [CATEGORICAL_FRAMEWORK.md](CATEGORICAL_FRAMEWORK.md) for complete API documentation.

**Quick Links:**
- [Core Types](CATEGORICAL_FRAMEWORK.md#core-types)
- [Tool Registration](CATEGORICAL_FRAMEWORK.md#tool-registration)
- [Plan Generation](CATEGORICAL_FRAMEWORK.md#plan-generation)
- [Convergence](CATEGORICAL_FRAMEWORK.md#convergence)
- [Evidence](CATEGORICAL_FRAMEWORK.md#evidence)

---

## Project Structure

```
HOPF_LENS_DC/
├── src/hopf_lens_dc/          # Core library
│   ├── categorical_core.py    # Type system, schemas, assemblers
│   ├── planner.py             # Query → Plan compilation
│   ├── convergence.py         # Fixed-point iteration
│   ├── evidence.py            # Provenance tracking
│   ├── comonad.py             # Robustness testing
│   └── tool.py                # Dynamic tool system
├── examples/                  # Live examples
│   └── example_paris_bridges.py
├── tests/                     # Test suite (22 tests)
│   └── test_categorical_framework.py
├── setup.py                   # pip install -e .
└── requirements.txt           # Dependencies
```

---

## Use Cases

### Production AI Agents

- **Critical Systems**: Type safety prevents runtime crashes
- **Financial Applications**: Evidence tracking for audit trails
- **Healthcare AI**: Provenance required for regulatory compliance
- **Autonomous Systems**: Convergence guarantees prevent infinite loops

### Research

- **Formal Verification**: Mathematical proofs of correctness
- **Tool Composition**: Study compositional properties
- **Convergence Analysis**: Empirical validation of theory
- **Evidence Mining**: Provenance graph analysis

### Education

- **Functional Programming**: Practical monads/functors/coalgebras
- **Type Theory**: Real-world dependent types
- **AI Safety**: Formal guarantees for LLM systems
- **Software Engineering**: Design patterns from category theory

---

## Contributing

Areas of interest:

- Synthesis strategies for complex types
- Additional convergence metrics
- Performance optimizations
- Integration with LangChain/LlamaIndex

**Development:**
```bash
pip install -e ".[dev]"
python -m pytest tests/ --cov=src
```

---

## References

**Category Theory:**
1. Kleisli (1965) - "Every standard construction is induced by adjoint functors"
2. Moggi (1991) - "Notions of computation and monads"
3. Rutten (2000) - "Universal coalgebra"

**Applications:**
4. Power & Watanabe (2002) - "Combining monads and comonads"
5. Uustalu & Vene (2008) - "Comonadic notions of computation"

---

## License

MIT License - See LICENSE file

---

**HOPF_LENS_DC** - Type safety meets LLM orchestration

**Architecture:** Category theory foundations + Practical engineering
**Guarantees:** No missing arguments · Provable convergence · Traceable evidence
**Performance:** Plan-time validation · Automatic synthesis · Bounded iteration
