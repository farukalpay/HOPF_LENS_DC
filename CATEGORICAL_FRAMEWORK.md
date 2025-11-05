# Categorical Tool Composition Framework

**A type-safe, formally verified framework for composable AI tool execution**

## Table of Contents

- [Overview](#overview)
- [Theoretical Foundation](#theoretical-foundation)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Testing](#testing)

---

## Overview

This framework implements AI tool composition using **category theory** to provide formal guarantees:

1. **No missing arguments**: Tools cannot be invoked with empty or incomplete argument dictionaries
2. **Automatic synthesis**: Missing arguments are synthesized via left Kan extensions
3. **Compositional plans**: Tools compose via free monoidal category (sequential ∘, parallel ⊗)
4. **Convergence guarantees**: Iteration proven contractive under metric space structure
5. **Evidence traceability**: Every claim factors through a source via natural transformations
6. **Robustness testing**: Counterfactual attacks modeled as comonad

### Key Benefits

- **Type Safety**: All tools have explicit schemas; argument mismatches caught before execution
- **Eliminates Empty Dicts**: The `search_web({})` bug is impossible by construction
- **Composable**: Tools compose via category-theoretic operations
- **Convergent**: Iteration guaranteed to reach fixed point or terminate
- **Traceable**: Full provenance chain from claims to sources

---

## Theoretical Foundation

### 1. Tools as Kleisli Morphisms

Each tool is a morphism in the Kleisli category of an effect monad:

```
f: A × C → E[B]
```

Where:
- **A** = required arguments (product type)
- **C** = shared context
- **E** = effect monad (HTTP, parsing, timeouts)
- **B** = result type

#### Arity Schema: Ar(f)

Every tool registers an **arity schema** Ar(f) = A specifying its argument product:

```python
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("limit", int, required=False, default=10)
```

This is the **domain object** in the category.

#### Argument Assembler: μ_f: C → A

Each tool has an **assembler** μ_f that constructs arguments from context:

```python
assembler = DirectAssembler(schema, mappings={"query": "query"})
```

The assembler is a **total function**:
- Returns `Effect[A]` on success
- Returns `Effect.fail(...)` if projections don't exist

#### Invocation Rule

A tool can **only** be invoked if:

```
has_limit(C → A)  ⟺  ∀ arg ∈ A, ∃ π_arg: C → arg
```

If the limit doesn't exist, we attempt **Kan synthesis**.

---

### 2. Left Kan Extension (Synthesis)

When arguments can't be assembled from context, we compute:

```
A' := Lan_U(C)
```

Where:
- **U**: forgetful functor (forgets structure to strings)
- **Lan_U**: left Kan extension (minimal structure synthesizing A from C)

#### Example

```
Query: "List 3 landmark bridges"
Missing: k (count parameter)
Synthesis: Extract "3" from query → k := 3
```

The synthesizer looks for patterns:
- Numbers in query → integer arguments
- Query text → query string arguments
- Keywords → boolean flags

---

### 3. Planner Functor: P: Q → Free(T)

Planning is a functor from **query objects** to the **free monoidal category** over tools:

```
P: Q → Free(T)
```

#### Free Monoidal Structure

Plans are built from:
- **Identity**: `id: T → T`
- **Sequential composition**: `f ∘ g`
- **Parallel composition**: `f ⊗ g`

#### Naturality Square

For any binding β: Q → Q', the following commutes:

```
    Q ----P---→ Free(T)
    |            |
    β           bind
    ↓            ↓
    Q' ---P---→ Free(T')
```

This ensures: **binding then planning = planning then binding**

---

### 4. Convergence as Coalgebra

Answers form a **metric space** (X, d) where convergence is defined as:

```
γ: X → F(X)
```

The coalgebra unfolds answer states:

```
x₀ → F(x₀) → F²(x₀) → ... → x*
```

#### Contraction Property

The endofunctor F is contractive:

```
d(F(x), F(y)) ≤ λ · d(x, y)   where λ < 1
```

By Banach fixed-point theorem, iteration **converges to unique fixed point**.

#### Metrics

We use a composite metric:

```
d(a₁, a₂) = 0.7 · d_semantic(a₁, a₂) + 0.3 · d_confidence(a₁, a₂)
```

Where:
- **d_semantic**: Jaccard distance on token sets
- **d_confidence**: Absolute difference in confidence scores

#### Stopping Criteria

Iteration stops when:

```
d(xₙ₊₁, xₙ) < ε   AND   fragility(xₙ) < τ
```

---

### 5. Evidence as Natural Transformation

Evidence is a natural transformation:

```
ε: Answer ⇒ Citations
```

For every answer, we compute:

```
∫^i Hom(claimᵢ, sourceⱼ)
```

The **coend** counts claim→source morphisms.

#### Policy Enforcement

We reject answers where:

```
∫^i Hom(claimᵢ, sourceⱼ) = 0
```

(No evidence = no claims factored through sources)

#### Morphism Structure

```
Claim ---[strength=0.9]--→ Source
  |
  | derived_from
  ↓
Claim'
```

Each morphism has:
- Claim ID
- Source ID
- Strength ∈ [0, 1]
- Extraction method

---

### 6. Counterfactuals as Comonad

Attacks are modeled in the **coKleisli category** of comonad W:

```
W[A] = (A, Context, [Perturbations])
```

#### Comonad Laws

1. **Counit**: `ε: W → Id`
   ```python
   w.extract() == original_value
   ```

2. **Comultiplication**: `δ: W → W∘W`
   ```python
   w.duplicate() == W[W[A]]
   ```

3. **Associativity**: `δ∘δ = (fmap δ)∘δ`

#### Attack Execution

For each attack:

1. Wrap answer in comonad W
2. Apply perturbation via `extend`
3. Extract result via counit ε
4. Verify counit law: ε∘δ = id

#### Stability Score

```
stability(original, perturbed) =
    0.7 · jaccard_similarity(text) +
    0.3 · (1 - |Δconfidence|)
```

Only accept if stability ≥ threshold.

---

## Core Components

### Module Structure

```
categorical_core.py       # Kleisli morphisms, schemas, assemblers
planner.py               # Planner functor P: Q → Free(T)
convergence.py           # Coalgebra, metrics, fixed-point iteration
evidence.py              # Natural transformation ε: Answer ⇒ Citations
comonad.py               # Comonad W for counterfactual attacks
```

### Key Classes

#### `ToolMorphism`

Represents a tool as a Kleisli morphism:

```python
tool = ToolMorphism(
    name="search_web",
    schema=arity_schema,
    assembler=argument_assembler,
    func=implementation,
    effects=[EffectType.HTTP, EffectType.IO]
)
```

#### `CategoricalToolRegistry`

Registry enforcing limit checks:

```python
registry = CategoricalToolRegistry()
registry.register(name, schema, assembler, func)

# Only invokes if limits exist
result = registry.invoke(name, context, use_synthesis=True)
```

#### `PlannerFunctor`

Maps queries to plans:

```python
planner = PlannerFunctor(registry)
query = QueryObject.from_text("List 3 bridges in Paris")
plan = planner.map_query(query, context)

# Execute plan
result = plan.execute(registry, context)
```

#### `AnswerCoalgebra`

Iterates to fixed point:

```python
coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)
final_answer = coalgebra.iterate(initial, max_iterations=10)
```

#### `Evidence`

Tracks claims and sources:

```python
evidence = Evidence()
evidence.add_claim(claim)
evidence.add_source(source)
evidence.add_morphism(claim_id, source_id, strength=0.9)

# Compute coend
count = evidence.compute_coend()
```

#### `CounterfactualExecutor`

Executes attacks:

```python
executor = CounterfactualExecutor()
results = executor.execute_attack_suite(answer, attacks)

scorer = RobustnessScorer()
robustness = scorer.compute_robustness(results)
```

---

## Usage Guide

### Basic Tool Registration

```python
from categorical_core import (
    CategoricalToolRegistry, AritySchema, DirectAssembler,
    ToolMorphism, Effect, EffectType
)

# 1. Create registry
registry = CategoricalToolRegistry()

# 2. Define schema
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("limit", int, required=False, default=10)

# 3. Create assembler
assembler = DirectAssembler(schema)

# 4. Implement function
def search_impl(args: Dict[str, Any], ctx: Context) -> Effect[Dict]:
    query = args["query"]
    limit = args["limit"]
    # ... implementation ...
    return Effect.pure({"results": [...]})

# 5. Register tool
tool = ToolMorphism(
    name="search_web",
    schema=schema,
    assembler=assembler,
    func=search_impl,
    effects=[EffectType.HTTP]
)
registry.tools["search_web"] = tool

# 6. Invoke (with automatic limit checking and synthesis)
context = Context(query="test")
result = registry.invoke("search_web", context, use_synthesis=True)
```

### Planning and Execution

```python
from planner import PlannerFunctor, QueryObject

# Create planner
planner = PlannerFunctor(registry)

# Parse query
query = QueryObject.from_text("Search for Paris landmarks")
context = Context(query=query.text)

# Generate plan
plan = planner.map_query(query, context)

# Validate
valid, errors = plan.validate(registry, context)

# Execute
if valid:
    result = plan.execute(registry, context)
```

### Convergence Loop

```python
from convergence import (
    Answer, AnswerCoalgebra, AnswerEndofunctor,
    SemanticDriftMetric, ConvergenceChecker
)

# Setup
metric = SemanticDriftMetric()
functor = AnswerEndofunctor(contraction_factor=0.7)
coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

# Initial answer
initial = Answer(text="...", confidence=0.5)

# Iterate to fixed point
final = coalgebra.iterate(initial, max_iterations=10)

# Check convergence
checker = ConvergenceChecker()
converged, metrics = checker.check_convergence(prev, final)
```

### Evidence Extraction

```python
from evidence import (
    ContextAwareExtractor, EvidencePolicy
)

# Extract citations
extractor = ContextAwareExtractor()
evidence = extractor.extract_citations(answer.text, context)

# Validate
policy = EvidencePolicy(
    min_claims=3,
    min_sources=1,
    min_morphisms=3
)
passes, violations = policy.check(evidence)
```

### Counterfactual Testing

```python
from comonad import (
    AttackGenerator, CounterfactualExecutor,
    RobustnessScorer
)

# Generate attacks
generator = AttackGenerator()
attacks = generator.generate_attacks(answer, k=3)

# Execute
executor = CounterfactualExecutor()
results = executor.execute_attack_suite(answer, attacks)

# Score
scorer = RobustnessScorer()
robustness = scorer.compute_robustness(results)
fragility = scorer.compute_fragility(results)
```

---

## Examples

### Complete Example: Paris Bridges

See `example_paris_bridges.py` for a full walkthrough demonstrating:

1. Tool registration with schemas
2. Limit checking (✓ query exists, ✗ k missing)
3. Kan synthesis (extract k=3 from query)
4. Plan generation (search ∘ extract ∘ dedupe)
5. Kleisli execution
6. Evidence extraction
7. Convergence checking
8. Counterfactual attacks

**Run it:**

```bash
python3 example_paris_bridges.py
```

**Expected output:**

```
Query: List 3 landmark bridges in Paris with a one-line fact each.

Answer:
1. Pont Neuf: The Pont Neuf is the oldest standing bridge...
2. Pont Alexandre III: The Pont Alexandre III is an arch bridge...
3. Pont de la Concorde: The Pont de la Concorde was built...

Metrics:
  Confidence: 0.800
  Evidence coend: 2
  Robustness: 0.969
  Fragility: 0.031
```

---

## API Reference

### categorical_core

#### `AritySchema`

```python
schema = AritySchema()
schema.add_arg(name, type, required=True, default=None, description="")

# Check if limit exists
has_limit, missing = schema.has_limit(context)

# Validate assembled args
valid, errors = schema.validate(args_dict)
```

#### `Context`

```python
context = Context(query="...", memory={}, metadata={})

# Check projection exists
has_proj = context.has_projection("key")

# Apply projection
value = context.project("key")

# Extend context
new_context = context.extend("key", value)
```

#### `Effect[T]`

```python
# Create effects
effect = Effect.pure(value)
effect = Effect.fail(error_message)

# Check success
if effect.is_success():
    result = effect.value

# Monadic bind
effect2 = effect.bind(lambda x: Effect.pure(x + 1))
```

#### `KanSynthesizer`

```python
synthesizer = KanSynthesizer()

# Synthesize missing args
result = synthesizer.synthesize(context, schema, missing_args)

if result.is_success():
    synthesized = result.value  # Dict[str, Any]
```

### planner

#### `QueryObject`

```python
query = QueryObject.from_text("Search for Paris")
query.query_type  # QueryType.FACTUAL
query.required_capabilities  # {"search"}
```

#### `Plan`

```python
plan = planner.map_query(query, context)

# Validate
valid, errors = plan.validate(registry, context)

# Execute
result = plan.execute(registry, context)
```

### convergence

#### `Answer`

```python
answer = Answer(
    text="...",
    confidence=0.8,
    evidence={...},
    iteration=0
)
```

#### `AnswerMetric`

```python
metric = SemanticDriftMetric()
distance = metric.distance(answer1, answer2)

# Or composite
metric = CompositeMetric([
    (SemanticDriftMetric(), 0.7),
    (ConfidenceMetric(), 0.3)
])
```

### evidence

#### `Evidence`

```python
evidence = Evidence()
evidence.add_claim(Claim(id="c1", text="..."))
evidence.add_source(Source(id="s1", type=SourceType.WEB))
evidence.add_morphism("c1", "s1", strength=0.9)

# Compute coend
coend = evidence.compute_coend()

# Validate
valid, errors = evidence.validate()
```

### comonad

#### `ContextComonad[A]`

```python
w = ContextComonad(value=answer)

# Counit
extracted = w.extract()

# Duplicate
ww = w.duplicate()

# Extend
w2 = w.extend(lambda x: process(x))
```

#### `CounterfactualExecutor`

```python
executor = CounterfactualExecutor(stability_threshold=0.7)

result = executor.execute_attack(answer, attack)
# result.passed: bool
# result.stability_score: float
```

---

## Testing

Run the test suite:

```bash
python3 test_categorical_framework.py
```

### Key Tests

1. **test_empty_dict_rejected**: Ensures tools cannot be invoked with empty args
2. **test_kan_synthesis_for_integer**: Validates number extraction from query
3. **test_limit_exists_check**: Verifies limit checking logic
4. **test_evidence_missing_sources**: Ensures claims without sources are rejected
5. **test_end_to_end_no_empty_dicts**: Critical integration test

### Test Coverage

- ✓ Argument validation
- ✓ Limit checking
- ✓ Kan synthesis
- ✓ Tool invocation
- ✓ Planning
- ✓ Convergence
- ✓ Evidence extraction
- ✓ Comonad laws
- ✓ Integration

---

## Implementation Notes

### Design Decisions

1. **Total functions everywhere**: No partial functions; all failures explicit via Effect monad
2. **Explicit schemas**: No magic; every tool declares its interface
3. **Immutable contexts**: Contexts are extended, not mutated
4. **Lazy synthesis**: Only synthesize when needed
5. **Early validation**: Check limits before execution

### Performance Considerations

- **Caching**: Planner caches plans by query
- **Parallel execution**: Plans can specify parallel composition (⊗)
- **Incremental**: Coalgebra stops early if converged

### Limitations

- **Synthesis heuristics**: Current synthesizer uses simple pattern matching
- **Evidence extraction**: Uses keyword overlap; could use LLM
- **Metric choice**: Current metrics are simple; could be learned

### Future Extensions

1. **Dependent types**: More expressive schemas
2. **Effect tracking**: Track all effects through composition
3. **Probabilistic**: Model uncertainty throughout
4. **Learning**: Learn synthesis patterns, metrics, attacks

---

## References

### Category Theory

- **Kleisli categories**: Kleisli, H. (1965). "Every standard construction is induced by a pair of adjoint functors"
- **Monads**: Moggi, E. (1991). "Notions of computation and monads"
- **Coalgebras**: Rutten, J. (2000). "Universal coalgebra: a theory of systems"
- **Kan extensions**: Mac Lane, S. (1971). "Categories for the Working Mathematician"

### Applications

- Power, J. & Watanabe, H. (2002). "Combining a monad and a comonad"
- Uustalu, T. & Vene, V. (2008). "Comonadic notions of computation"

---

## License

This framework is part of the HOPF_LENS_DC project.

---

## Contact

For questions or contributions, please see the main README.
