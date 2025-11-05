# HOPF_LENS_DC

**Category-Theoretic Framework for LLM Tool Composition with Formal Guarantees**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-22%20passing-brightgreen.svg)]()

## Overview

HOPF_LENS_DC is a research-grade framework for LLM orchestration that provides mathematically rigorous guarantees for tool composition, convergence, and evidence traceability. The framework combines category theory foundations with practical tool execution, eliminating entire classes of runtime errors through type-level guarantees.

### Core Contributions

1. **Type-Safe Tool Composition**: Tools modeled as Kleisli morphisms with explicit schemas prevent missing-argument bugs
2. **Automatic Parameter Synthesis**: Left Kan extensions fill missing parameters from natural language context
3. **Provable Convergence**: Coalgebra-based iteration with contraction mapping guarantees
4. **Evidence Traceability**: Natural transformations ensure all claims factor through verified sources
5. **Robustness Validation**: Comonad-based counterfactual testing proves answer stability

---

## Installation

### From Source

\`\`\`bash
git clone https://github.com/farukalpay/HOPF_LENS_DC.git
cd HOPF_LENS_DC
pip install -e .
\`\`\`

### Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Requirements:
- Python >= 3.8
- OpenAI API >= 1.0.0
- requests >= 2.28.0
- beautifulsoup4 >= 4.11.0

---

## Quick Start

### Categorical Framework (Type-Safe Execution)

\`\`\`python
import os
from hopf_lens_dc import (
    CategoricalToolRegistry,
    AritySchema,
    DirectAssembler,
    Context,
    PlannerFunctor,
    QueryObject,
)

# Initialize registry
registry = CategoricalToolRegistry()

# Define tool with explicit schema
schema = AritySchema()
schema.add_arg("query", str, required=True)
schema.add_arg("limit", int, required=False, default=10)

# Register tool (enforces limit checking)
assembler = DirectAssembler(schema)
registry.register("search_web", schema, assembler, implementation_func)

# Create planner and execute
planner = PlannerFunctor(registry)
query = QueryObject.from_text("List 3 bridges in Paris")
context = Context(query=query.text)

plan = planner.map_query(query, context)
result = plan.execute(registry, context)
\`\`\`

### Dynamic Tool System (Runtime Flexibility)

\`\`\`bash
python -m hopf_lens_dc.tool --api-key YOUR_KEY --query "What are the main landmarks in Paris?"
\`\`\`

Or programmatically:

\`\`\`python
from hopf_lens_dc.tool import hopf_lens_dc

result = hopf_lens_dc(
    query="What are the main landmarks in Paris?",
    api_key=os.getenv("OPENAI_API_KEY")
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.3f}")
\`\`\`

---

## Examples

### Paris Bridges Example

Demonstrates complete workflow with all categorical components:

\`\`\`bash
python examples/example_paris_bridges.py
\`\`\`

**Demonstrates:**
- Typed tool registration with arity schemas
- Limit checking and validation
- Left Kan synthesis for missing parameter \`k=3\`
- Free monoidal composition: \`search ∘ extract ∘ dedupe\`
- Kleisli category execution with effect tracking
- Evidence extraction via natural transformation
- Coalgebra-based convergence iteration
- Counterfactual robustness testing

**Expected Output:**

\`\`\`
Query: List 3 landmark bridges in Paris with a one-line fact each.

Answer:
1. Pont Neuf: The oldest standing bridge across the Seine, completed in 1607.
2. Pont Alexandre III: Built for the 1900 Exposition Universelle.
3. Pont de la Concorde: Constructed with stones from the Bastille.

Metrics:
  Confidence: 0.800
  Evidence coend: 2
  Robustness: 0.969
  Fragility: 0.031
\`\`\`

---

## Architecture

### Categorical Framework Components

#### 1. Kleisli Morphisms (categorical_core.py)

Tools as typed morphisms \`f: A × C → E[B]\` where:
- **A**: Required argument product (explicit schema)
- **C**: Shared context object
- **E**: Effect monad (HTTP, parsing, timeouts)
- **B**: Result type

\`\`\`python
# Every tool requires explicit schema
schema = AritySchema()
schema.add_arg("query", str, required=True)

# Enforces totality - no partial functions
assembler = DirectAssembler(schema)

# Can only invoke if limit exists: ∀ arg ∈ A, ∃ π: C → arg
can_invoke, missing = tool.can_invoke(context)
\`\`\`

#### 2. Left Kan Extension (Automatic Synthesis)

When projection \`π: C → A\` doesn't exist, compute \`Lan_U(C)\`:

\`\`\`python
synthesizer = KanSynthesizer()
result = synthesizer.synthesize(context, schema, missing_args)

# Example: "List 3 bridges" → extracts k=3
# Query contains "3" → synthesizes integer parameter
\`\`\`

#### 3. Planner Functor (planner.py)

Maps queries to free monoidal category: \`P: Q → Free(T)\`

**Sequential composition (∘):**
\`\`\`python
plan = search_tool ∘ extract_tool ∘ dedupe_tool
\`\`\`

**Parallel composition (⊗):**
\`\`\`python
plan = tool1 ⊗ tool2 ⊗ tool3
\`\`\`

#### 4. Coalgebra Convergence (convergence.py)

Iteration as coalgebra \`γ: X → F(X)\` on metric space:

\`\`\`python
# Answer space with metric d
metric = CompositeMetric([
    (SemanticDriftMetric(), 0.7),
    (ConfidenceMetric(), 0.3)
])

# Coalgebra with contractive endofunctor
functor = AnswerEndofunctor(contraction_factor=0.7)
coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

# Iterate to fixed point
final_answer = coalgebra.iterate(initial, max_iterations=10)
\`\`\`

#### 5. Evidence as Natural Transformation (evidence.py)

Every claim must factor through source: \`ε: Answer ⇒ Citations\`

\`\`\`python
evidence = Evidence()
evidence.add_claim(claim)
evidence.add_source(source)
evidence.add_morphism(claim_id, source_id, strength=0.9)

# Coend must be non-zero
coend = evidence.compute_coend()  # ∫^i Hom(claim_i, source_j)
assert coend > 0, "No evidence - rejected"
\`\`\`

#### 6. Comonad for Attacks (comonad.py)

Counterfactual testing in coKleisli category:

\`\`\`python
# Comonad W with counit and comultiplication
w = ContextComonad(value=answer)

# Counit law: ε ∘ δ = id
assert w.extract() == answer

# Execute attacks
executor = CounterfactualExecutor()
results = executor.execute_attack_suite(answer, attacks)

# Compute robustness
scorer = RobustnessScorer()
robustness = scorer.compute_robustness(results)
\`\`\`

---

## Mathematical Foundations

### Category Theory Concepts

| Concept | Implementation | Guarantee |
|---------|---------------|-----------|
| Kleisli Category | Tools as \`f: A → E[B]\` | Explicit effects, total functions |
| Monads | Effect monad E | Composable error handling |
| Functors | Planner \`P: Q → Free(T)\` | Structure-preserving mapping |
| Natural Transformations | Evidence \`ε: Answer ⇒ Citations\` | Commutes with morphisms |
| Coalgebras | Convergence \`γ: X → F(X)\` | Fixed-point iteration |
| Kan Extensions | Left adjoint \`S ⊣ U\` | Universal synthesis property |
| Free Monoidal Categories | Plan composition | Sequential (∘) and parallel (⊗) |
| Comonads | Context comonad W | Counterfactual validation |

### Formal Properties

**Theorem 1 (No Missing Arguments):**
\`\`\`
∀ tool f, ∀ context C:
  invoke(f, C) succeeds ⟺ ∃ μ_f: C → A such that μ_f is total
\`\`\`

**Theorem 2 (Convergence):**
\`\`\`
If F: X → X is contractive with factor λ < 1, then:
  ∃! x* ∈ X such that F(x*) = x*
  and lim_{n→∞} F^n(x₀) = x* for any x₀
\`\`\`

**Theorem 3 (Evidence Completeness):**
\`\`\`
∀ answer A, ∀ claim c ∈ claims(A):
  ∃ source s ∈ sources(A), ∃ morphism m ∈ Hom(c, s)
\`\`\`

---

## Project Structure

\`\`\`
HOPF_LENS_DC/
├── src/hopf_lens_dc/          # Main library source
│   ├── __init__.py            # Package exports
│   ├── categorical_core.py    # Kleisli morphisms, schemas, assemblers
│   ├── planner.py             # Planner functor P: Q → Free(T)
│   ├── convergence.py         # Coalgebra, metrics, fixed points
│   ├── evidence.py            # Natural transformation ε
│   ├── comonad.py             # Comonad W for counterfactuals
│   └── tool.py                # Dynamic tool system (original)
├── examples/                  # Example scripts
│   └── example_paris_bridges.py
├── tests/                     # Test suite
│   └── test_categorical_framework.py
├── docs/                      # Documentation
│   └── CATEGORICAL_FRAMEWORK.md
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
\`\`\`

---

## Testing

Run the comprehensive test suite:

\`\`\`bash
python -m pytest tests/
\`\`\`

Or directly:

\`\`\`bash
python tests/test_categorical_framework.py
\`\`\`

**Test Coverage:**
- Argument validation (empty dict prevention)
- Limit checking and existence proofs
- Kan synthesis for various types
- Tool invocation and composition
- Plan generation and validation
- Evidence extraction and policies
- Convergence metrics and criteria
- Comonad laws (counit, comultiplication)
- Integration tests

**Result:** 22/22 tests passing

---

## Configuration

### Convergence Parameters

\`\`\`python
TAU_A = 0.02    # Semantic drift threshold
TAU_C = 0.01    # Confidence improvement threshold
TAU_NU = 0.15   # Maximum allowed fragility
\`\`\`

### Execution Limits

\`\`\`python
K_ATTACK = 3    # Counterfactual probes per iteration
K_EXEC = 4      # Tasks per batch
T_MAX = 10      # Maximum iterations
TIME_BUDGET_MS = 60000  # 60 second timeout
\`\`\`

---

## Documentation

### Core Documentation

- **[CATEGORICAL_FRAMEWORK.md](CATEGORICAL_FRAMEWORK.md)**: Complete theoretical foundations, proofs, and API reference
- **[examples/example_paris_bridges.py](examples/example_paris_bridges.py)**: Annotated walkthrough
- **[tests/test_categorical_framework.py](tests/test_categorical_framework.py)**: Test specifications

---

## Use Cases

### Research Applications

- Formal verification of LLM agent systems
- Compositional tool design with provable properties
- Convergence analysis of iterative reasoning
- Evidence-based decision making with traceability

### Production Systems

- Type-safe LLM agents for critical applications
- Traceable AI decision systems
- Robust pipelines with formal guarantees
- Fault-tolerant tool composition

### Education

- Category theory through practical implementation
- Functional programming patterns in AI systems
- Mathematical foundations of LLM orchestration
- Type theory and formal methods

---

## Contributing

Contributions are welcome. Areas of interest:

- Extended synthesizers for complex types
- Additional metrics for convergence
- More sophisticated attack strategies
- Integration with other frameworks
- Performance optimizations

**Development Setup:**

\`\`\`bash
pip install -e ".[dev]"
python -m pytest tests/ --cov=src
\`\`\`

---

## License

MIT License - See LICENSE file for details

---

## References

### Category Theory

1. Kleisli, H. (1965). "Every standard construction is induced by a pair of adjoint functors"
2. Moggi, E. (1991). "Notions of computation and monads"
3. Rutten, J. (2000). "Universal coalgebra: a theory of systems"
4. Mac Lane, S. (1971). "Categories for the Working Mathematician"

### Applications

5. Power, J. & Watanabe, H. (2002). "Combining a monad and a comonad"
6. Uustalu, T. & Vene, V. (2008). "Comonadic notions of computation"

---

## Citation

If you use this framework in academic work, please cite:

\`\`\`bibtex
@software{hopf_lens_dc_2024,
  title = {HOPF_LENS_DC: Categorical Tool Composition Framework},
  author = {HOPF_LENS_DC Contributors},
  year = {2024},
  url = {https://github.com/farukalpay/HOPF_LENS_DC}
}
\`\`\`

---

## Contact

- Issues: [GitHub Issues](https://github.com/farukalpay/HOPF_LENS_DC/issues)
- Documentation: [CATEGORICAL_FRAMEWORK.md](CATEGORICAL_FRAMEWORK.md)
- Examples: [examples/](examples/)

---

**HOPF_LENS_DC** - Category Theory for LLM Orchestration
