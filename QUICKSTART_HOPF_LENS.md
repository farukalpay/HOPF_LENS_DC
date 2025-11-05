# HOPF/Lens Framework - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Run the Tests

Verify the implementation works:

```bash
python tests/test_hopf_lens.py
```

**Expected output**:
```
======================================================================
HOPF/Lens Framework - Comprehensive Test Suite
======================================================================
...
Ran 34 tests in 0.003s
OK
======================================================================
Tests run: 34
Successes: 34
Failures: 0
Errors: 0
======================================================================
```

✓ All 34 tests pass

### 2. Run the Worked Example

See schema evolution in action:

```bash
python examples/example_hopf_lens_weather.py
```

**What it demonstrates**:
- Schema evolution: `{city, units: {C,F}}` → `{location: {city, country}, units: {metric, imperial}}`
- Lens construction with PutGet/GetPut laws
- Kan extension migration
- Argument synthesis with defaults
- Soundness verification

### 3. Run the CLI Demo

Experience all components:

```bash
# All scenarios
python examples/cli_hopf_lens_demo.py --mode all

# Or individual scenarios:
python examples/cli_hopf_lens_demo.py --mode evolution   # Schema evolution
python examples/cli_hopf_lens_demo.py --mode correction  # Self-correction
python examples/cli_hopf_lens_demo.py --mode soundness   # Property verification
python examples/cli_hopf_lens_demo.py --mode hopf        # HOPF morphisms
```

---

## 📚 Core Concepts (10 minutes)

### Concept 1: Schema as Category

```python
from src.hopf_lens_dc.hopf_lens import JSONSchema, JSONType

# Define a schema (object in category Sch)
schema = JSONSchema(
    name="user",
    type=JSONType.OBJECT,
    properties={
        "name": JSONSchema(name="name", type=JSONType.STRING),
        "age": JSONSchema(name="age", type=JSONType.INTEGER)
    },
    required={"name"}
)
```

**Mathematical meaning**: Object in category **Sch** of JSON schemas.

### Concept 2: Schema Edits (Morphisms)

```python
from src.hopf_lens_dc.hopf_lens import SchemaEdit, EditType

# Schema edit σ: S_old → S_new
edit = SchemaEdit(
    edit_type=EditType.RENAME,
    source_path=["old_name"],
    target_path=["new_name"],
    cost=0.5
)
```

**Mathematical meaning**: Morphism in **Sch**.

### Concept 3: Lenses (Bidirectional Transformations)

```python
from src.hopf_lens_dc.hopf_lens import create_lens_for_edit

# Create lens from edit
lens = create_lens_for_edit(edit)

# Forward: get(source) → target
target = lens.get({"old_name": "value"})
# → {"new_name": "value"}

# Backward: put(source, target) → source
source = lens.put({"old_name": "old"}, {"new_name": "new"})
# → {"old_name": "new"}

# Verify laws
lens.validate_put_get(source, target)  # PutGet law
lens.validate_get_put(source)          # GetPut law
```

**Mathematical meaning**: Lens **L = (get, put)** with laws.

### Concept 4: Argument Synthesis

```python
from src.hopf_lens_dc.hopf_lens import ArgumentSynthesizer, Contract
from src.hopf_lens_dc.categorical_core import Context

# Define contract
contract = Contract(
    schema=new_schema,
    predicate=lambda v, c: "name" in v and len(v["name"]) > 0,
    description="Non-empty name"
)

# Synthesize from partial arguments
synthesizer = ArgumentSynthesizer()
context = Context(query="John Doe", metadata={"default_age": 30})

result = synthesizer.synthesize(
    partial_args={"old_name": "John"},  # Partial/old format
    context=context,
    source_schema=old_schema,
    target_schema=new_schema,
    contract=contract
)

# Result:
{
    "arguments": {"name": "John", "age": 30},  # Complete new format
    "cost": 0.8,                                # Synthesis cost
    "edits": ["rename", "default_add"]         # Applied edits
}
```

**Mathematical meaning**: **SYN_σ(â, c) ∈ arg min |ρ|** s.t. **(ρ(â,c), c) ⊨ Φ**

### Concept 5: HOPF Morphisms

```python
from src.hopf_lens_dc.hopf_lens import HOPFToolMorphism
from src.hopf_lens_dc.categorical_core import Effect

# Define tool
def weather_api(args: dict, ctx: Context) -> Effect:
    city = args["city"]
    units = args.get("units", "metric")
    return Effect.pure({
        "city": city,
        "temperature": 15,
        "units": units
    })

# Create HOPF morphism
morphism = HOPFToolMorphism(
    name="get_weather",
    source_schema=old_schema,
    target_schema=new_schema,
    contract=contract,
    func=weather_api
)

# Invoke with partial arguments (automatic repair!)
result = morphism.invoke(
    partial_args={"city": "Paris"},  # Missing fields!
    context=context
)

# Result: Synthesis fills missing fields, validates, executes
if result.is_success():
    print(result.value)  # {"city": "Paris", "temperature": 15, ...}
```

**Mathematical meaning**: **t♯: A × C → E[B]** with extension **t̄♯: A_⊥ × C → E[B]**

### Concept 6: Self-Correction Coalgebra

```python
from src.hopf_lens_dc.hopf_lens import (
    CorrectionState, CorrectionEndofunctor, SelfCorrectionCoalgebra
)

# Initial state
initial = CorrectionState(
    context=Context(query="task"),
    plan=["step1", "step2"],
    max_retries=10
)

# Endofunctor with contraction
functor = CorrectionEndofunctor(contraction_factor=0.7)  # λ < 1

# Coalgebra
coalgebra = SelfCorrectionCoalgebra(
    functor=functor,
    max_iterations=10,
    epsilon=0.01
)

# Correction step
def step(state: CorrectionState) -> CorrectionState:
    # Your correction logic
    if error_detected(state):
        state.plan = fix_plan(state.plan)
    else:
        state.last_result = "success"
        state.last_error = None
    return state

# Iterate to fixed point
result = coalgebra.iterate(initial, step)

# Convergence guaranteed by Banach theorem!
if result.is_success():
    print(f"Converged to: {result.value.last_result}")
```

**Mathematical meaning**: Coalgebra **α: X → F(X)** with **lfp(Φ) = ⊔ₙ Φⁿ(⊥)**

---

## 🎯 Real-World Example

### Scenario: Evolving a Search API

**Old API (v1)**:
```json
{
  "query": "machine learning",
  "max_results": 10
}
```

**New API (v2)**:
```json
{
  "search": {
    "query": "machine learning",
    "filters": {
      "max_results": 10,
      "sort_by": "relevance"
    }
  }
}
```

**Implementation**:

```python
from src.hopf_lens_dc.hopf_lens import *
from src.hopf_lens_dc.categorical_core import *

# 1. Define schemas
old_schema = JSONSchema(
    name="search_v1",
    type=JSONType.OBJECT,
    properties={
        "query": JSONSchema(name="query", type=JSONType.STRING),
        "max_results": JSONSchema(name="max_results", type=JSONType.INTEGER)
    },
    required={"query"}
)

new_schema = JSONSchema(
    name="search_v2",
    type=JSONType.OBJECT,
    properties={
        "search": JSONSchema(
            name="search",
            type=JSONType.OBJECT,
            properties={
                "query": JSONSchema(name="query", type=JSONType.STRING),
                "filters": JSONSchema(
                    name="filters",
                    type=JSONType.OBJECT,
                    properties={
                        "max_results": JSONSchema(
                            name="max_results",
                            type=JSONType.INTEGER,
                            default=10
                        ),
                        "sort_by": JSONSchema(
                            name="sort_by",
                            type=JSONType.STRING,
                            default="relevance"
                        )
                    }
                )
            },
            required={"query"}
        )
    },
    required={"search"}
)

# 2. Create tool morphism
def search_impl(args: dict, ctx: Context) -> Effect:
    search_params = args["search"]
    query = search_params["query"]
    filters = search_params.get("filters", {})

    # Your search implementation
    results = perform_search(query, filters)

    return Effect.pure({"results": results})

contract = Contract(
    schema=new_schema,
    predicate=lambda v, c: "search" in v and "query" in v["search"],
    description="Valid search with query"
)

morphism = HOPFToolMorphism(
    name="search",
    source_schema=old_schema,
    target_schema=new_schema,
    contract=contract,
    func=search_impl
)

# 3. Old client calls still work!
old_call = {"query": "ML papers", "max_results": 5}
context = Context(query="Find ML papers")

result = morphism.invoke(old_call, context)

# Automatically migrated to:
# {
#   "search": {
#     "query": "ML papers",
#     "filters": {
#       "max_results": 5,
#       "sort_by": "relevance"  # Default added
#     }
#   }
# }

if result.is_success():
    print(result.value["results"])
```

**Benefits**:
- ✓ Old clients automatically work with new API
- ✓ Type-safe migration
- ✓ Contract validation
- ✓ Cost tracking
- ✓ Formal guarantees

---

## 🔬 Verification

### Verify Soundness

```python
from src.hopf_lens_dc.hopf_lens import SoundnessChecker

# 1. Contract soundness: (SYN_σ(â,c), c) ⊨ Φ
sound, error = SoundnessChecker.check_contract_soundness(
    synthesizer=ArgumentSynthesizer(),
    partial_args={"query": "test"},
    context=context,
    source_schema=old_schema,
    target_schema=new_schema,
    contract=contract
)
print(f"Contract sound: {sound}")  # True

# 2. Preservation: SYN_id(a,c) = a
preserved = SoundnessChecker.check_preservation(
    synthesizer=ArgumentSynthesizer(),
    total_args={"query": "test", "max_results": 10},
    context=context,
    schema=old_schema
)
print(f"Preserved: {preserved}")  # True

# 3. Idempotence: SYN(SYN(â,c),c) = SYN(â,c)
idempotent = SoundnessChecker.check_idempotence(
    synthesizer=ArgumentSynthesizer(),
    partial_args={"query": "test"},
    context=context,
    source_schema=old_schema,
    target_schema=new_schema
)
print(f"Idempotent: {idempotent}")  # True
```

---

## 📊 Performance

### Edit Costs

Default costs (configurable):

```python
EditType.RENAME: 0.5
EditType.NEST: 1.0
EditType.ENUM_MAP: 0.7
EditType.COERCION: 0.8
EditType.DEFAULT_ADD: 0.3
```

### Complexity

- Schema validation: **O(n)** where n = number of fields
- Kan extension: **O(e)** where e = number of edits
- Synthesis: **O(e·f)** where f = number of fields
- Convergence: **O(k·λⁿ)** where λ < 1 (geometric convergence)

---

## 🛠️ Advanced Usage

### Custom Edit Costs

```python
synthesizer = ArgumentSynthesizer()
synthesizer.default_costs[EditType.RENAME] = 0.2  # Cheaper rename
synthesizer.default_costs[EditType.NEST] = 2.0     # Expensive nest
```

### Custom Contracts

```python
def age_contract(value: dict, ctx: Context) -> bool:
    # Complex validation
    age = value.get("age", 0)
    country = ctx.metadata.get("country", "US")

    # Different age requirements by country
    min_age = 21 if country == "US" else 18
    return age >= min_age

contract = Contract(
    schema=schema,
    predicate=age_contract,
    description="Legal age for country"
)
```

### Convergence Metrics

```python
def semantic_distance(s1: CorrectionState, s2: CorrectionState) -> float:
    # Custom metric
    if s1.last_result != s2.last_result:
        return compute_semantic_diff(s1.last_result, s2.last_result)
    return 0.0

coalgebra = SelfCorrectionCoalgebra(
    functor=CorrectionEndofunctor(contraction_factor=0.6),
    max_iterations=50,
    epsilon=0.001
)

result = coalgebra.iterate(initial, step_fn, semantic_distance)
```

---

## 📖 Further Reading

- **Full Documentation**: See `HOPF_LENS_FRAMEWORK.md`
- **Mathematical Spec**: Original HOPF/Lens DC specification
- **Tests**: `tests/test_hopf_lens.py` for all examples
- **Examples**: `examples/` directory

---

## ✅ Summary

You now know:

1. ✓ How to define JSON schemas as category objects
2. ✓ How to create schema edits (morphisms)
3. ✓ How to use lenses for bidirectional transformations
4. ✓ How to synthesize arguments from partial inputs
5. ✓ How to create HOPF morphisms with automatic repair
6. ✓ How to use self-correction coalgebras
7. ✓ How to verify soundness properties

**Next steps**:
- Read the full documentation
- Run the examples
- Adapt to your use case
- Verify your properties

**The HOPF/Lens framework provides mathematical guarantees for schema evolution!**
