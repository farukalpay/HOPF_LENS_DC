# HOPF/Lens Framework v0.2 - Formalization & Fixes

**Status**: Planning document
**Version**: 0.2
**Date**: 2025-11-05

## Critical Fixes Required

### 1. Fix `unit_system` Enum Mapping Bug

**Problem**: In Scenario 1, `unit_system` is being set to query string instead of enum-mapped value.

**Root Cause**:
```python
# Current buggy flow:
partial = {"temp_unit": "C"}
# After NEST: {"location": {"city": "Paris"}, "temp_unit": "C"}
# After RENAME: {"location": {"city": "Paris"}, "unit_system": "C"}
# After ENUM_MAP: Should be "metric" but synthesis fills with query!
```

**Fix**: Two-phase synthesis
1. **Shape phase**: Apply structural edits (NEST, RENAME)
2. **Value phase**: Apply value transformations (ENUM_MAP, COERCION)

### 2. Separate Shape and Value Transformations

#### Shape Edits (Structure)
- RENAME: field path changes
- NEST: create nested objects
- UNNEST: flatten objects
- FIELD_ADD: add new field slots
- FIELD_REMOVE: remove field slots

**Invariant**: Shape edits preserve value types, only change structure.

#### Value Edits (Content)
- ENUM_MAP: transform enum values
- COERCION: type conversions
- DEFAULT_ADD: fill missing values
- NORMALIZE: canonicalize values

**Invariant**: Value edits preserve structure, only change content.

### 3. Formalize Edit Specifications

#### EditSpec JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "EditSpec",
  "type": "object",
  "properties": {
    "edit_id": {"type": "string"},
    "edit_type": {
      "enum": ["rename", "nest", "unnest", "enum_map", "coercion", "default_add"]
    },
    "phase": {"enum": ["shape", "value"]},
    "source_path": {
      "type": "array",
      "items": {"type": "string"}
    },
    "target_path": {
      "type": "array",
      "items": {"type": "string"}
    },
    "parameters": {"type": "object"},
    "cost": {"type": "number", "minimum": 0},
    "preconditions": {
      "type": "array",
      "items": {"type": "string"}
    },
    "postconditions": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["edit_id", "edit_type", "phase", "source_path", "target_path", "cost"]
}
```

#### Example EditSpec

```json
{
  "edit_id": "enum_map_units",
  "edit_type": "enum_map",
  "phase": "value",
  "source_path": ["temp_unit"],
  "target_path": ["unit_system"],
  "parameters": {
    "mapping": {
      "C": "metric",
      "F": "imperial"
    },
    "strict": true,
    "fallback": "metric"
  },
  "cost": 0.7,
  "preconditions": [
    "source_field_exists(temp_unit)",
    "source_value_in_domain({C, F})"
  ],
  "postconditions": [
    "target_field_exists(unit_system)",
    "target_value_in_range({metric, imperial})"
  ]
}
```

### 4. RepairPolicy Specification

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "RepairPolicy",
  "type": "object",
  "properties": {
    "policy_id": {"type": "string"},
    "allowed_edits": {
      "type": "array",
      "items": {"$ref": "#/definitions/EditSpec"}
    },
    "cost_model": {
      "type": "object",
      "properties": {
        "edit_costs": {
          "type": "object",
          "additionalProperties": {"type": "number"}
        },
        "budget": {"type": "number"},
        "optimization": {"enum": ["minimize_cost", "minimize_edits", "weighted"]}
      }
    },
    "context_mapping": {
      "type": "object",
      "description": "How to extract defaults from context",
      "properties": {
        "field_extractors": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "extractor": {"type": "string"},
              "context_path": {"type": "string"},
              "fallback": {}
            }
          }
        }
      }
    },
    "cross_field_constraints": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "constraint_id": {"type": "string"},
          "fields": {"type": "array", "items": {"type": "string"}},
          "predicate": {"type": "string"},
          "repair_strategy": {"type": "string"}
        }
      }
    }
  }
}
```

#### Example RepairPolicy

```json
{
  "policy_id": "weather_v1_to_v2",
  "cost_model": {
    "edit_costs": {
      "nest": 1.0,
      "rename": 0.5,
      "enum_map": 0.7,
      "default_add": 0.3
    },
    "budget": 5.0,
    "optimization": "minimize_cost"
  },
  "context_mapping": {
    "field_extractors": {
      "location.country": {
        "extractor": "metadata",
        "context_path": "metadata.country",
        "fallback": "US"
      },
      "unit_system": {
        "extractor": "preference",
        "context_path": "metadata.units",
        "fallback": "metric"
      }
    }
  },
  "cross_field_constraints": [
    {
      "constraint_id": "consistent_units",
      "fields": ["unit_system", "temperature"],
      "predicate": "units_match(unit_system, temperature.unit)",
      "repair_strategy": "convert_temperature"
    }
  ]
}
```

### 5. SynthesisContract Specification

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SynthesisContract",
  "type": "object",
  "properties": {
    "contract_id": {"type": "string"},
    "schema_ref": {"type": "string"},
    "predicates": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "predicate_id": {"type": "string"},
          "type": {"enum": ["type_check", "range_check", "cross_field", "context_dependent"]},
          "expression": {"type": "string"},
          "error_message": {"type": "string"}
        }
      }
    },
    "invariants": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "invariant_id": {"type": "string"},
          "property": {"enum": ["soundness", "preservation", "idempotence", "monotonicity"]},
          "formula": {"type": "string"}
        }
      }
    }
  }
}
```

### 6. Convergence Metric Specification

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ConvergenceMetric",
  "type": "object",
  "properties": {
    "metric_id": {"type": "string"},
    "metric_type": {"enum": ["semantic_distance", "edit_distance", "cost_distance", "hybrid"]},
    "distance_function": {
      "type": "object",
      "properties": {
        "implementation": {"type": "string"},
        "properties": {
          "type": "object",
          "properties": {
            "symmetric": {"type": "boolean"},
            "triangle_inequality": {"type": "boolean"},
            "lipschitz_constant": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      }
    },
    "convergence_criteria": {
      "type": "object",
      "properties": {
        "epsilon": {"type": "number", "minimum": 0},
        "max_iterations": {"type": "integer", "minimum": 1},
        "early_stop_conditions": {
          "type": "array",
          "items": {"type": "string"}
        }
      }
    }
  }
}
```

---

## Implementation Plan (Checklist Format)

### Phase 1: Fix Critical Bugs ✓ PRIORITY

- [ ] **1.1** Implement two-phase synthesis (shape → value)
- [ ] **1.2** Fix enum mapping to apply in value phase
- [ ] **1.3** Ensure context extraction happens before value phase
- [ ] **1.4** Add test case for unit_system bug
- [ ] **1.5** Verify fix with Scenario 1

### Phase 2: Formalize Specifications

- [ ] **2.1** Implement EditSpec JSON schema
- [ ] **2.2** Implement RepairPolicy JSON schema
- [ ] **2.3** Implement SynthesisContract JSON schema
- [ ] **2.4** Implement ConvergenceMetric JSON schema
- [ ] **2.5** Create JSON validator for all specs

### Phase 3: Mathematical Axioms & Invariants

- [ ] **3.1** Define lens law checkers (PutGet, GetPut, PutPut)
- [ ] **3.2** Define edit composition axioms
- [ ] **3.3** Define semiring axioms (associativity, distributivity)
- [ ] **3.4** Implement invariant checkers
- [ ] **3.5** Add property-based tests (Hypothesis framework)

### Phase 4: Enhanced Convergence

- [ ] **4.1** Implement precise distance metrics
- [ ] **4.2** Measure empirical Lipschitz constant
- [ ] **4.3** Add contraction verification
- [ ] **4.4** Implement adaptive step sizing
- [ ] **4.5** Add convergence rate estimation

### Phase 5: Error Categories & E Monad

- [ ] **5.1** Define error categories (type, contract, resource, timeout)
- [ ] **5.2** Implement extended E monad with diagnostics
- [ ] **5.3** Add error recovery strategies
- [ ] **5.4** Implement Kleisli composition with error handling
- [ ] **5.5** Add error accumulation (Applicative validation)

### Phase 6: Benchmark & Testing Infrastructure

- [ ] **6.1** Create benchmark harness
- [ ] **6.2** Define performance baselines
- [ ] **6.3** Add edge case test suite
- [ ] **6.4** Implement fuzz testing
- [ ] **6.5** Add regression test suite

### Phase 7: Tool Server Manifest

- [ ] **7.1** Define Manifest JSON schema
- [ ] **7.2** Implement schema versioning
- [ ] **7.3** Add capability negotiation
- [ ] **7.4** Implement backward compatibility matrix
- [ ] **7.5** Add migration path documentation

---

## Detailed Technical Specs

### Two-Phase Synthesis Algorithm

```python
def two_phase_synthesis(
    partial_args: Dict[str, Any],
    context: Context,
    source_schema: JSONSchema,
    target_schema: JSONSchema,
    edit_spec: List[EditSpec],
    repair_policy: RepairPolicy,
    contract: SynthesisContract
) -> Effect[Dict[str, Any]]:
    """
    Two-phase synthesis with shape/value separation.

    Phase 1 (Shape):
      1. Apply NEST/UNNEST edits
      2. Apply RENAME edits
      3. Apply FIELD_ADD/REMOVE edits
      4. Result: correct structure, possibly wrong values

    Phase 2 (Value):
      1. Extract defaults from context per RepairPolicy
      2. Apply ENUM_MAP transformations
      3. Apply COERCION transformations
      4. Apply DEFAULT_ADD for missing values
      5. Apply NORMALIZE transformations

    Post-processing:
      1. Validate against SynthesisContract
      2. Compute total cost
      3. Return synthesized arguments
    """
```

### Lipschitz Constant Estimation

```python
def estimate_lipschitz_constant(
    functor: CorrectionEndofunctor,
    metric: Callable[[CorrectionState, CorrectionState], float],
    sample_size: int = 100
) -> float:
    """
    Empirically estimate Lipschitz constant λ.

    Method:
      1. Sample random pairs (x₁, x₂) from state space
      2. Compute d(F(x₁), F(x₂)) / d(x₁, x₂)
      3. Return max ratio (upper bound on λ)

    Property:
      If λ < 1, Banach theorem guarantees unique fixed point.
    """
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

# Lens laws
@given(st.dictionaries(st.text(), st.text()))
def test_lens_put_get_law(a: dict):
    """∀a,b: get(put(a,b)) = b"""
    lens = create_lens_for_edit(rename_edit)
    b = lens.get(a)
    assert lens.get(lens.put(a, b)) == b

@given(st.dictionaries(st.text(), st.text()))
def test_lens_get_put_law(a: dict):
    """∀a: put(a, get(a)) ≈ a"""
    lens = create_lens_for_edit(rename_edit)
    assert lens.put(a, lens.get(a)) == a

# Semiring axioms
@given(st.lists(st.builds(SchemaEdit)))
def test_semiring_associativity(edits: List[SchemaEdit]):
    """∀e₁,e₂,e₃: (e₁⊗e₂)⊗e₃ = e₁⊗(e₂⊗e₃)"""
    semiring = EditSemiring()
    # Test associativity
    ...

# Contract soundness
@given(st.dictionaries(st.text(), st.integers()))
def test_contract_soundness_property(partial: dict):
    """∀â,c: (SYN(â,c), c) ⊨ Φ"""
    synthesizer = ArgumentSynthesizer()
    context = Context(query="test")
    result = synthesizer.synthesize(partial, context, ...)
    assert contract.check(result.value["arguments"], context)[0]
```

---

## Next Immediate Steps

**Step 1**: Fix the `unit_system` bug
```bash
# Create fix branch
git checkout -b fix/two-phase-synthesis

# Implement two-phase synthesis
# Add test case
# Verify fix
```

**Step 2**: Create JSON spec validators
```bash
# Implement EditSpec, RepairPolicy, etc.
# Add JSON Schema validation
# Create example specs
```

**Step 3**: Add property-based tests
```bash
pip install hypothesis
# Add lens law tests
# Add semiring axiom tests
# Add contract soundness tests
```

**Step 4**: Measure Lipschitz constants
```bash
# Implement estimation algorithm
# Run experiments
# Document convergence rates
```

---

## Questions for Clarification

1. **E Monad Design**: Do you want:
   - Simple `Ok | Err` (current)
   - Multi-error accumulation (Applicative)
   - Structured diagnostics with error codes

2. **Convergence Metric**: Preferred distance function:
   - Hamming distance (field-wise diff)
   - Levenshtein distance (edit distance)
   - Semantic embedding distance
   - Custom domain-specific

3. **Benchmark Suite**: Focus areas:
   - Performance (latency, throughput)
   - Correctness (edge cases, fuzz testing)
   - Scalability (large schemas, long edit sequences)

4. **Tool Manifest**: Required features:
   - Version negotiation protocol
   - Capability discovery
   - Migration path specification
   - Backward compatibility matrix

Please let me know your preferences and I'll proceed with implementation!
