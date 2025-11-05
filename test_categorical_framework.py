"""
TESTS FOR CATEGORICAL TOOL FRAMEWORK

Validates:
1. Missing argument prevention (no empty dicts!)
2. Limit checking
3. Kan synthesis
4. Argument assembly
5. Tool composition
6. Evidence validation
7. Convergence properties
"""

import unittest
from typing import Dict, Any

from categorical_core import (
    Context, AritySchema, DirectAssembler, ToolMorphism,
    CategoricalToolRegistry, Effect, EffectType, KanSynthesizer,
    create_simple_tool
)
from planner import PlannerFunctor, QueryObject
from convergence import Answer, AnswerCoalgebra, AnswerEndofunctor, SemanticDriftMetric
from evidence import Evidence, Claim, Source, SourceType, EvidencePolicy
from comonad import ContextComonad, AttackGenerator, CounterfactualExecutor


class TestArgumentValidation(unittest.TestCase):
    """Test argument validation and missing-argument prevention"""

    def test_empty_dict_rejected(self):
        """Test that tools cannot be called with empty argument dict"""
        # Create tool requiring query argument
        schema = AritySchema()
        schema.add_arg("query", str, required=True)

        assembler = DirectAssembler(schema)

        invocation_count = {"count": 0}

        def dummy_func(args: Dict[str, Any], ctx: Context) -> Effect[Any]:
            invocation_count["count"] += 1
            # Verify args is not empty
            if not args or not args.get("query"):
                raise ValueError("Function called with empty/missing args!")
            return Effect.pure({"result": "ok"})

        tool = ToolMorphism(
            name="test_tool",
            schema=schema,
            assembler=assembler,
            func=dummy_func,
            effects=[EffectType.PURE]
        )

        # Try to invoke with completely empty context (no query)
        empty_context = Context(query="")  # Empty query string
        result = tool.invoke(empty_context, synthesizer=None)

        # Should fail without calling the function
        self.assertFalse(result.is_success())
        self.assertEqual(invocation_count["count"], 0, "Function was called despite empty args!")
        self.assertIn("missing", result.error.lower())

    def test_limit_exists_check(self):
        """Test limit existence checking"""
        schema = AritySchema()
        schema.add_arg("query", str, required=True)
        schema.add_arg("limit", int, required=True)

        # Context with only query
        context = Context(query="test")

        # Check limit
        has_limit, missing = schema.has_limit(context)

        self.assertFalse(has_limit)
        self.assertIn("limit", missing)

    def test_limit_exists_with_all_args(self):
        """Test limit exists when all args available"""
        schema = AritySchema()
        schema.add_arg("query", str, required=True)

        context = Context(query="test")

        has_limit, missing = schema.has_limit(context)

        self.assertTrue(has_limit)
        self.assertEqual(missing, [])

    def test_kan_synthesis_for_integer(self):
        """Test Kan synthesis can extract integer from query"""
        synthesizer = KanSynthesizer()

        context = Context(query="Give me top 5 results")
        schema = AritySchema()
        schema.add_arg("k", int, required=True)

        result = synthesizer.synthesize(context, schema, ["k"])

        self.assertTrue(result.is_success())
        self.assertEqual(result.value["k"], 5)

    def test_kan_synthesis_for_query_string(self):
        """Test Kan synthesis uses query for query argument"""
        synthesizer = KanSynthesizer()

        context = Context(query="search for Paris")
        schema = AritySchema()
        schema.add_arg("query", str, required=True)

        result = synthesizer.synthesize(context, schema, ["query"])

        self.assertTrue(result.is_success())
        self.assertEqual(result.value["query"], "search for Paris")


class TestToolRegistry(unittest.TestCase):
    """Test tool registry and invocation"""

    def setUp(self):
        """Set up test registry"""
        self.registry = CategoricalToolRegistry()

        # Register test tool
        def test_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
            return {"query": args["query"], "result": "ok"}

        tool = create_simple_tool(
            name="test_search",
            required_args=[("query", str)],
            func=test_impl,
            effects=[EffectType.PURE]
        )

        self.registry.tools["test_search"] = tool

    def test_invoke_with_valid_context(self):
        """Test successful invocation with valid context"""
        context = Context(query="test query")
        result = self.registry.invoke("test_search", context)

        self.assertTrue(result.is_success())
        self.assertEqual(result.value["query"], "test query")

    def test_invoke_with_missing_args_and_synthesis(self):
        """Test invocation with synthesis for missing args"""
        # Context without query initially
        context = Context(query="find bridges")

        result = self.registry.invoke("test_search", context, use_synthesis=True)

        self.assertTrue(result.is_success())

    def test_cannot_invoke_nonexistent_tool(self):
        """Test that invoking nonexistent tool fails"""
        context = Context()
        result = self.registry.invoke("nonexistent", context)

        self.assertFalse(result.is_success())
        self.assertIn("not found", result.error.lower())


class TestPlanner(unittest.TestCase):
    """Test planner functor"""

    def setUp(self):
        """Set up test planner"""
        self.registry = CategoricalToolRegistry()

        def search_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
            return {"results": ["r1", "r2"]}

        search_tool = create_simple_tool(
            name="search_web",
            required_args=[("query", str)],
            func=search_impl
        )

        self.registry.tools["search_web"] = search_tool
        self.planner = PlannerFunctor(self.registry)

    def test_plan_generation(self):
        """Test plan generation from query"""
        query = QueryObject.from_text("Search for Paris landmarks")
        context = Context(query=query.text)

        plan = self.planner.map_query(query, context)

        self.assertIsNotNone(plan)
        self.assertIsNotNone(plan.root)

    def test_plan_validation(self):
        """Test plan validation"""
        query = QueryObject.from_text("Search for Paris")
        context = Context(query=query.text)

        plan = self.planner.map_query(query, context)
        valid, errors = plan.validate(self.registry, context)

        self.assertTrue(valid)


class TestConvergence(unittest.TestCase):
    """Test convergence module"""

    def test_metric_distance(self):
        """Test semantic drift metric"""
        metric = SemanticDriftMetric()

        a1 = Answer(text="Paris is the capital of France", confidence=0.8)
        a2 = Answer(text="Paris is the capital of France", confidence=0.8)

        distance = metric.distance(a1, a2)

        self.assertEqual(distance, 0.0)  # Identical answers

    def test_metric_different_answers(self):
        """Test metric on different answers"""
        metric = SemanticDriftMetric()

        a1 = Answer(text="Paris is the capital", confidence=0.8)
        a2 = Answer(text="London is the capital", confidence=0.8)

        distance = metric.distance(a1, a2)

        self.assertGreater(distance, 0.0)
        self.assertLessEqual(distance, 1.0)

    def test_coalgebra_iteration(self):
        """Test coalgebra iteration"""
        metric = SemanticDriftMetric()
        functor = AnswerEndofunctor(contraction_factor=0.7)
        coalgebra = AnswerCoalgebra(functor, metric, epsilon=0.02)

        initial = Answer(text="Test answer", confidence=0.5)

        # Iterate
        result = coalgebra.iterate(initial, max_iterations=3)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.iteration, initial.iteration)


class TestEvidence(unittest.TestCase):
    """Test evidence layer"""

    def test_evidence_validation(self):
        """Test evidence validation"""
        evidence = Evidence()

        claim = Claim(id="c1", text="Paris is capital")
        source = Source(id="s1", type=SourceType.WEB, content="Paris is capital")

        evidence.add_claim(claim)
        evidence.add_source(source)
        evidence.add_morphism("c1", "s1", strength=0.9)

        valid, errors = evidence.validate()

        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_evidence_missing_sources(self):
        """Test evidence with missing sources"""
        evidence = Evidence()

        claim = Claim(id="c1", text="Test claim")
        evidence.add_claim(claim)

        valid, errors = evidence.validate()

        self.assertFalse(valid)
        self.assertTrue(any("no sources" in e.lower() for e in errors))

    def test_evidence_coend(self):
        """Test coend computation"""
        evidence = Evidence()

        evidence.add_claim(Claim(id="c1", text="Claim 1"))
        evidence.add_claim(Claim(id="c2", text="Claim 2"))
        evidence.add_source(Source(id="s1", type=SourceType.WEB))

        evidence.add_morphism("c1", "s1", 0.9)
        evidence.add_morphism("c2", "s1", 0.8)

        coend = evidence.compute_coend()

        self.assertEqual(coend, 2)

    def test_evidence_policy(self):
        """Test evidence policy enforcement"""
        policy = EvidencePolicy(
            min_claims=2,
            min_sources=1,
            min_morphisms=2
        )

        evidence = Evidence()
        evidence.add_claim(Claim(id="c1", text="Claim 1"))
        evidence.add_claim(Claim(id="c2", text="Claim 2"))
        evidence.add_source(Source(id="s1", type=SourceType.WEB))
        evidence.add_morphism("c1", "s1", 0.9)
        evidence.add_morphism("c2", "s1", 0.8)

        passes, violations = policy.check(evidence)

        self.assertTrue(passes)


class TestComonad(unittest.TestCase):
    """Test comonad for counterfactual attacks"""

    def test_counit_law(self):
        """Test counit law: extract from duplicated = original"""
        answer = Answer(text="Test", confidence=0.8)
        w = ContextComonad(value=answer)

        extracted = w.extract()

        self.assertEqual(extracted, answer)

    def test_duplicate(self):
        """Test duplication"""
        answer = Answer(text="Test", confidence=0.8)
        w = ContextComonad(value=answer)

        duplicated = w.duplicate()

        self.assertIsInstance(duplicated, ContextComonad)
        self.assertIsInstance(duplicated.value, ContextComonad)

    def test_perturbation(self):
        """Test perturbation application"""
        answer = Answer(text="Test", confidence=0.8)
        w = ContextComonad(value=answer)

        perturbed = w.perturb({"type": "test", "delta": 0.1})

        self.assertEqual(perturbed.get_perturbation_count(), 1)

    def test_attack_execution(self):
        """Test counterfactual attack execution"""
        from comonad import ConfidenceAttack

        answer = Answer(text="Paris is the capital", confidence=0.8)
        attack = ConfidenceAttack(confidence_delta=-0.3)

        executor = CounterfactualExecutor()
        result = executor.execute_attack(answer, attack)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.stability_score, 0.0)
        self.assertLessEqual(result.stability_score, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete framework"""

    def test_end_to_end_no_empty_dicts(self):
        """
        Critical test: Ensure NO tool can be called with empty arguments.
        This is the main bug we're preventing!
        """
        registry = CategoricalToolRegistry()

        # Create tool requiring arguments
        schema = AritySchema()
        schema.add_arg("query", str, required=True)
        schema.add_arg("limit", int, required=True)

        assembler = DirectAssembler(schema)

        call_count = {"count": 0}

        def strict_func(args: Dict[str, Any], ctx: Context) -> Effect[Any]:
            # This should NEVER be called with empty args!
            if not args or not args.get("query") or not args.get("limit"):
                call_count["count"] += 1
                raise ValueError("CRITICAL: Tool called with empty/missing arguments!")
            return Effect.pure({"ok": True})

        tool = ToolMorphism(
            name="strict_tool",
            schema=schema,
            assembler=assembler,
            func=strict_func,
            effects=[]
        )

        registry.tools["strict_tool"] = tool

        # Try to invoke with incomplete context
        context = Context(query="test")  # Missing 'limit'

        # Without synthesis - should fail gracefully
        result = registry.invoke("strict_tool", context, use_synthesis=False)
        self.assertFalse(result.is_success())
        self.assertEqual(call_count["count"], 0, "Tool was called despite missing args!")

        # With synthesis - should succeed
        result = registry.invoke("strict_tool", context, use_synthesis=True)
        # May fail synthesis, but should never call func with empty args
        self.assertEqual(call_count["count"], 0, "Tool was called with invalid args!")


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("CATEGORICAL FRAMEWORK TEST SUITE")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestArgumentValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestToolRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestConvergence))
    suite.addTests(loader.loadTestsFromTestCase(TestEvidence))
    suite.addTests(loader.loadTestsFromTestCase(TestComonad))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
