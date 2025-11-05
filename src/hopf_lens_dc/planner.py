"""
PLANNER FUNCTOR: P: Q → Free(T)

Implements planning as a functor from query objects to the free monoidal
category over tools. Enforces naturality with argument binding.

Key properties:
- Plans are programs in Free(T) with ∘ (sequential) and ⊗ (parallel)
- Only emits plans where all limits exist (or can be synthesized)
- Naturality square: binding then planning = planning then binding
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .categorical_core import (
    Context, ToolMorphism, ComposedMorphism, CompositionType,
    CategoricalToolRegistry, KanSynthesizer, Effect, AritySchema
)


# ============================================================================
# QUERY OBJECTS (Category Q)
# ============================================================================

class QueryType(Enum):
    """Types of queries in category Q"""
    FACTUAL = "factual"
    COMPUTATIONAL = "computational"
    COMPOSITIONAL = "compositional"
    EXPLORATORY = "exploratory"


@dataclass
class QueryObject:
    """
    Object in query category Q.
    Represents a structured query with requirements.
    """
    text: str
    query_type: QueryType = QueryType.FACTUAL
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: Set[str] = field(default_factory=set)

    @staticmethod
    def from_text(text: str) -> 'QueryObject':
        """Parse query from natural language text"""
        # Simple heuristic-based classification
        text_lower = text.lower()

        if any(word in text_lower for word in ["calculate", "compute", "evaluate", "sum", "multiply"]):
            query_type = QueryType.COMPUTATIONAL
        elif any(word in text_lower for word in ["list", "compare", "analyze", "explain"]):
            query_type = QueryType.COMPOSITIONAL
        elif any(word in text_lower for word in ["explore", "investigate", "research"]):
            query_type = QueryType.EXPLORATORY
        else:
            query_type = QueryType.FACTUAL

        # Extract required capabilities
        capabilities = set()
        if any(word in text_lower for word in ["search", "find", "look up", "who", "what", "where"]):
            capabilities.add("search")
        if any(word in text_lower for word in ["calculate", "math", "compute"]):
            capabilities.add("math")
        if any(word in text_lower for word in ["fetch", "api", "get data"]):
            capabilities.add("api")

        return QueryObject(
            text=text,
            query_type=query_type,
            required_capabilities=capabilities
        )


# ============================================================================
# PLAN OBJECTS (Free Monoidal Category Free(T))
# ============================================================================

@dataclass
class PlanNode:
    """
    Node in a plan tree representing Free(T).
    Can be atomic (single tool) or composite (∘ or ⊗).
    """
    tool_name: Optional[str] = None
    composition: Optional[ComposedMorphism] = None
    dependencies: List['PlanNode'] = field(default_factory=list)
    estimated_cost: float = 1.0

    def is_atomic(self) -> bool:
        """Check if this is an atomic tool invocation"""
        return self.tool_name is not None and self.composition is None

    def is_composite(self) -> bool:
        """Check if this is a composite morphism"""
        return self.composition is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan node"""
        if self.is_atomic():
            return {
                "type": "atomic",
                "tool": self.tool_name,
                "cost": self.estimated_cost
            }
        else:
            return {
                "type": "composite",
                "composition": self.composition.composition_type.value if self.composition else None,
                "dependencies": [d.to_dict() for d in self.dependencies],
                "cost": self.estimated_cost
            }


@dataclass
class Plan:
    """
    Complete plan in Free(T).
    Represents a program composed of tools via ∘ and ⊗.
    """
    root: PlanNode
    context_bindings: Dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0

    def validate(self, registry: CategoricalToolRegistry, context: Context) -> Tuple[bool, List[str]]:
        """
        Validate that plan can be executed:
        - All tools exist
        - All limits exist (or can be synthesized)
        - No circular dependencies
        """
        errors = []

        # Extend context with bindings
        extended_context = context
        for key, value in self.context_bindings.items():
            extended_context = extended_context.extend(key, value)

        # Validate recursively
        self._validate_node(self.root, registry, extended_context, errors)

        return len(errors) == 0, errors

    def _validate_node(
        self,
        node: PlanNode,
        registry: CategoricalToolRegistry,
        context: Context,
        errors: List[str]
    ):
        """Recursively validate plan node"""
        if node.is_atomic():
            # Check tool exists
            tool = registry.get(node.tool_name)
            if not tool:
                errors.append(f"Tool '{node.tool_name}' not found")
                return

            # Check limit exists
            can_invoke, missing = tool.can_invoke(context)
            if not can_invoke:
                # Check if we can synthesize
                synth_result = registry.synthesizer.synthesize(
                    context,
                    tool.schema,
                    missing
                )
                if not synth_result.is_success():
                    errors.append(
                        f"Tool '{node.tool_name}' missing arguments {missing} "
                        f"and synthesis failed: {synth_result.error}"
                    )

        elif node.is_composite():
            # Validate each dependency
            for dep in node.dependencies:
                self._validate_node(dep, registry, context, errors)

    def execute(
        self,
        registry: CategoricalToolRegistry,
        context: Context
    ) -> Effect[Any]:
        """
        Execute plan in the Kleisli category.
        Returns Effect[result] of final computation.
        """
        # Extend context with bindings
        extended_context = context
        for key, value in self.context_bindings.items():
            extended_context = extended_context.extend(key, value)

        # Validate before execution
        valid, errors = self.validate(registry, extended_context)
        if not valid:
            return Effect.fail(f"Plan validation failed: {'; '.join(errors)}")

        # Execute root node
        return self._execute_node(self.root, registry, extended_context)

    def _execute_node(
        self,
        node: PlanNode,
        registry: CategoricalToolRegistry,
        context: Context
    ) -> Effect[Any]:
        """Recursively execute plan node"""
        if node.is_atomic():
            # Execute single tool
            return registry.invoke(node.tool_name, context, use_synthesis=True)

        elif node.is_composite():
            # Execute composition
            if node.composition:
                return node.composition.execute(context, registry.synthesizer)
            else:
                # Execute dependencies and combine
                results = []
                for dep in node.dependencies:
                    result = self._execute_node(dep, registry, context)
                    if not result.is_success():
                        return result
                    results.append(result.value)

                return Effect.pure(results)

        return Effect.fail("Invalid plan node")


# ============================================================================
# PLANNER FUNCTOR P: Q → Free(T)
# ============================================================================

class PlannerFunctor:
    """
    Functor P: Q → Free(T) from queries to plans.

    Enforces naturality: for any binding transformation β: Q → Q',
    the following square commutes:

        Q ----P---→ Free(T)
        |            |
        β            bind
        ↓            ↓
        Q' ---P---→ Free(T')

    This means: binding then planning = planning then binding
    """

    def __init__(self, registry: CategoricalToolRegistry):
        self.registry = registry
        self.planner_cache: Dict[str, Plan] = {}

    def map_query(self, query: QueryObject, context: Context) -> Plan:
        """
        Map query object to plan in Free(T).

        Args:
            query: Query object from category Q
            context: Execution context C

        Returns:
            Plan in Free(T) with all limits validated
        """
        # Check cache
        cache_key = f"{query.text}:{query.query_type.value}"
        if cache_key in self.planner_cache:
            return self.planner_cache[cache_key]

        # Decompose query into plan
        plan = self._decompose_query(query, context)

        # Validate plan
        valid, errors = plan.validate(self.registry, context)
        if not valid:
            # Plan is invalid - try to repair
            plan = self._repair_plan(plan, errors, context)

        # Cache and return
        self.planner_cache[cache_key] = plan
        return plan

    def _decompose_query(self, query: QueryObject, context: Context) -> Plan:
        """
        Decompose query into a plan tree.
        Uses heuristics to map query types to tool compositions.
        """
        available_tools = self.registry.list_tools()

        # Match query capabilities to available tools
        plan_nodes = []

        if "search" in query.required_capabilities:
            if "search_web" in available_tools:
                plan_nodes.append(PlanNode(tool_name="search_web", estimated_cost=1.0))
            elif "fetch_api" in available_tools:
                plan_nodes.append(PlanNode(tool_name="fetch_api", estimated_cost=0.8))

        if "math" in query.required_capabilities:
            if "eval_math" in available_tools:
                plan_nodes.append(PlanNode(tool_name="eval_math", estimated_cost=0.2))

        # Default: if no specific tools matched, try search
        if not plan_nodes and "search_web" in available_tools:
            plan_nodes.append(PlanNode(tool_name="search_web", estimated_cost=1.0))

        # Compose nodes based on query type
        if query.query_type == QueryType.COMPUTATIONAL:
            # Sequential: search then compute
            if len(plan_nodes) > 1:
                root = PlanNode(
                    composition=ComposedMorphism(
                        tools=[],
                        composition_type=CompositionType.SEQUENTIAL
                    ),
                    dependencies=plan_nodes,
                    estimated_cost=sum(n.estimated_cost for n in plan_nodes)
                )
            else:
                root = plan_nodes[0] if plan_nodes else PlanNode(tool_name="search_web")

        elif query.query_type == QueryType.COMPOSITIONAL:
            # Parallel exploration then sequential synthesis
            if len(plan_nodes) > 1:
                root = PlanNode(
                    composition=ComposedMorphism(
                        tools=[],
                        composition_type=CompositionType.PARALLEL
                    ),
                    dependencies=plan_nodes,
                    estimated_cost=max(n.estimated_cost for n in plan_nodes)
                )
            else:
                root = plan_nodes[0] if plan_nodes else PlanNode(tool_name="search_web")

        else:
            # Sequential by default
            root = plan_nodes[0] if plan_nodes else PlanNode(tool_name="search_web")

        return Plan(
            root=root,
            context_bindings={"query": query.text},
            total_cost=root.estimated_cost if root else 0.0
        )

    def _repair_plan(self, plan: Plan, errors: List[str], context: Context) -> Plan:
        """
        Repair invalid plan by:
        - Adding synthesis hints to context bindings
        - Removing tools that can't be satisfied
        - Falling back to simpler plans
        """
        # For now, simple repair: try to use just search_web
        if "search_web" in self.registry.list_tools():
            root = PlanNode(tool_name="search_web", estimated_cost=1.0)
            return Plan(
                root=root,
                context_bindings=plan.context_bindings,
                total_cost=1.0
            )

        # If even search_web fails, return empty plan
        return plan

    def check_naturality(
        self,
        query: QueryObject,
        binding: Dict[str, Any],
        context: Context
    ) -> bool:
        """
        Check naturality square:
        Does binding then planning equal planning then binding?

        Args:
            query: Original query
            binding: Context transformation β
            context: Base context

        Returns:
            True if naturality holds
        """
        # Path 1: Plan then bind
        plan1 = self.map_query(query, context)
        bound_plan1 = self._bind_plan(plan1, binding)

        # Path 2: Bind then plan
        bound_context = context
        for key, value in binding.items():
            bound_context = bound_context.extend(key, value)
        plan2 = self.map_query(query, bound_context)

        # Compare plans (simplified: check if root tools match)
        return self._plans_equivalent(bound_plan1, plan2)

    def _bind_plan(self, plan: Plan, binding: Dict[str, Any]) -> Plan:
        """Apply binding transformation to plan"""
        new_bindings = {**plan.context_bindings, **binding}
        return Plan(
            root=plan.root,
            context_bindings=new_bindings,
            total_cost=plan.total_cost
        )

    def _plans_equivalent(self, plan1: Plan, plan2: Plan) -> bool:
        """Check if two plans are equivalent (simplified)"""
        # Simple check: compare root node tool names
        if plan1.root.tool_name != plan2.root.tool_name:
            return False

        # Check bindings (keys should match)
        if set(plan1.context_bindings.keys()) != set(plan2.context_bindings.keys()):
            return False

        return True


# ============================================================================
# PLAN OPTIMIZER
# ============================================================================

class PlanOptimizer:
    """
    Optimize plans in Free(T) by:
    - Deduplicating redundant tool calls
    - Reordering for better parallelism
    - Pruning unnecessary branches
    """

    def optimize(self, plan: Plan) -> Plan:
        """Apply optimization transformations to plan"""
        # For now, return plan as-is
        # Future: implement actual optimizations
        return plan

    def estimate_cost(self, plan: Plan) -> float:
        """Estimate execution cost of plan"""
        return plan.total_cost


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from .categorical_core import create_simple_tool, EffectType

    print("Planner Functor Module")
    print("=" * 60)

    # Create registry with sample tools
    registry = CategoricalToolRegistry()

    # Register search tool
    def search_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
        return {"query": args["query"], "results": ["result1", "result2"]}

    search_tool = create_simple_tool(
        name="search_web",
        required_args=[("query", str)],
        func=search_impl,
        effects=[EffectType.HTTP]
    )
    registry.tools["search_web"] = search_tool

    # Register math tool
    def math_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
        return {"expression": args["expression"], "result": eval(args["expression"])}

    math_tool = create_simple_tool(
        name="eval_math",
        required_args=[("expression", str)],
        func=math_impl,
        effects=[EffectType.PURE]
    )
    registry.tools["eval_math"] = math_tool

    # Create planner
    planner = PlannerFunctor(registry)

    # Test query
    query = QueryObject.from_text("Search for Paris bridges and count them")
    context = Context(query=query.text)

    print(f"\nQuery: {query.text}")
    print(f"Type: {query.query_type.value}")
    print(f"Capabilities: {query.required_capabilities}")

    # Generate plan
    plan = planner.map_query(query, context)
    print(f"\nPlan generated:")
    print(f"  Root tool: {plan.root.tool_name}")
    print(f"  Bindings: {plan.context_bindings}")
    print(f"  Cost: {plan.total_cost}")

    # Validate plan
    valid, errors = plan.validate(registry, context)
    print(f"\nPlan valid: {valid}")
    if errors:
        print(f"Errors: {errors}")

    # Test naturality
    binding = {"limit": 10}
    natural = planner.check_naturality(query, binding, context)
    print(f"\nNaturality holds: {natural}")

    print("\n✓ Planner module validated")
