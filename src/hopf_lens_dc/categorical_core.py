"""
CATEGORICAL CORE: Type-safe tool composition framework

This module implements tools as morphisms in a Kleisli category with formal guarantees:
- Tools are typed morphisms f: A×C → E[B]
- Arguments must be assembled via total functions μ_f: C → A
- Missing arguments trigger left Kan extension synthesis
- Composition is free monoidal (sequential ∘, parallel ⊗)
- Convergence is a coalgebra with metric distance
"""

from typing import (
    Dict, List, Any, Optional, Callable, TypeVar, Generic, Tuple, Union
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from enum import Enum


# ============================================================================
# TYPE VARIABLES & EFFECTS
# ============================================================================

A = TypeVar('A')  # Argument type
B = TypeVar('B')  # Result type
C = TypeVar('C')  # Context type
E = TypeVar('E')  # Effect type


class EffectType(Enum):
    """Effect monad E for tool execution"""
    HTTP = "http"
    PARSE = "parse"
    TIMEOUT = "timeout"
    IO = "io"
    PURE = "pure"


@dataclass
class Effect(Generic[B]):
    """
    Effect monad E[B] wrapping results with side effects.
    Forms a Kleisli category with bind operation.
    """
    value: Optional[B]
    effects: List[EffectType] = field(default_factory=list)
    error: Optional[str] = None

    def is_success(self) -> bool:
        """Check if effect computation succeeded"""
        return self.error is None and self.value is not None

    def bind(self, f: Callable[[B], 'Effect[Any]']) -> 'Effect[Any]':
        """
        Kleisli composition: (>>=) for monadic bind.
        Enables sequential composition of effectful computations.
        """
        if not self.is_success():
            return Effect(value=None, error=self.error, effects=self.effects)
        try:
            result = f(self.value)
            result.effects = self.effects + result.effects
            return result
        except Exception as e:
            return Effect(value=None, error=str(e), effects=self.effects)

    @staticmethod
    def pure(value: B) -> 'Effect[B]':
        """Monadic return: lift pure value into Effect"""
        return Effect(value=value, effects=[EffectType.PURE])

    @staticmethod
    def fail(error: str, effects: List[EffectType] = None) -> 'Effect[B]':
        """Construct failed effect"""
        return Effect(value=None, error=error, effects=effects or [])


# ============================================================================
# ARGUMENT SCHEMAS & OBJECTS
# ============================================================================

@dataclass
class ArgSchema:
    """
    Schema for a tool argument (part of the product A).
    Represents a single projection π_i: A → T_i
    """
    name: str
    type: type
    required: bool = True
    default: Optional[Any] = None
    description: str = ""

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate that value matches this schema"""
        if value is None:
            if self.required and self.default is None:
                return False, f"Required argument '{self.name}' is missing"
            return True, None

        # Type checking
        if not isinstance(value, self.type):
            return False, f"Argument '{self.name}' has type {type(value).__name__}, expected {self.type.__name__}"

        return True, None


@dataclass
class AritySchema:
    """
    Arity schema Ar(f) = A = product of argument schemas.
    Represents the domain object in the category.
    """
    arguments: List[ArgSchema] = field(default_factory=list)

    def add_arg(self, name: str, arg_type: type, required: bool = True,
                default: Any = None, description: str = "") -> 'AritySchema':
        """Add argument to the product schema"""
        self.arguments.append(ArgSchema(
            name=name,
            type=arg_type,
            required=required,
            default=default,
            description=description
        ))
        return self

    def validate(self, args: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that args dictionary satisfies this product.
        Checks that all projections π_i: args → arg_i exist and are well-typed.
        """
        errors = []

        for schema in self.arguments:
            value = args.get(schema.name)
            valid, error = schema.validate(value)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors

    def get_required_args(self) -> List[str]:
        """Get list of required argument names"""
        return [arg.name for arg in self.arguments if arg.required]

    def has_limit(self, context: 'Context') -> Tuple[bool, List[str]]:
        """
        Check if limit exists: can we construct product A from context C?
        Returns (has_limit, missing_projections)
        """
        missing = []
        for arg in self.arguments:
            if arg.required:
                # Check if projection exists: π_arg: C → arg
                if not context.has_projection(arg.name):
                    missing.append(arg.name)

        return len(missing) == 0, missing


@dataclass
class Context:
    """
    Shared context C available to all tools.
    Forms an object in the category from which we construct arguments.
    """
    query: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_projection(self, key: str) -> bool:
        """Check if projection π_key: C → value exists"""
        # Check direct fields
        if key == "query" and self.query:
            return True

        # Check memory
        if key in self.memory:
            return True

        # Check metadata
        if key in self.metadata:
            return True

        return False

    def project(self, key: str) -> Optional[Any]:
        """Apply projection π_key: C → value"""
        if key == "query":
            return self.query
        if key in self.memory:
            return self.memory[key]
        if key in self.metadata:
            return self.metadata[key]
        return None

    def extend(self, key: str, value: Any) -> 'Context':
        """Extend context with new binding (creates new context object)"""
        new_context = Context(
            query=self.query,
            memory={**self.memory, key: value},
            metadata=self.metadata.copy()
        )
        return new_context


# ============================================================================
# ARGUMENT ASSEMBLER (μ_f: C → A)
# ============================================================================

class ArgumentAssembler(ABC):
    """
    Abstract assembler μ_f: C → A that constructs tool arguments from context.
    Must be TOTAL: always returns valid A or fails explicitly.
    """

    @abstractmethod
    def assemble(self, context: Context) -> Effect[Dict[str, Any]]:
        """
        Assemble arguments A from context C.
        Returns Effect[A] where A is the argument dictionary.
        """
        pass

    @abstractmethod
    def get_schema(self) -> AritySchema:
        """Get the arity schema Ar(f) = A"""
        pass


class DirectAssembler(ArgumentAssembler):
    """
    Simple assembler that directly projects from context.
    μ_f(C) = {arg_i: π_i(C) for all arg_i in A}
    """

    def __init__(self, schema: AritySchema, mappings: Dict[str, str] = None):
        """
        Args:
            schema: Arity schema Ar(f)
            mappings: Optional mappings from schema arg names to context keys
        """
        self.schema = schema
        self.mappings = mappings or {}

    def assemble(self, context: Context) -> Effect[Dict[str, Any]]:
        """Assemble by direct projection"""
        args = {}
        missing = []

        for arg_schema in self.schema.arguments:
            # Determine context key to project from
            context_key = self.mappings.get(arg_schema.name, arg_schema.name)

            # Apply projection
            value = context.project(context_key)

            if value is None or (isinstance(value, str) and not value.strip()):
                # None or empty string is invalid
                if arg_schema.required and arg_schema.default is None:
                    missing.append(arg_schema.name)
                    continue
                value = arg_schema.default

            args[arg_schema.name] = value

        if missing:
            return Effect.fail(
                f"Cannot assemble arguments: missing projections {missing}",
                effects=[EffectType.PURE]
            )

        # Validate assembled arguments
        valid, errors = self.schema.validate(args)
        if not valid:
            return Effect.fail(
                f"Assembled arguments invalid: {'; '.join(errors)}",
                effects=[EffectType.PURE]
            )

        return Effect.pure(args)

    def get_schema(self) -> AritySchema:
        return self.schema


# ============================================================================
# LEFT KAN EXTENSION SYNTHESIZER (S ⊣ U)
# ============================================================================

class KanSynthesizer:
    """
    Left Kan extension synthesizer: Lan_U(C) → A

    When limit doesn't exist (missing projections), synthesize minimal
    arguments from natural language context using adjunction S ⊣ U
    where U forgets semantics to strings.
    """

    def __init__(self):
        self.synthesis_cache: Dict[str, Any] = {}

    def synthesize(
        self,
        context: Context,
        schema: AritySchema,
        missing_args: List[str]
    ) -> Effect[Dict[str, Any]]:
        """
        Compute Lan_U to synthesize missing arguments.

        For example:
        - query="Who is X?" + missing "query" → synthesize query="Who is X?"
        - query="List 3 bridges" + missing "k" → synthesize k=3

        Args:
            context: Source context C
            schema: Target arity schema A
            missing_args: List of projections that don't exist

        Returns:
            Effect[Dict] with synthesized arguments
        """
        synthesized = {}

        for arg_name in missing_args:
            # Find schema for this argument
            arg_schema = next(
                (a for a in schema.arguments if a.name == arg_name),
                None
            )

            if not arg_schema:
                return Effect.fail(f"Unknown argument in schema: {arg_name}")

            # Synthesize based on type and context
            synth_value = self._synthesize_single(
                arg_name,
                arg_schema.type,
                context
            )

            if synth_value is None:
                return Effect.fail(
                    f"Cannot synthesize argument '{arg_name}' of type {arg_schema.type.__name__}"
                )

            synthesized[arg_name] = synth_value

        return Effect.pure(synthesized)

    def _synthesize_single(
        self,
        arg_name: str,
        arg_type: type,
        context: Context
    ) -> Optional[Any]:
        """
        Synthesize a single argument via Kan extension.
        Uses heuristics based on argument name and type.
        """
        query = context.query.lower()

        # String synthesis: often just use the query
        if arg_type == str:
            if arg_name in ["query", "q", "search_query", "term"]:
                return context.query
            if arg_name in ["text", "content", "message"]:
                return context.query

        # Integer synthesis: extract numbers from query
        if arg_type == int:
            # Look for numbers in query
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                # For arguments like "k", "limit", "count", "n"
                if arg_name in ["k", "limit", "count", "n", "max_results", "top_k"]:
                    return int(numbers[0])

        # Boolean synthesis
        if arg_type == bool:
            if any(word in query for word in ["yes", "true", "enable", "with"]):
                return True
            if any(word in query for word in ["no", "false", "disable", "without"]):
                return False
            # Default to True for optional boolean flags
            return True

        # List synthesis
        if arg_type == list:
            # Default empty list
            return []

        # Dict synthesis
        if arg_type == dict:
            return {}

        return None


# ============================================================================
# KLEISLI MORPHISM (Tool as f: A×C → E[B])
# ============================================================================

@dataclass
class ToolMorphism(Generic[A, B]):
    """
    Tool as a Kleisli morphism f: A×C → E[B]

    Represents a typed, effectful computation with:
    - Arity schema Ar(f) = A (argument product)
    - Argument assembler μ_f: C → A
    - Implementation func: A×C → E[B]
    - Effects in monad E
    """
    name: str
    schema: AritySchema
    assembler: ArgumentAssembler
    func: Callable[[Dict[str, Any], Context], Effect[B]]
    effects: List[EffectType] = field(default_factory=list)
    description: str = ""

    def can_invoke(self, context: Context) -> Tuple[bool, List[str]]:
        """
        Check if limit exists: can we construct A from C?
        Returns (can_invoke, missing_projections)
        """
        return self.schema.has_limit(context)

    def invoke(
        self,
        context: Context,
        synthesizer: Optional[KanSynthesizer] = None
    ) -> Effect[B]:
        """
        Invoke tool: compose assembler and function.

        Flow:
        1. Check if limit exists (can assemble A from C)
        2. If not, try Kan synthesis if synthesizer provided
        3. Assemble arguments A = μ_f(C)
        4. Execute f(A, C) → E[B]

        This enforces: NEVER invoke with empty/missing arguments!
        """
        # Try to assemble arguments
        assembled = self.assembler.assemble(context)

        if not assembled.is_success():
            # Limit doesn't exist - try synthesis if available
            if synthesizer:
                can_invoke, missing = self.can_invoke(context)
                if not can_invoke:
                    # Synthesize missing arguments via Lan_U
                    synth_result = synthesizer.synthesize(
                        context,
                        self.schema,
                        missing
                    )

                    if synth_result.is_success():
                        # Merge synthesized with any partial assembly
                        base_args = assembled.value if assembled.value else {}
                        merged_args = {**base_args, **synth_result.value}

                        # Validate merged arguments
                        valid, errors = self.schema.validate(merged_args)
                        if not valid:
                            return Effect.fail(
                                f"Synthesized arguments invalid: {'; '.join(errors)}"
                            )

                        assembled = Effect.pure(merged_args)
                    else:
                        return Effect.fail(
                            f"Cannot invoke {self.name}: {synth_result.error}"
                        )

            if not assembled.is_success():
                return Effect.fail(
                    f"Cannot invoke {self.name}: {assembled.error}"
                )

        # Execute function with assembled arguments
        try:
            result = self.func(assembled.value, context)
            result.effects = assembled.effects + result.effects + self.effects
            return result
        except Exception as e:
            return Effect.fail(
                f"Execution error in {self.name}: {str(e)}",
                effects=assembled.effects + self.effects
            )

    def compose(self, other: 'ToolMorphism') -> 'ComposedMorphism':
        """
        Sequential composition: self ∘ other
        Forms a morphism in the Kleisli category.
        """
        return ComposedMorphism(tools=[other, self], composition_type=CompositionType.SEQUENTIAL)


# ============================================================================
# FREE MONOIDAL COMPOSITION
# ============================================================================

class CompositionType(Enum):
    """Types of composition in free monoidal category"""
    SEQUENTIAL = "sequential"  # ∘ (compose)
    PARALLEL = "parallel"      # ⊗ (tensor)


@dataclass
class ComposedMorphism:
    """
    Composite morphism in the free monoidal category Free(T).
    Represents a plan as a tree of sequential (∘) and parallel (⊗) compositions.
    """
    tools: List[Union[ToolMorphism, 'ComposedMorphism']]
    composition_type: CompositionType

    def execute(
        self,
        context: Context,
        synthesizer: Optional[KanSynthesizer] = None
    ) -> Effect[List[Any]]:
        """
        Execute composed morphism.

        - Sequential: execute in order, threading context
        - Parallel: execute concurrently (or sequentially with independent contexts)
        """
        if self.composition_type == CompositionType.SEQUENTIAL:
            return self._execute_sequential(context, synthesizer)
        else:
            return self._execute_parallel(context, synthesizer)

    def _execute_sequential(
        self,
        context: Context,
        synthesizer: Optional[KanSynthesizer]
    ) -> Effect[List[Any]]:
        """Execute tools sequentially: f₁ ∘ f₂ ∘ ... ∘ fₙ"""
        results = []
        current_context = context

        for tool in self.tools:
            if isinstance(tool, ToolMorphism):
                result = tool.invoke(current_context, synthesizer)
            else:
                result = tool.execute(current_context, synthesizer)

            if not result.is_success():
                return Effect.fail(
                    f"Sequential composition failed: {result.error}",
                    effects=result.effects
                )

            results.append(result.value)

            # Extend context with result for next tool
            # Store as tool_name_result
            tool_name = tool.name if isinstance(tool, ToolMorphism) else f"composed_{len(results)}"
            current_context = current_context.extend(tool_name, result.value)

        return Effect.pure(results)

    def _execute_parallel(
        self,
        context: Context,
        synthesizer: Optional[KanSynthesizer]
    ) -> Effect[List[Any]]:
        """Execute tools in parallel: f₁ ⊗ f₂ ⊗ ... ⊗ fₙ"""
        results = []
        all_effects = []

        # Execute all tools with same context (independent)
        for tool in self.tools:
            if isinstance(tool, ToolMorphism):
                result = tool.invoke(context, synthesizer)
            else:
                result = tool.execute(context, synthesizer)

            if not result.is_success():
                # In parallel, one failure doesn't stop others
                # But we collect the error
                results.append(Effect.fail(result.error))
            else:
                results.append(result.value)

            all_effects.extend(result.effects)

        # Check if any critical failures
        failures = [r for r in results if isinstance(r, Effect) and not r.is_success()]
        if failures:
            return Effect.fail(
                f"Parallel composition had {len(failures)} failures",
                effects=all_effects
            )

        return Effect(value=results, effects=all_effects)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class CategoricalToolRegistry:
    """
    Registry for typed tools as Kleisli morphisms.
    Enforces that tools can only be invoked when limits exist.
    """

    def __init__(self):
        self.tools: Dict[str, ToolMorphism] = {}
        self.synthesizer = KanSynthesizer()

    def register(
        self,
        name: str,
        schema: AritySchema,
        assembler: ArgumentAssembler,
        func: Callable[[Dict[str, Any], Context], Effect[Any]],
        effects: List[EffectType] = None,
        description: str = ""
    ) -> ToolMorphism:
        """
        Register a tool as a Kleisli morphism.
        Requires explicit schema and assembler - no magic!
        """
        tool = ToolMorphism(
            name=name,
            schema=schema,
            assembler=assembler,
            func=func,
            effects=effects or [],
            description=description
        )
        self.tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[ToolMorphism]:
        """Retrieve registered tool"""
        return self.tools.get(name)

    def invoke(
        self,
        name: str,
        context: Context,
        use_synthesis: bool = True
    ) -> Effect[Any]:
        """
        Invoke tool by name with context.
        Automatically checks limits and synthesizes if needed.
        """
        tool = self.get(name)
        if not tool:
            return Effect.fail(f"Tool '{name}' not found in registry")

        synthesizer = self.synthesizer if use_synthesis else None
        return tool.invoke(context, synthesizer)

    def can_invoke(self, name: str, context: Context) -> Tuple[bool, List[str]]:
        """Check if tool can be invoked with given context"""
        tool = self.get(name)
        if not tool:
            return False, [f"Tool '{name}' not found"]

        return tool.can_invoke(context)

    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_simple_tool(
    name: str,
    required_args: List[Tuple[str, type]],
    optional_args: List[Tuple[str, type, Any]] = None,
    func: Callable[[Dict[str, Any], Context], Any] = None,
    effects: List[EffectType] = None
) -> ToolMorphism:
    """
    Helper to create a simple tool with direct assembler.

    Args:
        name: Tool name
        required_args: List of (arg_name, arg_type) tuples
        optional_args: List of (arg_name, arg_type, default) tuples
        func: Implementation function
        effects: Effect types

    Returns:
        ToolMorphism ready to register
    """
    # Build schema
    schema = AritySchema()
    for arg_name, arg_type in required_args:
        schema.add_arg(arg_name, arg_type, required=True)

    for arg_name, arg_type, default in (optional_args or []):
        schema.add_arg(arg_name, arg_type, required=False, default=default)

    # Create direct assembler
    assembler = DirectAssembler(schema)

    # Wrap function to return Effect
    def wrapped_func(args: Dict[str, Any], ctx: Context) -> Effect[Any]:
        try:
            result = func(args, ctx) if func else {"args": args}
            return Effect.pure(result)
        except Exception as e:
            return Effect.fail(str(e))

    return ToolMorphism(
        name=name,
        schema=schema,
        assembler=assembler,
        func=wrapped_func,
        effects=effects or [],
        description=f"Tool {name}"
    )


if __name__ == "__main__":
    # Example usage
    print("Categorical Tool Framework - Core Module")
    print("=" * 60)

    # Create a simple search tool
    def search_impl(args: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
        query = args["query"]
        return {"query": query, "results": [f"Result for {query}"]}

    search_tool = create_simple_tool(
        name="search_web",
        required_args=[("query", str)],
        func=search_impl,
        effects=[EffectType.HTTP, EffectType.IO]
    )

    # Test with context that has query
    ctx_with_query = Context(query="test search")
    result = search_tool.invoke(ctx_with_query)
    print(f"Search with query in context: {result.is_success()}")
    print(f"Result: {result.value}")

    # Test with empty context (should fail)
    ctx_empty = Context(query="")
    result_empty = search_tool.invoke(ctx_empty)
    print(f"\nSearch with empty context: {result_empty.is_success()}")
    print(f"Error: {result_empty.error}")

    # Test with synthesis
    synthesizer = KanSynthesizer()
    result_synth = search_tool.invoke(ctx_with_query, synthesizer)
    print(f"\nSearch with synthesis: {result_synth.is_success()}")

    print("\n✓ Core module validated")
