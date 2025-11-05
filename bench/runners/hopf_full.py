"""
HOPF_LENS_DC runner using the REAL categorical framework.

This implementation uses:
- AritySchema from categorical_core for type checking
- Context and projections for argument assembly
- KanSynthesizer for automatic argument extraction
- CategoricalToolRegistry for tool execution
- Effect monad for composable error handling
"""

import json
import time
from typing import Dict, Any, Optional
import openai
from bench.runners import BenchmarkResult
from bench.tools import TOOL_REGISTRY, reset_mock_db

# Import actual HOPF_LENS_DC categorical framework
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.hopf_lens_dc.categorical_core import (
    CategoricalToolRegistry,
    AritySchema,
    DirectAssembler,
    Context,
    Effect,
    EffectType,
    KanSynthesizer,
    ToolMorphism
)

INPUT_TOKEN_COST = 0.00003
OUTPUT_TOKEN_COST = 0.00006


class HOPFCategoricalRunner:
    """HOPF_LENS_DC runner with full categorical guarantees."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tool_calls: int = 6,
        timeout_s: int = 30
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s
        self.synthesizer = KanSynthesizer()

        # Build categorical tool registry with proper schemas
        self.registry = self._build_categorical_registry()

    def _build_categorical_registry(self) -> CategoricalToolRegistry:
        """Build categorical tool registry from benchmark tools."""
        registry = CategoricalToolRegistry()

        # Register eval_math
        math_schema = AritySchema()
        math_schema.add_arg("expression", str, required=True)

        def math_impl(args: Dict[str, Any], ctx: Context) -> Effect[Dict]:
            from bench.tools import eval_math
            result = eval_math(args)
            if result.success:
                return Effect.pure(result.result)
            else:
                return Effect.fail(result.error)

        math_assembler = DirectAssembler(math_schema, {"expression": "expression"})

        math_tool = ToolMorphism(
            name="eval_math",
            schema=math_schema,
            assembler=math_assembler,
            func=math_impl,
            effects=[EffectType.PURE]
        )
        registry.tools["eval_math"] = math_tool

        # Register web_search
        search_schema = AritySchema()
        search_schema.add_arg("query", str, required=True)
        search_schema.add_arg("limit", int, required=False, default=5)

        def search_impl(args: Dict[str, Any], ctx: Context) -> Effect[Dict]:
            from bench.tools import web_search
            result = web_search(args)
            if result.success:
                return Effect.pure(result.result)
            else:
                return Effect.fail(result.error)

        search_assembler = DirectAssembler(
            search_schema,
            {"query": "query", "limit": "limit"}
        )

        search_tool = ToolMorphism(
            name="web_search",
            schema=search_schema,
            assembler=search_assembler,
            func=search_impl,
            effects=[EffectType.HTTP]
        )
        registry.tools["web_search"] = search_tool

        # Register crud_tool
        crud_schema = AritySchema()
        crud_schema.add_arg("operation", str, required=True)
        crud_schema.add_arg("entity_type", str, required=True)
        crud_schema.add_arg("entity_id", str, required=True)
        crud_schema.add_arg("data", dict, required=False, default={})

        def crud_impl(args: Dict[str, Any], ctx: Context) -> Effect[Dict]:
            from bench.tools import crud_tool
            result = crud_tool(args)
            if result.success:
                return Effect.pure(result.result)
            else:
                return Effect.fail(result.error)

        crud_assembler = DirectAssembler(
            crud_schema,
            {
                "operation": "operation",
                "entity_type": "entity_type",
                "entity_id": "entity_id",
                "data": "data"
            }
        )

        crud_tool = ToolMorphism(
            name="crud_tool",
            schema=crud_schema,
            assembler=crud_assembler,
            func=crud_impl,
            effects=[EffectType.IO]
        )
        registry.tools["crud_tool"] = crud_tool

        return registry

    def _get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions."""
        tool_descriptions = []

        for tool_name, (_, bench_schema) in TOOL_REGISTRY.items():
            required = ", ".join(f"{k}:{v.__name__}" for k, v in bench_schema.required_args.items())
            optional = ", ".join(
                f"{k}:{v[0].__name__}={v[1]}" for k, v in bench_schema.optional_args.items()
            )
            args_str = required
            if optional:
                args_str += f", optional: {optional}"

            tool_descriptions.append(f"- {tool_name}({args_str})")

        tools_text = "\n".join(tool_descriptions)

        return f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools_text}

To use a tool, respond with JSON in this format:
{{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}

After using tools, provide a final answer in this format:
{{"answer": "your final answer here"}}

Be concise and accurate."""

    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract tool call or answer."""
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            return None
        except json.JSONDecodeError:
            return None

    def run_task(self, task: Dict[str, Any]) -> BenchmarkResult:
        """Run task with categorical HOPF_LENS_DC guarantees."""
        task_id = task['task_id']
        description = task['description']
        start_time = time.time()

        reset_mock_db()

        # Create categorical context
        context = Context(query=description)

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": description}
        ]

        tool_calls = []
        validation_errors = []
        iterations = 0
        final_answer = None
        error = None
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            while iterations < self.max_tool_calls:
                iterations += 1

                elapsed = time.time() - start_time
                if elapsed > self.timeout_s:
                    error = "Timeout"
                    break

                # Get LLM response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=500
                )

                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens

                content = response.choices[0].message.content
                parsed = self._parse_llm_response(content)

                if parsed is None:
                    final_answer = content
                    break

                if "answer" in parsed:
                    final_answer = parsed["answer"]
                    break

                if "tool" in parsed and "args" in parsed:
                    tool_name = parsed["tool"]
                    llm_args = parsed["args"]

                    # Check if tool exists in categorical registry
                    if tool_name not in self.registry.tools:
                        error = f"Unknown tool: {tool_name}"
                        break

                    tool = self.registry.tools[tool_name]
                    schema = tool.schema

                    # Build context with LLM-provided args
                    current_context = context
                    for key, value in llm_args.items():
                        current_context = current_context.extend(key, value)

                    # TYPE CHECKING: Use has_limit to check if all required projections exist
                    has_limit, missing_projections = schema.has_limit(current_context)

                    validation_errs = []
                    if not has_limit:
                        # Missing required arguments - record validation errors
                        for missing in missing_projections:
                            validation_errs.append(f"Missing required argument: {missing}")

                        # SYNTHESIS: Try Kan extension to fill missing args
                        synth_result = self.synthesizer.synthesize(
                            current_context, schema, missing_projections
                        )

                        if synth_result.is_success():
                            # Add synthesized args to context
                            for key, value in synth_result.value.items():
                                current_context = current_context.extend(key, value)
                                llm_args[key] = value  # Also update for recording

                            # Re-check has_limit after synthesis
                            has_limit, missing_projections = schema.has_limit(current_context)
                            if has_limit:
                                validation_errs = []  # Clear errors if synthesis succeeded

                    # Execute tool via categorical registry
                    result = self.registry.invoke(
                        tool_name, current_context, use_synthesis=True
                    )

                    # Record tool call with validation status
                    is_valid = len(validation_errs) == 0 and result.is_success()
                    tool_calls.append({
                        'iteration': iterations,
                        'tool_name': tool_name,
                        'args': llm_args,
                        'is_valid': is_valid,
                        'validation_errors': validation_errs,
                        'success': result.is_success(),
                        'result': result.value if result.is_success() else None,
                        'error': result.error if not result.is_success() else None
                    })

                    if validation_errs:
                        validation_errors.extend(validation_errs)

                    # Add result to messages
                    if result.is_success():
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"Tool result: {json.dumps(result.value)}"
                        })
                    else:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"Tool error: {result.error}"
                        })
                else:
                    final_answer = content
                    break

            if final_answer is None and iterations >= self.max_tool_calls:
                final_answer = f"Max iterations reached ({self.max_tool_calls})"
                error = "Max iterations"

        except Exception as e:
            error = str(e)
            final_answer = None

        latency_ms = (time.time() - start_time) * 1000
        estimated_cost = (
            total_input_tokens * INPUT_TOKEN_COST / 1000 +
            total_output_tokens * OUTPUT_TOKEN_COST / 1000
        )

        success = final_answer is not None and error is None

        return BenchmarkResult(
            task_id=task_id,
            runner_type="hopf_full",
            success=success,
            final_answer=final_answer,
            num_iterations=iterations,
            tool_calls=tool_calls,
            validation_errors=validation_errors,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            error=error
        )


def create_runner(api_key: str, **kwargs):
    """Factory function to create categorical HOPF runner."""
    return HOPFCategoricalRunner(api_key=api_key, **kwargs)
