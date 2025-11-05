"""
HOPF_LENS_DC runner with type checks DISABLED.

This ablation skips:
- Schema validation (has_limit checks)
- Argument type checking
- Missing argument detection

But keeps:
- Dynamic synthesis
- Convergence iteration
- Evidence tracking
"""

import json
import time
from typing import Dict, Any, Optional
import openai
from bench.runners import BenchmarkResult
from bench.tools import get_tool, reset_mock_db

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.hopf_lens_dc.categorical_core import (
    Context,
    KanSynthesizer
)

INPUT_TOKEN_COST = 0.00003
OUTPUT_TOKEN_COST = 0.00006


class HOPFNoTypeChecksRunner:
    """HOPF runner with type checking disabled."""

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

    def _get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions."""
        from bench.tools import TOOL_REGISTRY

        tool_descriptions = []
        for tool_name, (_, schema) in TOOL_REGISTRY.items():
            required = ", ".join(f"{k}:{v.__name__}" for k, v in schema.required_args.items())
            optional = ", ".join(
                f"{k}:{v[0].__name__}={v[1]}" for k, v in schema.optional_args.items()
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
        """Run task WITHOUT type checking (ablation)."""
        task_id = task['task_id']
        description = task['description']
        start_time = time.time()

        reset_mock_db()

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
                    args = parsed["args"]

                    try:
                        # NO TYPE CHECKING - just call the tool directly
                        tool_func, schema = get_tool(tool_name)

                        # Still try synthesis if args are missing
                        for key, value in args.items():
                            context = context.extend(key, value)

                        # Check if synthesis is needed (but don't validate)
                        missing_args = [
                            arg for arg in schema.required_args.keys()
                            if arg not in args
                        ]

                        if missing_args:
                            synth_result = self.synthesizer.synthesize(
                                context, schema, missing_args
                            )
                            if synth_result.is_success():
                                args.update(synth_result.value)

                        # Execute tool WITHOUT validation
                        result = tool_func(args)

                        # Record tool call
                        # Note: We still check result.is_valid from the tool itself
                        # but we didn't do categorical validation
                        tool_calls.append({
                            'iteration': iterations,
                            'tool_name': tool_name,
                            'args': args,
                            'is_valid': result.is_valid,  # From tool execution
                            'validation_errors': result.validation_errors,
                            'success': result.success,
                            'result': result.result,
                            'error': result.error
                        })

                        if result.validation_errors:
                            validation_errors.extend(result.validation_errors)

                        if result.success:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": f"Tool result: {json.dumps(result.result)}"
                            })
                        else:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": f"Tool error: {result.error}"
                            })

                    except Exception as e:
                        error = f"Tool execution error: {str(e)}"
                        break
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
            runner_type="hopf_no_typechecks",
            success=success,
            final_answer=final_answer,
            num_iterations=iterations,
            tool_calls=tool_calls,
            validation_errors=validation_errors,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            error=error
        )


def create_runner(api_key: str, **kwargs) -> HOPFNoTypeChecksRunner:
    """Factory function to create HOPF no-typecheck runner."""
    return HOPFNoTypeChecksRunner(api_key=api_key, **kwargs)
