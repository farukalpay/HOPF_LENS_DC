"""
LangGraph-style baseline: single agent with tool-calling loop.

This is a simple ReAct-style agent that:
1. Receives a query
2. Decides which tool to call via LLM
3. Executes the tool
4. Repeats until answer is complete or max iterations reached

No type checking, no synthesis - just raw LLM tool calling.
"""

import json
import time
from typing import Dict, Any, List, Optional
import openai
from bench.runners import BenchmarkResult
from bench.tools import get_tool, TOOL_REGISTRY, reset_mock_db


# Token cost estimates (GPT-4 pricing as proxy)
INPUT_TOKEN_COST = 0.00003  # $0.03 per 1K tokens
OUTPUT_TOKEN_COST = 0.00006  # $0.06 per 1K tokens


class LangGraphBaseline:
    """Simple tool-calling loop baseline."""

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

    def _get_system_prompt(self) -> str:
        """Get system prompt with tool descriptions."""
        tool_descriptions = []
        for tool_name, (_, schema) in TOOL_REGISTRY.items():
            required = ", ".join(f"{k}:{v.__name__}" for k, v in schema.required_args.items())
            optional = ", ".join(
                f"{k}:{v[0].__name__}={v[1]}" for k, v in schema.optional_args.items()
            )
            args_str = required
            if optional:
                args_str += f", optional: {optional}"

            tool_descriptions.append(
                f"- {tool_name}({args_str}): {schema.description}"
            )

        tools_text = "\n".join(tool_descriptions)

        return f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools_text}

To use a tool, respond with JSON in this format:
{{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}

After using tools, provide a final answer in this format:
{{"answer": "your final answer here"}}

Be concise and accurate. Only call tools when necessary."""

    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract tool call or answer."""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

            # Try to find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)

            return None
        except json.JSONDecodeError:
            return None

    def run_task(self, task: Dict[str, Any]) -> BenchmarkResult:
        """Run a single task using LangGraph-style loop."""
        task_id = task['task_id']
        description = task['description']
        start_time = time.time()

        reset_mock_db()  # Reset state

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

                # Check timeout
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

                # Parse response
                parsed = self._parse_llm_response(content)

                if parsed is None:
                    # Could not parse - treat as final answer
                    final_answer = content
                    break

                # Check if it's a tool call or final answer
                if "answer" in parsed:
                    final_answer = parsed["answer"]
                    break

                if "tool" in parsed and "args" in parsed:
                    tool_name = parsed["tool"]
                    args = parsed["args"]

                    # Execute tool
                    try:
                        tool_func, schema = get_tool(tool_name)
                        result = tool_func(args)

                        # Record tool call
                        tool_calls.append({
                            'iteration': iterations,
                            'tool_name': tool_name,
                            'args': args,
                            'is_valid': result.is_valid,
                            'validation_errors': result.validation_errors,
                            'success': result.success,
                            'result': result.result,
                            'error': result.error
                        })

                        # Track validation errors
                        if result.validation_errors:
                            validation_errors.extend(result.validation_errors)

                        # Add tool result to messages
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
                    # Malformed response
                    final_answer = content
                    break

            # If no answer yet, use last message
            if final_answer is None and iterations >= self.max_tool_calls:
                final_answer = f"Max iterations reached ({self.max_tool_calls})"
                error = "Max iterations"

        except Exception as e:
            error = str(e)
            final_answer = None

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        estimated_cost = (
            total_input_tokens * INPUT_TOKEN_COST / 1000 +
            total_output_tokens * OUTPUT_TOKEN_COST / 1000
        )

        success = final_answer is not None and error is None

        return BenchmarkResult(
            task_id=task_id,
            runner_type="langgraph",
            success=success,
            final_answer=final_answer,
            num_iterations=iterations,
            tool_calls=tool_calls,
            validation_errors=validation_errors,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            error=error
        )


def create_runner(api_key: str, **kwargs) -> LangGraphBaseline:
    """Factory function to create LangGraph baseline runner."""
    return LangGraphBaseline(api_key=api_key, **kwargs)
