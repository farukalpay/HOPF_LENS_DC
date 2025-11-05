"""
Benchmark runners for HOPF vs LangGraph comparison.

This module provides:
1. LangGraphBaseline: Simple tool-calling loop
2. HOPFFullRunner: Full HOPF_LENS_DC with all features
3. HOPFNoTypeChecksRunner: HOPF with type checks disabled
4. HOPFNoSynthesisRunner: HOPF with synthesis disabled
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class RunnerType(Enum):
    """Types of benchmark runners."""
    LANGGRAPH = "langgraph"
    HOPF_FULL = "hopf_full"
    HOPF_NO_TYPECHECKS = "hopf_no_typechecks"
    HOPF_NO_SYNTHESIS = "hopf_no_synthesis"


@dataclass
class BenchmarkResult:
    """Result of running a single task."""
    task_id: str
    runner_type: str
    success: bool
    final_answer: Any
    num_iterations: int
    tool_calls: list  # List of tool call records
    validation_errors: list  # List of validation errors
    latency_ms: float
    estimated_cost: float
    error: str = None

    @property
    def tool_validity_rate(self) -> float:
        """Fraction of tool calls with valid arguments."""
        if not self.tool_calls:
            return 0.0
        valid_calls = sum(1 for call in self.tool_calls if call.get('is_valid', False))
        return valid_calls / len(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'runner_type': self.runner_type,
            'success': self.success,
            'final_answer': self.final_answer,
            'num_iterations': self.num_iterations,
            'num_tool_calls': len(self.tool_calls),
            'tool_validity_rate': self.tool_validity_rate,
            'validation_errors': self.validation_errors,
            'latency_ms': self.latency_ms,
            'estimated_cost': self.estimated_cost,
            'error': self.error
        }
