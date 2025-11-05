"""
Benchmark tools with schema validation for HOPF vs LangGraph comparison.

This module provides three tools with strict schemas:
1. eval_math: Evaluate mathematical expressions
2. web_search: Mock web search with query and limit
3. crud_tool: CRUD operations with required arguments

All tools track:
- Validation errors (missing/extra/ill-typed args)
- Execution success/failure
- Call counts for cost estimation
"""

from typing import Dict, Any, List, Tuple
import json
import re
from dataclasses import dataclass
from enum import Enum


class ToolCallResult:
    """Result of a tool call with validation tracking."""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: str = None,
        validation_errors: List[str] = None,
        tool_name: str = None
    ):
        self.success = success
        self.result = result
        self.error = error
        self.validation_errors = validation_errors or []
        self.tool_name = tool_name

    @property
    def is_valid(self) -> bool:
        """True if no validation errors."""
        return len(self.validation_errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'validation_errors': self.validation_errors,
            'tool_name': self.tool_name,
            'is_valid': self.is_valid
        }


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    required_args: Dict[str, type]
    optional_args: Dict[str, Tuple[type, Any]]  # (type, default)
    description: str

    def validate(self, args: Dict[str, Any]) -> List[str]:
        """Validate arguments against schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for missing required arguments
        for arg_name, arg_type in self.required_args.items():
            if arg_name not in args:
                errors.append(f"Missing required argument: {arg_name}")
            elif not isinstance(args[arg_name], arg_type):
                errors.append(
                    f"Argument '{arg_name}' has wrong type: "
                    f"expected {arg_type.__name__}, got {type(args[arg_name]).__name__}"
                )

        # Check for extra arguments
        all_valid_args = set(self.required_args.keys()) | set(self.optional_args.keys())
        for arg_name in args.keys():
            if arg_name not in all_valid_args:
                errors.append(f"Extra/unknown argument: {arg_name}")

        # Check optional argument types
        for arg_name, (arg_type, _) in self.optional_args.items():
            if arg_name in args and not isinstance(args[arg_name], arg_type):
                errors.append(
                    f"Optional argument '{arg_name}' has wrong type: "
                    f"expected {arg_type.__name__}, got {type(args[arg_name]).__name__}"
                )

        return errors


# Tool schemas
EVAL_MATH_SCHEMA = ToolSchema(
    name="eval_math",
    required_args={"expression": str},
    optional_args={},
    description="Evaluate a mathematical expression. Requires: expression (str)"
)

WEB_SEARCH_SCHEMA = ToolSchema(
    name="web_search",
    required_args={"query": str},
    optional_args={"limit": (int, 5)},
    description="Search the web. Requires: query (str). Optional: limit (int, default=5)"
)

CRUD_SCHEMA = ToolSchema(
    name="crud_tool",
    required_args={
        "operation": str,  # must be one of: create, read, update, delete
        "entity_type": str,
        "entity_id": str
    },
    optional_args={"data": (dict, {})},
    description=(
        "Perform CRUD operations. Requires: operation (str: create/read/update/delete), "
        "entity_type (str), entity_id (str). Optional: data (dict)"
    )
)


def eval_math(args: Dict[str, Any]) -> ToolCallResult:
    """Evaluate a mathematical expression.

    Args:
        args: Must contain 'expression' (str)

    Returns:
        ToolCallResult with validation and execution status
    """
    errors = EVAL_MATH_SCHEMA.validate(args)

    if errors:
        return ToolCallResult(
            success=False,
            error="Validation failed",
            validation_errors=errors,
            tool_name="eval_math"
        )

    expression = args['expression']

    try:
        # Safe evaluation of basic arithmetic
        # Remove whitespace and validate characters
        safe_expr = expression.strip()
        if not re.match(r'^[0-9+\-*/().\s]+$', safe_expr):
            return ToolCallResult(
                success=False,
                error=f"Invalid characters in expression: {expression}",
                validation_errors=[],
                tool_name="eval_math"
            )

        result = eval(safe_expr, {"__builtins__": {}}, {})

        return ToolCallResult(
            success=True,
            result={"expression": expression, "result": result},
            validation_errors=[],
            tool_name="eval_math"
        )
    except Exception as e:
        return ToolCallResult(
            success=False,
            error=f"Execution error: {str(e)}",
            validation_errors=[],
            tool_name="eval_math"
        )


def web_search(args: Dict[str, Any]) -> ToolCallResult:
    """Mock web search that returns synthetic results.

    Args:
        args: Must contain 'query' (str), optional 'limit' (int, default=5)

    Returns:
        ToolCallResult with validation and execution status
    """
    errors = WEB_SEARCH_SCHEMA.validate(args)

    if errors:
        return ToolCallResult(
            success=False,
            error="Validation failed",
            validation_errors=errors,
            tool_name="web_search"
        )

    query = args['query']
    limit = args.get('limit', 5)

    # Generate mock results based on query
    results = []
    for i in range(limit):
        results.append({
            'title': f'Result {i+1} for "{query}"',
            'snippet': f'This is a snippet about {query}. Result number {i+1}.',
            'url': f'https://example.com/{query.replace(" ", "_")}/{i+1}'
        })

    return ToolCallResult(
        success=True,
        result={'query': query, 'results': results, 'count': len(results)},
        validation_errors=[],
        tool_name="web_search"
    )


# Mock database for CRUD operations
_MOCK_DB: Dict[str, Dict[str, Dict[str, Any]]] = {}


def crud_tool(args: Dict[str, Any]) -> ToolCallResult:
    """Perform CRUD operations on mock entities.

    Args:
        args: Must contain:
            - operation (str): one of 'create', 'read', 'update', 'delete'
            - entity_type (str): type of entity (e.g., 'user', 'product')
            - entity_id (str): unique identifier
            - data (dict, optional): data for create/update operations

    Returns:
        ToolCallResult with validation and execution status
    """
    errors = CRUD_SCHEMA.validate(args)

    if errors:
        return ToolCallResult(
            success=False,
            error="Validation failed",
            validation_errors=errors,
            tool_name="crud_tool"
        )

    operation = args['operation']
    entity_type = args['entity_type']
    entity_id = args['entity_id']
    data = args.get('data', {})

    # Validate operation
    valid_ops = {'create', 'read', 'update', 'delete'}
    if operation not in valid_ops:
        return ToolCallResult(
            success=False,
            error=f"Invalid operation: {operation}. Must be one of {valid_ops}",
            validation_errors=[],
            tool_name="crud_tool"
        )

    # Initialize entity_type in DB if needed
    if entity_type not in _MOCK_DB:
        _MOCK_DB[entity_type] = {}

    try:
        if operation == 'create':
            if entity_id in _MOCK_DB[entity_type]:
                return ToolCallResult(
                    success=False,
                    error=f"Entity {entity_type}/{entity_id} already exists",
                    validation_errors=[],
                    tool_name="crud_tool"
                )
            _MOCK_DB[entity_type][entity_id] = data
            result = {'operation': 'create', 'entity_type': entity_type, 'entity_id': entity_id, 'data': data}

        elif operation == 'read':
            if entity_id not in _MOCK_DB[entity_type]:
                return ToolCallResult(
                    success=False,
                    error=f"Entity {entity_type}/{entity_id} not found",
                    validation_errors=[],
                    tool_name="crud_tool"
                )
            result = {
                'operation': 'read',
                'entity_type': entity_type,
                'entity_id': entity_id,
                'data': _MOCK_DB[entity_type][entity_id]
            }

        elif operation == 'update':
            if entity_id not in _MOCK_DB[entity_type]:
                return ToolCallResult(
                    success=False,
                    error=f"Entity {entity_type}/{entity_id} not found",
                    validation_errors=[],
                    tool_name="crud_tool"
                )
            _MOCK_DB[entity_type][entity_id].update(data)
            result = {
                'operation': 'update',
                'entity_type': entity_type,
                'entity_id': entity_id,
                'data': _MOCK_DB[entity_type][entity_id]
            }

        elif operation == 'delete':
            if entity_id not in _MOCK_DB[entity_type]:
                return ToolCallResult(
                    success=False,
                    error=f"Entity {entity_type}/{entity_id} not found",
                    validation_errors=[],
                    tool_name="crud_tool"
                )
            deleted_data = _MOCK_DB[entity_type].pop(entity_id)
            result = {
                'operation': 'delete',
                'entity_type': entity_type,
                'entity_id': entity_id,
                'deleted_data': deleted_data
            }

        return ToolCallResult(
            success=True,
            result=result,
            validation_errors=[],
            tool_name="crud_tool"
        )

    except Exception as e:
        return ToolCallResult(
            success=False,
            error=f"Execution error: {str(e)}",
            validation_errors=[],
            tool_name="crud_tool"
        )


def reset_mock_db():
    """Reset the mock database. Used between benchmark tasks."""
    global _MOCK_DB
    _MOCK_DB = {}


# Tool registry
TOOL_REGISTRY = {
    'eval_math': (eval_math, EVAL_MATH_SCHEMA),
    'web_search': (web_search, WEB_SEARCH_SCHEMA),
    'crud_tool': (crud_tool, CRUD_SCHEMA)
}


def get_tool(tool_name: str):
    """Get tool function and schema by name."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]
