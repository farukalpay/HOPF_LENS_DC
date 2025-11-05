"""
Generate 150-task benchmark dataset for HOPF vs LangGraph comparison.

Tasks are divided into:
- 50 math tasks (eval_math tool)
- 50 web search tasks (web_search tool)
- 50 CRUD tasks (crud_tool, may require 2-3 calls)

Each task has:
- task_id: unique identifier
- description: natural language query
- ground_truth: expected result or validation function
- expected_tools: list of tools that should be called
- max_tool_calls: maximum expected tool calls (1-3)
"""

import json
import random
from typing import Dict, Any, List


def generate_math_tasks(start_id: int) -> List[Dict[str, Any]]:
    """Generate 50 math evaluation tasks."""
    tasks = []

    # Simple arithmetic (20 tasks)
    operations = [
        ('+', lambda a, b: a + b),
        ('-', lambda a, b: a - b),
        ('*', lambda a, b: a * b),
        ('/', lambda a, b: a / b)
    ]

    for i in range(20):
        a, b = random.randint(1, 100), random.randint(1, 100)
        if i % 4 == 3:  # division
            b = random.randint(1, 20)  # smaller divisor
        op_symbol, op_func = operations[i % 4]
        result = op_func(a, b)

        tasks.append({
            'task_id': f'math_{start_id + i:03d}',
            'description': f'Calculate {a} {op_symbol} {b}',
            'ground_truth': {
                'type': 'exact_match',
                'expected_result': result,
                'expression': f'{a}{op_symbol}{b}'
            },
            'expected_tools': ['eval_math'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    # Multi-step arithmetic (15 tasks)
    for i in range(15):
        a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)
        expressions = [
            (f'{a} + {b} * {c}', a + b * c),
            (f'({a} + {b}) * {c}', (a + b) * c),
            (f'{a} * {b} + {c}', a * b + c),
        ]
        expr, result = expressions[i % 3]

        tasks.append({
            'task_id': f'math_{start_id + 20 + i:03d}',
            'description': f'What is {expr}?',
            'ground_truth': {
                'type': 'exact_match',
                'expected_result': result,
                'expression': expr
            },
            'expected_tools': ['eval_math'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    # Word problems (15 tasks)
    word_problems = [
        ('sum of {} and {}', '{} + {}', lambda a, b: a + b),
        ('difference between {} and {}', '{} - {}', lambda a, b: a - b),
        ('product of {} and {}', '{} * {}', lambda a, b: a * b),
        ('{} divided by {}', '{} / {}', lambda a, b: a / b),
        ('{} plus {} times {}', '{} + {} * {}', lambda a, b, c: a + b * c),
    ]

    for i in range(15):
        if i < 12:  # 2-operand
            a, b = random.randint(1, 100), random.randint(1, 100)
            if i % 4 == 3:  # division
                b = random.randint(1, 20)
            desc_template, expr_template, func = word_problems[i % 4]
            description = desc_template.format(a, b)
            expression = expr_template.format(a, b)
            result = func(a, b)
        else:  # 3-operand
            a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)
            desc_template, expr_template, func = word_problems[4]
            description = desc_template.format(a, b, c)
            expression = expr_template.format(a, b, c)
            result = func(a, b, c)

        tasks.append({
            'task_id': f'math_{start_id + 35 + i:03d}',
            'description': f'Compute the {description}',
            'ground_truth': {
                'type': 'exact_match',
                'expected_result': result,
                'expression': expression
            },
            'expected_tools': ['eval_math'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    return tasks


def generate_search_tasks(start_id: int) -> List[Dict[str, Any]]:
    """Generate 50 web search tasks."""
    tasks = []

    topics = [
        'machine learning', 'quantum computing', 'climate change', 'renewable energy',
        'artificial intelligence', 'blockchain', 'space exploration', 'gene therapy',
        'autonomous vehicles', 'cybersecurity', 'neural networks', 'data science',
        'cloud computing', 'edge computing', 'IoT devices', 'quantum encryption',
        'fusion energy', 'CRISPR', 'nanotechnology', 'robotics', 'virtual reality',
        'augmented reality', '5G networks', 'satellite internet', 'carbon capture'
    ]

    # Simple search with default limit (20 tasks)
    for i in range(20):
        topic = topics[i % len(topics)]

        tasks.append({
            'task_id': f'search_{start_id + i:03d}',
            'description': f'Search for information about {topic}',
            'ground_truth': {
                'type': 'schema_check',
                'required_fields': ['query', 'results', 'count'],
                'query_contains': topic.split()[0]  # at least first word
            },
            'expected_tools': ['web_search'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    # Search with explicit limit (20 tasks)
    for i in range(20):
        topic = topics[i % len(topics)]
        limit = random.choice([3, 5, 10, 15])

        tasks.append({
            'task_id': f'search_{start_id + 20 + i:03d}',
            'description': f'Find {limit} results about {topic}',
            'ground_truth': {
                'type': 'schema_check',
                'required_fields': ['query', 'results', 'count'],
                'query_contains': topic.split()[0],
                'expected_count': limit
            },
            'expected_tools': ['web_search'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    # Two-step search and refine (10 tasks)
    for i in range(10):
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 5) % len(topics)]

        tasks.append({
            'task_id': f'search_{start_id + 40 + i:03d}',
            'description': f'Search for {topic1} and then find 3 results about {topic2}',
            'ground_truth': {
                'type': 'multi_tool',
                'expected_tool_sequence': ['web_search', 'web_search'],
                'final_query_contains': topic2.split()[0]
            },
            'expected_tools': ['web_search'],
            'max_tool_calls': 2,
            'timeout_s': 30
        })

    return tasks


def generate_crud_tasks(start_id: int) -> List[Dict[str, Any]]:
    """Generate 50 CRUD operation tasks."""
    tasks = []

    entity_types = ['user', 'product', 'order', 'document', 'record']

    # Single CRUD operations (20 tasks)
    operations = ['create', 'read', 'update', 'delete']
    for i in range(20):
        op = operations[i % 4]
        entity_type = entity_types[i % len(entity_types)]
        entity_id = f'test_{i:03d}'

        if op == 'create':
            description = f'Create a new {entity_type} with id {entity_id} and name "Test Item {i}"'
            ground_truth = {
                'type': 'crud_check',
                'operation': 'create',
                'entity_type': entity_type,
                'entity_id': entity_id
            }
        elif op == 'read':
            description = f'Read the {entity_type} with id {entity_id}'
            ground_truth = {
                'type': 'crud_check',
                'operation': 'read',
                'entity_type': entity_type,
                'entity_id': entity_id
            }
        elif op == 'update':
            description = f'Update the {entity_type} with id {entity_id} to set status to "active"'
            ground_truth = {
                'type': 'crud_check',
                'operation': 'update',
                'entity_type': entity_type,
                'entity_id': entity_id
            }
        else:  # delete
            description = f'Delete the {entity_type} with id {entity_id}'
            ground_truth = {
                'type': 'crud_check',
                'operation': 'delete',
                'entity_type': entity_type,
                'entity_id': entity_id
            }

        tasks.append({
            'task_id': f'crud_{start_id + i:03d}',
            'description': description,
            'ground_truth': ground_truth,
            'expected_tools': ['crud_tool'],
            'max_tool_calls': 1,
            'timeout_s': 30
        })

    # Two-step CRUD (15 tasks: create then read)
    for i in range(15):
        entity_type = entity_types[i % len(entity_types)]
        entity_id = f'multi_{i:03d}'

        tasks.append({
            'task_id': f'crud_{start_id + 20 + i:03d}',
            'description': (
                f'Create a {entity_type} with id {entity_id} and name "Item {i}", '
                f'then read it back to verify'
            ),
            'ground_truth': {
                'type': 'multi_crud',
                'operations': ['create', 'read'],
                'entity_type': entity_type,
                'entity_id': entity_id
            },
            'expected_tools': ['crud_tool'],
            'max_tool_calls': 2,
            'timeout_s': 30
        })

    # Three-step CRUD (15 tasks: create, update, read)
    for i in range(15):
        entity_type = entity_types[i % len(entity_types)]
        entity_id = f'complex_{i:03d}'

        tasks.append({
            'task_id': f'crud_{start_id + 35 + i:03d}',
            'description': (
                f'Create a {entity_type} with id {entity_id}, '
                f'then update its status to "completed", '
                f'then read it to confirm the update'
            ),
            'ground_truth': {
                'type': 'multi_crud',
                'operations': ['create', 'update', 'read'],
                'entity_type': entity_type,
                'entity_id': entity_id
            },
            'expected_tools': ['crud_tool'],
            'max_tool_calls': 3,
            'timeout_s': 30
        })

    return tasks


def main():
    """Generate all 150 tasks and save to tasks.jsonl."""
    random.seed(42)  # Fixed seed for reproducibility

    tasks = []
    tasks.extend(generate_math_tasks(0))
    tasks.extend(generate_search_tasks(50))
    tasks.extend(generate_crud_tasks(100))

    # Shuffle tasks to mix categories
    random.Random(42).shuffle(tasks)

    # Re-assign sequential IDs
    for i, task in enumerate(tasks):
        task['task_id'] = f'task_{i:03d}'

    # Save to JSONL
    output_path = 'bench/tasks.jsonl'
    with open(output_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

    print(f'Generated {len(tasks)} tasks')
    print(f'Saved to {output_path}')

    # Print statistics
    tool_counts = {}
    for task in tasks:
        for tool in task['expected_tools']:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print('\nTool distribution:')
    for tool, count in sorted(tool_counts.items()):
        print(f'  {tool}: {count}')

    call_counts = {}
    for task in tasks:
        max_calls = task['max_tool_calls']
        call_counts[max_calls] = call_counts.get(max_calls, 0) + 1

    print('\nTool call distribution:')
    for calls, count in sorted(call_counts.items()):
        print(f'  {calls} calls: {count} tasks')


if __name__ == '__main__':
    main()
