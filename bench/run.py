"""
Main benchmark runner for HOPF vs LangGraph comparison.

Runs 150 tasks across 4 configurations:
1. LangGraph baseline
2. HOPF_LENS_DC full
3. HOPF without type checks
4. HOPF without synthesis

Computes 5 metrics:
1. Success rate (exact match or task-specific checker)
2. Tool-argument validity rate (no missing/extra/ill-typed args)
3. Average iterations to convergence
4. End-to-end latency
5. API cost
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import csv
from datetime import datetime
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.runners.langgraph_baseline import create_runner as create_langgraph
from bench.runners.hopf_full import create_runner as create_hopf_full
from bench.runners.hopf_no_typechecks import create_runner as create_hopf_no_typechecks
from bench.runners.hopf_no_synthesis import create_runner as create_hopf_no_synthesis
from bench.tools import reset_mock_db


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(tasks_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line.strip()))
    return tasks


def evaluate_answer(task: Dict[str, Any], result: Any) -> bool:
    """Evaluate if the final answer matches ground truth.

    Args:
        task: Task definition with ground_truth
        result: BenchmarkResult object

    Returns:
        True if answer is correct
    """
    if not result.success:
        return False

    ground_truth = task['ground_truth']
    gt_type = ground_truth['type']

    if gt_type == 'exact_match':
        # Check if answer contains the expected result
        expected = ground_truth['expected_result']
        answer_str = str(result.final_answer).lower()
        expected_str = str(expected).lower()

        # Check if expected value appears in answer
        return expected_str in answer_str

    elif gt_type == 'schema_check':
        # Check if tool was called with correct schema
        if not result.tool_calls:
            return False

        last_call = result.tool_calls[-1]
        if not last_call.get('success', False):
            return False

        # Check required fields
        if 'required_fields' in ground_truth:
            call_result = last_call.get('result', {})
            for field in ground_truth['required_fields']:
                if field not in call_result:
                    return False

        # Check query contains expected term
        if 'query_contains' in ground_truth:
            call_result = last_call.get('result', {})
            query = call_result.get('query', '').lower()
            expected_term = ground_truth['query_contains'].lower()
            if expected_term not in query:
                return False

        # Check expected count
        if 'expected_count' in ground_truth:
            call_result = last_call.get('result', {})
            count = call_result.get('count', 0)
            if count != ground_truth['expected_count']:
                return False

        return True

    elif gt_type == 'multi_tool':
        # Check if correct sequence of tools was called
        expected_seq = ground_truth['expected_tool_sequence']
        actual_seq = [call['tool_name'] for call in result.tool_calls]

        if len(actual_seq) < len(expected_seq):
            return False

        # Check if expected sequence appears in actual
        for i in range(len(actual_seq) - len(expected_seq) + 1):
            if actual_seq[i:i+len(expected_seq)] == expected_seq:
                # Check final query
                if 'final_query_contains' in ground_truth:
                    last_call = result.tool_calls[-1]
                    call_result = last_call.get('result', {})
                    query = call_result.get('query', '').lower()
                    expected_term = ground_truth['final_query_contains'].lower()
                    return expected_term in query
                return True

        return False

    elif gt_type == 'crud_check':
        # Check if CRUD operation was executed
        if not result.tool_calls:
            return False

        for call in result.tool_calls:
            if call.get('tool_name') == 'crud_tool' and call.get('success', False):
                call_result = call.get('result', {})
                if call_result.get('operation') == ground_truth['operation']:
                    if call_result.get('entity_type') == ground_truth['entity_type']:
                        if call_result.get('entity_id') == ground_truth['entity_id']:
                            return True

        return False

    elif gt_type == 'multi_crud':
        # Check if sequence of CRUD operations was executed
        expected_ops = ground_truth['operations']
        actual_ops = []

        for call in result.tool_calls:
            if call.get('tool_name') == 'crud_tool' and call.get('success', False):
                call_result = call.get('result', {})
                actual_ops.append(call_result.get('operation'))

        # Check if expected sequence appears in actual
        for i in range(len(actual_ops) - len(expected_ops) + 1):
            if actual_ops[i:i+len(expected_ops)] == expected_ops:
                return True

        return False

    return False


def compute_metrics(results: List[Any], tasks: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics for a set of results.

    Returns:
        Dict with keys: success_rate, validity_rate, avg_iterations,
        avg_latency_ms, total_cost
    """
    if not results:
        return {
            'success_rate': 0.0,
            'validity_rate': 0.0,
            'avg_iterations': 0.0,
            'avg_latency_ms': 0.0,
            'total_cost': 0.0
        }

    # Success rate
    successes = sum(1 for r, t in zip(results, tasks) if evaluate_answer(t, r))
    success_rate = successes / len(results)

    # Validity rate
    total_calls = sum(len(r.tool_calls) for r in results)
    valid_calls = sum(
        sum(1 for call in r.tool_calls if call.get('is_valid', False))
        for r in results
    )
    validity_rate = valid_calls / total_calls if total_calls > 0 else 0.0

    # Average iterations
    avg_iterations = sum(r.num_iterations for r in results) / len(results)

    # Average latency
    avg_latency_ms = sum(r.latency_ms for r in results) / len(results)

    # Total cost
    total_cost = sum(r.estimated_cost for r in results)

    return {
        'success_rate': success_rate,
        'validity_rate': validity_rate,
        'avg_iterations': avg_iterations,
        'avg_latency_ms': avg_latency_ms,
        'total_cost': total_cost,
        'num_tasks': len(results),
        'num_successes': successes,
        'num_valid_calls': valid_calls,
        'num_total_calls': total_calls
    }


def save_results(
    all_results: Dict[str, List[Any]],
    tasks: List[Dict[str, Any]],
    artifacts_dir: str,
    results_dir: str
):
    """Save results to CSV and JSON files."""
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save per-task results to CSV
    csv_path = os.path.join(results_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'task_id', 'runner_type', 'success', 'evaluated_correct',
            'num_iterations', 'num_tool_calls', 'tool_validity_rate',
            'num_validation_errors', 'latency_ms', 'estimated_cost', 'error'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for runner_type, results in all_results.items():
            for result, task in zip(results, tasks):
                writer.writerow({
                    'task_id': result.task_id,
                    'runner_type': runner_type,
                    'success': result.success,
                    'evaluated_correct': evaluate_answer(task, result),
                    'num_iterations': result.num_iterations,
                    'num_tool_calls': len(result.tool_calls),
                    'tool_validity_rate': result.tool_validity_rate,
                    'num_validation_errors': len(result.validation_errors),
                    'latency_ms': result.latency_ms,
                    'estimated_cost': result.estimated_cost,
                    'error': result.error or ''
                })

    print(f'Saved per-task results to {csv_path}')

    # Compute aggregate metrics
    metrics = {}
    for runner_type, results in all_results.items():
        metrics[runner_type] = compute_metrics(results, tasks)

    # Save aggregate metrics to JSON
    json_path = os.path.join(results_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Saved aggregate metrics to {json_path}')

    # Save raw traces to artifacts
    for runner_type, results in all_results.items():
        trace_path = os.path.join(artifacts_dir, f'{runner_type}_traces.jsonl')
        with open(trace_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + '\n')

        print(f'Saved {runner_type} traces to {trace_path}')

    # Print summary table
    print('\n' + '='*80)
    print('BENCHMARK RESULTS SUMMARY')
    print('='*80)
    print(f'{"Runner":<25} {"Success":<10} {"Validity":<10} {"Iters":<8} {"Latency(ms)":<12} {"Cost($)":<10}')
    print('-'*80)

    for runner_type in ['langgraph', 'hopf_full', 'hopf_no_typechecks', 'hopf_no_synthesis']:
        m = metrics[runner_type]
        print(f'{runner_type:<25} {m["success_rate"]*100:>7.2f}%  {m["validity_rate"]*100:>7.2f}%  '
              f'{m["avg_iterations"]:>6.2f}  {m["avg_latency_ms"]:>10.1f}  {m["total_cost"]:>8.4f}')

    print('='*80)

    # Determine winner
    baseline = metrics['langgraph']
    hopf_full = metrics['hopf_full']

    success_delta = (hopf_full['success_rate'] - baseline['success_rate']) * 100
    validity_delta = (baseline['validity_rate'] - hopf_full['validity_rate']) * 100  # Lower is better
    latency_ratio = hopf_full['avg_latency_ms'] / baseline['avg_latency_ms']
    cost_ratio = hopf_full['total_cost'] / baseline['total_cost']

    print('\nCOMPARISON: HOPF_FULL vs LANGGRAPH')
    print(f'  Success rate delta: {success_delta:+.2f} pp')
    print(f'  Invalid call reduction: {validity_delta:+.2f} pp')
    print(f'  Latency ratio: {latency_ratio:.2f}x')
    print(f'  Cost ratio: {cost_ratio:.2f}x')

    # Win criteria: ≥5pp higher success OR ≥20pp fewer invalid calls, with non-inferior latency/cost (±5%)
    win = False
    if success_delta >= 5 or validity_delta >= 20:
        if latency_ratio <= 1.05 and cost_ratio <= 1.05:
            win = True

    print(f'\nWINNER: {"HOPF_LENS_DC" if win else "TIE/LANGGRAPH"}')
    print('='*80)


def main():
    parser = argparse.ArgumentParser(description='Run HOPF vs LangGraph benchmark')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use (default: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature (default: 0.2)')
    parser.add_argument('--max-tool-calls', type=int, default=6, help='Max tool calls per task (default: 6)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per task in seconds (default: 30)')
    parser.add_argument('--tasks', type=str, default='bench/tasks.jsonl', help='Path to tasks file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num-tasks', type=int, default=None, help='Number of tasks to run (default: all)')
    parser.add_argument('--runners', type=str, default='all',
                       help='Comma-separated list of runners (langgraph,hopf_full,hopf_no_typechecks,hopf_no_synthesis) or "all"')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('Error: OpenAI API key not provided. Set --api-key or OPENAI_API_KEY environment variable.')
        sys.exit(1)

    # Set random seed
    random.seed(args.seed)

    # Load tasks
    tasks = load_tasks(args.tasks)
    if args.num_tasks:
        tasks = tasks[:args.num_tasks]

    print(f'Loaded {len(tasks)} tasks')
    print(f'Model: {args.model}')
    print(f'Temperature: {args.temperature}')
    print(f'Max tool calls: {args.max_tool_calls}')
    print(f'Timeout: {args.timeout}s')
    print(f'Seed: {args.seed}')

    # Determine which runners to use
    if args.runners == 'all':
        runner_names = ['langgraph', 'hopf_full', 'hopf_no_typechecks', 'hopf_no_synthesis']
    else:
        runner_names = [r.strip() for r in args.runners.split(',')]

    # Create runners
    runner_factories = {
        'langgraph': create_langgraph,
        'hopf_full': create_hopf_full,
        'hopf_no_typechecks': create_hopf_no_typechecks,
        'hopf_no_synthesis': create_hopf_no_synthesis
    }

    runner_kwargs = {
        'api_key': api_key,
        'model': args.model,
        'temperature': args.temperature,
        'max_tool_calls': args.max_tool_calls,
        'timeout_s': args.timeout
    }

    # Run benchmarks
    all_results = {}

    for runner_name in runner_names:
        print(f'\n{"="*80}')
        print(f'Running {runner_name.upper()}')
        print(f'{"="*80}')

        runner = runner_factories[runner_name](**runner_kwargs)
        results = []

        for i, task in enumerate(tasks):
            print(f'[{i+1}/{len(tasks)}] {task["task_id"]}: {task["description"][:60]}...')

            try:
                result = runner.run_task(task)
                results.append(result)

                status = '✓' if result.success else '✗'
                print(f'  {status} {result.num_iterations} iters, {len(result.tool_calls)} calls, '
                      f'{result.latency_ms:.0f}ms')

            except Exception as e:
                print(f'  ERROR: {str(e)}')
                # Create failed result
                from bench.runners import BenchmarkResult
                result = BenchmarkResult(
                    task_id=task['task_id'],
                    runner_type=runner_name,
                    success=False,
                    final_answer=None,
                    num_iterations=0,
                    tool_calls=[],
                    validation_errors=[],
                    latency_ms=0.0,
                    estimated_cost=0.0,
                    error=str(e)
                )
                results.append(result)

        all_results[runner_name] = results

    # Save results
    artifacts_dir = 'bench/artifacts'
    results_dir = 'bench/results'
    save_results(all_results, tasks, artifacts_dir, results_dir)


if __name__ == '__main__':
    main()
