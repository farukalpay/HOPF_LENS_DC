#!/usr/bin/env python3
"""
Example 1: Simple Math Query
Demonstrates basic categorical pipeline with eval_math tool
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hopf_lens_dc.tool import hopf_lens_dc_categorical

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    query = "Calculate the sum of 15 and 27"

    print("=" * 80)
    print("EXAMPLE 1: Simple Math Query")
    print("=" * 80)
    print(f"Query: {query}\n")

    result = hopf_lens_dc_categorical(query, api_key, time_budget_ms=20000)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Answer: {result['answer']}")
    print(f"✓ Confidence: {result['confidence']:.3f}")
    print(f"✓ Tools used: {', '.join(result['tools_used'])}")
    print(f"✓ Execution time: {result['elapsed_seconds']:.2f}s")
    print(f"✓ Convergence: {result['convergence']['converged']}")
    print(f"✓ Evidence quality: {result['evidence']['coend']} morphisms")
    print(f"✓ Robustness: {result['robustness']['score']:.3f}")

if __name__ == "__main__":
    main()
