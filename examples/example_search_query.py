#!/usr/bin/env python3
"""
Example 2: Web Search Query
Demonstrates Kleisli composition with search_web tool
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

    query = "What are the benefits of category theory in software engineering?"

    print("=" * 80)
    print("EXAMPLE 2: Web Search Query")
    print("=" * 80)
    print(f"Query: {query}\n")

    result = hopf_lens_dc_categorical(query, api_key, time_budget_ms=30000)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Answer: {result['answer'][:200]}...")
    print(f"✓ Confidence: {result['confidence']:.3f}")
    print(f"✓ Tools used: {', '.join(result['tools_used'])}")
    print(f"✓ Execution time: {result['elapsed_seconds']:.2f}s")
    print(f"✓ Convergence: {result['convergence']['converged']} ({result['convergence']['iterations']} iterations)")
    print(f"✓ Claims: {len(result['evidence']['claims'])}")
    print(f"✓ Sources: {len(result['evidence']['sources'])}")
    print(f"✓ Evidence morphisms: {result['evidence']['coend']}")
    print(f"✓ Robustness: {result['robustness']['score']:.3f}")
    print(f"✓ Attacks tested: {len(result['robustness']['attacks'])}")

if __name__ == "__main__":
    main()
