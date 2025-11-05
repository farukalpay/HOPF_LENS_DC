.PHONY: bench bench-quick bench-full clean-bench help install-bench

# Load environment variables from .env if it exists
-include .env
export

help:
	@echo "HOPF_LENS_DC Benchmark Targets:"
	@echo ""
	@echo "  make bench           - Run full 150-task benchmark (all 4 runners)"
	@echo "  make bench-quick     - Run quick 10-task benchmark for testing"
	@echo "  make bench-full      - Run full benchmark with detailed output"
	@echo "  make install-bench   - Install benchmark dependencies"
	@echo "  make clean-bench     - Clean benchmark artifacts and results"
	@echo ""
	@echo "Environment variables (set in .env or override on command line):"
	@echo "  OPENAI_API_KEY       - OpenAI API key (required)"
	@echo "  BENCHMARK_MODEL      - Model to use (default: gpt-4o)"
	@echo "  BENCHMARK_TEMPERATURE - Temperature (default: 0.2)"
	@echo "  BENCHMARK_SEED       - Random seed (default: 42)"

install-bench:
	@echo "Installing benchmark dependencies..."
	pip install -r bench/requirements.txt
	@echo "Done!"

bench: bench-full

bench-quick:
	@echo "Running quick 10-task benchmark..."
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "Error: OPENAI_API_KEY not set. Copy .env.template to .env and set your API key."; \
		exit 1; \
	fi
	python bench/run.py \
		--api-key "$$OPENAI_API_KEY" \
		--model "$${BENCHMARK_MODEL:-gpt-4o}" \
		--temperature "$${BENCHMARK_TEMPERATURE:-0.2}" \
		--max-tool-calls "$${BENCHMARK_MAX_TOOL_CALLS:-6}" \
		--timeout "$${BENCHMARK_TIMEOUT:-30}" \
		--seed "$${BENCHMARK_SEED:-42}" \
		--num-tasks 10

bench-full:
	@echo "Running full 150-task benchmark..."
	@echo "This may take 30-60 minutes and cost approximately \$5-10 in API calls."
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "Error: OPENAI_API_KEY not set. Copy .env.template to .env and set your API key."; \
		exit 1; \
	fi
	python bench/run.py \
		--api-key "$$OPENAI_API_KEY" \
		--model "$${BENCHMARK_MODEL:-gpt-4o}" \
		--temperature "$${BENCHMARK_TEMPERATURE:-0.2}" \
		--max-tool-calls "$${BENCHMARK_MAX_TOOL_CALLS:-6}" \
		--timeout "$${BENCHMARK_TIMEOUT:-30}" \
		--seed "$${BENCHMARK_SEED:-42}"
	@echo ""
	@echo "Benchmark complete!"
	@echo "Results saved to:"
	@echo "  - bench/results/results.csv (per-task results)"
	@echo "  - bench/results/metrics.json (aggregate metrics)"
	@echo "  - bench/artifacts/*_traces.jsonl (detailed traces)"

clean-bench:
	@echo "Cleaning benchmark artifacts and results..."
	rm -rf bench/artifacts/*.jsonl
	rm -rf bench/results/*.csv bench/results/*.json
	@echo "Done!"
