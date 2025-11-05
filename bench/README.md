# HOPF vs LangGraph Benchmark

This directory contains a comprehensive benchmark comparing HOPF_LENS_DC against a LangGraph-style baseline.

## Quick Start

```bash
# Install dependencies
make install-bench

# Set up API key
cp ../.env.template ../.env
# Edit .env and add your OPENAI_API_KEY

# Run quick test (10 tasks)
make bench-quick

# Run full benchmark (150 tasks)
make bench
```

## Structure

```
bench/
├── run.py                     # Main benchmark orchestrator
├── generate_tasks.py          # Task dataset generator
├── tasks.jsonl                # 150 benchmark tasks (generated)
├── requirements.txt           # Pinned dependencies
├── tools/
│   └── __init__.py           # Tool implementations (eval_math, web_search, crud_tool)
├── runners/
│   ├── __init__.py           # Shared types
│   ├── langgraph_baseline.py  # LangGraph-style baseline
│   ├── hopf_full.py          # Full HOPF_LENS_DC
│   ├── hopf_no_typechecks.py # HOPF w/o type checking
│   └── hopf_no_synthesis.py  # HOPF w/o synthesis
├── results/                   # Generated metrics
│   ├── metrics.json          # Aggregate statistics
│   └── results.csv           # Per-task results
└── artifacts/                 # Generated traces
    ├── langgraph_traces.jsonl
    ├── hopf_full_traces.jsonl
    ├── hopf_no_typechecks_traces.jsonl
    └── hopf_no_synthesis_traces.jsonl
```

## Metrics

The benchmark computes five key metrics:

1. **Success Rate:** Percentage of tasks with correct output
2. **Tool Validity Rate:** Percentage of tool calls with valid arguments
3. **Average Iterations:** Average iterations until convergence
4. **Average Latency:** Average wall-clock time per task (ms)
5. **Total Cost:** Total estimated API cost ($)

## Configurations

1. **LangGraph Baseline:** Simple ReAct loop with no type checking
2. **HOPF_LENS_DC Full:** Complete categorical framework
3. **HOPF w/o Type Checks:** Ablation with validation disabled
4. **HOPF w/o Synthesis:** Ablation with Kan synthesis disabled

## Task Dataset

150 tasks across three categories:

- **Math (50):** Arithmetic expressions
- **Search (50):** Web search queries
- **CRUD (50):** Database operations

Each task has:
- Deterministic ground truth
- 1-3 expected tool calls
- 30-second timeout

## Usage

### Run All Configurations

```bash
python run.py --api-key $OPENAI_API_KEY
```

### Run Specific Configuration

```bash
# Just LangGraph
python run.py --api-key $OPENAI_API_KEY --runners langgraph

# Just HOPF full
python run.py --api-key $OPENAI_API_KEY --runners hopf_full

# Multiple configurations
python run.py --api-key $OPENAI_API_KEY --runners langgraph,hopf_full
```

### Custom Parameters

```bash
python run.py \
  --api-key $OPENAI_API_KEY \
  --model gpt-4o \
  --temperature 0.2 \
  --max-tool-calls 6 \
  --timeout 30 \
  --seed 42 \
  --num-tasks 10  # Run subset
```

## Regenerating Tasks

```bash
python generate_tasks.py
```

This creates a fresh `tasks.jsonl` with 150 tasks using seed 42.

## Results Analysis

After running the benchmark:

```bash
# View aggregate metrics
cat results/metrics.json | python -m json.tool

# View per-task results
head -20 results/results.csv

# View failure traces
grep '"success": false' artifacts/langgraph_traces.jsonl | head -5
```

## Cost Estimation

- **Quick test (10 tasks):** ~$0.30-0.60
- **Full benchmark (150 tasks):** ~$5-10

Costs vary based on:
- Model used (gpt-4o is expensive)
- Number of iterations per task
- Token counts in responses

## Win Criteria

HOPF_LENS_DC "wins" if:
- ≥5 pp higher success rate, OR
- ≥20 pp fewer invalid tool calls

AND:
- Latency within ±5%
- Cost within ±5%

## Security

 **Never commit `.env` files or API keys!**

All sensitive data must be in environment variables.
