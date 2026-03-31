# delta-prune contradiction benchmark (RAG-style)

**PyPI note:** a normal `pip install delta-prune` installs only the **wheel** (library code). The **`benchmarks/`** directory ships in the **source distribution (sdist)** on PyPI and in git — see the main [README](../README.md) section *PyPI: wheel vs source distribution*.

Measures **answer accuracy** when retrieved chunks contain a **leading misleading** passage plus **correct** passages, comparing:

1. **Baseline** — all chunks concatenated (contradictions present).
2. **After `filter_chunks`** — same chunks passed through `DeltaPrune` (default: `strategy=prune`).

## Why chunk order matters

`filter_chunks` with `prune` removes the **older** (lower index) side of each detected conflict. Tasks are authored so the **first** chunk is wrong and **later** chunks are right; pruning drops the bad lead when it contradicts the good evidence.

## Requirements

- Editable install from repo root: `pip install -e ".[ollama]"` (or `[openai]` / use `claude` backend).
- A running model (e.g. Ollama) matching `--model`.

## Run

From repository root:

```bash
pip install -e ".[ollama]"
python benchmarks/run_benchmark.py --backend ollama --model qwen2.5:14b --strategy prune
```

Full JSON artifact:

```bash
python benchmarks/run_benchmark.py --backend ollama --model qwen2.5:14b --out results/benchmark_run.json
```

Options: `--limit N`, `--strategy annotate|report`, `--locale ja`, `--data path/to/tasks.json`.

## Scoring

**Gold substring match** (normalized): the reference answer string must appear in the model response (case-insensitive, punctuation stripped). This is a cheap, reproducible proxy; swap for LLM-as-judge or RAGAS later for papers.

## Extending

- Add rows to `data/tasks.json` (same schema).
- For RAGAS / RGB-style integration, keep this harness as the **injection + prune** layer and plug their evaluators on top of `baseline_answer` / `pruned_answer`.
