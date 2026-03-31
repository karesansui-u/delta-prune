# delta-prune contradiction benchmark (RAG-style)

**PyPI note:** a normal `pip install delta-prune` installs only the **wheel** (library code). The **`benchmarks/`** directory ships in the **source distribution (sdist)** on PyPI and in git — see the main [README](../README.md) section *PyPI: wheel vs source distribution*.

Measures **answer accuracy** when retrieved chunks contain a **leading misleading** passage plus **correct** passages, comparing:

1. **Baseline** — all chunks concatenated (contradictions present).
2. **After `filter_chunks`** — same chunks passed through `DeltaPrune` (default: `strategy=prune`).

## Why chunk order matters (mechanism)

`filter_chunks` with `prune` removes the **older** (lower index) side of each detected conflict. Tasks in `data/tasks.json` are authored so the **first** chunk is misleading and **later** chunks are correct; when the conflict detector fires, pruning drops the bad lead and keeps the good evidence.

This is **not** claiming that “first in the list” equals “wrong in production RAG.” It is an **alignment trick** between the benchmark author and the current prune rule.

## Assumptions & papers (external validity)

Before using these metrics in a **paper or product comparison**, state explicitly:

1. **Index ≠ trust in real RAG.** Retrievers rank by similarity, not chronology or authority. A false chunk may appear **last** or in the **middle**; then `prune` may remove a **true** early chunk if the model judges it “older” relative to a later lie, or fail to help if ordering does not match the tool’s tie-break rule.
2. **What this benchmark measures:** under a **fixed, disclosed chunk order** (misleading first), does `filter_chunks` improve gold-match accuracy versus passing the same chunks unpruned? That is a **stress test of the middleware in a favorable ordering**, not a claim about arbitrary retrieval order.
3. **Index-independent evaluation (future work):** for stronger claims, add conditions where misleading chunks appear at **random positions**, use **`annotate`** instead of `prune`, add **recency / source-priority** metadata to the pruner, or re-rank after contradiction detection.

If the paper’s methods section states (2) clearly, reviewers can place the result correctly. See also `run_benchmark.py` JSON `meta.chunk_order_note`.

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
