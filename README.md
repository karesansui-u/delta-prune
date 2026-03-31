# delta-prune

LLM context contradiction detector and pruner.

**Problem**: When LLMs accumulate contradictory information in their context, reasoning accuracy collapses (GPT-4o-mini: 100% → 10%, Gemini: 100% → 0%). This isn't a context length problem — it's a contradiction problem. Making the window bigger doesn't help.

**Solution**: Scan **(1) chat messages** or **(2) RAG retrieval chunks** for contradictions *before* you call the downstream LLM. Remove or annotate them. Same strategies (`prune` / `annotate` / `report`) for both.

### When to use which API

| Use case | API | Input |
|----------|-----|--------|
| Agents, chatbots, multi-turn dialogue | `prune(messages)` | `[{"role":"user","content":"..."}, ...]` |
| RAG, enterprise search, doc Q&A | `prune.filter_chunks(chunks)` | `list[str]` (one string per retrieved chunk) |

**v0.3.0**: `filter_chunks()` and `ChunkPruneResult` — RAG path shares the same conflict detector and optional embedding pre-filter as chat mode.

**Paper**: ["Cognitive Sleep for LLMs: How Contradiction Metabolism Prevents Context Rot"](https://doi.org/10.5281/zenodo.19322371)

## Install

```bash
pip install delta-prune                  # core (zero dependencies)
pip install delta-prune[fast]            # with embedding pre-filter (recommended for long contexts)
pip install delta-prune[openai]        # with OpenAI backend
pip install delta-prune[ollama]          # with Ollama Python client (local models)
```

## Quick Start

```python
from delta_prune import DeltaPrune
from delta_prune.llm import ClaudeCLI

prune = DeltaPrune(llm=ClaudeCLI())

messages = [
    {"role": "user", "content": "My favorite food is curry."},
    {"role": "assistant", "content": "Curry, nice!"},
    {"role": "user", "content": "I hate curry. I like ramen."},
    {"role": "assistant", "content": "Ramen it is!"},
    {"role": "user", "content": "What's my favorite food?"},
]

result = prune(messages)
clean_messages = result.messages  # contradictions annotated
print(f"delta = {result.delta}")  # contradiction density
print(f"conflicts = {len(result.conflicts)}")
```

## RAG: retrieved chunks

Use the same `DeltaPrune` instance and strategy for **retrieval chunks** (plain strings). Each non-empty chunk is treated as one factual unit — **no** per-chunk claim-extraction LLM (unlike chat mode). Pairwise checks use the same conflict detector; pass `embedding=` and `max_llm_pairs` when you retrieve many chunks.

```python
from delta_prune import DeltaPrune, ChunkPruneResult
from delta_prune.llm import ClaudeCLI

prune = DeltaPrune(llm=ClaudeCLI(), strategy="prune", locale="en")

chunks = [
    "Product ships in 3–5 business days.",
    "All orders arrive next day guaranteed.",  # may contradict previous chunk
]
result = prune.filter_chunks(chunks)
assert isinstance(result, ChunkPruneResult)

context = "\n\n".join(result.filtered_chunks)  # feed to your answer-generation LLM
# result.delta, result.conflicts, result.has_conflicts
```

| Strategy | `filter_chunks` behavior |
|----------|--------------------------|
| `"annotate"` (default) | Prepend one chunk that lists contradictions; then original chunks |
| `"prune"` | Drop older conflicting chunks; order preserved for the rest |
| `"report"` | Return the input list unchanged (detect only) |

## Strategies

| Strategy | Behavior |
|----------|----------|
| `"annotate"` (default) | Add a system message listing detected contradictions |
| `"prune"` | Remove messages containing the older side of contradictions |
| `"report"` | Detect only, return original messages unchanged |

```python
prune = DeltaPrune(llm=llm, strategy="prune")     # remove old contradictions
prune = DeltaPrune(llm=llm, strategy="annotate")   # add context annotation
prune = DeltaPrune(llm=llm, strategy="report")     # detect only, no changes
```

## LLM Backends

```python
from delta_prune.llm import ClaudeCLI, OllamaLLM, OpenAILLM

# Claude CLI (subscription, $0)
prune = DeltaPrune(llm=ClaudeCLI(model="sonnet"))

# Local Ollama
prune = DeltaPrune(llm=OllamaLLM(model="gemma3:27b"))

# OpenAI API
prune = DeltaPrune(llm=OpenAILLM(model="gpt-4o-mini"))
```

## Performance: Embedding Pre-Filter

For long **chats** or many **RAG chunks**, use an embedding to avoid O(n²) LLM calls (same `DeltaPrune` constructor applies to both `__call__` and `filter_chunks`):

```python
from delta_prune.embedding import SentenceTransformerEmbedding

prune = DeltaPrune(
    llm=ClaudeCLI(),
    embedding=SentenceTransformerEmbedding(),  # local, free
    similarity_threshold=0.7,                  # only check similar pairs
    max_llm_pairs=30,                          # hard cap on LLM calls
)
```

| Mode | 20 claims | 50 claims | 100 claims |
|------|-----------|-----------|------------|
| No embedding (O(n²)) | 190 LLM calls | 1,225 calls | 4,950 calls |
| With embedding | ~10-30 LLM calls | ~10-30 calls | ~10-30 calls |

Install with: `pip install delta-prune[fast]`

## Language

English is the default. Japanese is available:

```python
prune = DeltaPrune(llm=llm, locale="ja")  # Japanese prompts
```

## How It Works

**Chat** (`prune(messages)`):

```
messages (raw conversation)
    ↓
① Extract: pull factual claims from each message (LLM)
    → [("likes curry", turn=0), ("hates curry, likes ramen", turn=2)]
    ↓
② Pre-filter (optional): embedding similarity → keep only similar pairs
    ↓
③ Detect: LLM checks filtered pairs for contradictions
    ↓
④ Resolve: prune / annotate / report → PruneResult
```

**RAG** (`filter_chunks(chunks)`):

```
list[str] (retrieved chunks)
    ↓
① Each non-empty chunk → one claim (no extraction LLM)
    ↓
② Pre-filter (optional): same as chat
    ↓
③ Detect / ④ Resolve → ChunkPruneResult.filtered_chunks
```

## Delta Score

`result.delta` = contradiction density (conflicts / claims). 0.0 = clean, higher = more contradictions.

Based on the [survival equation](https://doi.org/10.5281/zenodo.19322371): S = μ × e^(-δ×k). Reducing δ has an exponential effect on reasoning quality.

## Background

Based on research showing that **context rot is caused by contradiction accumulation, not context length**. Tested across 8 LLM models with statistically significant results (Kruskal-Wallis p=0.027, complete rank separation). See [DeltaZero](https://github.com/karesansui-u/delta-zero) for the full research.

## Benchmark (contradiction RAG)

Reproducible **baseline vs `filter_chunks`** accuracy on a small English task set (misleading chunk first). See [`benchmarks/README.md`](benchmarks/README.md).

**From a git checkout** (recommended):

```bash
pip install -e ".[ollama]"
python benchmarks/run_benchmark.py --backend ollama --model qwen2.5:14b --out results/run.json
```

### PyPI: wheel vs source distribution (sdist)

| Artifact | What you get |
|----------|----------------|
| **Wheel** (default `pip install delta-prune`) | Only the importable package under `site-packages`. **No `benchmarks/` tree** — you cannot run `benchmarks/run_benchmark.py` from that install alone. |
| **sdist** (`.tar.gz` on [PyPI](https://pypi.org/project/delta-prune/#files)) | Full project source as shipped by the release: **`benchmarks/`**, **`tests/`**, docs, and `pyproject.toml`. Use this if you want the harness without cloning Git. |

**Run the benchmark from a downloaded sdist** (example):

```bash
pip download delta-prune --no-binary delta-prune -d /tmp/dp
tar -xzf /tmp/dp/delta_prune-*.tar.gz -C /tmp/dp
cd /tmp/dp/delta_prune-*
pip install -e ".[ollama]"
python benchmarks/run_benchmark.py --backend ollama --model qwen2.5:14b
```

Alternatively clone [the repository](https://github.com/karesansui-u/delta-prune); behavior is the same as extracting the sdist.

## Publishing (maintainers)

PyPI uploads run on **GitHub Actions** using the repo secret **`__TOKEN__`** (PyPI API token). See [`docs/RELEASE.md`](docs/RELEASE.md) for steps. **Coding agents:** see [`AGENTS.md`](AGENTS.md) for the same facts in agent-oriented form.

## Changelog

### 0.3.0

- **RAG API**: `DeltaPrune.filter_chunks(chunks) -> ChunkPruneResult` with `filtered_chunks`, `delta`, `conflicts` (same strategies and optional embedding pre-filter as chat mode).
- README: use-case table (chat vs RAG), install extra `delta-prune[ollama]`.

### 0.2.x

- Chat-only pipeline: claim extraction, conflict detection, prune / annotate / report; EN/JA prompts; optional embedding pre-filter.
