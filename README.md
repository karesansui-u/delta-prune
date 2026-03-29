# delta-prune

LLM context contradiction detector and pruner.

**Problem**: When LLMs accumulate contradictory information in their context, reasoning accuracy collapses (GPT-4o-mini: 100% → 10%, Gemini: 100% → 0%). This isn't a context length problem — it's a contradiction problem. Making the window bigger doesn't help.

**Solution**: Scan the conversation for contradictions before sending to the LLM. Remove or annotate them. Three lines of code.

**Paper**: ["Cognitive Sleep for LLMs: How Contradiction Metabolism Prevents Context Rot"](https://doi.org/10.5281/zenodo.19322371)

## Install

```bash
pip install delta-prune                  # core (zero dependencies)
pip install delta-prune[fast]            # with embedding pre-filter (recommended)
pip install delta-prune[openai]          # with OpenAI backend
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

For conversations with many messages, use an embedding to avoid O(n²) LLM calls:

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

```
messages (raw conversation)
    ↓
① Extract: pull factual claims from each message (LLM)
    → [("likes curry", turn=0), ("hates curry, likes ramen", turn=2)]
    ↓
② Pre-filter (optional): embedding similarity → keep only similar pairs
    → [("curry:like" ↔ "curry:hate")] — skip unrelated pairs
    ↓
③ Detect: LLM checks filtered pairs for contradictions
    → [conflict: "food preference changed"]
    ↓
④ Resolve: prune / annotate / report
    ↓
⑤ Return: clean messages + delta score + conflict details
```

## Delta Score

`result.delta` = contradiction density (conflicts / claims). 0.0 = clean, higher = more contradictions.

Based on the [survival equation](https://doi.org/10.5281/zenodo.19322371): S = μ × e^(-δ×k). Reducing δ has an exponential effect on reasoning quality.

## Background

Based on research showing that **context rot is caused by contradiction accumulation, not context length**. Tested across 8 LLM models with statistically significant results (Kruskal-Wallis p=0.027, complete rank separation). See [DeltaZero](https://github.com/karesansui-u/delta-zero) for the full research.
