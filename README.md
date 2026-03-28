# delta-prune

LLM context contradiction detector and pruner.

**Problem**: When LLMs accumulate contradictory information in their context, reasoning accuracy collapses (GPT-4o-mini: 100% → 10%, Gemini: 100% → 0%). This isn't a context length problem — it's a contradiction problem. Making the window bigger doesn't help.

**Solution**: Scan the conversation for contradictions before sending to the LLM. Remove or annotate them. Three lines of code.

## Usage

```python
from delta_prune import DeltaPrune
from delta_prune.llm import ClaudeCLI

prune = DeltaPrune(llm=ClaudeCLI())

messages = [
    {"role": "user", "content": "好きな食べ物はカレーです"},
    {"role": "assistant", "content": "カレーですね！"},
    {"role": "user", "content": "カレーは嫌いです。ラーメンが好き"},
    {"role": "assistant", "content": "ラーメンですね！"},
    {"role": "user", "content": "私の好きな食べ物は？"},
]

result = prune(messages)
clean_messages = result.messages  # contradictions resolved
print(f"delta = {result.delta}")  # contradiction density
print(f"conflicts = {len(result.conflicts)}")
```

## Strategies

| Strategy | Behavior |
|----------|----------|
| `"annotate"` (default) | Add a system message listing detected contradictions |
| `"prune"` | Remove messages containing the older side of contradictions |
| `"report"` | Detect only, return original messages unchanged |

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

## How it works

```
messages (raw conversation)
    ↓
① Extract: pull factual claims from each message
    → [("likes curry", turn=0), ("hates curry, likes ramen", turn=2)]
    ↓
② Detect: pairwise contradiction check
    → [("curry:like" ⇔ "curry:hate", conflict)]
    ↓
③ Prune/Annotate: resolve based on recency
    → remove turn=0, or add system annotation
    ↓
④ Return: clean messages + delta score
```

## Delta score

`result.delta` = contradiction density (conflicts / claims). 0.0 = clean, higher = more contradictions. Based on the [DeltaZero](https://github.com/karesansui-u/delta-zero) survival equation research.

## Background

Based on research showing that **context rot is caused by contradiction accumulation, not context length**. Tested across 8 LLM models with statistically significant results (p=0.01). See [DeltaZero](https://github.com/karesansui-u/delta-zero) for the full research.
