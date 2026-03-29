# delta-prune Design Document

## What This Is

delta-prune is a lightweight middleware that detects and resolves contradictions in LLM conversation context before sending to any API. Based on [DeltaZero](https://github.com/karesansui-u/delta-zero) research showing that context rot is caused by contradiction accumulation (p<0.001, d=8.80).

**Paper**: ["Cognitive Sleep for LLMs"](https://doi.org/10.5281/zenodo.19322371)

## Current State (v0.1.0)

### What works
- `DeltaPrune(llm=...) → result.messages` — 3-line API
- 3 strategies: annotate (add system message), prune (remove old), report (detect only)
- 3 LLM backends: ClaudeCLI, OllamaLLM, OpenAILLM
- 6 unit tests (FakeLLM), all passing
- Apache 2.0 license (patent defense clause)

### Architecture
```
src/delta_prune/
  __init__.py      # Public API: DeltaPrune, PruneResult, Claim, Conflict
  pruner.py        # DeltaPrune class — orchestrates extract → detect → resolve
  extractor.py     # Extract factual claims from messages via LLM
  resolver.py      # Detect contradictions between claim pairs via LLM
  llm.py           # LLM adapters (ClaudeCLI, OllamaLLM, OpenAILLM)
  llm_parser.py    # JSON extraction from LLM output

tests/
  test_pruner.py   # 6 tests using FakeLLM
```

### Data flow
```
messages: list[dict]          # OpenAI format [{"role": "user", "content": "..."}]
    ↓
extract_claims(messages, llm) # LLM call per message → list[Claim]
    ↓
detect_conflicts(claims, llm) # LLM call per pair → list[Conflict]  ← O(n²) PROBLEM
    ↓
apply_strategy(messages, conflicts)  # prune / annotate / report
    ↓
PruneResult(messages, conflicts, claims, delta)
```

---

## Problems to Fix (Priority Order)

### Problem 1: Prompts are Japanese-only
All LLM prompts in extractor.py and resolver.py are in Japanese. The paper and README are English. Users who `pip install delta-prune` will get Japanese prompts regardless of their language.

**Files to change**:
- `src/delta_prune/extractor.py` — EXTRACT_PROMPT (lines 14-40)
- `src/delta_prune/resolver.py` — CONFLICT_PROMPT (lines 14-46)

**Design decision**: Default to English. Add `locale` parameter to DeltaPrune constructor for Japanese support.

```python
# Target API
prune = DeltaPrune(llm=ClaudeCLI())              # English (default)
prune = DeltaPrune(llm=ClaudeCLI(), locale="ja")  # Japanese
```

**Implementation**:
- Create `prompts.py` with EXTRACT_PROMPT_EN, EXTRACT_PROMPT_JA, CONFLICT_PROMPT_EN, CONFLICT_PROMPT_JA
- Extractor and Resolver accept `locale` parameter
- DeltaPrune passes `locale` through

### Problem 2: O(n²) LLM calls for conflict detection
`detect_conflicts()` checks every pair of claims. 20 claims = 190 LLM calls. 50 claims = 1225 calls. Unusable for real conversations.

**Solution**: Embedding-based pre-filter. Only send semantically similar pairs to LLM.

```
Before (O(n²)):
  All claim pairs → LLM conflict check

After (O(n²) embedding + O(k) LLM):
  All claim pairs → cosine similarity → top-k most similar pairs → LLM conflict check
```

Embedding is ~1000x faster and ~1000x cheaper than LLM calls. With a similarity threshold, we can reduce LLM calls from n² to ~10-30 regardless of n.

**Files to change**:
- `src/delta_prune/resolver.py` — add embedding pre-filter
- `src/delta_prune/embedding.py` — new file, embedding adapters

**Embedding options** (in order of preference):
1. `sentence-transformers` (local, free, ~100ms per batch)
2. OpenAI embeddings API (fast, cheap but not free)
3. No embedding (fallback to O(n²), for users who don't want the dependency)

**Design**:
```python
class DeltaPrune:
    def __init__(
        self,
        llm: LLM,
        embedding: Embedding | None = None,  # Optional: enables fast pre-filter
        similarity_threshold: float = 0.7,   # Only LLM-check pairs above this
        max_llm_pairs: int = 30,             # Hard cap on LLM calls
        ...
    )
```

If `embedding` is None, fall back to O(n²) (small conversations) or raise a warning (large conversations).

**Implementation steps**:
1. Create `embedding.py` with `Embedding` protocol + `SentenceTransformerEmbedding` adapter
2. In `resolver.py`, add `filter_candidate_pairs(claims, embedding, threshold)` function
3. In `pruner.py`, pass embedding to resolver
4. Add `sentence-transformers` as optional dependency: `pip install delta-prune[fast]`

### Problem 3: Not on PyPI
Users can't `pip install delta-prune`. The paper links to GitHub but installation requires cloning.

**Steps**:
1. Verify `pyproject.toml` is correct (already done)
2. `python -m build` → creates wheel
3. `twine upload dist/*` → publishes to PyPI

**Pre-publish checklist**:
- [ ] Problem 1 (English prompts) is fixed
- [ ] Problem 2 (O(n²)) is fixed
- [ ] All tests pass
- [ ] README has install instructions and working example
- [ ] Version is 0.2.0 (significant changes from 0.1.0)

---

## Implementation Plan

### Phase 1: English prompts (30 min)
1. Create `src/delta_prune/prompts.py`
   - EXTRACT_PROMPT_EN, EXTRACT_PROMPT_JA
   - CONFLICT_PROMPT_EN, CONFLICT_PROMPT_JA
   - `get_extract_prompt(locale)`, `get_conflict_prompt(locale)`
2. Update `extractor.py`: accept `locale`, use prompts.py
3. Update `resolver.py`: accept `locale`, use prompts.py
4. Update `pruner.py`: accept `locale` in constructor, pass through
5. Update `__init__.py` if new public types
6. Add tests for English prompts
7. Update FakeLLM to handle English keywords

### Phase 2: Embedding pre-filter (1-2 hours)
1. Create `src/delta_prune/embedding.py`
   - `Embedding` protocol: `encode(texts: list[str]) -> list[list[float]]`
   - `SentenceTransformerEmbedding(model="all-MiniLM-L6-v2")`
   - `cosine_similarity(a, b) -> float`
2. Update `resolver.py`
   - Add `filter_candidate_pairs(claims, embedding, threshold, max_pairs)` function
   - `detect_conflicts()` accepts optional `embedding` parameter
   - If embedding provided: pre-filter, then LLM on filtered pairs only
   - If not provided: O(n²) fallback with warning for large n
3. Update `pruner.py`: pass embedding through
4. Update `pyproject.toml`: add `[project.optional-dependencies] fast = ["sentence-transformers"]`
5. Add tests:
   - Test with FakeEmbedding (returns predictable vectors)
   - Test that pre-filter reduces pair count
   - Test that O(n²) fallback still works
6. Update README with `pip install delta-prune[fast]`

### Phase 3: PyPI publish (30 min)
1. Bump version to 0.2.0 in pyproject.toml
2. Update README:
   - Installation: `pip install delta-prune` or `pip install delta-prune[fast]`
   - Quick start example (English)
   - Strategy comparison table
   - Performance note (with/without embedding)
3. `python -m build`
4. `twine upload dist/*`
5. Verify: `pip install delta-prune && python -c "from delta_prune import DeltaPrune; print('OK')"`

---

## File-by-File Change Summary

| File | Phase | Change |
|------|-------|--------|
| `src/delta_prune/prompts.py` | 1 | **NEW** — English + Japanese prompts |
| `src/delta_prune/extractor.py` | 1 | Use prompts.py, accept locale |
| `src/delta_prune/resolver.py` | 1+2 | Use prompts.py, accept locale, add embedding pre-filter |
| `src/delta_prune/pruner.py` | 1+2 | Accept locale + embedding in constructor |
| `src/delta_prune/embedding.py` | 2 | **NEW** — Embedding protocol + SentenceTransformer adapter |
| `src/delta_prune/__init__.py` | 1+2 | Export new types if needed |
| `src/delta_prune/llm.py` | — | No changes |
| `src/delta_prune/llm_parser.py` | — | No changes |
| `tests/test_pruner.py` | 1+2 | Add English prompt tests + embedding filter tests |
| `pyproject.toml` | 2+3 | Add optional deps, bump version |
| `README.md` | 3 | Update install + usage |

---

## API After Changes (v0.2.0)

```python
from delta_prune import DeltaPrune
from delta_prune.llm import ClaudeCLI, OpenAILLM
from delta_prune.embedding import SentenceTransformerEmbedding

# Minimal (English, no embedding, O(n²) for small conversations)
prune = DeltaPrune(llm=ClaudeCLI())
result = prune(messages)

# With embedding pre-filter (recommended for conversations > 20 messages)
prune = DeltaPrune(
    llm=OpenAILLM(model="gpt-4o-mini"),
    embedding=SentenceTransformerEmbedding(),  # local, free
    similarity_threshold=0.7,
    max_llm_pairs=30,
)
result = prune(messages)

# Japanese
prune = DeltaPrune(llm=ClaudeCLI(), locale="ja")

# Access results
result.messages      # cleaned messages
result.conflicts     # list of detected contradictions
result.claims        # list of extracted claims
result.delta         # contradiction density (0.0 = clean)
result.has_conflicts # bool
```

---

## Testing Strategy

| Test | Type | What it verifies |
|------|------|-----------------|
| test_no_conflict | Unit (FakeLLM) | Clean messages pass through |
| test_detect_contradiction | Unit (FakeLLM) | Contradictions are found |
| test_prune_strategy | Unit (FakeLLM) | Old messages removed |
| test_annotate_strategy | Unit (FakeLLM) | System annotation added |
| test_empty_messages | Unit (FakeLLM) | Edge case |
| test_greeting_only | Unit (FakeLLM) | No claims extracted |
| test_english_prompts | Unit (FakeLLM) | English locale works | **NEW** |
| test_japanese_prompts | Unit (FakeLLM) | Japanese locale works | **NEW** |
| test_embedding_filter | Unit (FakeEmb) | Pre-filter reduces pairs | **NEW** |
| test_embedding_fallback | Unit (FakeLLM) | No embedding → O(n²) works | **NEW** |
| test_max_llm_pairs | Unit (FakeEmb) | Hard cap is respected | **NEW** |

---

## Relationship to delta-zero

delta-prune is **not** a subset of delta-zero. They solve different problems at different layers:

| | delta-prune | delta-zero |
|---|---|---|
| When | Pre-send (before LLM call) | Idle time (between conversations) |
| State | Stateless (processes message list) | Stateful (4-layer memory + SQLite + ChromaDB) |
| Scope | Single conversation | Persistent knowledge across sessions |
| Dependency | LLM adapter only | Full stack (SQLite, ChromaDB, scheduler) |
| User | Any LLM API developer | DeltaZero deployment |

Future integration path: delta-prune could use delta-zero's Resolver as a backend, but this is not planned for v0.2.0.

---

## How to Continue This Work

If you are an LLM picking up this task:

1. Read this DESIGN.md first
2. Read the current code: `src/delta_prune/*.py` (511 lines total)
3. Run tests: `cd /Users/sunagawa/Project/delta-prune && python -m pytest tests/ -v`
4. Implement Phase 1, 2, 3 in order
5. Each phase should end with all tests passing
6. Do not change the public API signature of `DeltaPrune.__call__` — it must stay `(messages) -> PruneResult`
7. Do not add dependencies that aren't optional (core must work with just the stdlib + an LLM adapter)

### Key constraints
- Core package has ZERO required dependencies (LLM adapters import lazily)
- `sentence-transformers` is optional (`[fast]` extra)
- English is default, Japanese is opt-in
- All prompts go in prompts.py, nowhere else
- Tests must work without any LLM or embedding model (FakeLLM/FakeEmbedding only)
