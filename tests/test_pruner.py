"""Tests for DeltaPrune using fake LLM and embedding."""

from __future__ import annotations

import json

import pytest

from delta_prune import DeltaPrune, Claim
from delta_prune.resolver import filter_candidate_pairs


# ============================================================================
# Fakes
# ============================================================================

class FakeLLM:
    """Keyword-based fake LLM for testing. Handles both EN and JA prompts."""

    def generate(self, prompt: str) -> str | None:
        # Extractor responses (JA)
        if "情報の分類者" in prompt:
            return self._extract_ja(prompt)

        # Extractor responses (EN)
        if "information classifier" in prompt:
            return self._extract_en(prompt)

        # Resolver responses (JA)
        if "整合性チェッカー" in prompt:
            return self._resolve(prompt)

        # Resolver responses (EN)
        if "consistency checker" in prompt:
            return self._resolve_en(prompt)

        return None

    def _extract_ja(self, prompt: str) -> str:
        if "カレー" in prompt and "嫌い" not in prompt:
            return json.dumps([{"claim": "好きな食べ物はカレー", "type": "preference", "reasoning": "食の好み"}])
        if "ラーメン" in prompt or "カレーは嫌い" in prompt:
            return json.dumps([{"claim": "好きな食べ物はラーメン、カレーは嫌い", "type": "preference", "reasoning": "食の好み変化"}])
        if "渋谷区" in prompt:
            return json.dumps([{"claim": "東京都渋谷区に住んでいる", "type": "fact", "reasoning": "住所"}])
        if "おはよう" in prompt:
            return json.dumps([])
        return json.dumps([])

    def _extract_en(self, prompt: str) -> str:
        if "curry" in prompt.lower() and "hate" not in prompt.lower():
            return json.dumps([{"claim": "favorite food is curry", "type": "preference", "reasoning": "food preference"}])
        if "ramen" in prompt.lower() or "hate curry" in prompt.lower():
            return json.dumps([{"claim": "favorite food is ramen, hates curry", "type": "preference", "reasoning": "food preference changed"}])
        if "shibuya" in prompt.lower():
            return json.dumps([{"claim": "lives in Shibuya, Tokyo", "type": "fact", "reasoning": "address"}])
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return json.dumps([])
        return json.dumps([])

    def _resolve(self, prompt: str) -> str:
        if "カレー" in prompt and "ラーメン" in prompt:
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "直接矛盾",
                "summary": "好きな食べ物がカレーからラーメンに変化",
            })
        return json.dumps({
            "has_conflict": False,
            "conflict_type": "矛盾なし",
            "summary": None,
        })

    def _resolve_en(self, prompt: str) -> str:
        if "curry" in prompt.lower() and "ramen" in prompt.lower():
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "direct_contradiction",
                "summary": "food preference changed from curry to ramen",
            })
        return json.dumps({
            "has_conflict": False,
            "conflict_type": "no_conflict",
            "summary": None,
        })


class FakeEmbedding:
    """Fake embedding that returns predictable vectors based on keywords.

    Claims containing "food"/"curry"/"ramen" get similar vectors.
    Claims about "address"/"shibuya" get different vectors.
    """

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            if any(w in lower for w in ["food", "curry", "ramen", "カレー", "ラーメン", "食べ物"]):
                vectors.append([0.9, 0.1, 0.0])
            elif any(w in lower for w in ["address", "shibuya", "渋谷", "住"]):
                vectors.append([0.0, 0.1, 0.9])
            else:
                vectors.append([0.3, 0.3, 0.3])
        return vectors


# ============================================================================
# Existing tests (v0.1.0 — must still pass)
# ============================================================================

def test_no_conflict():
    """Messages without contradictions pass through unchanged."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, locale="ja")

    messages = [
        {"role": "user", "content": "私は東京都渋谷区に住んでいます。"},
        {"role": "assistant", "content": "渋谷区ですね！"},
    ]

    result = prune(messages)
    assert not result.has_conflicts
    assert result.delta == 0.0
    assert len(result.messages) == 2


def test_detect_contradiction():
    """Contradicting food preferences are detected."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="report", locale="ja")

    messages = [
        {"role": "user", "content": "好きな食べ物はカレーです。"},
        {"role": "assistant", "content": "カレーですね！"},
        {"role": "user", "content": "カレーは嫌いです。ラーメンが好きです。"},
        {"role": "assistant", "content": "ラーメンですね！"},
    ]

    result = prune(messages)
    assert result.has_conflicts
    assert len(result.conflicts) == 1
    assert result.conflicts[0].conflict_type == "直接矛盾"
    assert result.delta > 0


def test_prune_strategy():
    """Prune strategy removes the older contradicting message."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="prune", locale="ja")

    messages = [
        {"role": "user", "content": "好きな食べ物はカレーです。"},
        {"role": "assistant", "content": "カレーですね！"},
        {"role": "user", "content": "カレーは嫌いです。ラーメンが好きです。"},
        {"role": "assistant", "content": "ラーメンですね！"},
    ]

    result = prune(messages)
    assert result.has_conflicts
    # The older message (turn 0, "カレーです") should be removed
    assert len(result.messages) == 3
    assert "カレーは嫌い" in result.messages[1]["content"]


def test_annotate_strategy():
    """Annotate strategy adds a system message about contradictions."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="annotate", locale="ja")

    messages = [
        {"role": "user", "content": "好きな食べ物はカレーです。"},
        {"role": "assistant", "content": "カレーですね！"},
        {"role": "user", "content": "カレーは嫌いです。ラーメンが好きです。"},
    ]

    result = prune(messages)
    assert result.has_conflicts
    # Should have original 3 + 1 annotation = 4 messages
    assert len(result.messages) == 4
    system_msgs = [m for m in result.messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "矛盾" in system_msgs[0]["content"]


def test_empty_messages():
    """Empty message list returns empty result."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm)

    result = prune([])
    assert not result.has_conflicts
    assert result.delta == 0.0


def test_greeting_only():
    """Messages with only greetings produce no claims."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, locale="ja")

    messages = [
        {"role": "user", "content": "おはよう"},
        {"role": "assistant", "content": "おはようございます！"},
    ]

    result = prune(messages)
    assert not result.has_conflicts
    assert len(result.claims) == 0


# ============================================================================
# Phase 1 tests: English prompts
# ============================================================================

def test_english_prompts():
    """English locale detects contradictions correctly."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="report", locale="en")

    messages = [
        {"role": "user", "content": "My favorite food is curry."},
        {"role": "assistant", "content": "Curry, nice!"},
        {"role": "user", "content": "I hate curry. I like ramen."},
        {"role": "assistant", "content": "Ramen it is!"},
    ]

    result = prune(messages)
    assert result.has_conflicts
    assert len(result.conflicts) == 1
    assert result.conflicts[0].conflict_type == "direct_contradiction"


def test_english_annotate():
    """English annotate produces English system message."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="annotate", locale="en")

    messages = [
        {"role": "user", "content": "My favorite food is curry."},
        {"role": "assistant", "content": "Curry, nice!"},
        {"role": "user", "content": "I hate curry. I like ramen."},
    ]

    result = prune(messages)
    assert result.has_conflicts
    system_msgs = [m for m in result.messages if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert "contradiction" in system_msgs[0]["content"].lower()


# ============================================================================
# Phase 2 tests: Embedding pre-filter
# ============================================================================

def test_embedding_filter():
    """Embedding pre-filter only selects similar claim pairs."""
    claims = [
        Claim(text="favorite food is curry", type="preference", source_turn=0, role="user"),
        Claim(text="lives in Shibuya, Tokyo", type="fact", source_turn=1, role="user"),
        Claim(text="favorite food is ramen, hates curry", type="preference", source_turn=2, role="user"),
    ]

    embedding = FakeEmbedding()
    pairs = filter_candidate_pairs(claims, embedding, threshold=0.5, max_pairs=30)

    # curry ↔ ramen should be similar (both food vectors)
    # curry ↔ shibuya should NOT be similar
    assert (0, 2) in pairs
    assert (0, 1) not in pairs


def test_embedding_fallback():
    """Without embedding, O(n²) fallback works normally."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm, strategy="report", locale="en", embedding=None)

    messages = [
        {"role": "user", "content": "My favorite food is curry."},
        {"role": "assistant", "content": "Nice!"},
        {"role": "user", "content": "I hate curry. I like ramen."},
    ]

    result = prune(messages)
    assert result.has_conflicts


def test_max_llm_pairs():
    """max_llm_pairs caps the number of pairs sent to LLM."""
    claims = [
        Claim(text=f"food item {i}", type="preference", source_turn=i, role="user")
        for i in range(10)
    ]

    embedding = FakeEmbedding()  # All food → all similar
    pairs = filter_candidate_pairs(claims, embedding, threshold=0.5, max_pairs=5)

    assert len(pairs) <= 5


def test_embedding_with_prune():
    """Full pipeline with embedding pre-filter detects and prunes correctly."""
    llm = FakeLLM()
    embedding = FakeEmbedding()
    prune = DeltaPrune(
        llm=llm,
        strategy="report",
        locale="en",
        embedding=embedding,
        similarity_threshold=0.5,
    )

    messages = [
        {"role": "user", "content": "My favorite food is curry."},
        {"role": "assistant", "content": "Nice!"},
        {"role": "user", "content": "I live in Shibuya."},
        {"role": "assistant", "content": "Cool area!"},
        {"role": "user", "content": "I hate curry. I like ramen."},
    ]

    result = prune(messages)
    assert result.has_conflicts
    assert len(result.conflicts) == 1
    assert "curry" in result.conflicts[0].summary.lower() or "ramen" in result.conflicts[0].summary.lower()


# ============================================================================
# Integration tests (skipped in CI, run locally with: pytest -m integration)
# ============================================================================

@pytest.mark.integration
def test_integration_ollama():
    """Real LLM produces parseable output."""
    from delta_prune.llm import OllamaLLM

    llm = OllamaLLM(model="gemma3:12b")
    prune = DeltaPrune(llm=llm, strategy="report", locale="en")

    messages = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Blue is nice!"},
        {"role": "user", "content": "I hate blue. Red is my favorite color."},
    ]

    result = prune(messages)
    # Should detect at least one conflict
    assert result.has_conflicts
    assert len(result.claims) >= 2
