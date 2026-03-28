"""Tests for DeltaPrune using a fake LLM."""

from __future__ import annotations

import json

from delta_prune import DeltaPrune, Claim, Conflict
from delta_prune.llm import LLM


class FakeLLM:
    """Keyword-based fake LLM for testing."""

    def generate(self, prompt: str) -> str | None:
        # Extractor responses
        if "情報の分類者" in prompt:
            if "カレー" in prompt and "嫌い" not in prompt:
                return json.dumps([{"claim": "好きな食べ物はカレー", "type": "preference", "reasoning": "食の好み"}])
            if "ラーメン" in prompt or "カレーは嫌い" in prompt:
                return json.dumps([{"claim": "好きな食べ物はラーメン、カレーは嫌い", "type": "preference", "reasoning": "食の好み変化"}])
            if "渋谷区" in prompt:
                return json.dumps([{"claim": "東京都渋谷区に住んでいる", "type": "fact", "reasoning": "住所"}])
            if "おはよう" in prompt:
                return json.dumps([])
            return json.dumps([])

        # Resolver responses
        if "整合性チェッカー" in prompt:
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

        return None


def test_no_conflict():
    """Messages without contradictions pass through unchanged."""
    llm = FakeLLM()
    prune = DeltaPrune(llm=llm)

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
    prune = DeltaPrune(llm=llm, strategy="report")

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
    prune = DeltaPrune(llm=llm, strategy="prune")

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
    prune = DeltaPrune(llm=llm, strategy="annotate")

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
    prune = DeltaPrune(llm=llm)

    messages = [
        {"role": "user", "content": "おはよう"},
        {"role": "assistant", "content": "おはようございます！"},
    ]

    result = prune(messages)
    assert not result.has_conflicts
    assert len(result.claims) == 0
