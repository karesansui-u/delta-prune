"""Prompt templates for extraction and conflict detection.

All prompts live here. Extractor and resolver import from this module.
"""

from __future__ import annotations

EXTRACT_PROMPT_EN = """You are an information classifier.

Extract factual claims, rules, preferences, and attributes from the following message.
Return ONLY a JSON array. Extract each fact individually.

[Message]
{text}

[
  {{
    "claim": "extracted fact or rule (one sentence)",
    "type": "fact | rule | preference",
    "reasoning": "why this is important (one sentence)"
  }}
]

[Rules]
- Do not combine multiple facts into one claim
- Do not extract greetings, small talk, or meta-information (return empty array [])
- If the message contains no extractable information, return empty array []
"""

EXTRACT_PROMPT_JA = """あなたは情報の分類者です。
必ず日本語で回答してください。

以下のメッセージから、事実・ルール・好み・属性を **1つずつ個別に** 抽出してください。
JSON配列のみで回答してください。

【メッセージ】
{text}

[
  {{
    "claim": "抽出した事実やルール（1文）",
    "type": "fact | rule | preference",
    "reasoning": "なぜ重要か（1文）"
  }}
]

【ルール】
- 1つのclaimに複数の事実を含めないこと
- 挨拶・雑談・メタ情報は抽出しない（空配列 [] を返す）
- メッセージに抽出すべき情報がなければ空配列 [] を返す
"""

CONFLICT_PROMPT_EN = """You are a knowledge consistency checker.

Determine whether the following two claims contradict each other.

[Claim A] (message {turn_a})
{claim_a}

[Claim B] (message {turn_b})
{claim_b}

Respond ONLY in the following JSON format.

{{
  "has_conflict": true | false,
  "conflict_type": "direct_contradiction | temporal_change | scope_difference | no_conflict",
  "summary": "one-sentence summary of the conflict. null if no conflict"
}}

[Criteria]
- direct_contradiction: incompatible claims about the same subject (e.g., likes vs dislikes)
- temporal_change: same subject's state changed over time (e.g., single → married)
- scope_difference: different claims under different conditions (not a contradiction)
- no_conflict: different topics or compatible claims
- If topics are unrelated, always return "no_conflict"
"""

CONFLICT_PROMPT_JA = """あなたは知識の整合性チェッカーです。
必ず日本語で回答してください。

以下の2つの主張が矛盾しているか判定してください。

【主張 A】(発言 {turn_a})
{claim_a}

【主張 B】(発言 {turn_b})
{claim_b}

以下のJSON形式のみで回答してください。

{{
  "has_conflict": true | false,
  "conflict_type": "直接矛盾 | 時間的変化 | 条件違い | 矛盾なし",
  "summary": "矛盾の要約（1文）。矛盾なしの場合は null"
}}

【判定基準】
- 直接矛盾: 同じ対象について、両立しない主張（例: 好き vs 嫌い、12人 vs 5000人）
- 時間的変化: 同じ対象の状態が時間で変化（例: 独身 → 既婚）
- 条件違い: 異なる条件下での異なる主張（矛盾ではない）
- 矛盾なし: 異なるトピック、または両立する主張
- テーマが無関係な場合は必ず「矛盾なし」とすること
"""

PROMPTS = {
    "en": {"extract": EXTRACT_PROMPT_EN, "conflict": CONFLICT_PROMPT_EN},
    "ja": {"extract": EXTRACT_PROMPT_JA, "conflict": CONFLICT_PROMPT_JA},
}


def get_prompt(locale: str, kind: str) -> str:
    """Get a prompt template by locale and kind.

    Args:
        locale: "en" or "ja"
        kind: "extract" or "conflict"

    Returns:
        Prompt template string with {text} or {turn_a}/{claim_a}/{turn_b}/{claim_b} placeholders
    """
    if locale not in PROMPTS:
        raise ValueError(f"Unsupported locale: {locale!r}. Supported: {list(PROMPTS.keys())}")
    if kind not in PROMPTS[locale]:
        raise ValueError(f"Unknown prompt kind: {kind!r}. Supported: {list(PROMPTS[locale].keys())}")
    return PROMPTS[locale][kind]
