"""Extract factual claims from chat messages.

Simplified from DeltaZero's EnhancedExtractor — no persistent storage,
operates on transient message lists.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta_prune.llm import LLM
from delta_prune.llm_parser import parse_json_array

EXTRACT_PROMPT = """あなたは情報の分類者です。
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


@dataclass
class Claim:
    """A single factual claim extracted from a message."""

    text: str
    type: str  # "fact", "rule", "preference"
    source_turn: int  # message index in the conversation
    role: str  # "user" or "assistant"


def extract_claims(
    messages: list[dict[str, str]],
    llm: LLM,
    roles: tuple[str, ...] = ("user",),
) -> list[Claim]:
    """Extract all factual claims from a list of messages.

    Args:
        messages: OpenAI-format messages [{"role": "user", "content": "..."}]
        llm: LLM adapter for extraction
        roles: which roles to extract from (default: user only)

    Returns:
        List of Claims with source_turn metadata
    """
    claims: list[Claim] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role not in roles or not content.strip():
            continue

        prompt = EXTRACT_PROMPT.format(text=content)
        output = llm.generate(prompt)
        if not output:
            continue

        parsed = parse_json_array(output)
        for item in parsed:
            claim_text = item.get("claim", "")
            claim_type = item.get("type", "fact")
            if claim_text:
                claims.append(Claim(
                    text=claim_text,
                    type=claim_type,
                    source_turn=i,
                    role=role,
                ))

    return claims
