"""Detect contradictions between claims.

Simplified from DeltaZero's Resolver — no persistent storage,
operates on in-memory claim pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta_prune.extractor import Claim
from delta_prune.llm import LLM
from delta_prune.llm_parser import parse_json

CONFLICT_PROMPT = """あなたは知識の整合性チェッカーです。
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


@dataclass
class Conflict:
    """A detected contradiction between two claims."""

    claim_a: Claim
    claim_b: Claim
    conflict_type: str
    summary: str


def detect_conflicts(claims: list[Claim], llm: LLM) -> list[Conflict]:
    """Detect contradictions between all claim pairs.

    Uses a simple O(n²) pairwise comparison. For typical conversation
    lengths (<100 claims), this is fast enough.
    """
    conflicts: list[Conflict] = []

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]

            prompt = CONFLICT_PROMPT.format(
                turn_a=a.source_turn,
                claim_a=a.text,
                turn_b=b.source_turn,
                claim_b=b.text,
            )

            output = llm.generate(prompt)
            if not output:
                continue

            parsed = parse_json(output)
            if not parsed.get("has_conflict"):
                continue

            conflicts.append(Conflict(
                claim_a=a,
                claim_b=b,
                conflict_type=parsed.get("conflict_type", "直接矛盾"),
                summary=parsed.get("summary", ""),
            ))

    return conflicts
