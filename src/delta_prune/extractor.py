"""Extract factual claims from chat messages.

Simplified from DeltaZero's EnhancedExtractor — no persistent storage,
operates on transient message lists.
"""

from __future__ import annotations

from dataclasses import dataclass

from delta_prune.llm import LLM
from delta_prune.llm_parser import parse_json_array
from delta_prune.prompts import get_prompt


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
    locale: str = "en",
) -> list[Claim]:
    """Extract all factual claims from a list of messages.

    Args:
        messages: OpenAI-format messages [{"role": "user", "content": "..."}]
        llm: LLM adapter for extraction
        roles: which roles to extract from (default: user only)
        locale: "en" or "ja"

    Returns:
        List of Claims with source_turn metadata
    """
    claims: list[Claim] = []
    prompt_template = get_prompt(locale, "extract")

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role not in roles or not content.strip():
            continue

        prompt = prompt_template.format(text=content)
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
