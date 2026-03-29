"""Detect contradictions between claims.

Simplified from DeltaZero's Resolver — no persistent storage,
operates on in-memory claim pairs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from delta_prune.extractor import Claim
from delta_prune.llm import LLM
from delta_prune.llm_parser import parse_json
from delta_prune.prompts import get_prompt

if TYPE_CHECKING:
    from delta_prune.embedding import Embedding


@dataclass
class Conflict:
    """A detected contradiction between two claims."""

    claim_a: Claim
    claim_b: Claim
    conflict_type: str
    summary: str


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def filter_candidate_pairs(
    claims: list[Claim],
    embedding: Embedding,
    threshold: float = 0.7,
    max_pairs: int = 30,
) -> list[tuple[int, int]]:
    """Pre-filter claim pairs by embedding similarity.

    Returns indices of claim pairs that exceed the similarity threshold,
    sorted by similarity (highest first), capped at max_pairs.
    """
    texts = [c.text for c in claims]
    vectors = embedding.encode(texts)

    scored_pairs: list[tuple[float, int, int]] = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            sim = _cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                scored_pairs.append((sim, i, j))

    scored_pairs.sort(reverse=True)
    return [(i, j) for _, i, j in scored_pairs[:max_pairs]]


def detect_conflicts(
    claims: list[Claim],
    llm: LLM,
    locale: str = "en",
    embedding: Embedding | None = None,
    similarity_threshold: float = 0.7,
    max_llm_pairs: int = 30,
) -> list[Conflict]:
    """Detect contradictions between claims.

    If embedding is provided, pre-filters pairs by cosine similarity
    to reduce LLM calls. Otherwise falls back to O(n²) pairwise comparison.
    """
    n = len(claims)
    prompt_template = get_prompt(locale, "conflict")

    if embedding is not None:
        pairs = filter_candidate_pairs(claims, embedding, similarity_threshold, max_llm_pairs)
    else:
        total_pairs = n * (n - 1) // 2
        if total_pairs > 100:
            warnings.warn(
                f"No embedding provided: checking {total_pairs} claim pairs via LLM (O(n²)). "
                f"Consider using an embedding for conversations with >20 claims. "
                f"Install with: pip install delta-prune[fast]",
                stacklevel=2,
            )
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    conflicts: list[Conflict] = []

    for i, j in pairs:
        a, b = claims[i], claims[j]

        prompt = prompt_template.format(
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
            conflict_type=parsed.get("conflict_type", "direct_contradiction"),
            summary=parsed.get("summary", ""),
        ))

    return conflicts
