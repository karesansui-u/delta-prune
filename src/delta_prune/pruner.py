"""Core pruning logic: remove or annotate contradictions in message lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from delta_prune.extractor import Claim, extract_claims
from delta_prune.llm import LLM
from delta_prune.resolver import Conflict, detect_conflicts

if TYPE_CHECKING:
    from delta_prune.embedding import Embedding


@dataclass
class PruneResult:
    """Result of a prune operation."""

    messages: list[dict[str, str]]
    conflicts: list[Conflict]
    claims: list[Claim]
    delta: float  # contradiction density: conflicts / claims

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


@dataclass
class ChunkPruneResult:
    """Result of RAG chunk filtering (`filter_chunks`)."""

    filtered_chunks: list[str]
    conflicts: list[Conflict]
    claims: list[Claim]
    delta: float

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0


def _chunks_to_claims(chunks: list[str]) -> list[Claim]:
    """One claim per non-empty chunk; source_turn is the chunk index."""
    claims: list[Claim] = []
    for i, text in enumerate(chunks):
        if not text or not text.strip():
            continue
        claims.append(
            Claim(
                text=text.strip(),
                type="rag_chunk",
                source_turn=i,
                role="user",
            )
        )
    return claims


class DeltaPrune:
    """LLM context contradiction detector and pruner.

    Usage:
        prune = DeltaPrune(llm=ClaudeCLI())
        result = prune(messages)
        clean_messages = result.messages  # contradictions resolved
        print(result.delta)              # contradiction density
    """

    def __init__(
        self,
        llm: LLM,
        strategy: str = "annotate",
        extract_roles: tuple[str, ...] = ("user",),
        locale: str = "en",
        embedding: Embedding | None = None,
        similarity_threshold: float = 0.7,
        max_llm_pairs: int = 30,
    ) -> None:
        """
        Args:
            llm: LLM adapter for extraction and conflict detection
            strategy: "prune" (remove old), "annotate" (add context), "report" (detect only)
            extract_roles: which message roles to extract claims from
            locale: "en" (default) or "ja" — controls prompt language
            embedding: optional embedding adapter for fast pre-filtering
            similarity_threshold: cosine similarity threshold for pre-filter (0.0-1.0)
            max_llm_pairs: maximum number of pairs to send to LLM for conflict check
        """
        self._llm = llm
        self._strategy = strategy
        self._extract_roles = extract_roles
        self._locale = locale
        self._embedding = embedding
        self._similarity_threshold = similarity_threshold
        self._max_llm_pairs = max_llm_pairs

    def __call__(self, messages: list[dict[str, str]]) -> PruneResult:
        """Analyze and clean a message list."""
        # Step 1: Extract claims from all messages
        claims = extract_claims(messages, self._llm, self._extract_roles, self._locale)

        if len(claims) < 2:
            return PruneResult(
                messages=messages,
                conflicts=[],
                claims=claims,
                delta=0.0,
            )

        # Step 2: Detect contradictions
        conflicts = detect_conflicts(
            claims,
            self._llm,
            locale=self._locale,
            embedding=self._embedding,
            similarity_threshold=self._similarity_threshold,
            max_llm_pairs=self._max_llm_pairs,
        )

        # Step 3: Calculate delta (contradiction density)
        delta = len(conflicts) / len(claims) if claims else 0.0

        if not conflicts:
            return PruneResult(
                messages=messages,
                conflicts=[],
                claims=claims,
                delta=delta,
            )

        # Step 4: Apply strategy
        if self._strategy == "prune":
            cleaned = self._apply_prune(messages, conflicts)
        elif self._strategy == "annotate":
            cleaned = self._apply_annotate(messages, conflicts)
        else:  # "report"
            cleaned = list(messages)

        return PruneResult(
            messages=cleaned,
            conflicts=conflicts,
            claims=claims,
            delta=delta,
        )

    def filter_chunks(self, chunks: list[str]) -> ChunkPruneResult:
        """Detect contradictions across RAG retrieval chunks and filter or annotate.

        Each non-empty chunk is treated as one factual unit (no per-chunk extraction LLM).
        Uses the same conflict detection and strategies as chat mode: ``prune`` removes
        chunks that hold the older side of a contradiction; ``annotate`` prepends a
        consistency note chunk; ``report`` returns the original list unchanged.

        Chunk order is preserved except for removals. Empty strings are kept but do not
        participate in pairwise checks.
        """
        if not chunks:
            return ChunkPruneResult(
                filtered_chunks=[],
                conflicts=[],
                claims=[],
                delta=0.0,
            )

        claims = _chunks_to_claims(chunks)

        if len(claims) < 2:
            return ChunkPruneResult(
                filtered_chunks=list(chunks),
                conflicts=[],
                claims=claims,
                delta=0.0,
            )

        conflicts = detect_conflicts(
            claims,
            self._llm,
            locale=self._locale,
            embedding=self._embedding,
            similarity_threshold=self._similarity_threshold,
            max_llm_pairs=self._max_llm_pairs,
        )

        delta = len(conflicts) / len(claims) if claims else 0.0

        if not conflicts:
            return ChunkPruneResult(
                filtered_chunks=list(chunks),
                conflicts=[],
                claims=claims,
                delta=delta,
            )

        if self._strategy == "prune":
            filtered = self._apply_chunk_prune(chunks, conflicts)
        elif self._strategy == "annotate":
            filtered = self._apply_chunk_annotate(chunks, conflicts)
        else:
            filtered = list(chunks)

        return ChunkPruneResult(
            filtered_chunks=filtered,
            conflicts=conflicts,
            claims=claims,
            delta=delta,
        )

    def _apply_chunk_prune(
        self,
        chunks: list[str],
        conflicts: list[Conflict],
    ) -> list[str]:
        indices_to_remove: set[int] = set()
        for conflict in conflicts:
            older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            indices_to_remove.add(older.source_turn)
        return [c for i, c in enumerate(chunks) if i not in indices_to_remove]

    def _apply_chunk_annotate(
        self,
        chunks: list[str],
        conflicts: list[Conflict],
    ) -> list[str]:
        if self._locale == "ja":
            lines: list[str] = []
            for conflict in conflicts:
                older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
                newer = max(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
                lines.append(
                    f"- {conflict.conflict_type}: "
                    f"チャンク{older.source_turn}「{older.text}」→ "
                    f"チャンク{newer.source_turn}「{newer.text}」に変化"
                )
            header = (
                "[RAGコンテキスト整合性チェック]\n"
                "取得チャンク間で以下の矛盾が検出されました。後ろ（インデックスが大きい）のチャンクを優先してください。\n"
                + "\n".join(lines)
            )
        else:
            lines = []
            for conflict in conflicts:
                older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
                newer = max(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
                lines.append(
                    f"- {conflict.conflict_type}: "
                    f"chunk {older.source_turn} \"{older.text}\" → "
                    f"chunk {newer.source_turn} \"{newer.text}\""
                )
            header = (
                "[RAG context consistency check]\n"
                "Contradictions were detected between retrieved chunks. "
                "Prioritize the later chunk (higher index):\n"
                + "\n".join(lines)
            )
        return [header] + list(chunks)

    def _apply_prune(
        self,
        messages: list[dict[str, str]],
        conflicts: list[Conflict],
    ) -> list[dict[str, str]]:
        """Remove messages containing the older side of each contradiction."""
        turns_to_remove: set[int] = set()

        for conflict in conflicts:
            # Keep the newer claim, remove the older one
            older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            turns_to_remove.add(older.source_turn)

        return [
            msg for i, msg in enumerate(messages)
            if i not in turns_to_remove
        ]

    def _apply_annotate(
        self,
        messages: list[dict[str, str]],
        conflicts: list[Conflict],
    ) -> list[dict[str, str]]:
        """Add a system message summarizing detected contradictions."""
        if self._locale == "ja":
            return self._annotate_ja(messages, conflicts)
        return self._annotate_en(messages, conflicts)

    def _annotate_en(
        self,
        messages: list[dict[str, str]],
        conflicts: list[Conflict],
    ) -> list[dict[str, str]]:
        annotations: list[str] = []
        for conflict in conflicts:
            older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            newer = max(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            annotations.append(
                f"- {conflict.conflict_type}: "
                f"message {older.source_turn} \"{older.text}\" → "
                f"message {newer.source_turn} \"{newer.text}\""
            )

        annotation_msg = {
            "role": "system",
            "content": (
                "[Context consistency check]\n"
                "The following contradictions were detected. Prioritize the latest information:\n"
                + "\n".join(annotations)
            ),
        }

        result = list(messages)
        last_user_idx = len(result) - 1
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                last_user_idx = i
                break

        result.insert(last_user_idx, annotation_msg)
        return result

    def _annotate_ja(
        self,
        messages: list[dict[str, str]],
        conflicts: list[Conflict],
    ) -> list[dict[str, str]]:
        annotations: list[str] = []
        for conflict in conflicts:
            older = min(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            newer = max(conflict.claim_a, conflict.claim_b, key=lambda c: c.source_turn)
            annotations.append(
                f"- {conflict.conflict_type}: "
                f"発言{older.source_turn}「{older.text}」→ "
                f"発言{newer.source_turn}「{newer.text}」に変化"
            )

        annotation_msg = {
            "role": "system",
            "content": (
                "[コンテキスト整合性チェック]\n"
                "以下の矛盾が検出されました。最新の情報を優先してください:\n"
                + "\n".join(annotations)
            ),
        }

        result = list(messages)
        last_user_idx = len(result) - 1
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                last_user_idx = i
                break

        result.insert(last_user_idx, annotation_msg)
        return result
