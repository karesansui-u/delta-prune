"""Core pruning logic: remove or annotate contradictions in message lists."""

from __future__ import annotations

from dataclasses import dataclass, field

from delta_prune.extractor import Claim, extract_claims
from delta_prune.llm import LLM
from delta_prune.resolver import Conflict, detect_conflicts


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
    ) -> None:
        """
        Args:
            llm: LLM adapter for extraction and conflict detection
            strategy: "prune" (remove old), "annotate" (add context), "report" (detect only)
            extract_roles: which message roles to extract claims from
        """
        self._llm = llm
        self._strategy = strategy
        self._extract_roles = extract_roles

    def __call__(self, messages: list[dict[str, str]]) -> PruneResult:
        """Analyze and clean a message list."""
        # Step 1: Extract claims from all messages
        claims = extract_claims(messages, self._llm, self._extract_roles)

        if len(claims) < 2:
            return PruneResult(
                messages=messages,
                conflicts=[],
                claims=claims,
                delta=0.0,
            )

        # Step 2: Detect contradictions
        conflicts = detect_conflicts(claims, self._llm)

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

        # Insert annotation before the last user message
        result = list(messages)
        last_user_idx = len(result) - 1
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                last_user_idx = i
                break

        result.insert(last_user_idx, annotation_msg)
        return result
