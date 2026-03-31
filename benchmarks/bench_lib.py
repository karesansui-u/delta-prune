"""Shared logic for the contradiction RAG benchmark (no CLI)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from delta_prune import DeltaPrune
from delta_prune.llm import LLM


@dataclass
class Task:
    id: str
    question: str
    gold_answer: str
    chunks_in_retrieval_order: list[str]


@dataclass
class TaskResult:
    task_id: str
    baseline_answer: str | None
    pruned_answer: str | None
    baseline_correct: bool
    pruned_correct: bool
    baseline_ctx_chars: int
    pruned_ctx_chars: int
    num_conflicts_after_prune_check: int
    delta_after_prune: float
    filtered_chunk_count: int


def load_tasks(path: Path) -> list[Task]:
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    out: list[Task] = []
    for row in raw:
        out.append(
            Task(
                id=row["id"],
                question=row["question"],
                gold_answer=row["gold_answer"],
                chunks_in_retrieval_order=list(row["chunks_in_retrieval_order"]),
            )
        )
    return out


def normalize_for_match(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def score_answer_contains_gold(gold: str, response: str | None) -> bool:
    """Heuristic: gold (normalized) appears as substring of response (normalized)."""
    if not response:
        return False
    g = normalize_for_match(gold)
    r = normalize_for_match(response)
    if not g or not r:
        return False
    return g in r


def build_qa_prompt(context: str, question: str) -> str:
    return (
        "You are a careful assistant. Answer using ONLY the context below. "
        "If the context is insufficient or contradictory, give the best factually "
        "supported answer from the context and keep the answer short.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def run_one_task(
    *,
    task: Task,
    llm: LLM,
    pruner: DeltaPrune,
) -> TaskResult:
    chunks = list(task.chunks_in_retrieval_order)
    baseline_ctx = "\n\n".join(chunks)

    chunk_result = pruner.filter_chunks(chunks)
    pruned_ctx = "\n\n".join(chunk_result.filtered_chunks)

    base_prompt = build_qa_prompt(baseline_ctx, task.question)
    prune_prompt = build_qa_prompt(pruned_ctx, task.question)

    baseline_ans = llm.generate(base_prompt)
    pruned_ans = llm.generate(prune_prompt)

    return TaskResult(
        task_id=task.id,
        baseline_answer=baseline_ans,
        pruned_answer=pruned_ans,
        baseline_correct=score_answer_contains_gold(task.gold_answer, baseline_ans),
        pruned_correct=score_answer_contains_gold(task.gold_answer, pruned_ans),
        baseline_ctx_chars=len(baseline_ctx),
        pruned_ctx_chars=len(pruned_ctx),
        num_conflicts_after_prune_check=len(chunk_result.conflicts),
        delta_after_prune=chunk_result.delta,
        filtered_chunk_count=len(chunk_result.filtered_chunks),
    )


def aggregate(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "n_tasks": 0,
            "baseline_accuracy": None,
            "pruned_accuracy": None,
            "delta_accuracy_pp": None,
        }
    b = sum(1 for r in results if r.baseline_correct)
    p = sum(1 for r in results if r.pruned_correct)
    return {
        "n_tasks": n,
        "baseline_accuracy": b / n,
        "pruned_accuracy": p / n,
        "delta_accuracy_pp": (p - b) / n * 100.0,
        "baseline_correct": b,
        "pruned_correct": p,
    }
