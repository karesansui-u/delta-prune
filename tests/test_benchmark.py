"""Unit tests for benchmarks/bench_lib.py (no live LLM)."""

from __future__ import annotations

from pathlib import Path

import bench_lib


def test_score_answer_contains_gold():
    assert bench_lib.score_answer_contains_gold("8", "The result is 8.")
    assert bench_lib.score_answer_contains_gold("Paris", "paris")
    assert not bench_lib.score_answer_contains_gold("8", "nine")
    assert not bench_lib.score_answer_contains_gold("8", None)


def test_load_tasks_schema():
    data = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "tasks.json"
    tasks = bench_lib.load_tasks(data)
    assert len(tasks) >= 1
    t = tasks[0]
    assert t.id
    assert t.chunks_in_retrieval_order


def test_aggregate_empty():
    assert bench_lib.aggregate([])["n_tasks"] == 0


def test_aggregate_counts():
    r = [
        bench_lib.TaskResult(
            task_id="a",
            baseline_answer="x",
            pruned_answer="y",
            baseline_correct=True,
            pruned_correct=False,
            baseline_ctx_chars=1,
            pruned_ctx_chars=1,
            num_conflicts_after_prune_check=0,
            delta_after_prune=0.0,
            filtered_chunk_count=1,
        ),
        bench_lib.TaskResult(
            task_id="b",
            baseline_answer="x",
            pruned_answer="y",
            baseline_correct=False,
            pruned_correct=True,
            baseline_ctx_chars=1,
            pruned_ctx_chars=1,
            num_conflicts_after_prune_check=0,
            delta_after_prune=0.0,
            filtered_chunk_count=1,
        ),
    ]
    s = bench_lib.aggregate(r)
    assert s["baseline_accuracy"] == 0.5
    assert s["pruned_accuracy"] == 0.5
    assert s["delta_accuracy_pp"] == 0.0
