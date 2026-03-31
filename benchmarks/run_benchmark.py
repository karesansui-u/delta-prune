#!/usr/bin/env python3
"""RAG-style benchmark: contradictory chunks vs delta-prune filtered chunks.

Chunk order: misleading chunk FIRST (lower index), trustworthy chunks AFTER.
delta-prune ``strategy=prune`` drops the older side of detected conflicts, which
removes the bad lead chunk when it contradicts later correct chunks.

Usage (from repo root)::

    pip install -e ".[ollama]"
    python benchmarks/run_benchmark.py --backend ollama --model qwen2.5:14b

Outputs JSON summary to stdout; use ``--out`` for a full report file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import bench_lib
from delta_prune import DeltaPrune


def _make_llm(backend: str, model: str):
    if backend == "ollama":
        from delta_prune.llm import OllamaLLM

        return OllamaLLM(model=model)
    if backend == "claude":
        from delta_prune.llm import ClaudeCLI

        return ClaudeCLI(model=model)
    if backend == "openai":
        from delta_prune.llm import OpenAILLM

        return OpenAILLM(model=model)
    raise SystemExit(f"Unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="delta-prune RAG contradiction benchmark")
    parser.add_argument(
        "--data",
        type=Path,
        default=_BENCH / "data" / "tasks.json",
        help="Path to tasks JSON",
    )
    parser.add_argument("--backend", choices=("ollama", "claude", "openai"), default="ollama")
    parser.add_argument("--model", default="qwen2.5:14b", help="Model id for the backend")
    parser.add_argument(
        "--strategy",
        choices=("prune", "annotate", "report"),
        default="prune",
        help="delta-prune strategy for filter_chunks",
    )
    parser.add_argument("--locale", default="en", choices=("en", "ja"))
    parser.add_argument("--limit", type=int, default=0, help="Max tasks (0 = all)")
    parser.add_argument("--out", type=Path, default=None, help="Write full JSON results here")
    args = parser.parse_args()

    tasks = bench_lib.load_tasks(args.data)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    llm = _make_llm(args.backend, args.model)
    pruner = DeltaPrune(llm=llm, strategy=args.strategy, locale=args.locale)

    results: list[bench_lib.TaskResult] = []
    for task in tasks:
        results.append(bench_lib.run_one_task(task=task, llm=llm, pruner=pruner))

    summary = bench_lib.aggregate(results)
    report = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "model": args.model,
            "strategy": args.strategy,
            "locale": args.locale,
            "data": str(args.data),
            "chunk_order_note": (
                "Misleading chunk is first (older index); prune removes it when "
                "it contradicts later correct chunks."
            ),
            "external_validity": (
                "Index order here is authored to match prune semantics; real RAG "
                "ranking is usually unrelated to truth/recency. Cite as controlled "
                "ordering unless you add random-position or annotate-only conditions."
            ),
        },
        "summary": summary,
        "tasks": [
            {
                "task_id": r.task_id,
                "baseline_correct": r.baseline_correct,
                "pruned_correct": r.pruned_correct,
                "num_conflicts": r.num_conflicts_after_prune_check,
                "delta": r.delta_after_prune,
                "filtered_chunk_count": r.filtered_chunk_count,
                "baseline_answer_preview": (r.baseline_answer or "")[:200],
                "pruned_answer_preview": (r.pruned_answer or "")[:200],
            }
            for r in results
        ],
    }

    print(json.dumps({"summary": summary, "meta": report["meta"]}, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        full = {
            **report,
            "tasks_full": [
                {
                    "task_id": r.task_id,
                    "baseline_answer": r.baseline_answer,
                    "pruned_answer": r.pruned_answer,
                    "baseline_correct": r.baseline_correct,
                    "pruned_correct": r.pruned_correct,
                    "baseline_ctx_chars": r.baseline_ctx_chars,
                    "pruned_ctx_chars": r.pruned_ctx_chars,
                    "num_conflicts": r.num_conflicts_after_prune_check,
                    "delta": r.delta_after_prune,
                    "filtered_chunk_count": r.filtered_chunk_count,
                }
                for r in results
            ],
        }
        args.out.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
