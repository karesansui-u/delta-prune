# delta-prune — notes for coding agents (Claude Code, Cursor, etc.)

## PyPI publishing

- **Packages are published from GitHub Actions**, not from a developer laptop by default.
- Workflow: [`.github/workflows/publish-pypi.yml`](.github/workflows/publish-pypi.yml)
- **Trigger**
  - Manual: GitHub → **Actions** → **Publish to PyPI** → **Run workflow**
  - Or push a git tag matching `v*` (e.g. `v0.4.0`)
- **Credential**: repository secret **`__TOKEN__`** on `karesansui-u/delta-prune` — value is a PyPI API token (`pypi-…`). Do **not** commit tokens; do **not** paste them into chat.
- The action uses `pypa/gh-action-pypi-publish`; username `__token__` is implied when using an API token.

## Release checklist

1. Bump `version` in [`pyproject.toml`](pyproject.toml).
2. Update [`README.md`](README.md) **Changelog** if user-facing behavior changed.
3. Commit and push to `main`.
4. Run the workflow (or tag `v*`). PyPI will serve the version built from the repo at that run.

## Local install / tests

- Editable: `pip install -e ".[ollama]"` from repo root (or `[fast]`, `[openai]` as needed).
- Tests: `pytest tests/ -k "not ollama"` (skip real Ollama integration unless intended).

## Contradiction RAG benchmark

- **Purpose**: Compare **baseline** (misleading chunk + truth in context) vs **`filter_chunks`** on answer accuracy (gold substring heuristic).
- **Docs**: [`benchmarks/README.md`](benchmarks/README.md)
- **Run** (real LLM): `python benchmarks/run_benchmark.py --backend ollama --model <name>`
- **Data**: [`benchmarks/data/tasks.json`](benchmarks/data/tasks.json) — misleading chunk is **first** so `strategy=prune` removes it when conflicts are detected.
