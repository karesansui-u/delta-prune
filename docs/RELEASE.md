# Releasing delta-prune to PyPI

## Where credentials live

| What | Where |
|------|--------|
| PyPI API token | GitHub repository **Settings → Secrets and variables → Actions** |
| Secret name | **`__TOKEN__`** (value = full `pypi-…` string from [pypi.org account API tokens](https://pypi.org/manage/account/token/)) |

The token is **not** stored in this git repository.

## How packages get to PyPI

GitHub Actions workflow **Publish to PyPI** (file: `.github/workflows/publish-pypi.yml`):

1. Checks out `main` (or the ref for a tag push).
2. Runs `python -m build` → `dist/*.whl` and `sdist`.
3. Uploads with `pypa/gh-action-pypi-publish` using `secrets.__TOKEN__`.

## How to publish a new version

1. Set `version` in `pyproject.toml`.
2. Update `README.md` changelog if needed.
3. Push to `main`.
4. Either:
   - **Actions → Publish to PyPI → Run workflow**, or
   - Create and push a tag: `git tag v0.x.y && git push origin v0.x.y`

Then confirm on [pypi.org/project/delta-prune](https://pypi.org/project/delta-prune/).

## Optional: upload from your machine

Only if you have `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` (or `~/.pypirc`) configured locally:

```bash
python -m build && twine upload dist/*
```

Prefer CI so the same artifact is reproducible from git.
