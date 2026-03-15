# Release Process

This repository uses a tag-based package release workflow.

## Before Tagging

1. Update [`src/spectral_library/_version.py`](src/spectral_library/_version.py)
   and [`pyproject.toml`](pyproject.toml) if the version changes.
2. Add a new entry to [`CHANGELOG.md`](CHANGELOG.md).
3. Add tagged release notes under `docs/releases/<version>.md`.
4. Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
python3 -m pip install build
python3 scripts/build_distribution.py
```

## Tagging

Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## What CI Does

The release workflow:

1. builds wheel and sdist artifacts
2. installs each artifact in a clean virtual environment
3. runs `spectral-library --help`
4. runs a minimal prepare, validate, map, and batch-map smoke workflow
5. publishes release artifacts and GitHub release notes for the tag

The release-notes body is loaded from `docs/releases/<version>.md`.
