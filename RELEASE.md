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
PYTHONPATH=src python3 scripts/run_full_library_benchmarks.py \
  --prepared-root /path/to/prepared/runtime \
  --neighbor-estimator simplex_mixture \
  --knn-backend numpy \
  --k 10 \
  --max-test-rows 512 \
  --output-root build/full-library-benchmarks \
  --thresholds benchmarks/default_thresholds.json \
  --fail-on-thresholds
python3 -m pip install build
python3 scripts/build_distribution.py
```

## Tagging

Create and push a version tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## What CI Does

The release workflow:

1. builds wheel and sdist artifacts
2. installs each artifact in a clean virtual environment
3. runs `spectral-library --help`
4. runs a minimal prepare, validate, map, and batch-map smoke workflow
5. publishes release artifacts and GitHub release notes for the tag

The release-notes body is loaded from `docs/releases/<version>.md`.

The scheduled full-library benchmark workflow is separate from tagged release
publishing and uploads its own JSON/CSV artifacts when enabled.
