# Release Process

This page summarizes the production release workflow for `spectral-library`.

## Before Tagging

1. Update
   [`src/spectral_library/_version.py`](https://github.com/MarcYin/spectral_library/blob/main/src/spectral_library/_version.py)
   and
   [`pyproject.toml`](https://github.com/MarcYin/spectral_library/blob/main/pyproject.toml)
   if the version changes.
2. Add a new entry to
   [`CHANGELOG.md`](https://github.com/MarcYin/spectral_library/blob/main/CHANGELOG.md).
3. Add release notes under `docs/releases/<version>.md`.
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

## CI Responsibilities

The tagged release workflow:

1. builds wheel and sdist artifacts
2. installs each artifact in a clean environment
3. runs public CLI smoke tests
4. publishes to PyPI through trusted publishing
5. creates the GitHub release using the matching release-notes file

The scheduled full-library benchmark workflow is separate from tagged package
release publishing. It runs from
[`full-library-benchmarks.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/full-library-benchmarks.yml)
when the repository variable `FULL_LIBRARY_PREPARED_ROOT` is configured.

The docs site is published separately from pushes to `main` through the GitHub
Pages workflow.
