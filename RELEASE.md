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
python3 scripts/package_prepared_runtime.py \
  --prepared-root /path/to/prepared/runtime \
  --output-dir dist
```
5. Confirm the following GitHub Actions checks are green on `main` or the
   release branch head:
   - `Package Checks`
   - `Security Checks`
   - `CodeQL`
6. Confirm the repository settings required by release automation are enabled:
   - PyPI trusted publishing for the `pypi` environment
   - GitHub Pages source = `GitHub Actions`
   - dependency graph and code scanning enabled for the repository

## Tagging

Create and push a version tag:

```bash
git tag v0.6.0
git push origin v0.6.0
```

## What CI Does

The release workflow:

1. builds wheel and sdist artifacts
2. generates CycloneDX SBOM files for the wheel and sdist install environments
3. installs each artifact in a clean virtual environment
4. runs `spectral-library --help`
5. runs a minimal prepare, validate, map, and batch-map smoke workflow
6. writes GitHub build-provenance and SBOM attestations for the release files
7. publishes release artifacts and GitHub release notes for the tag

The release-notes body is loaded from `docs/releases/<version>.md`.

Pre-built runtime tarballs are packaged separately with
`scripts/package_prepared_runtime.py` and attached to the GitHub Release after
the tagged package workflow succeeds.

The scheduled full-library benchmark workflow is separate from tagged release
publishing and uploads its own JSON/CSV artifacts when enabled.

Weekly dependency freshness and security scanning are handled separately by:

- `.github/workflows/security-checks.yml`
- `.github/workflows/codeql.yml`
- `.github/dependabot.yml`

All workflow action dependencies are pinned to immutable SHAs. Update them
through reviewed pull requests instead of floating major tags.
