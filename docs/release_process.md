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
5. Confirm the GitHub Actions security gates are green:
   - `Package Checks`
   - `Security Checks`
   - `CodeQL`
6. Confirm the repository-level release settings are still configured:
   - PyPI trusted publishing on the `pypi` environment
   - GitHub Pages publishing from GitHub Actions
   - dependency graph and code scanning enabled

## Tagging

Create and push a version tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

## CI Responsibilities

The tagged release workflow:

1. builds wheel and sdist artifacts
2. generates CycloneDX SBOM files for the wheel and sdist install environments
3. installs each artifact in a clean environment
4. runs public CLI smoke tests
5. writes GitHub build-provenance and SBOM attestations
6. publishes to PyPI through trusted publishing
7. creates the GitHub release using the matching release-notes file

The scheduled full-library benchmark workflow is separate from tagged package
release publishing. It runs from
[`full-library-benchmarks.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/full-library-benchmarks.yml)
when the repository variable `FULL_LIBRARY_PREPARED_ROOT` is configured.

The docs site is published separately from pushes to `main` through the GitHub
Pages workflow.

Security policy is enforced continuously outside the release job by:

- [`security-checks.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/security-checks.yml)
  for dependency review and `pip-audit`
- [`codeql.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/codeql.yml)
  for Python CodeQL analysis
- [`dependabot.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/dependabot.yml)
  for weekly GitHub Actions and Python dependency update PRs
