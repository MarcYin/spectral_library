# Security and Provenance

This repository now treats release security and artifact provenance as part of
the public production contract, not as ad hoc maintainer work.

## Security Gates

### Dependency Review

Pull requests are checked by
[`security-checks.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/security-checks.yml)
with GitHub's dependency review action.

- New vulnerable dependency changes fail the PR at `high` severity or above.
- The workflow also posts a PR summary when the review fails.

### `pip-audit`

The same security workflow audits two dependency surfaces:

- the runtime install set from `project.dependencies`
- the docs build toolchain from `project.dependencies + project.optional-dependencies.docs`

The workflow exports the requirements directly from
[`pyproject.toml`](https://github.com/MarcYin/spectral_library/blob/main/pyproject.toml)
and then runs `pip-audit --strict`.

### CodeQL

[`codeql.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/codeql.yml)
runs GitHub CodeQL analysis for Python on pushes to `main`, pull requests to
`main`, manual dispatches, and a weekly schedule.

The scoped config in
[`codeql-config.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/codeql/codeql-config.yml)
analyzes:

- `src/`
- `scripts/`

It intentionally excludes generated outputs, docs, examples, and tests.

### Dependency Freshness

[`dependabot.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/dependabot.yml)
keeps both GitHub Actions and Python dependencies on a weekly review cycle.

## Release Provenance

Tagged releases use
[`release-package.yml`](https://github.com/MarcYin/spectral_library/blob/main/.github/workflows/release-package.yml)
to produce more than just a wheel and sdist.

The release workflow now:

1. builds the wheel and sdist
2. generates CycloneDX SBOMs for the wheel and sdist install environments
3. smoke-tests both artifacts in clean virtual environments
4. writes GitHub build-provenance attestations for the release artifacts
5. writes GitHub SBOM attestations for both artifact types
6. publishes to PyPI and creates the GitHub release on version tags

The release artifact set therefore includes:

- the wheel
- the sdist
- `spectral-library-wheel.sbom.cdx.json`
- `spectral-library-sdist.sbom.cdx.json`

## Scope Notes

- Optional ANN extras are still platform-dependent, especially `scann`.
  Runtime support and smoke coverage remain documented in
  [CLI Reference](cli_reference.md) and [Getting Started](mapping_quickstart.md).
- `confidence_score` remains heuristic. Security and provenance checks improve
  trust in the package and build chain, not scientific calibration of outputs.
- The docs site is published separately from package release, but it is covered
  by the main package/docs CI path.
