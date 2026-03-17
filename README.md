# Spectral Library

`spectral-library` is a Python package and CLI for:

1. preparing row-aligned runtime artifacts from a SIAC spectral-library export,
2. mapping source-sensor reflectance to target sensors,
3. reconstructing VNIR, SWIR, or full hyperspectral outputs with retrieval-based
   spectral mapping.

The public names are:

- distribution: `spectral-library`
- import package: `spectral_library`
- CLI: `spectral-library`
- license: `MIT` for repository software and repository-authored docs

## Start Here

- published docs site:
  [https://marcyin.github.io/spectral_library/](https://marcyin.github.io/spectral_library/)
- overview:
  [`docs/index.md`](docs/index.md)
- getting started:
  [`docs/mapping_quickstart.md`](docs/mapping_quickstart.md)
- mathematical foundations:
  [`docs/theory.md`](docs/theory.md)
- official sensor examples:
  [`docs/official_sensor_examples.md`](docs/official_sensor_examples.md)
- public CLI reference:
  [`docs/cli_reference.md`](docs/cli_reference.md)
- public Python API reference:
  [`docs/python_api_reference.md`](docs/python_api_reference.md)
- security and provenance:
  [`docs/security_provenance.md`](docs/security_provenance.md)
- prepared-runtime contract:
  [`docs/prepared_runtime_contract.md`](docs/prepared_runtime_contract.md)
- internal maintainer docs:
  [`docs/internal_overview.md`](docs/internal_overview.md)

## Install

Install from this repository in a Python `3.9+` environment:

```bash
python3 -m pip install .
```

Supported Python versions in CI are `3.9`, `3.10`, `3.11`, and `3.12`.

The same documentation set is published to GitHub Pages from this repository.

For the retained internal normalization and QA commands, install the optional
internal-build dependencies:

```bash
python3 -m pip install ".[internal-build]"
```

This extra is also required if you want to regenerate the bundled official
sensor example assets with `scripts/build_official_mapping_examples.py`.

For the static documentation site and GitHub Pages build path, install:

```bash
python3 -m pip install ".[docs]"
```

The published docs site is built with MkDocs from [mkdocs.yml](mkdocs.yml).

For the optional SciPy `cKDTree` shortlist backend, install:

```bash
python3 -m pip install ".[knn]"
```

Additional optional ANN backends are available with:

```bash
python3 -m pip install ".[knn-faiss]"
python3 -m pip install ".[knn-pynndescent]"
python3 -m pip install ".[knn-scann]"
```

## Minimal Example

Prepare a mapping runtime:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --knn-index-backend faiss \
  --output-root build/mapping_runtime
```

Validate it:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

Map one reflectance sample:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query.csv \
  --output-mode target_sensor \
  --output path/to/mapped.csv
```

Map many samples from one CSV:

```bash
spectral-library map-reflectance-batch \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query_batch.csv \
  --output-mode target_sensor \
  --output path/to/mapped_batch.csv
```

The detailed quickstart, supported CSV layouts, Python API examples, SRF JSON
schema, and prepared-runtime contract are documented in
[`docs/mapping_quickstart.md`](docs/mapping_quickstart.md).

For larger prepared runtimes, the repository also ships
[`scripts/run_full_library_benchmarks.py`](scripts/run_full_library_benchmarks.py)
plus the scheduled/manual GitHub Actions workflow
[`full-library-benchmarks.yml`](.github/workflows/full-library-benchmarks.yml).
Use `--max-test-rows` to keep held-out benchmark runs bounded on large
prepared libraries.

For production release hardening, the repository now also ships:

- dependency review and `pip-audit` gates in
  [`security-checks.yml`](.github/workflows/security-checks.yml)
- Python CodeQL analysis in
  [`codeql.yml`](.github/workflows/codeql.yml)
- weekly dependency updates in
  [`.github/dependabot.yml`](.github/dependabot.yml)
- release SBOM and provenance attestations in
  [`release-package.yml`](.github/workflows/release-package.yml)

Workflow dependencies are pinned to immutable GitHub Action SHAs, and
Dependabot is configured to move those pins forward through reviewed pull
requests.

For reproducible cross-sensor examples built from official MODIS, Sentinel-2A,
Landsat 8, and Landsat 9 response functions, see
[`docs/official_sensor_examples.md`](docs/official_sensor_examples.md) and the
bundled example assets in
[`examples/official_mapping`](examples/official_mapping/README.md). Those
official examples expect the previously composed full SIAC library recorded in
the example manifest, rather than a vendored mini runtime.
