# spectral-library

`spectral-library` is a Python package and CLI for:

- preparing stable row-aligned mapping runtimes from SIAC-style hyperspectral exports
- mapping source-sensor reflectance to target sensors
- reconstructing VNIR, SWIR, or full `400-2500 nm` spectra with retrieval-based spectral mapping

Project identity:

- distribution: `spectral-library`
- import package: `spectral_library`
- public CLI: `spectral-library`
- internal maintainer CLI: `spectral-library-internal`
- current version: `0.5.0`
- license: `MIT` for repository software and repository-authored docs; referenced or redistributed third-party datasets, metadata, and derived artifacts may remain subject to upstream terms

Documentation:

- published site: [https://marcyin.github.io/spectral_library/](https://marcyin.github.io/spectral_library/)
- getting started: [docs/mapping_quickstart.md](docs/mapping_quickstart.md)
- theory: [docs/theory.md](docs/theory.md)
- official sensor examples: [docs/official_sensor_examples.md](docs/official_sensor_examples.md)
- CLI reference: [docs/cli_reference.md](docs/cli_reference.md)
- Python API reference: [docs/python_api_reference.md](docs/python_api_reference.md)
- prepared runtime contract: [docs/prepared_runtime_contract.md](docs/prepared_runtime_contract.md)
- security and provenance: [docs/security_provenance.md](docs/security_provenance.md)
- release process: [docs/release_process.md](docs/release_process.md)

## Install

Install the public package in Python `3.9` to `3.14`:

```bash
python3 -m pip install .
```

Optional extras:

- docs site: `python3 -m pip install ".[docs]"`
- SciPy `cKDTree` KNN backend: `python3 -m pip install ".[knn]"`
- FAISS backend: `python3 -m pip install ".[knn-faiss]"`
- PyNNDescent backend: `python3 -m pip install ".[knn-pynndescent]"`
- ScaNN backend: `python3 -m pip install ".[knn-scann]"`
- all ANN extras: `python3 -m pip install ".[knn-all]"`
- internal build and example-regeneration tools: `python3 -m pip install ".[internal-build]"`

## Quickstart

Download a pre-built runtime (77,125 spectra, MODIS/Sentinel-2/Landsat 8/9):

```bash
spectral-library download-prepared-library \
  --output-root build/mapping_runtime
```

Or build your own from a SIAC-style export:

```bash
spectral-library build-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --source-sensor SENSOR_B \
  --knn-index-backend faiss \
  --output-root build/mapping_runtime
```

Validate it:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

Map one sample:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query.csv \
  --output-mode target_sensor \
  --neighbor-estimator simplex_mixture \
  --knn-backend faiss \
  --output path/to/mapped.csv
```

Map a batch and write diagnostics:

```bash
spectral-library map-reflectance-batch \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query_batch.csv \
  --output-mode target_sensor \
  --neighbor-estimator simplex_mixture \
  --knn-backend faiss \
  --diagnostics-output path/to/diagnostics.json \
  --neighbor-review-output path/to/neighbor_review.csv \
  --output path/to/mapped_batch.csv
```

Benchmark a prepared runtime:

```bash
python3 scripts/run_full_library_benchmarks.py \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --source-sensor SENSOR_B \
  --neighbor-estimator simplex_mixture \
  --knn-backend numpy \
  --knn-backend faiss \
  --max-test-rows 200 \
  --output-root build/benchmark_outputs
```

For CSV layouts, SRF JSON shape, batch self-exclusion, diagnostics, confidence policy, and Python examples, start with [docs/mapping_quickstart.md](docs/mapping_quickstart.md).

## Repository Layout

The repository is now organized around four package areas:

- `spectral_library.mapping`
  public mapping runtime, prepared-runtime build, and retrieval engine
- `spectral_library.distribution`
  runtime download helpers used by `download-prepared-library`
- `spectral_library.sources`
  maintainer-oriented source manifests, fetchers, fetch batching, and catalog assembly
- `spectral_library.normalization`
  maintainer-oriented normalization, coverage filtering, quality plots, and SIAC package export

Compatibility wrappers keep older flat root modules like `batch.py`,
`build_db.py`, `normalize.py`, and `runtime_download.py` importable while the
implementation lives in the new package layout.

## What The Package Ships

- Stable public Python API for `prepare_mapping_library(...)`, `build_mapping_library(...)`, `validate_prepared_library(...)`, and `SpectralMapper`
- Public CLI for runtime build, validation, single-sample mapping, batch mapping, and benchmarking
- Prepared runtime format with manifest, checksums, sensor schema, and optional persisted ANN indexes
- Exact `numpy` KNN plus optional `scipy_ckdtree`, `faiss`, `pynndescent`, and `scann` backends
- Multiple retrieval estimators including `mean`, `distance_weighted_mean`, and `simplex_mixture`
- Rich diagnostics with neighbor identities, distances, weights, source-band fits, and confidence routing

## Official Sensor Examples

The repository includes reproducible cross-sensor examples built from official:

- NASA MODIS response functions
- ESA Sentinel-2A response functions
- USGS Landsat 8 and Landsat 9 response functions

Those examples live in [examples/official_mapping/README.md](examples/official_mapping/README.md) and are documented in [docs/official_sensor_examples.md](docs/official_sensor_examples.md).

Important:

- the official example bundle expects the previously composed full SIAC library recorded in its manifest
- the docs example is a held-out evaluation against that full library, not a vendored miniature retrieval database
- Sentinel-2 uses `B8A` for the semantic `nir` band in the published example set

## Production Readiness

The repository includes the production-release work needed to ship the package:

- package build and smoke-test workflows
- GitHub Pages docs deployment
- dependency review, `pip-audit`, and Python CodeQL
- weekly Dependabot update PRs
- pinned GitHub Action SHAs across workflows
- SBOM generation and provenance attestations in the release workflow
- scheduled and manual full-library benchmark workflows

See [docs/security_provenance.md](docs/security_provenance.md), [SECURITY.md](SECURITY.md), and [RELEASE.md](RELEASE.md).

## Maintainers

Internal build and source-ingest material is intentionally separated from the public mapping package. If you are maintaining the full library build pipeline, start here:

- internal overview: [docs/internal_overview.md](docs/internal_overview.md)
- internal build pipeline: [docs/internal_build_pipeline.md](docs/internal_build_pipeline.md)
- release notes: [docs/releases/0.5.0.md](docs/releases/0.5.0.md)
