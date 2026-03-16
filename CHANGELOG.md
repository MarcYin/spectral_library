# Changelog

All notable changes to `spectral-library` will be documented in this file.

The format follows Keep a Changelog and the project uses semantic versioning
for its public Python API, CLI, and prepared-runtime contract.

## [0.2.0] - 2026-03-16

### Added

- Optional KNN backends:
  `scipy_ckdtree`, `faiss`, `pynndescent`, and `scann`
- Prepare-time persistence for supported ANN indexes and prepared-runtime
  manifest tracking through `knn_index_artifacts`
- Heuristic mapping confidence diagnostics at both segment and top-level result
  scope
- Full-library benchmark automation through
  [`scripts/run_full_library_benchmarks.py`](scripts/run_full_library_benchmarks.py),
  threshold configuration in [`benchmarks/default_thresholds.json`](benchmarks/default_thresholds.json),
  and the scheduled GitHub Actions workflow
  [`.github/workflows/full-library-benchmarks.yml`](.github/workflows/full-library-benchmarks.yml)
- Optional-backend GitHub Actions smoke coverage for persisted-index prepare and
  ANN query paths

### Changed

- The prepared-runtime schema is now `1.2.0`
- Runtime preparation can record interpolation repair summary statistics and
  persisted ANN index artifacts in `manifest.json`
- The public mapping API and CLI now expose row-exclusion controls, neighbor
  diagnostics, heuristic confidence scores, and backend selection for KNN
  search
- Official mapping examples now use the full prepared SIAC library with exact
  held-out row exclusion instead of the earlier toy catalogue

### Fixed

- The full-library benchmark runner no longer accumulates default estimator,
  backend, or `k` values when explicit CLI arguments are supplied
- Prepared-runtime checksum hashing now correctly includes file contents for all
  runtime artifacts, including persisted ANN index directories
- SIAC sparse-gap interpolation is now constrained and recorded, rather than
  silently extrapolating arbitrary internal gaps

## [0.1.0] - 2026-03-15

### Added

- Public mapping runtime preparation via `prepare_mapping_library(...)` and
  `spectral-library prepare-mapping-library`
- Prepared-runtime validation via `validate_prepared_library(...)` and
  `spectral-library validate-prepared-library`
- Retrieval-based spectral mapping through `SpectralMapper`,
  `spectral-library map-reflectance`, and
  `spectral-library map-reflectance-batch`
- Stable output modes:
  `target_sensor`, `vnir_spectrum`, `swir_spectrum`, and `full_spectrum`
- Batch mapping result type and batch CLI diagnostics output
- Regression-baseline benchmarking via `benchmark_mapping(...)` and
  `spectral-library benchmark-mapping`
- Prepared-runtime manifest and checksum validation, including row-alignment
  checks and schema-compatibility errors
- Public quickstart documentation in
  [`docs/mapping_quickstart.md`](docs/mapping_quickstart.md)
- Public docs index, CLI reference, Python API reference, prepared-runtime
  contract guide, and official-source MODIS/Sentinel-2A/Landsat 8/Landsat 9
  examples with generated visuals
- MkDocs documentation site with GitHub Pages deployment from this repo's
  GitHub Actions workflows
- Mathematical foundations documentation covering the forward model,
  segment-wise retrieval equations, overlap blending, and benchmark metrics
- Internal pipeline reference in
  [`docs/internal_build_pipeline.md`](docs/internal_build_pipeline.md)
- Release contract tests, package smoke tooling, and GitHub Actions workflows
  for package checks and tagged releases

### Changed

- Package metadata is now aligned to the public distribution name
  `spectral-library`
- The repository software is now distributed under the MIT License, while
  referenced third-party datasets and derived artifacts remain subject to
  their own upstream terms
- Runtime dependencies are separated from optional internal-build and
  acceleration dependencies
- Batch mapping now performs batched segment retrieval work internally instead
  of re-entering the full single-sample path for every sample

### Fixed

- Malformed prepared runtimes now fail through the public error types instead
  of leaking raw parser exceptions
- Missing `checksums.json`, duplicate row identities, row-index gaps, duplicate
  prepared `sensor_id` values, and corrupted array files are rejected
  explicitly
