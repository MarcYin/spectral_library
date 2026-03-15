# Changelog

All notable changes to `spectral-library` will be documented in this file.

The format follows Keep a Changelog and the project uses semantic versioning
for its public Python API, CLI, and prepared-runtime contract.

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
- Internal pipeline reference in
  [`docs/internal_build_pipeline.md`](docs/internal_build_pipeline.md)
- Release contract tests, package smoke tooling, and GitHub Actions workflows
  for package checks and tagged releases

### Changed

- Package metadata is now aligned to the public distribution name
  `spectral-library`
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
