# Changelog

All notable changes to `spectral-library` will be documented in this file.

The format follows Keep a Changelog and the project uses semantic versioning
for its public Python API, CLI, and prepared-runtime contract.

## [0.4.0] - 2026-04-05

### Changed

- prepared-runtime `sensor_schema.json` is now strictly `rsrf`-normalized,
  validates its own `schema_version` and `canonical_wavelength_grid`, and
  stores band responses only as `response_definition`
- custom SRF JSON bands loaded from `--srf-root` now require
  `response_definition`; legacy top-level `wavelength_nm` / `rsr` payloads are
  rejected
- official MODIS, Sentinel-2A, Landsat 8, and Landsat 9 SRF example JSONs and
  generation scripts now emit `response_definition` payloads to match the
  runtime contract

### Fixed

- invalid prepared `sensor_schema.json` payloads now fail as prepared-runtime
  validation errors instead of surfacing as generic sensor-schema errors
- migration, troubleshooting, CLI, and Python API docs now describe the real
  `0.2.x` to `0.4.0` rebuild requirement and `rsrf`-only runtime contract

### Removed

- compatibility with prepared runtimes using schema `1.2.0`
- fallback handling for legacy sampled-band payloads in custom sensor JSON and
  prepared-runtime `sensor_schema.json`

## [0.3.1] - 2026-04-05

### Changed

- aligned built-in spectral-mapping sensor ids with canonical `rsrf` names such
  as `sentinel-2a_msi`, `landsat-8_oli`, `landsat-9_oli2`, `terra_modis`, and
  platform-specific VIIRS ids
- resolved built-in mapping sensor schemas from `rsrf` instead of the repo's
  committed default SRF snapshots while keeping `--srf-root` available for
  custom local sensor JSON definitions
- allowed prepared runtimes to synthesize missing `source_<sensor>_{vnir,swir}`
  matrices on demand for supported canonical `rsrf` sensors

### Fixed

- release workflow smoke installs now consume the built wheel artifact directly
  instead of a hardcoded package version string
- official example assets, docs, manifests, and tests now use canonical `rsrf`
  sensor ids consistently

## [0.3.0] - 2026-04-04

### Added

- Rust-accelerated batch hot paths via PyO3/Rayon native extension
  (`spectral_library._mapping_rust`) for spectrum reconstruction,
  target-sensor projection, segment merging, and neighbor refinement
- `LinearSpectralMapper` for fixed `input @ weights + bias` high-throughput
  mapping of millions of pixels without per-sample KNN retrieval
- `BatchMappingArrayResult` dense result type for array-oriented batch mapping
- `map_reflectance_batch_arrays()` explicit dense batch path returning
  `BatchMappingArrayResult`
- `map_reflectance_debug()` and `map_reflectance_batch_debug()` for rich
  per-sample diagnostics, splitting the slim default from the debug path
- `compile_linear_mapper()` on `SpectralMapper` to compile a prepared runtime
  into a reusable linear mapper
- `map_reflectance_batch_to_zarr()` for streaming large batch results to
  Zarr stores
- `map_reflectance_batch_ndarray()` for direct ndarray input batch mapping
- cKDTree caching across batch calls for `scipy_ckdtree` backend
- Dense batch vectorization for segment retrieval, reconstruction, and
  target-sensor projection
- Cross-platform wheel builds (manylinux x86_64/aarch64, macOS x86_64/arm64,
  Windows AMD64) via cibuildwheel in GitHub Actions
- CLI `--output-format zarr` and `--output-chunk-size` for streaming batch
  output

### Changed

- `map_reflectance()` now returns slim results by default (no neighbor ids,
  no diagnostics); use `map_reflectance_debug()` for the full payload
- `map_reflectance_batch()` now returns `BatchMappingArrayResult` by default;
  use `map_reflectance_batch_debug()` for per-sample `MappingResult` objects
- Batch segment retrieval is fully vectorized instead of per-sample iteration
- Release workflow now builds platform-specific wheels with compiled Rust
  extension instead of pure-Python wheels

### Fixed

- Double `np.ascontiguousarray(np.asarray(...))` in `LinearSpectralMapper`
  init reduced to single call
- `assert` in persisted KNN index query path replaced with proper
  `MappingInputError` raise
- Redundant `.copy()` on wavelength array construction removed

## [0.2.1] - 2026-04-03

### Added

- New public CLI command `download-prepared-library` to fetch pre-built
  prepared runtimes from GitHub Releases or direct URLs
- New `spectral_library.runtime_download` module with `download_prepared_library()`
  Python API
- New `scripts/package_prepared_runtime.py` to package prepared runtimes into
  distributable tarballs with SHA-256 sidecar digests
- Pre-built runtime tarball (77,125 spectra, MODIS/Sentinel-2/Landsat 8/9)
  published as a GitHub Release asset

### Changed

- README quickstart now leads with `download-prepared-library`
- Mapping quickstart docs split into download and build-your-own paths
- CLI reference and Python API reference updated with new command

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
