# Spectral Library

`spectral-library` is a Python package and CLI for:

1. preparing a stable row-aligned runtime from a SIAC-style hyperspectral library
2. mapping source-sensor reflectance to target sensors
3. reconstructing VNIR, SWIR, or full `400-2500 nm` spectra with retrieval-based spectral mapping

Public names:

| Surface | Name |
| --- | --- |
| distribution | `spectral-library` |
| import package | `spectral_library` |
| public CLI | `spectral-library` |
| internal CLI | `spectral-library-internal` |

## Documentation

- docs site: [https://marcyin.github.io/spectral_library/](https://marcyin.github.io/spectral_library/)
- getting started: [`docs/mapping_quickstart.md`](docs/mapping_quickstart.md)
- theory: [`docs/theory.md`](docs/theory.md)
- official sensor examples: [`docs/official_sensor_examples.md`](docs/official_sensor_examples.md)
- CLI reference: [`docs/cli_reference.md`](docs/cli_reference.md)
- Python API reference: [`docs/python_api_reference.md`](docs/python_api_reference.md)
- runtime contract: [`docs/prepared_runtime_contract.md`](docs/prepared_runtime_contract.md)
- security and provenance: [`docs/security_provenance.md`](docs/security_provenance.md)
- security policy: [`SECURITY.md`](SECURITY.md)

## Install

Base install:

```bash
python3 -m pip install .
```

Supported Python versions in CI are `3.9`, `3.10`, `3.11`, and `3.12`.

Optional extras:

| Extra | Purpose |
| --- | --- |
| `.[docs]` | build the MkDocs site and GitHub Pages output |
| `.[knn]` | enable the SciPy `cKDTree` backend |
| `.[knn-faiss]` | enable the FAISS backend |
| `.[knn-pynndescent]` | enable the PyNNDescent backend |
| `.[knn-scann]` | enable the ScaNN backend on supported platforms |
| `.[internal-build]` | run retained SIAC-build and example-regeneration tooling |
| `.[accel]` | enable optional Rust-backed smoothing helpers |
| `.[dev]` | run tests, docs, and release tooling |

## Quickstart

Prepare a runtime:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

Validate it:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

Map one sample to a target sensor:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query.csv \
  --output-mode target_sensor \
  --k 10 \
  --output path/to/mapped.csv
```

Map a batch and export diagnostics:

```bash
spectral-library map-reflectance-batch \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/query_batch.csv \
  --output-mode target_sensor \
  --k 10 \
  --neighbor-estimator simplex_mixture \
  --output path/to/mapped_batch.csv \
  --diagnostics-output path/to/mapped_batch_diagnostics.json \
  --neighbor-review-output path/to/mapped_batch_neighbor_review.csv
```

The complete setup, CSV layouts, Python API examples, estimator options, and
ANN backends are documented in
[`docs/mapping_quickstart.md`](docs/mapping_quickstart.md).

## Production Readiness

The repository now includes:

- pinned GitHub Actions dependencies across all workflows
- dependency review, `pip-audit`, and CodeQL workflows
- release SBOM generation and GitHub provenance attestations
- a published security policy in [`SECURITY.md`](SECURITY.md)
- a clean MkDocs build path without `mkdocs-material`

The main remaining step is operational rather than code-related: run the hosted
GitHub workflows on `main` and on a release tag to confirm Pages, PyPI, and the
release attestation path all succeed in the live repository.

## Examples And Benchmarks

For reproducible MODIS, Sentinel-2A, Landsat 8, and Landsat 9 examples built
from official response functions, see
[`docs/official_sensor_examples.md`](docs/official_sensor_examples.md) and
[`examples/official_mapping`](examples/official_mapping/README.md).

For larger prepared runtimes, use
[`scripts/run_full_library_benchmarks.py`](scripts/run_full_library_benchmarks.py)
and the scheduled/manual workflow
[`full-library-benchmarks.yml`](.github/workflows/full-library-benchmarks.yml).

## Maintainers

Public users can ignore the internal build/design pages. Maintainer-oriented
material starts at [`docs/internal_overview.md`](docs/internal_overview.md).
