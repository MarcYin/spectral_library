# Getting Started

This guide is the fastest path from installation to a working mapping run.

It covers the public package workflow:

1. install the package
2. prepare a runtime
3. validate the runtime
4. map one sample
5. map a batch
6. call the Python API directly

## Before You Start

You need:

- Python `3.9` or newer
- a SIAC-style spectral export with:
  - `tabular/siac_spectra_metadata.csv`
  - `tabular/siac_normalized_spectra.csv`
- one or more sensor SRF JSON definitions

If you want a complete repository example, use the official-source bundle
documented in [Official Sensor Examples](official_sensor_examples.md).

## Step 1: Install

Install the package from the repository:

```bash
python3 -m pip install .
```

Optional extras:

| Extra | Use when |
| --- | --- |
| `.[docs]` | building the MkDocs site or GitHub Pages output |
| `.[knn]` | enabling the optional SciPy `cKDTree` backend |
| `.[knn-faiss]` | enabling the optional FAISS backend |
| `.[knn-pynndescent]` | enabling the optional PyNNDescent backend |
| `.[knn-scann]` | enabling the optional ScaNN backend on supported platforms |
| `.[internal-build]` | regenerating official examples or running retained SIAC-build tooling |
| `.[accel]` | using optional Rust-backed smoothing utilities |
| `.[dev]` | running tests and release tooling |

## Step 2: Get A Prepared Runtime

There are two ways to obtain a prepared runtime.

### Option A: Download a pre-built runtime (recommended)

Pre-built runtimes with 77,125 spectra and precomputed matrices for MODIS
Terra, Sentinel-2A, Landsat 8, and Landsat 9 are published as GitHub Release
assets.

```bash
spectral-library download-prepared-library \
  --output-root build/mapping_runtime
```

This fetches the latest release, verifies the SHA-256 digest, extracts the
runtime, and validates it.  You can pin a specific release with `--tag v0.2.0`
or point at any hosted tarball with `--url <URL>`.

### Option B: Build your own runtime

If you need different source sensors or a custom spectral library, build the
prepared runtime from a SIAC-style export:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --knn-index-backend faiss \
  --output-root build/mapping_runtime
```

What this does:

- loads and validates the sensor SRF definitions
- precomputes source-sensor simulation matrices
- optionally persists full-feature ANN indexes for supported backends
- writes row-aligned hyperspectral arrays and metadata
- emits `manifest.json`, `sensor_schema.json`, and `checksums.json`

The runtime format is documented in
[Prepared Runtime Contract](prepared_runtime_contract.md).

## Step 3: Validate The Runtime

Run validation before serving mappings or publishing a prepared runtime:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

Skip checksum hashing, but still require the full runtime layout:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime \
  --no-verify-checksums
```

## Step 4: Map A Single Sample

Map one source observation to a target sensor:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/source_reflectance.csv \
  --output-mode target_sensor \
  --k 10 \
  --output path/to/mapped_reflectance.csv
```

If you installed `.[knn]`, you can switch the shortlist search to SciPy's
`cKDTree` backend. This keeps the same distance metric, but can accelerate
larger searches, especially when many batch rows share the same valid-band
pattern:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/source_reflectance.csv \
  --output-mode target_sensor \
  --k 10 \
  --knn-backend scipy_ckdtree \
  --knn-eps 0.05 \
  --output path/to/mapped_reflectance.csv
```

`--knn-eps 0` keeps the tree search exact. Values above `0` allow approximate
shortlists.

Additional ANN backends use the same `--knn-backend` flag:

- `faiss`: FAISS HNSW search
- `pynndescent`: PyNNDescent approximate search
- `scann`: ScaNN approximate search

Those backends still re-rank the returned shortlist by exact RMS distance
before the estimator runs.

The result diagnostics also expose a heuristic `confidence_score` for the full
mapping and for each segment. It is based on valid-band coverage, neighbor
distances, source-space fit RMSE, and estimator weight concentration.

Current production interpretation policy:

- `high` / `accept`: `confidence_score >= 0.85`
- `medium` / `manual_review`: `0.60 <= confidence_score < 0.85`
- `low` / `reject`: `confidence_score < 0.60`

This policy is heuristic and conservative. It is useful for routing and QA, not
as a calibrated probability of correctness.

Reconstruct a full `400-2500 nm` spectrum instead:

```bash
spectral-library map-reflectance \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --input path/to/source_reflectance.csv \
  --output-mode full_spectrum \
  --k 10 \
  --output path/to/reconstructed_spectrum.csv
```

Single-sample input layouts:

| Layout | Columns |
| --- | --- |
| long | `band_id,reflectance[,valid]` |
| wide | one row with one column per source band, optional `valid_<band_id>` columns |

## Step 5: Map A Batch

Map many samples from one CSV:

```bash
spectral-library map-reflectance-batch \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --input path/to/source_reflectance_batch.csv \
  --output-mode target_sensor \
  --k 10 \
  --neighbor-estimator simplex_mixture \
  --output path/to/mapped_batch.csv \
  --diagnostics-output path/to/mapped_batch_diagnostics.json \
  --neighbor-review-output path/to/mapped_batch_neighbor_review.csv
```

Batch input layouts:

| Layout | Columns |
| --- | --- |
| long | `sample_id,band_id,reflectance[,valid][,exclude_row_id]` |
| wide | one row per sample, optional `sample_id`, optional `exclude_row_id`, one column per source band, optional `valid_<band_id>` columns |

Batch outputs:

| Output mode | Shape |
| --- | --- |
| `target_sensor` | `sample_id` plus one column per target band |
| `vnir_spectrum` / `swir_spectrum` / `full_spectrum` | `sample_id` plus `nm_<wavelength>` columns |

If only one target segment maps successfully, the missing target-band cells are
left blank in the batch CSV.

If `--neighbor-review-output` is set, the command also writes a long-form CSV
that records the retained neighbors, their distances, estimator weights, the
query band values, and the selected neighbors' source-band values for each
sample and segment.

If you are running a held-out evaluation against a shared prepared runtime, put
the exact prepared `row_id` to exclude in the optional `exclude_row_id`
column. The official MODIS/Sentinel-2/Landsat example uses this pattern so
each query removes only its own source row from the full library.

Pre-sorting the prepared rows by global spectral similarity is not a substitute
for a search backend. Retrieval runs independently in VNIR and NIR-SWIR
feature space, and can use different valid-band subsets per query, so a single
fixed row order does not make the actual KNN search cheaper.

## Step 6: Use The Python API

```python
from pathlib import Path

from spectral_library import SpectralMapper, prepare_mapping_library, validate_prepared_library

prepare_mapping_library(
    siac_root=Path("build/siac_library"),
    srf_root=Path("path/to/srfs"),
    output_root=Path("build/mapping_runtime"),
    source_sensors=["SENSOR_A"],
)

validate_prepared_library(Path("build/mapping_runtime"))

mapper = SpectralMapper(Path("build/mapping_runtime"))

result = mapper.map_reflectance(
    source_sensor="SENSOR_A",
    reflectance={"blue": 0.12, "nir": 0.34, "swir1": 0.27},
    output_mode="full_spectrum",
    k=10,
    exclude_row_ids=["source_id:spectrum_id:sample_name"],
)
```

For the stable imports and result objects, see
[Python API Reference](python_api_reference.md).

## Sensor SRF JSON Shape

Each file in `--srf-root` defines one sensor:

```json
{
  "sensor_id": "SENSOR_A",
  "bands": [
    {
      "band_id": "blue",
      "segment": "vnir",
      "wavelength_nm": [445.0, 450.0, 455.0],
      "rsr": [0.2, 1.0, 0.2]
    }
  ]
}
```

Band support must stay inside its declared segment:

- `vnir`: `400-1000 nm`
- `swir`: `800-2500 nm`

## Where To Go Next

- [Official Sensor Examples](official_sensor_examples.md)
  for end-to-end MODIS, Sentinel-2A, Landsat 8, and Landsat 9 runs
- [Mathematical Foundations](theory.md)
  for the retrieval equations behind the outputs
- [CLI Reference](cli_reference.md)
  for `--knn-index-backend`, ANN backend selection, and the full-library
  benchmark runner
- [CLI Reference](cli_reference.md)
  for every public command and flag
- [Prepared Runtime Contract](prepared_runtime_contract.md)
  for the stable on-disk runtime rules
