# Getting Started

This guide is the fastest path from installation to a working mapping run.

It covers the public package workflow:

1. install the package
2. prepare a runtime
3. validate the runtime
4. map one sample
5. map a batch
6. call the Python API directly

Repository package layout:

- `spectral_library.mapping`
  public mapping runtime, prepared-runtime build, and retrieval engine
- `spectral_library.distribution`
  runtime download helpers
- `spectral_library.sources`
  maintainer-oriented source acquisition and catalog tools
- `spectral_library.normalization`
  maintainer-oriented normalization and SIAC package-export tools

For public usage, the main entry points are the root `spectral_library` imports,
the `spectral_library.mapping` package, and `spectral_library.distribution`.

## Before You Start

You need:

- Python `3.9` or newer
- a SIAC-style spectral export with:
  - `tabular/siac_spectra_metadata.csv`
  - `tabular/siac_normalized_spectra.csv`
- canonical `rsrf` sensor ids for the sensors you want to precompute

If you want a complete repository example, use the official-source bundle
documented in [Official Sensor Examples](official_sensor_examples.md).

## Step 1: Install

Install the package from the repository:

```bash
python3 -m pip install .
```

Built-in canonical sensors now require `rsrf>=0.3.1`. On first use, `rsrf`
can bootstrap the matching canonical runtime data into its local cache
automatically. Set `RSRF_ROOT` only when you want to override that with a local
checkout or mirrored runtime root, and preseed `RSRF_CACHE_DIR` if you need the
same flow in an offline environment.

Optional extras:

| Extra | Use when |
| --- | --- |
| `.[docs]` | building the MkDocs site or GitHub Pages output |
| `.[knn]` | enabling the optional SciPy `cKDTree` backend |
| `.[knn-faiss]` | enabling the optional FAISS backend |
| `.[knn-pynndescent]` | enabling the optional PyNNDescent backend |
| `.[knn-scann]` | enabling the optional ScaNN backend on supported platforms |
| `.[zarr]` | writing chunked batch outputs to a Zarr store |
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
runtime, and validates it. You can pin a specific release with `--tag v0.6.3`
or point at any hosted tarball with `--url <URL>`.

### Option B: Build your own runtime

If you need different source sensors or a custom spectral library, build the
prepared runtime from a SIAC-style export:

```bash
spectral-library build-mapping-library \
  --siac-root build/siac_library \
  --source-sensor sentinel-2b_msi \
  --source-sensor snpp_viirs \
  --knn-index-backend faiss \
  --output-root build/mapping_runtime
```

If you have custom sensor definitions that are not available from `rsrf`, add
them with `--srf-root path/to/srfs`. Each file in that directory must be a
valid `rsrf_sensor_definition` JSON document. Band-level segment metadata must
be declared in `extensions.spectral_library.segment`.

What this does:

- resolves requested built-in sensors from `rsrf` and validates any extra local SRF JSON definitions
- precomputes source-sensor simulation matrices
- optionally persists full-feature ANN indexes for supported backends
- writes row-aligned hyperspectral arrays and metadata
- emits `manifest.json`, `sensor_schema.json`, and `checksums.json`

The runtime format is documented in
[Prepared Runtime Contract](prepared_runtime_contract.md).

Mapping commands can also fetch the default published runtime on demand. If you
omit `--prepared-root` from `map-reflectance`, `map-reflectance-batch`, or
`benchmark-mapping`, the CLI resolves the package-matched published runtime
into the user cache and reuses it on later runs. Keep using `--prepared-root`
when you want to point at a locally built or otherwise custom runtime.

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

By default, the mapping commands stay on the fast output path and do not build
the rich diagnostic payloads. If you request `--diagnostics-output` or
`--neighbor-review-output`, the CLI switches to debug mode and includes a
heuristic `confidence_score` for the full mapping and for each segment. It is
based on valid-band coverage, neighbor distances, source-space fit RMSE, and
estimator weight concentration.

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

??? example "Sample single-sample CSV (long format)"

    ```csv
    band_id,reflectance
    blue,0.0836
    green,0.1036
    red,0.0656
    nir,0.3978
    swir1,0.0956
    swir2,0.0350
    ```

??? example "Sample single-sample CSV (wide format)"

    ```csv
    blue,green,red,nir,swir1,swir2
    0.0836,0.1036,0.0656,0.3978,0.0956,0.0350
    ```

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
  --output path/to/mapped_batch.csv
```

Batch input layouts:

| Layout | Columns |
| --- | --- |
| long | `sample_id,band_id,reflectance[,valid][,exclude_row_id]` |
| wide | one row per sample, optional `sample_id`, optional `exclude_row_id`, one column per source band, optional `valid_<band_id>` columns |

??? example "Sample batch CSV (wide format)"

    ```csv
    sample_id,blue,green,red,nir,swir1,swir2
    blue_spruce_needles,0.0836,0.1036,0.0656,0.3978,0.0956,0.0350
    pale_brown_silty_loam,0.0937,0.1935,0.2837,0.3678,0.4842,0.4543
    tap_water,0.0279,0.0270,0.0265,0.0262,0.0210,0.0189
    asphalt_road,0.0731,0.0883,0.1032,0.1266,0.1951,0.2139
    ```

Batch outputs:

| Output mode | Shape |
| --- | --- |
| `target_sensor` | `sample_id` plus one column per target band |
| `vnir_spectrum` / `swir_spectrum` / `full_spectrum` | `sample_id` plus `nm_<wavelength>` columns |

If only one target segment maps successfully, the missing target-band cells are
left blank in the batch CSV.

If you need neighbor review or per-segment diagnostics for a batch run, add
`--diagnostics-output` and/or `--neighbor-review-output`. That forces the rich
debug path for the batch command.

For large spectral outputs, write directly to a Zarr store instead of emitting a
very wide CSV:

```bash
spectral-library map-reflectance-batch \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --input path/to/source_reflectance_batch.csv \
  --output-mode full_spectrum \
  --output-format zarr \
  --output-chunk-size 4096 \
  --output path/to/reconstructed_batch.zarr
```

The Zarr store contains `reflectance`, `sample_id`, `source_fit_rmse`, and
either `wavelength_nm` or `band_id`, depending on the selected output mode.
Chunked Zarr output currently does not emit the JSON diagnostics file or the
neighbor-review CSV.
The CLI streams wide input directly and also streams long-format input when rows
for each `sample_id` stay grouped together. Interleaved long-format rows fall
back to the materialized loader so the public semantics remain unchanged.

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

from spectral_library import SpectralMapper, build_mapping_library, validate_prepared_library

build_mapping_library(
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

For raster-scale production runs, compile a fixed dense mapper once and reuse it
across many chunks:

```python
import numpy as np

linear_mapper = mapper.compile_linear_mapper(
    source_sensor="SENSOR_A",
    target_sensor="SENSOR_B",
    output_mode="target_sensor",
    dtype="float32",
)

input_block = np.asarray(
    [
        [0.12, 0.34, 0.27],
        [0.10, 0.31, 0.25],
    ],
    dtype=np.float32,
)
mapped_block = linear_mapper.map_array(input_block, chunk_size=65536)
```

For large batch outputs that should stay on disk, use the Zarr export path:

```python
mapper.map_reflectance_batch_to_zarr(
    zarr_path=Path("build/reconstructed_batch.zarr"),
    source_sensor="SENSOR_A",
    reflectance_rows=input_block,
    output_mode="target_sensor",
    target_sensor="SENSOR_B",
    chunk_size=65536,
)
```

For the stable imports and result objects, see
[Python API Reference](python_api_reference.md).

## Sensor SRF JSON Shape

Use this only for custom sensors that are not already provided by `rsrf`. Each
file in `--srf-root` defines one `rsrf_sensor_definition` sensor document:

```json
{
  "schema_type": "rsrf_sensor_definition",
  "schema_version": "1.0.0",
  "sensor_id": "SENSOR_A",
  "bands": [
    {
      "band_id": "blue",
      "response_definition": {
        "kind": "sampled",
        "wavelength_nm": [445.0, 450.0, 455.0],
        "response": [0.2, 1.0, 0.2]
      },
      "extensions": {
        "spectral_library": {
          "segment": "vnir"
        }
      }
    },
    {
      "band_id": "green",
      "response_definition": {
        "kind": "band_spec",
        "center_wavelength_nm": 560.0,
        "fwhm_nm": 20.0
      },
      "extensions": {
        "spectral_library": {
          "segment": "vnir"
        }
      }
    }
  ]
}
```

Band support must stay inside the segment declared in
`extensions.spectral_library.segment`:

- `vnir`: `400-1000 nm`
- `swir`: `800-2500 nm`

## Where To Go Next

- [Official Sensor Examples](official_sensor_examples.md)
  for end-to-end MODIS, Sentinel-2A, Landsat 8, and Landsat 9 runs
- [Mathematical Foundations](theory.md)
  for the retrieval equations behind the outputs
- [CLI Reference](cli_reference.md)
  for every public command, flag, ANN backend selection, and the full-library
  benchmark runner
- [Python API Reference](python_api_reference.md)
  for stable imports, result objects, and error types
- [Prepared Runtime Contract](prepared_runtime_contract.md)
  for the stable on-disk runtime rules
- [Troubleshooting](troubleshooting.md)
  for common issues, confidence score interpretation, and backend selection
