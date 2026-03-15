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
| `.[internal-build]` | regenerating official examples or running retained SIAC-build tooling |
| `.[accel]` | using optional Rust-backed smoothing utilities |
| `.[dev]` | running tests and release tooling |

## Step 2: Prepare A Runtime

Build the prepared runtime once for the source sensors you care about.

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

What this does:

- loads and validates the sensor SRF definitions
- precomputes source-sensor simulation matrices
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
  --output path/to/mapped_batch.csv \
  --diagnostics-output path/to/mapped_batch_diagnostics.json
```

Batch input layouts:

| Layout | Columns |
| --- | --- |
| long | `sample_id,band_id,reflectance[,valid]` |
| wide | one row per sample, optional `sample_id`, one column per source band, optional `valid_<band_id>` columns |

Batch outputs:

| Output mode | Shape |
| --- | --- |
| `target_sensor` | `sample_id` plus one column per target band |
| `vnir_spectrum` / `swir_spectrum` / `full_spectrum` | `sample_id` plus `nm_<wavelength>` columns |

If only one target segment maps successfully, the missing target-band cells are
left blank in the batch CSV.

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
- `swir`: `900-2500 nm`

## Where To Go Next

- [Official Sensor Examples](official_sensor_examples.md)
  for end-to-end MODIS, Sentinel-2A, Landsat 8, and Landsat 9 runs
- [Mathematical Foundations](theory.md)
  for the retrieval equations behind the outputs
- [CLI Reference](cli_reference.md)
  for every public command and flag
- [Prepared Runtime Contract](prepared_runtime_contract.md)
  for the stable on-disk runtime rules
