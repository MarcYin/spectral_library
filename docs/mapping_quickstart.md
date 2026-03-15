# Mapping Quickstart

This guide covers the public package surface for `spectral-library`:

- preparing a mapping runtime,
- validating the prepared runtime,
- mapping one sample,
- mapping batches of samples,
- using the Python API.

Additional public docs:

- [`index.md`](index.md)
- [`official_sensor_examples.md`](official_sensor_examples.md)
- [`cli_reference.md`](cli_reference.md)
- [`python_api_reference.md`](python_api_reference.md)
- [`prepared_runtime_contract.md`](prepared_runtime_contract.md)

Published site:
[https://marcyin.github.io/spectral_library/](https://marcyin.github.io/spectral_library/)

## Package Identity

- distribution: `spectral-library`
- import package: `spectral_library`
- CLI: `spectral-library`

## Install

Install from this repository in a Python `3.9+` environment:

```bash
python3 -m pip install .
```

Supported Python versions in CI are:

- `3.9`
- `3.10`
- `3.11`
- `3.12`

Optional dependency groups:

- `.[internal-build]`
  for the retained normalization, plotting, and SIAC-build commands
- `.[docs]`
  for the static documentation site and GitHub Pages build path
- `.[accel]`
  for optional Rust-backed smoothing utilities used by internal scripts
- `.[dev]`
  for test and release tooling

## Prepare A Mapping Runtime

Build a prepared runtime from a SIAC export plus one or more sensor SRF JSON
definitions:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_library \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

## Validate The Prepared Runtime

Validate the runtime root and verify checksums:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

Skip file hashing but still require the full runtime layout:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime \
  --no-verify-checksums
```

## Map One Sample

Map source reflectance to a target sensor:

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

Single-sample input CSV formats:

- long format with `band_id,reflectance[,valid]`
- wide format with one row and one column per source band id

## Map A Batch

Map multiple samples from one CSV:

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

Batch input CSV formats:

- long format with `sample_id,band_id,reflectance[,valid]`
- wide format with one row per sample, optional `sample_id`, one column per
  source band id, and optional `valid_<band_id>` columns

Batch output CSV formats:

- `target_sensor`: one row per sample, `sample_id` plus one column per target
  band
- `vnir_spectrum`, `swir_spectrum`, `full_spectrum`: one row per sample,
  `sample_id` plus `nm_<wavelength>` columns

If a target-sensor batch sample only maps one segment successfully, unmapped
target-band columns are left blank for that sample.

## Benchmark Mapping

Benchmark the retrieval workflow against the regression baseline:

```bash
spectral-library benchmark-mapping \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --report path/to/benchmark.json
```

## Python API

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

single_result = mapper.map_reflectance(
    source_sensor="SENSOR_A",
    reflectance={"blue": 0.12, "nir": 0.34, "swir1": 0.27},
    output_mode="full_spectrum",
    k=10,
)

batch_result = mapper.map_reflectance_batch(
    source_sensor="SENSOR_A",
    sample_ids=["sample_a", "sample_b"],
    reflectance_rows=[
        {"blue": 0.12, "nir": 0.34, "swir1": 0.27},
        {"blue": 0.18, "nir": 0.29, "swir1": 0.22},
    ],
    output_mode="target_sensor",
    target_sensor="SENSOR_B",
    k=10,
)
```

## Sensor SRF JSON

Each sensor JSON file in `--srf-root` should define one sensor:

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

## Prepared Runtime Layout

The prepared runtime root is the public on-disk contract used by mapping:

- `manifest.json`
- `mapping_metadata.parquet`
- `hyperspectral_vnir.npy`
- `hyperspectral_swir.npy`
- `source_<sensor_id>_vnir.npy`
- `source_<sensor_id>_swir.npy`
- `sensor_schema.json`
- `checksums.json`

## Compatibility Policy

The stable `1.x` public contract covers:

- the Python API entry points documented in this guide
- the CLI commands documented in this guide
- the prepared-runtime schema version and required root layout
- the stable output modes:
  `target_sensor`, `vnir_spectrum`, `swir_spectrum`, `full_spectrum`

Additive optional parameters and additive manifest fields are allowed in minor
releases. Renames, removals, required-parameter additions, or schema-breaking
prepared-runtime changes require a major version.
