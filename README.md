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

## Start Here

- public install and usage guide:
  [`docs/mapping_quickstart.md`](docs/mapping_quickstart.md)
- internal SIAC build-system reference:
  [`docs/internal_build_pipeline.md`](docs/internal_build_pipeline.md)
- mapping design:
  [`docs/spectral_mapping_usage_plan.md`](docs/spectral_mapping_usage_plan.md)
- production release design:
  [`docs/production_release_standard_plan.md`](docs/production_release_standard_plan.md)

## Install

Install from this repository in a Python `3.9+` environment:

```bash
python3 -m pip install .
```

Supported Python versions in CI are `3.9`, `3.10`, `3.11`, and `3.12`.

For the retained internal normalization and QA commands, install the optional
internal-build dependencies:

```bash
python3 -m pip install ".[internal-build]"
```

## Minimal Example

Prepare a mapping runtime:

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
