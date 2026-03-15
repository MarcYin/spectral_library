# CLI Reference

The public CLI entry point is:

```bash
spectral-library
```

Global options:

- `--version`
  print the package version and exit
- `--json-errors`
  emit machine-readable JSON errors to `stderr`

## Public Mapping Commands

### `prepare-mapping-library`

Build the prepared runtime used by mapping.

Required arguments:

- `--siac-root`
  path to the SIAC-style export with
  `tabular/siac_spectra_metadata.csv` and
  `tabular/siac_normalized_spectra.csv`
- `--srf-root`
  directory containing one or more sensor SRF JSON files
- `--source-sensor`
  one or more source sensor ids to precompute as source-query arrays
- `--output-root`
  output directory for the prepared runtime

Optional arguments:

- `--dtype`
  floating-point array dtype, default `float32`

### `validate-prepared-library`

Validate a prepared runtime root.

Required arguments:

- `--prepared-root`
  runtime root created by `prepare-mapping-library`

Optional arguments:

- `--no-verify-checksums`
  skip file hashing while still requiring the full runtime layout

### `map-reflectance`

Map one sample from a source sensor either to a target sensor or to a
reconstructed spectral output.

Required arguments:

- `--prepared-root`
- `--source-sensor`
- `--input`
- `--output-mode`
  one of `target_sensor`, `vnir_spectrum`, `swir_spectrum`,
  `full_spectrum`
- `--output`

Conditionally required arguments:

- `--target-sensor`
  required when `--output-mode target_sensor`

Optional arguments:

- `--k`
  number of nearest neighbors, default `10`
- `--min-valid-bands`
  minimum valid source bands per segment, default `1`

Input CSV layouts:

- long format:
  `band_id,reflectance[,valid]`
- wide format:
  one row, one source-band column per band id, optional `valid_<band_id>`
  columns

Output CSV layouts:

- `target_sensor`:
  `band_id,segment,reflectance`
- spectral outputs:
  `wavelength_nm,reflectance`

### `map-reflectance-batch`

Map many samples from one CSV.

Required arguments:

- `--prepared-root`
- `--source-sensor`
- `--input`
- `--output-mode`
- `--output`

Conditionally required arguments:

- `--target-sensor`
  required when `--output-mode target_sensor`

Optional arguments:

- `--k`
- `--min-valid-bands`
- `--diagnostics-output`
  JSON summary output for the batch

Input CSV layouts:

- long format:
  `sample_id,band_id,reflectance[,valid]`
- wide format:
  `sample_id` plus one column per source band and optional
  `valid_<band_id>` columns

Output CSV layouts:

- `target_sensor`:
  one row per sample with `sample_id` plus one column per target band
- spectral outputs:
  one row per sample with `sample_id` plus `nm_<wavelength>` columns

### `benchmark-mapping`

Benchmark retrieval-based mapping against the regression baseline.

Required arguments:

- `--prepared-root`
- `--source-sensor`
- `--target-sensor`
- `--report`

Optional arguments:

- `--k`
  default `10`
- `--test-fraction`
  default `0.2`
- `--random-seed`
  default `0`

## Public Input Contracts

Sensor JSON files are documented in [Mapping Quickstart](mapping_quickstart.md)
and [Prepared Runtime Contract](prepared_runtime_contract.md).

The mathematical model behind the CLI outputs is documented in
[Mathematical Foundations](theory.md).

Example inputs and outputs using official MODIS, Sentinel-2A, Landsat 8, and
Landsat 9 responses are in
[Official Sensor Examples](official_sensor_examples.md).

## Retained Internal Commands

The CLI still ships internal data-acquisition and SIAC-build commands:

- `plan-matrix`
- `fetch-source`
- `fetch-batch`
- `assemble-database`
- `tidy-results`
- `normalize-sources`
- `plot-quality`
- `filter-coverage`
- `build-siac-library`

Those commands are documented in
[Internal Build Pipeline](internal_build_pipeline.md) because they are not part
of the stable public mapping contract.
