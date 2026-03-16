# CLI Reference

The public CLI entry point is:

```bash
spectral-library
```

## Global Options

| Option | Meaning |
| --- | --- |
| `--version` | print the package version and exit |
| `--json-errors` | emit machine-readable JSON errors to `stderr` |
| `--json-logs` | emit structured JSON log events to `stderr` |

## Public Commands

### `prepare-mapping-library`

Build the prepared runtime used by mapping.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--siac-root` | SIAC-style export root containing metadata and normalized spectra tables |
| `--srf-root` | directory containing sensor SRF JSON files |
| `--source-sensor` | one or more source sensor ids to precompute |
| `--output-root` | output directory for the prepared runtime |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--dtype` | floating-point output dtype, default `float32` |

### `validate-prepared-library`

Validate a prepared runtime root.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--prepared-root` | runtime root created by `prepare-mapping-library` |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--no-verify-checksums` | skip file hashing while still requiring the full runtime layout |

### `map-reflectance`

Map one sample from a source sensor to a target sensor or to a reconstructed
spectral output.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--prepared-root` | prepared runtime root |
| `--source-sensor` | source sensor id |
| `--input` | single-sample CSV |
| `--output-mode` | one of `target_sensor`, `vnir_spectrum`, `swir_spectrum`, `full_spectrum` |
| `--output` | output CSV path |

Conditionally required:

| Flag | Meaning |
| --- | --- |
| `--target-sensor` | required when `--output-mode target_sensor` |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--k` | nearest-neighbor count, default `10` |
| `--min-valid-bands` | minimum valid source bands per segment, default `1` |
| `--exclude-row-id` | exclude one or more prepared row ids from neighbor selection |
| `--exclude-sample-name` | exclude one or more prepared `sample_name` values from neighbor selection |

Single-sample input layouts:

| Layout | Columns |
| --- | --- |
| long | `band_id,reflectance[,valid]` |
| wide | one row with one source-band column per band id, optional `valid_<band_id>` columns |

Single-sample outputs:

| Output mode | CSV shape |
| --- | --- |
| `target_sensor` | `band_id,segment,reflectance` |
| spectral outputs | `wavelength_nm,reflectance` |

### `map-reflectance-batch`

Map many samples from one CSV.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--prepared-root` | prepared runtime root |
| `--source-sensor` | source sensor id |
| `--input` | batch CSV |
| `--output-mode` | public output mode |
| `--output` | output CSV path |

Conditionally required:

| Flag | Meaning |
| --- | --- |
| `--target-sensor` | required when `--output-mode target_sensor` |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--k` | nearest-neighbor count |
| `--min-valid-bands` | minimum valid source bands per segment |
| `--exclude-row-id` | exclude one or more prepared row ids for every batch sample |
| `--exclude-sample-name` | exclude one or more prepared `sample_name` values for every batch sample |
| `--self-exclude-sample-id` | exclude rows whose prepared `sample_name` matches each batch `sample_id` |
| `--diagnostics-output` | optional JSON summary path |

Batch input layouts:

| Layout | Columns |
| --- | --- |
| long | `sample_id,band_id,reflectance[,valid][,exclude_row_id]` |
| wide | `sample_id`, optional `exclude_row_id`, one column per source band, and optional `valid_<band_id>` columns |

If `exclude_row_id` is present in the batch CSV, the command excludes that one
exact prepared row for the corresponding sample before retrieval. This is the
recommended way to run held-out full-library examples without removing other
rows that share the same class or source.

Batch outputs:

| Output mode | CSV shape |
| --- | --- |
| `target_sensor` | one row per sample, `sample_id` plus one column per target band |
| spectral outputs | one row per sample, `sample_id` plus `nm_<wavelength>` columns |

### `benchmark-mapping`

Benchmark retrieval against the regression baseline.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--prepared-root` | prepared runtime root |
| `--source-sensor` | source sensor id |
| `--target-sensor` | target sensor id |
| `--report` | JSON output path |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--k` | nearest-neighbor count, default `10` |
| `--test-fraction` | held-out fraction, default `0.2` |
| `--random-seed` | split seed, default `0` |

## Error Behavior

Public commands provide:

- non-zero exit codes on failure
- human-readable errors by default
- JSON error envelopes with `--json-errors`
- newline-delimited JSON log events with `--json-logs`

The error envelope includes:

- `error_code`
- `message`
- `command`
- `context`

The JSON log envelope includes:

- `command`
- `event`
- `level`
- `timestamp`
- `context`

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Mathematical Foundations](theory.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)
- [Official Sensor Examples](official_sensor_examples.md)

## Internal Commands

The CLI still ships retained internal commands for source acquisition and SIAC
build workflows:

- `plan-matrix`
- `fetch-source`
- `fetch-batch`
- `assemble-database`
- `tidy-results`
- `normalize-sources`
- `plot-quality`
- `filter-coverage`
- `build-siac-library`

Those are documented in [Internal Build Pipeline](internal_build_pipeline.md)
because they are not part of the stable public mapping contract.
