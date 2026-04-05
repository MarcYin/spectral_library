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
| `--source-sensor` | one or more source sensor ids to precompute |
| `--output-root` | output directory for the prepared runtime |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--dtype` | floating-point output dtype, default `float32` |
| `--srf-root` | when provided, load extra local SRF JSON definitions alongside built-in `rsrf` sensors; each custom band must provide an `rsrf` `response_definition`, and legacy top-level sampled-band payloads are rejected |
| `--knn-index-backend` | optionally persist ANN indexes for `faiss`, `pynndescent`, or `scann` during prepare |

When you rely on built-in `rsrf` sensors instead of local JSON definitions, the
runtime environment must also expose the `rsrf` registry data. If your `rsrf`
install does not ship it, set `RSRF_ROOT` to an `rsrf` checkout before running
`prepare-mapping-library`.

### `download-prepared-library`

Download a pre-built prepared runtime from a GitHub Release or a direct URL.

Required arguments:

| Flag | Meaning |
| --- | --- |
| `--output-root` | local directory to extract the runtime into |

Optional arguments:

| Flag | Meaning |
| --- | --- |
| `--url` | direct URL to a `.tar.gz` runtime archive (skips GitHub Release lookup) |
| `--tag` | GitHub Release tag to download from (e.g. `v0.4.0`); defaults to latest |
| `--sha256` | expected SHA-256 hex digest for the archive |
| `--no-verify` | skip runtime validation after extraction |

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
| `--neighbor-estimator` | `mean`, `distance_weighted_mean`, or `simplex_mixture`, default `mean` |
| `--knn-backend` | shortlist search backend: `numpy`, `scipy_ckdtree`, `faiss`, `pynndescent`, or `scann`; default `numpy` |
| `--knn-eps` | approximation slack / search-accuracy knob for supported ANN backends; `0` keeps `scipy_ckdtree` exact |
| `--exclude-row-id` | exclude one or more prepared row ids from neighbor selection |
| `--exclude-sample-name` | exclude one or more prepared `sample_name` values from neighbor selection |
| `--diagnostics-output` | optional JSON summary path |
| `--neighbor-review-output` | optional CSV path with one row per segment-neighbor review record |

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

If `--neighbor-review-output` is set, the command also writes a CSV review table
with one row per segment-neighbor pair. It includes the sample id, segment
status, neighbor rank, distance, estimator weight, query band values, and the
selected neighbor's source-band values. The review rows also record
`knn_backend` and `knn_eps`.

The JSON diagnostics also include a heuristic `confidence_score` at the overall
mapping level and per segment. It is derived from valid-band coverage, neighbor
distances, estimator weight concentration, and source-space fit RMSE. Treat it
as a ranking aid, not a calibrated probability. The diagnostics also include
`confidence_policy` with the current routing thresholds:

- `high` / `accept`: `>= 0.85`
- `medium` / `manual_review`: `>= 0.60` and `< 0.85`
- `low` / `reject`: `< 0.60`

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
| `--neighbor-estimator` | `mean`, `distance_weighted_mean`, or `simplex_mixture` |
| `--knn-backend` | shortlist search backend: `numpy`, `scipy_ckdtree`, `faiss`, `pynndescent`, or `scann` |
| `--knn-eps` | approximation slack / search-accuracy knob for supported ANN backends |
| `--exclude-row-id` | exclude one or more prepared row ids for every batch sample |
| `--exclude-sample-name` | exclude one or more prepared `sample_name` values for every batch sample |
| `--self-exclude-sample-id` | exclude rows whose prepared `sample_name` matches each batch `sample_id` |
| `--diagnostics-output` | optional JSON summary path |
| `--neighbor-review-output` | optional CSV path with one row per sample-segment-neighbor review record |

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

If `--neighbor-review-output` is set, the command writes a long-form CSV with
one row per sample, segment, and retained neighbor. This is the easiest public
way to audit which candidates were shortlisted and how the estimator reweighted
them.

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
| `--max-test-rows` | optional cap on held-out rows, default `0` in the CLI and `None` in Python |
| `--random-seed` | split seed, default `0` |
| `--neighbor-estimator` | retrieval estimator to benchmark: `mean`, `distance_weighted_mean`, or `simplex_mixture`; default `mean` |
| `--knn-backend` | shortlist search backend: `numpy`, `scipy_ckdtree`, `faiss`, `pynndescent`, or `scann`; default `numpy` |
| `--knn-eps` | approximation slack / search-accuracy knob for supported ANN backends |

Optional backend install extras:

```bash
python3 -m pip install "spectral-library[knn]"
python3 -m pip install "spectral-library[knn-faiss]"
python3 -m pip install "spectral-library[knn-pynndescent]"
python3 -m pip install "spectral-library[knn-scann]"
```

## Benchmark Automation

For larger prepared runtimes, the repository also ships a multi-scenario
benchmark runner:

```bash
PYTHONPATH=src python scripts/run_full_library_benchmarks.py \
  --prepared-root path/to/prepared_runtime \
  --neighbor-estimator simplex_mixture \
  --knn-backend numpy \
  --k 10 \
  --max-test-rows 512 \
  --output-root build/full-library-benchmarks \
  --thresholds benchmarks/default_thresholds.json \
  --fail-on-thresholds
```

It writes per-scenario JSON reports under `runs/`, plus aggregate
`summary.csv`, `summary.json`, and `reports.json`.

For non-`numpy` backends, the runner also records a same-scenario comparison to
the exact `numpy` baseline and can fail when backend drift exceeds the
`baseline_deltas` thresholds in `benchmarks/default_thresholds.json`.

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | General failure (invalid input, runtime error, or validation failure) |
| `2` | Argument parsing error (missing required flags, unknown options) |

All non-zero exits produce a human-readable message on `stderr`. Use
`--json-errors` for machine-readable error envelopes.

## Error Codes

Each structured error carries an `error_code` field for programmatic handling:

| `error_code` | Raised by | Meaning |
| --- | --- | --- |
| `invalid_sensor_schema` | `prepare-mapping-library` | Sensor schema data from `rsrf` or local JSON is malformed, unsupported, or violates band rules |
| `prepare_failed` | `prepare-mapping-library` | Runtime build failed (missing SIAC files, array errors) |
| `invalid_prepared_library` | `validate-prepared-library`, `map-reflectance` | Runtime layout or checksums are invalid |
| `prepared_library_incompatible` | `validate-prepared-library`, `map-reflectance` | Runtime schema version is not supported by this package version |
| `invalid_mapping_input` | `map-reflectance`, `map-reflectance-batch` | Input reflectance is invalid, sensor not found, or no valid segments |

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
- `elapsed_ms` on completion and failure events

Event values:

- `command_started`
- `command_completed`
- `command_failed`

Level values:

- `info` for start and completion
- `error` for failures

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [Mathematical Foundations](theory.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)
- [Official Sensor Examples](official_sensor_examples.md)

## Internal Commands

Source acquisition and library-packaging workflows are intentionally separated from
the public mapping CLI. Maintainers should use:

- `spectral-library-internal --help`

That maintainer-only entrypoint exposes the retained internal commands:

- `plan-matrix`
- `fetch-source`
- `fetch-batch`
- `assemble-database`
- `tidy-results`
- `normalize-sources`
- `plot-quality`
- `filter-coverage`
- `build-library-package`

Legacy direct invocation through `spectral-library <internal-command>` still
works for repository automation, but those commands are hidden from the public
help and are not part of the stable public mapping contract. See
[Internal Build Pipeline](internal_build_pipeline.md) for the maintained
workflow.
