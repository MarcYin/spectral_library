# Troubleshooting

This page covers common issues, their causes, and how to resolve them.

## Validation Failures

### `invalid_prepared_library`

**Symptom:** `validate-prepared-library` exits with a non-zero code and reports
missing files or inconsistent metadata.

**Common causes:**

| Cause | Fix |
| --- | --- |
| Incomplete prepare run | Re-run `build-mapping-library` from scratch |
| Missing `.npy` file for a declared source sensor | Re-run mapping or prepare; the runtime can synthesize missing matrices for supported canonical `rsrf` sensors |
| Checksum mismatch after manual file edits | Re-prepare the runtime; never edit runtime files by hand |
| Corrupt download or copy | Re-copy the runtime directory and validate again |

**Quick check:**

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime \
  --no-verify-checksums
```

If `--no-verify-checksums` passes but full validation fails, the layout is
correct but a file was modified after prepare.

### `prepared_library_incompatible`

**Symptom:** The mapper refuses to load a runtime built by an older package
version.

**Fix:** Re-prepare the runtime with the current package version. The schema
version is stored in `manifest.json` under `schema_version`.

| Package version | Schema version |
| --- | --- |
| 0.1.x | `1.0.0` |
| 0.2.x | `1.2.0` |
| current worktree | `3.1.0` |

## Mapping Errors

### `invalid_mapping_input`

**Symptom:** `map-reflectance` or the Python API raises `MappingInputError`.

**Common causes:**

- **Unknown sensor id:** The `--source-sensor` or `--target-sensor` value does
  not match any sensor defined in the prepared runtime's `sensor_schema.json`
  and cannot be resolved from `rsrf`. Check spelling and use canonical `rsrf`
  ids such as `sentinel-2b_msi`, `landsat-9_oli2`, or `snpp_viirs`.
- **Empty reflectance:** The input CSV has no valid reflectance values, or all
  bands are marked invalid.
- **No valid segment:** Both VNIR and SWIR segments have fewer valid bands than
  `--min-valid-bands` (default `1`). This usually means the input band ids do
  not match the sensor schema.

**Debugging steps:**

1. Inspect the sensor schema:
   ```bash
   cat build/mapping_runtime/sensor_schema.json | python3 -m json.tool
   ```
2. Verify your input band ids match the `band_id` values in the schema.
3. Check that reflectance values are numeric and in reflectance units
   (typically `0.0` to `1.0`).

### `invalid_sensor_schema`

**Symptom:** `build-mapping-library` fails while resolving a sensor schema.

**Common causes:**

- Missing required fields (`sensor_id`, `bands`, `band_id`, `segment`,
  `response_definition`)
- Duplicate `band_id` within one sensor
- `wavelength_nm` not strictly increasing
- No positive RSR values
- Positive RSR support outside the declared segment bounds
  (`vnir: 400-1000 nm`, `swir: 800-2500 nm`)
- `rsrf>=0.3.1` could not resolve its canonical runtime data. On first use,
  `rsrf` may need network access to populate its cache. In offline or mirrored
  environments, set `RSRF_ROOT` to a prepared `rsrf` runtime root or preseed
  `RSRF_CACHE_DIR`.

## Confidence Score Interpretation

The `confidence_score` in mapping diagnostics is a bounded heuristic between
`0.0` and `1.0`. It is **not** a calibrated probability.

### What the score combines

| Component | Measures |
| --- | --- |
| Valid-band coverage | Fraction of source bands available for retrieval |
| Mean neighbor distance | How close the query is to its nearest library spectra |
| Source-fit RMSE | How well the estimator reproduces the query in source-band space |
| Weight concentration | Whether one neighbor dominates (high concentration) or weights are spread |

### Production routing policy

| Label | Score range | Recommended action |
| --- | --- | --- |
| `high` / `accept` | `>= 0.85` | Use directly in downstream workflows |
| `medium` / `manual_review` | `0.60` to `0.85` | Inspect diagnostics; may be usable depending on application |
| `low` / `reject` | `< 0.60` | Do not use without manual verification |

### When scores are low

- **Few valid bands:** The query has missing bands, reducing retrieval
  information. Check that your input includes all expected bands.
- **Novel material:** The query spectrum may not be well-represented in the
  library. Consider whether the library coverage includes your target material.
- **Scale issues:** Reflectance values outside the `0.0-1.0` range can inflate
  distances. Verify your input is in reflectance units, not raw DN or
  percentage.

## KNN Backend Selection

The package supports five search backends. All produce the same final result
(re-ranked by exact RMS distance), but differ in speed characteristics.

| Backend | Install extra | Best for | Notes |
| --- | --- | --- | --- |
| `numpy` | *(included)* | Small runtimes (< 10k rows), debugging | No approximation; always exact |
| `scipy_ckdtree` | `.[knn]` | Medium runtimes (10k-100k rows) | Tree-based; `knn_eps=0` for exact, `> 0` for approximate |
| `faiss` | `.[knn-faiss]` | Large runtimes (> 100k rows) | HNSW index; supports persisted indexes |
| `pynndescent` | `.[knn-pynndescent]` | Large runtimes, pure Python | Approximate; supports persisted indexes |
| `scann` | `.[knn-scann]` | Very large runtimes on Linux x86_64 | Google ScaNN; supports persisted indexes |

### Persisted indexes

`faiss`, `pynndescent`, and `scann` support persisting ANN indexes at prepare
time with `--knn-index-backend`. Persisted indexes skip the index-build step
at query time, making the first query faster.

Persisted indexes are only used when:

- The query uses the **full** segment feature set (no missing bands)
- The query uses the **full** candidate pool (no per-query row exclusions)

Queries with missing bands or exclusions fall back to on-the-fly search
automatically.

### Choosing `knn_eps`

- `knn_eps=0`: Exact search. Use when result reproducibility matters.
- `knn_eps=0.01-0.05`: Mild approximation. Good default for production batch
  runs where small distance differences are acceptable.
- `knn_eps > 0.1`: Aggressive approximation. Only recommended for rapid
  prototyping or very large runtimes where speed is critical.

The final result is always re-ranked by exact distance, so moderate
approximation in the shortlist rarely changes the output.

## Common Setup Issues

### `ModuleNotFoundError: No module named 'spectral_library'`

Ensure the package is installed:

```bash
python3 -m pip install .
```

Or, if running from the repository without installing:

```bash
PYTHONPATH=src python3 -m spectral_library.cli --version
```

### Optional backend not found

If you see `ImportError` when using `--knn-backend scipy_ckdtree`:

```bash
python3 -m pip install "spectral-library[knn]"
```

Replace `knn` with the appropriate extra for your backend:

```bash
python3 -m pip install "spectral-library[knn-faiss]"
python3 -m pip install "spectral-library[knn-pynndescent]"
python3 -m pip install "spectral-library[knn-scann]"  # Linux x86_64 only
```

### ScaNN not available on macOS

ScaNN only supports Linux x86_64. On macOS, use `faiss` or `pynndescent`
instead.

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [CLI Reference](cli_reference.md)
- [Python API Reference](python_api_reference.md)
- [Mathematical Foundations](theory.md)
