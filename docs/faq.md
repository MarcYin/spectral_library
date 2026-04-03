# Frequently Asked Questions

## General

### What does this package do?

`spectral-library` maps reflectance between sensors and reconstructs
hyperspectral spectra. Given an observation from one sensor (e.g., Landsat 8),
it retrieves the most similar spectra from a hyperspectral library and uses
them to predict what another sensor (e.g., Sentinel-2A) would observe, or to
reconstruct the full 400-2500 nm spectrum.

### Is this a machine learning model?

No. The package uses nearest-neighbor retrieval from a spectral library, not a
trained regression or neural network. The library spectra serve as a physical
prior. An optional linear regression baseline is included only for
benchmarking, not for production mapping.

### What sensors are supported?

Any sensor whose spectral response function (SRF) can be expressed as
wavelength/RSR pairs on the 400-2500 nm range. The package ships official
examples for MODIS Terra, Sentinel-2A MSI, Landsat 8 OLI, and Landsat 9 OLI,
but you can define any sensor by writing an SRF JSON file.

### What spectral range is covered?

The canonical grid spans **400-2500 nm at 1 nm spacing** (2101 bands). This
covers visible, near-infrared, and shortwave infrared. The grid is split into
two retrieval segments:

- VNIR: 400-1000 nm
- SWIR: 800-2500 nm

The 800-1000 nm overlap ensures continuity across the detector transition.

## Prepared Runtimes

### How large is a prepared runtime?

Size depends on the number of library rows and source sensors. For a library
with ~30,000 rows and four source sensors, expect roughly 1-2 GB. The
hyperspectral arrays are the largest files.

### Can I share a prepared runtime between machines?

Yes. The runtime is a self-contained directory of `.npy` arrays, `.parquet`
metadata, and JSON configuration. Copy the entire directory and validate it on
the target machine:

```bash
spectral-library validate-prepared-library --prepared-root path/to/runtime
```

### Do I need to rebuild runtimes when I upgrade the package?

Only when the schema version changes. Check the
[migration guide](migration.md) for version-specific instructions.

## Mapping

### When should I use `target_sensor` vs `full_spectrum`?

- **`target_sensor`:** You need reflectance values at the exact bands of a
  specific target sensor. This is the most common use case for cross-sensor
  harmonization.
- **`full_spectrum`:** You need the continuous 400-2500 nm spectrum. Useful for
  spectral analysis, visualization, or feeding into models that expect
  hyperspectral input.
- **`vnir_spectrum` / `swir_spectrum`:** You only need one spectral segment.

### What does `k` control?

`k` is the number of nearest neighbors retrieved from the library. Larger `k`
produces smoother, more averaged reconstructions. Smaller `k` preserves more
spectral detail but increases sensitivity to individual library spectra.

The default `k=10` is a reasonable starting point. Values between 5 and 20
cover most use cases.

### What does `exclude_row_id` do?

It removes specific library rows from the neighbor search. The primary use case
is held-out evaluation: if your query spectrum is also in the library, exclude
it so the retrieval does not trivially find itself.

### Can I map from Sensor A to Sensor A?

Yes. This is valid and useful for quality assessment. If you map a sensor's
observation back to itself, the result should closely match the input. Large
deviations indicate the query is not well-represented in the library.

## Performance

### How fast is mapping?

Single-sample mapping with `numpy` backend completes in milliseconds for
typical library sizes. Batch mapping scales linearly with sample count. For
runtimes with > 100k rows, consider `scipy_ckdtree` or `faiss` to accelerate
the neighbor search.

### Which KNN backend should I use?

See the [backend selection guide](troubleshooting.md#knn-backend-selection) in
the troubleshooting page for a detailed comparison table.

**Quick rule of thumb:**

- **< 10k rows:** `numpy` (default, no dependencies)
- **10k-100k rows:** `scipy_ckdtree` with `knn_eps=0`
- **> 100k rows:** `faiss` with a persisted index

## Data Requirements

### What format does the SIAC root need to be in?

The SIAC root must contain:

- `tabular/siac_spectra_metadata.csv` - row metadata
- `tabular/siac_normalized_spectra.csv` - spectra on the 400-2500 nm grid with
  `nm_400`, `nm_401`, ..., `nm_2500` columns

### Can I use my own spectral library?

Yes, as long as it is formatted as a SIAC-style export. The key requirement is
that spectra are sampled on the canonical 1 nm grid from 400-2500 nm. See the
[internal build pipeline](internal_build_pipeline.md) for how the official
library is assembled from multiple sources.

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [CLI Reference](cli_reference.md)
- [Python API Reference](python_api_reference.md)
