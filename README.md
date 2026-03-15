# Spectral Library

`spectral-library` is a Python package and CLI for:

1. building curated SIAC spectral-library exports from heterogeneous source
   datasets,
2. preparing row-aligned runtime artifacts for spectral mapping,
3. mapping source-sensor reflectance to target sensors or reconstructed
   hyperspectral outputs with a retrieval-based workflow.

The public names for the current package are:

- distribution: `spectral-library`
- import package: `spectral_library`
- CLI: `spectral-library`

## Quickstart

Install the package from this repository in a Python `3.9+` environment:

```bash
python3 -m pip install .
```

When the package is published, the distribution name is `spectral-library`.

Prepare a mapping runtime from a SIAC export plus one or more sensor SRF JSON
definitions:

```bash
spectral-library prepare-mapping-library \
  --siac-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37 \
  --srf-root path/to/srfs \
  --source-sensor SENSOR_A \
  --output-root build/mapping_runtime
```

Validate the prepared runtime and verify its checksums:

```bash
spectral-library validate-prepared-library \
  --prepared-root build/mapping_runtime
```

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

Benchmark retrieval-based mapping against a regression baseline:

```bash
spectral-library benchmark-mapping \
  --prepared-root build/mapping_runtime \
  --source-sensor SENSOR_A \
  --target-sensor SENSOR_B \
  --report path/to/benchmark.json
```

The `map-reflectance` input CSV accepts either:

- long format with `band_id,reflectance[,valid]`
- wide format with one row and one column per source band id

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
result = mapper.map_reflectance(
    source_sensor="SENSOR_A",
    reflectance={"blue": 0.12, "nir": 0.34, "swir1": 0.27},
    output_mode="full_spectrum",
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

The prepared runtime root contains the public on-disk contract for mapping:

- `manifest.json`
- `mapping_metadata.parquet`
- `hyperspectral_vnir.npy`
- `hyperspectral_swir.npy`
- `source_<sensor_id>_vnir.npy`
- `source_<sensor_id>_swir.npy`
- `sensor_schema.json`
- `checksums.json`

## Internal Build System

The repository also retains the internal end-to-end SIAC build pipeline used to
produce the canonical library export. That build system:

1. fetches or reuses raw source assets,
2. normalizes them onto a shared `400-2500 nm` / `1 nm` reflectance grid,
3. applies source-aware QA, filtering, and repair stages,
4. exports a SIAC package with spectra, metadata, prototypes, diagnostics, and
   review plots.

## Canonical Build

The current canonical build kept in `build/` is:

- pipeline root:
  - `build/real_siac_pipeline_full_raw`
- final SIAC package:
  - `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`

Current package summary from the latest end-to-end run:

- total exported spectra: `77,125`
- labeled spectra: `71,193`
- unlabeled spectra: `5,932`
- package source rows: `28`
- exported spectral sources: `26`
- excluded source IDs: `ghisacasia_v001`, `understory_estonia_czech`
- excluded individual spectra: `37` `santa_barbara_urban_reflectance` spectra
- suspicious spectra after the final review pass: `187` (`0.24%`)

Latest top-level QA outputs:

- normalized QA:
  - `build/real_siac_pipeline_full_raw/11_source_artifacts_fixed/plots/quality`
- landcover QA:
  - `build/real_siac_pipeline_full_raw/11_source_artifacts_fixed/landcover_analysis/plots`
- SIAC package plots:
  - `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/plots`
- suspicious-spectrum review:
  - `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/full_review`

## Repository Structure

Important repo paths:

- `manifests/sources.csv`
  - curated source inventory, fetch adapters, status, landing URLs, and notes
- `manifests/siac_excluded_spectra.csv`
  - export-time spectrum-level exclusions
- `src/spectral_library/`
  - reusable fetch, normalize, export, and CLI code
- `scripts/build_real_siac_library_from_scratch.py`
  - canonical full raw-to-SIAC build driver
- `scripts/build_real_siac_library.py`
  - cache-first wrapper around the canonical full build driver
- `scripts/`
  - source-specific repair, filtering, plotting, and review stages
- `build/`
  - raw caches, latest pipeline outputs, and latest exported SIAC package

## End-to-End Workflow

The canonical end-to-end driver is:

```bash
MPLCONFIGDIR=build/.mplconfig PYTHONPATH=src python3 scripts/build_real_siac_library_from_scratch.py \
  --fallback-raw-roots build/local_sources_full_raw,build/local_sources_vegetation_all,build/local_sources \
  --raw-sources-root build/local_sources_full_raw \
  --pipeline-root build/real_siac_pipeline_full_raw \
  --output-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37
```

The driver executes the following stages.

### 0. Source Acquisition And Cache Reuse

Driver:

- `scripts/build_real_siac_library_from_scratch.py`

Behavior:

- starts from a fresh raw source root
- reuses any already downloaded source tree from the configured fallback roots
- seeds auth-only or manually downloaded sources from `build/mannual_download_data`
- only falls back to live fetch when a source is not already cached
- records one fetch log per source under `00_fetch_logs/`

In the current retained build, the full raw source set was rebuilt from scratch
using the public source inventory plus the manual bundles in
`build/mannual_download_data`.

### 1. Raw Normalization

Command:

- `spectral-library normalize-sources`

Output:

- `01_normalized_raw`

What it does:

- parses heterogeneous source formats
- converts all spectra to a shared `400-2500 nm` / `1 nm` grid
- writes:
  - `tabular/spectra_metadata.csv`
  - `tabular/normalized_spectra.csv`
  - `tabular/source_summary.csv`
  - `tabular/normalization_failures.csv`
  - `tabular/wavelength_grid.csv`

Current raw normalization summary from the latest run:

- downloaded sources attempted: `28`
- normalized sources: `26`
- normalized spectra before filtering: `140,549`
- failure rows: `515`

### 2. Coverage Filter

Command:

- `spectral-library filter-coverage --min-coverage 0.8`

Output:

- `02_cov80`

Rule:

- retain only spectra with at least `80%` of the `400-2500 nm` grid populated

Latest result:

- input spectra: `140,549`
- retained spectra: `138,110`
- dropped spectra: `2,439`

### 3. EMIT And Santa Artifact Repair

Script:

- `scripts/repair_emit_santa_artifacts.py`

Output:

- `03_emit_santa_fixed`

What it fixes:

- EMIT false shoulder / spike behavior in the second deep water absorption band
  near `1850-1950 nm`
- Santa Barbara longwave tail drift and related absorption/tail artifacts

Current result:

- repaired spectra: `1,121`
- dominant source in this stage:
  - `santa_barbara_urban_reflectance`: `1,048` spectra

### 4. GHISACASIA Deep-Band And Tail Repair

Script:

- `scripts/repair_ghisacasia_artifacts.py`

Output:

- `04_ghisacasia_fixed`

What it fixes:

- `1400 nm` and `1900 nm` absorption-window corruption
- unstable `>2400 nm` tail behavior

Current result:

- repaired spectra: `355`
- source:
  - `ghisacasia_v001`

### 5. Global Visible-Band Outlier Repair

Script:

- `scripts/robust_visible_outlier_fix.py`

Output:

- `05_visible_fixed`

What it does:

- runs a robust Whittaker smoother as an outlier detector on all spectra
- replaces only flagged visible-band regions (`400-700 nm`)
- uses blended replacement rather than hard overwrites

Current result:

- detected spectra: `66,157`
- spectra with visible repairs applied: `20,394`

This is a generic cleanup stage, not a source blacklist.

### 6. Source-Specific Deep-Absorption Smoothing

Script:

- `scripts/curate_source_absorption_rules.py`

Output:

- `06_absorption_smoothed`

What it does:

- applies targeted deep-water and longwave smoothing windows to sources already
  known to have systematic artifacts
- uses blended replacement windows rather than hard transitions

Current source list for this stage:

- `ghisacasia_v001`
- `hyspiri_ground_targets`
- `ngee_arctic_2018`
- `santa_barbara_urban_reflectance`

### 7. Native Range Filter

Script:

- `scripts/filter_by_native_range.py`

Output:

- `07_native_410_2400`

Rule:

- retain only spectra with:
  - `native_min_nm <= 410`
  - `native_max_nm >= 2400`

Why:

- the final grid remains `400-2500 nm`
- this filter allows small missing edges but removes spectra whose native source
  support is too truncated at the blue or longwave ends

Latest result:

- input spectra: `138,110`
- retained spectra: `73,679`

### 8. Vegetation Outlier Repair

Script:

- `scripts/repair_vegetation_outliers.py`

Output:

- `08_vegetation_outliers_fixed`

What it does:

- targeted repair pass for persistent vegetation outliers
- currently focused on `understory_estonia_czech`
- repairs `1400 nm`, `1900 nm`, and `>2400 nm` windows using source-specific
  interpolation / extrapolation rules with blending

Latest result:

- repaired spectra: `465`
- repaired source:
  - `understory_estonia_czech`

### 9. Vegetation Water-Band Spike Repair

Script:

- `scripts/repair_vegetation_water_band_spikes.py`

Output:

- `09_vegetation_waterfixed`

What it does:

- detects vegetation spectra with upward spikes in the `1400 nm` and `1900 nm`
  water absorption windows
- uses outside-neighbor interpolation plus blended replacement
- also applies point-spike cleanup to `understory_estonia_czech`

Latest result:

- repaired spectra: `329`
- repaired windows: `3,299`

### 10. Curated Subset Filter

Script:

- `scripts/filter_curated_subset_rules.py`

Output:

- `10_subsetfiltered`

Current explicit subset rule:

- remove the `bssl` `600-3000 nm`, `5 nm` subset

Reason:

- it creates a false onset in pooled soil means at `600 nm`
- the full-range `bssl` subset is retained

### 11. Remaining Source-Artifact Repair

Script:

- `scripts/repair_remaining_source_artifacts.py`

Output:

- `11_source_artifacts_fixed`

This is the final source-aware cleanup stage before SIAC export. It contains
the accumulated targeted fixes that are too specific for the earlier generic
stages.

Current stage summary from the latest run:

- repair rows: `1,190`

Source-level counts in the current retained build:

- `santa_barbara_urban_reflectance`: `1,065`
- `branch_tree_spectra_boreal_temperate`: `28`
- `bssl`: `20`
- `hyspiri_ground_targets`: `18`
- `ngee_arctic_2018`: `4`
- `emit_adjusted_vegetation` / `emit_l2a_surface`: `2`
- `sispec`: `1`
- `understory_estonia_czech`: `1`

### 12. Landcover Analysis And QA

Scripts:

- `scripts/landcover_analysis.py`
- `spectral-library plot-quality`

Outputs:

- `11_source_artifacts_fixed/landcover_analysis/`
- `11_source_artifacts_fixed/plots/quality/`

What it does:

- assigns optional landcover labels
- computes cluster and outlier diagnostics
- writes QA plots for coverage, parser mix, native ranges, and source counts

Latest normalized QA:

- mean coverage fraction: `0.998608`
- p05 coverage fraction: `0.999048`

### 13. SIAC Export

Command:

- `spectral-library build-siac-library`

Output:

- `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`

Export behavior:

- exports the final normalized spectra and metadata
- keeps landcover labels as optional annotations, not as a hard inclusion filter
- retains excluded-source metadata in:
  - `siac_manifest_sources`
  - `siac_source_summary`
  - `siac_excluded_sources`
- retains excluded individual spectra in:
  - `siac_excluded_spectra`

### 14. Full Suspicious-Spectrum Review

Script:

- `scripts/full_processed_spectra_review.py`

Output:

- `siac_spectral_library_.../full_review`

What it does:

- runs the final suspicious-spectrum detector on the exported SIAC package
- scores spectra using jumps, residuals, source-relative thresholds, and
  source-specific detector logic

Latest review result:

- suspicious spectra: `187`
- suspicious fraction: `0.24%`

Current largest remaining sources:

- `ossl`: `51`
- `usgs_v7`: `29`
- `bssl`: `28`

## Parser And Normalization Safeguards

The normalization layer includes several hard parser safeguards so auxiliary
series do not leak into the reflectance product.

### Auxiliary Statistics Are Not Parsed As Reflectance

The parser explicitly filters out statistical or uncertainty series such as:

- `std`
- `stdv`
- `stdev`
- `stddev`
- `stderr`
- `uncert`
- `uncertainty`
- `error`
- `err`
- `variance`
- `var`
- `sigma`
- `rmse`
- `mad`

This applies to:

- row-wide tabular spectra
- generic column-wise tabular series
- workbook-based band matrices
- netCDF variable selection

### USGS Auxiliary Files Are Ignored

USGS parser safeguards:

- skip `errorbars/` members in the ASCII archive
- skip standalone wavelength/bandpass auxiliary files as spectra
- parse only the actual spectral members

### Reflectance-Only Parsing For Mixed Files

Examples:

- `hyspiri_ground_targets`
  - parse `Reflect. %`
  - do not ingest `Norm. DN` calibration columns as reflectance

## Verified Scale-Factor Handling

Scale-factor overrides that are source-backed rather than inferred are recorded
in:

- `docs/scale_factor_verification.md`

The currently documented verified overrides are:

- `understory_estonia_czech`: force `1.0`
- `ossl`: force `100.0`
- `ghisacasia_v001`: force `100.0`
- `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016`: force `100.0`
- `hyspiri_ground_targets`: allow only source-backed `1` or `100`

## Source-Specific Corrections

This section summarizes the main source-specific rules that currently affect the
end-to-end build.

### Scale And Parser Corrections

| Source | Problem | Current action |
|---|---|---|
| `understory_estonia_czech` | parser previously misdetected scale because corrupted bands influenced the heuristic | force `value_scale_hint = 1.0` |
| `ossl` | raw VisNIR columns are percent-like and should not float between `1` and `100` | force `value_scale_hint = 100.0` |
| `ghisacasia_v001` | official documentation describes percent reflectance | force `value_scale_hint = 100.0` |
| `ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016` | source metadata explicitly says `Percent_Reflectance_0_100` | force `value_scale_hint = 100.0` |
| `hyspiri_ground_targets` | raw package mixes ratio and percent products | allow only `1` and `100`; reject `1000/10000`-style parser artifacts; ingest reflectance instead of DN-style columns |
| `usgs_v7` | archive contains auxiliary error-bar and wavelength files | skip those members and parse only spectral files |

### Landcover Classification Corrections

These are not spectral repairs; they are source-aware label corrections used by
`scripts/landcover_analysis.py`.

| Source | Current classification rule |
|---|---|
| `hyspiri_ground_targets` | always label as `soil` with `hyspiri:playa_dry_lake` |
| `drylands_emit` | use `metadata_json.level_1`: `soil -> soil`, `pv -> vegetation`, `npv -> unclassified` |
| `antarctic_vegetation_speclib` | rock/lava-associated samples are reassigned away from vegetation |
| `usgs_v7` | perturbed vegetation variants such as shift/const synthetic variants are not treated as valid vegetation prototypes |

### Spectral Repair And Filtering Rules

| Source | Problem pattern | Current action |
|---|---|---|
| `santa_barbara_urban_reflectance` | widespread post-`800 nm` roughness plus longwave tail artifacts | strong robust smoothing after `850 nm` with linear blend across `750-850 nm`, plus absorption/tail cleanup |
| `santa_barbara_urban_reflectance` | persistent bad individual spectra | exclude `37` specific spectra listed in `manifests/siac_excluded_spectra.csv` |
| `ghisacasia_v001` | deep water absorption corruption and unstable longwave tail | targeted `1400`, `1900`, and `>2400 nm` repair, then exclude the whole source from exported SIAC spectra while keeping metadata |
| `understory_estonia_czech` | severe `1400`, `1900`, and tail artifacts even after repair | targeted vegetation repair stages run, but the whole source is excluded from exported SIAC spectra while keeping metadata |
| `emit_adjusted_vegetation` | second absorption-band shoulder / spike artifacts and visible splice issues | targeted EMIT repair plus final visible cleanup |
| `emit_l2a_surface` | same EMIT family visible / shoulder issues on selected spectra | targeted EMIT repair plus final visible cleanup |
| `sispec` | localized second absorption-band anomaly in a small number of cases | targeted absorption-window cleanup |
| `bssl` | longwave-only `600-3000 nm` subset causes pooled-mean artifacts; some remaining tail roughness | exclude the longwave-only subset, then apply targeted tail repair to flagged spectra |
| `branch_tree_spectra_boreal_temperate` | longwave tail roughness after `~2300 nm` | targeted tail repair |
| `hyspiri_ground_targets` | longwave tail roughness and mixed-unit parsing edge cases | source-backed scale handling plus targeted tail repair |
| `ngee_arctic_2018` | unstable longwave tail | source-specific smoothing and targeted tail repair |
| `natural_snow_twigs` | possible tail instability after `~2350 nm` | targeted tail repair path is available in the final repair stage |
| `usgs_v7` | local out-of-range windows and visible splice/jump artifacts in a small subset | localized segment repair in the final stage |

## Export-Time Exclusions

The final SIAC export intentionally excludes some spectra or sources while still
keeping their metadata.

### Excluded Sources

- `ghisacasia_v001`
- `understory_estonia_czech`

These remain visible in:

- `tabular/siac_manifest_sources.csv`
- `tabular/siac_source_summary.csv`
- `tabular/siac_excluded_sources.csv`

### Excluded Individual Spectra

- `37` `santa_barbara_urban_reflectance` spectra

These remain logged in:

- `tabular/siac_excluded_spectra.csv`

## QA Outputs

The most important QA products are:

### Normalized Dataset QA

- `plots/quality/source_counts.png`
- `plots/quality/source_failure_rates.png`
- `plots/quality/native_spacing_hist.png`
- `plots/quality/native_range_scatter.png`
- `plots/quality/normalized_coverage_hist.png`
- `plots/quality/failure_reasons.png`
- `plots/quality/parser_counts.png`
- `plots/quality/quality_metrics.json`

### Landcover QA

- `landcover_analysis/plots/classification_counts.png`
- `landcover_analysis/plots/cluster_scatter.png`
- `landcover_analysis/plots/cluster_signatures.png`
- `landcover_analysis/plots/outlier_rates.png`
- `landcover_analysis/plots/typical_signatures.png`

### Final Package QA

- `plots/landcover_mean_curves.png`
- `plots/top_source_mean_curves.png`
- `plots/source_counts.png`
- `plots/source_mean_coverage.png`
- `plots/landcover_counts.png`

### Final Suspicious-Spectrum Review

- `full_review/review_summary.json`
- `full_review/flagged_suspicious_spectra.csv`
- `full_review/top_suspicious_full.png`
- `full_review/top_suspicious_visible.png`
- `full_review/source_suspicious_plots/`

## Running The Build Locally

Editable install:

```bash
python -m pip install -e .
```

Useful commands:

```bash
spectral-library normalize-sources --manifest manifests/sources.csv --results-root build/local_sources_full_raw --output-root build/test_normalized
spectral-library filter-coverage --normalized-root build/test_normalized --output-root build/test_cov80 --min-coverage 0.8
spectral-library plot-quality --normalized-root build/test_normalized --output-root build/test_normalized/plots/quality
spectral-library build-siac-library --manifest manifests/sources.csv --normalized-root build/test_normalized --output-root build/test_siac --exclude-source-ids ghisacasia_v001,understory_estonia_czech --exclude-spectra-csv manifests/siac_excluded_spectra.csv
```

Full raw-to-SIAC rebuild:

```bash
MPLCONFIGDIR=build/.mplconfig PYTHONPATH=src python3 scripts/build_real_siac_library_from_scratch.py
```

Cached rebuild from existing source trees:

```bash
MPLCONFIGDIR=build/.mplconfig PYTHONPATH=src python3 scripts/build_real_siac_library.py
```

By default this cache-first wrapper writes to separate `_cached` roots under
`build/`, so it does not overwrite the retained canonical full-raw build.

## Current Limitations

- `ghisacasia_v001` is still excluded from exported spectra because the source
  remains problematic even after repair, although its metadata are retained.
- `understory_estonia_czech` is still excluded from exported spectra because it
  remains unstable after aggressive targeted repair.
- The final suspicious-spectrum review still flags a small residual set, mostly
  from `ossl`, `usgs_v7`, and `bssl`.
- The native-range filter is a policy choice. It keeps the final grid at
  `400-2500 nm`, but only retains spectra with native support extending to at
  least `410 nm` on the blue end and `2400 nm` on the longwave end.

## Reference Files

- source inventory:
  - `manifests/sources.csv`
- individual spectrum exclusions:
  - `manifests/siac_excluded_spectra.csv`
- scale verification log:
  - `docs/scale_factor_verification.md`
- suspicious-source review:
  - `docs/remaining_suspicious_source_review_20260315.md`
- latest pipeline summary:
  - `build/real_siac_pipeline_full_raw/build_summary.json`
- latest package summary:
  - `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/build_summary.json`
