# Internal Build Pipeline

This document is an internal reference for the retained SIAC build pipeline in
this repository. It is not part of the public `spectral-library` package
contract.

Public package installation and mapping usage are documented in
[`mapping_quickstart.md`](mapping_quickstart.md).

## Purpose

The internal pipeline:

1. fetches or reuses raw source datasets,
2. normalizes them onto the shared `400-2500 nm` / `1 nm` grid,
3. applies source-aware filtering and repair stages,
4. exports the canonical SIAC package used as mapping input.

## Canonical Build Roots

Current retained build roots:

- pipeline root:
  `build/real_siac_pipeline_full_raw`
- final SIAC package:
  `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37`

## Repository Paths

Important internal paths:

- `manifests/sources.csv`
  curated source inventory, fetch adapters, source status, and notes
- `manifests/siac_excluded_spectra.csv`
  export-time spectrum exclusions
- `scripts/build_real_siac_library_from_scratch.py`
  canonical raw-to-SIAC build driver
- `scripts/build_real_siac_library.py`
  cache-first wrapper around the canonical driver
- `scripts/`
  source-specific repair, filtering, plotting, and review utilities
- `build/`
  cached source trees, pipeline artifacts, QA outputs, and SIAC exports

## Canonical End-To-End Command

```bash
MPLCONFIGDIR=build/.mplconfig PYTHONPATH=src python3 scripts/build_real_siac_library_from_scratch.py \
  --fallback-raw-roots build/local_sources_full_raw,build/local_sources_vegetation_all,build/local_sources \
  --raw-sources-root build/local_sources_full_raw \
  --pipeline-root build/real_siac_pipeline_full_raw \
  --output-root build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37
```

## Pipeline Stages

The current retained pipeline is organized as:

1. source acquisition and cache reuse
2. raw normalization
3. coverage filter
4. EMIT and Santa artifact repair
5. GHISACASIA deep-band and tail repair
6. global visible-band outlier repair
7. source-specific deep-absorption smoothing
8. native range filter
9. vegetation outlier repair
10. vegetation water-band spike repair
11. curated subset filter
12. remaining source-artifact repair
13. landcover analysis and QA
14. SIAC export
15. suspicious-spectrum review

Representative internal commands used by those stages:

- `spectral-library-internal normalize-sources`
- `spectral-library-internal filter-coverage --min-coverage 0.8`
- `spectral-library-internal plot-quality`
- `spectral-library-internal build-siac-library`

Representative repair and review scripts:

- `scripts/repair_emit_santa_artifacts.py`
- `scripts/repair_ghisacasia_artifacts.py`
- `scripts/robust_visible_outlier_fix.py`
- `scripts/curate_source_absorption_rules.py`
- `scripts/filter_by_native_range.py`
- `scripts/repair_vegetation_outliers.py`
- `scripts/repair_vegetation_water_band_spikes.py`
- `scripts/filter_curated_subset_rules.py`
- `scripts/repair_remaining_source_artifacts.py`
- `scripts/landcover_analysis.py`

## QA Outputs

Current top-level QA locations in the retained build:

- normalized QA:
  `build/real_siac_pipeline_full_raw/11_source_artifacts_fixed/plots/quality`
- landcover QA:
  `build/real_siac_pipeline_full_raw/11_source_artifacts_fixed/landcover_analysis/plots`
- SIAC package plots:
  `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/plots`
- suspicious-spectrum review:
  `build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37/full_review`

## Notes

- The internal build pipeline remains repository-specific and may change without
  the compatibility guarantees applied to the public mapping package.
- The public prepared-runtime contract begins after SIAC export, at
  `prepare-mapping-library`.
