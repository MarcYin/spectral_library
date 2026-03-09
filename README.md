# Spectral Library Database Scaffold

This repository is a manifest-driven scaffold for building a spectral-library
catalog incrementally. The current focus is making the source inventory and
build process GitHub Actions-friendly while adding a shared normalization layer
for the locally downloaded libraries.

## What exists now

- [manifests/sources.csv](manifests/sources.csv): curated source inventory
  - `landing_url` keeps the citation or dataset page
  - `download_url` optionally pins the current best direct asset URL for GitHub Actions fetch jobs
- [plan.md](plan.md): ingest and normalization plan
- `spectral-library` CLI:
  - `plan-matrix`
  - `fetch-source`
  - `fetch-batch`
  - `assemble-database`
  - `tidy-results`
  - `normalize-sources`
  - `plot-quality`
  - `filter-coverage`
  - `build-siac-library`
- GitHub Actions workflow:
  - `.github/workflows/build-database.yml`

## Build model

The workflow is split into three stages:

1. `plan`
   - reads `manifests/sources.csv`
   - filters sources by tier, status, and adapter
   - emits a job matrix for GitHub Actions
2. `fetch`
   - runs one job per source
   - stores per-source results under a stable cache path
   - uploads the source result as an artifact
3. `assemble`
   - merges all fetched results
   - writes tabular outputs, Parquet files, and `catalog.duckdb`

This keeps the build incremental and avoids one large monolithic job.

## Local usage

```bash
python -m pip install -e .
spectral-library plan-matrix --manifest manifests/sources.csv --tiers tier1,tier2 --statuses planned
spectral-library fetch-source --manifest manifests/sources.csv --source-id ossl --fetch-mode metadata
spectral-library fetch-batch --manifest manifests/sources.csv --output-root build/local_sources --fetch-mode assets --continue-on-error --clean-output
spectral-library assemble-database --manifest manifests/sources.csv --results-root build/sources --output-root build/assembled
spectral-library normalize-sources --manifest manifests/sources.csv --results-root build/local_sources --output-root build/normalized
spectral-library filter-coverage --normalized-root build/normalized --output-root build/normalized_cov80 --min-coverage 0.8
spectral-library plot-quality --normalized-root build/normalized --output-root build/normalized/plots
spectral-library build-siac-library --manifest manifests/sources.csv --normalized-root build/normalized_rebuild_v9_final --output-root build/siac_spectral_library_v1
```

`fetch-batch` writes one cleaned directory per source under the output root:

- `metadata/` for fetcher metadata JSON
- `docs/` for supporting PDF or DOCX files
- `data/` for downloaded spectral assets

`tidy-results` can be run afterward to reorganize an existing results tree into
the same layout.

## Current output scope

The repository now has two output layers.

Source catalog build:
- `sources`
- `fetch_results`
- `artifacts`
- `source_build_status` view

Normalization build:
- `wavelength_grid`
- `spectra_metadata`
- `normalized_spectra`
- `normalization_failures`
- `source_summary`

Quality plots:
- `source_counts.png`
- `source_failure_rates.png`
- `native_spacing_hist.png`
- `native_range_scatter.png`
- `normalized_coverage_hist.png`
- `failure_reasons.png`
- `parser_counts.png`

Coverage filtering:
- `filter-coverage` rewrites `spectra_metadata.csv` and `normalized_spectra.csv`
  using a minimum retained coverage threshold on the shared `400-2500 nm` grid
- `wavelength_grid.csv` and `normalization_failures.csv` are copied through
- `source_summary.csv` is rebuilt from the retained spectra

SIAC export:
- `build-siac-library` keeps the full curated normalized dataset and carries
  land-cover labels only as optional annotations
- it writes SIAC-ready metadata, spectra, source summaries, land-cover summaries
  for labeled subsets, pooled/source/source-balanced prototypes, and a DuckDB package
- the package keeps the same `400-2500 nm` / `1 nm` grid and is intended to be
  built from the curated final normalized archive

The normalization pass currently handles the common local formats already in
the workspace:
- ENVI `.hdr` + binary companions
- ECOSTRESS-style ASCII `.txt`
- tabular `.csv` spectra in row-wide, column-wide, and long-table layouts
- spectral archives in `.zip`, including USGS ASCII bundles and mixed bundles
  where spectral tables live beside metadata or quality CSVs
- workbook-based libraries in `.xlsx`, including row-wide tables and
  band-matrix sheets
- netCDF `.nc` spectra, including single-spectrum and `obs x wavelength`
  reflectance arrays
- `.rds` sources that expand into tabular spectra via `rdata`

Archive-only, spreadsheet-only, and other unsupported assets are logged to
`normalization_failures` rather than being silently skipped.

## Provider notes

- `ecosis_package` sources now resolve `https://ecosis.org/api/package/<slug>`
  and download the package resource files in asset mode. If a manifest
  `download_url` is present, it is treated as a preferred resource but does not
  suppress the rest of the package downloads.
- `specchio_client` sources are query-driven rather than file-list driven. The
  adapter expects `SPECCHIO_CLIENT_JAR` and optional runtime settings in the
  environment, then writes descriptor metadata in metadata mode or `space_###.csv`
  exports in assets mode when `SPECCHIO_QUERY_JSON` is provided.
