# GitHub Actions Build Design

## Objective

Build the spectral-library database incrementally on GitHub Actions instead of
requiring one local machine to fetch and stage every source.

## Workflow shape

### 1. Plan job

- reads `manifests/sources.csv`
- filters by:
  - `tier`
  - `status`
  - `fetch_adapter`
  - optional explicit `source_id`
- writes a matrix JSON payload

### 2. Fetch job

- runs as a matrix over `source_id`
- restores a per-source cache
- fetches metadata or assets for one source
- uploads the result directory as an artifact

### 3. Assemble job

- downloads all per-source artifacts
- merges them with the manifest
- writes:
  - tabular CSV outputs
  - Parquet outputs
  - `catalog.duckdb`

## Why this works well on GitHub Actions

- source jobs fail independently instead of failing the whole build early
- sources can be retried one at a time
- caches can avoid repeated metadata downloads
- the catalog can grow over time as new parsers are added

## Current limitations

- only a few adapters are implemented:
  - `static_http`
  - `zenodo_api`
  - `ecosis_package` metadata only
  - `manual_portal`
  - `earthdata_lpdaac` routed to manual review for now
- spectral parsing and resampling are not implemented yet
- large binary asset downloads may exceed GitHub Actions artifact or cache limits

## Recommended near-term extension

1. Add source-specific parsers for the first Tier 1 libraries.
2. Emit raw spectral tables per source.
3. Add a second workflow phase or command for normalization to
   `400-2500 nm` / `1 nm`.
4. Publish assembled database artifacts from workflow runs or GitHub Releases.
