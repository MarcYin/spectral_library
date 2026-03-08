# Spectral Library Fetch + Normalization Plan

## Goal

Build a standalone spectral-library database from the libraries listed in
`comprehensive_spectral_library_catalog.md`, with:

- reproducible fetch steps,
- preserved raw source files and provenance,
- a normalized reflectance product on a shared wavelength grid,
- a queryable database for SIAC-oriented spectral mapping.

This plan assumes the catalog file is the source inventory and that the first
target is land-cover optical / hyperspectral reflectance data.

## Key Decisions

### 1. Keep raw and normalized data as separate layers

The database should have two representations for each spectrum:

- `raw/native`: original wavelengths, original values, original metadata.
- `normalized/core`: reflectance resampled onto a common wavelength grid.

Raw data must never be overwritten by normalization.

### 2. Use one canonical core grid for SIAC

Use a canonical normalized grid of:

- wavelength range: `400-2500 nm`
- spacing: `1 nm`
- total bands: `2101`

Reason:

- the catalog itself notes that exact `300-2500 nm` coverage is uncommon,
- many strong libraries begin at `350 nm` or `400 nm`,
- `400-2500 nm` is the safest common denominator across the broadest set of
  listed VSWIR resources.

Important:

- this is a harmonized sampling grid, not a claim that the original instrument
  had native `1 nm` spectral resolution,
- no extrapolation should be performed beyond the source wavelength coverage,
- out-of-range bands must be stored as `null` plus a coverage mask.

### 3. Store only reflectance in the core product

The catalog mixes:

- reflectance libraries,
- emissivity products,
- image-derived spectra,
- archives and portals,
- derived / manually adjusted products.

The first normalized database should target `reflectance` only.
Non-reflectance data should either:

- stay in raw storage only, or
- be routed into separate modality-specific tables later.

## Source Tiers

The catalog should be split into implementation tiers.

### Tier 1: direct public VSWIR libraries with strong normalization value

Start here first.

- USGS Spectral Library v7
- ECOSTRESS Spectral Library v1.0
- ASTER Spectral Library v2.0
- Open Soil Spectral Library (OSSL)
- ICRAF Spectral Library
- Brazilian Soil Spectral Library (BSSL)
- LUCAS Soil spectral data
- HSDOS soil library
- ProbeField pre-processed spectra
- vegetation libraries from Zenodo
- SLUM
- KLUM
- WaRM
- SISpec
- natural snow samples dataset

These are the best candidates for the first reusable ingestion pipeline because
they are direct libraries or datasets with explicit spectral content.

### Tier 2: direct datasets with partial coverage or domain-specific limits

Fetch after Tier 1 is stable.

- SeaSWIR
- GLORIA
- PANTHYR O1BE hyperspectral water reflectance
- Urban Materials Spectral Library v1.0
- German image spectral library of urban surface materials
- Brussels/APEX/HyMap urban image-derived datasets
- GHISACONUS
- GHISACASIA
- EcoSIS-hosted direct packages

These need special handling because they are:

- partial-range datasets,
- image-derived rather than field/lab-native,
- Earthdata or portal-backed,
- domain-specific enough that class mapping and quality checks will vary.

### Tier 3: portals, archives, and derived products

Do not ingest these as if they were primary raw libraries.

- EcoSIS portal
- SPECCHIO
- Australian National Spectral Database portal
- SeaBASS
- WaterHypernet
- EMIT derived surface repositories
- earthlib
- repackaged urban compilations

For Tier 3, create manifests and provider adapters first. Only ingest specific
datasets after the exact downloadable assets and licenses are confirmed.

## Fetch Strategy

Use a provider-oriented fetch architecture instead of writing one script per
dataset from scratch.

### Provider adapters

Implement these fetch adapters:

1. `static_http`
2. `zenodo_api`
3. `pangaea`
4. `ecosis_package`
5. `earthdata_lpdaac`
6. `manual_portal`

Each source in the catalog should be mapped to one adapter type.

### Manifest-driven fetching

Create one manifest record per source with fields such as:

- `source_id`
- `name`
- `provider`
- `resource_type`
- `spectral_type`
- `landing_url`
- `download_url`
- `license`
- `auth_mode`
- `expected_format`
- `priority`
- `parser`
- `notes`

This allows reproducible downloading and avoids hard-coding URLs inside scripts.

### Raw landing zone

Fetched files should land in a raw cache with checksums:

- `data/raw/<source_id>/...`
- checksum file per artifact
- fetch log per run

Do not modify files in the raw cache.

## Parsing and Standardization

Every fetched dataset should be converted into a source-independent intermediate
schema before normalization.

### Intermediate spectrum schema

For each spectrum, extract:

- `sample_id`
- `source_id`
- `source_record_id`
- `sample_name`
- `material_class`
- `material_subclass`
- `measurement_type`
- `instrument`
- `platform`
- `native_wavelength_nm`
- `native_value`
- `native_unit`
- `native_start_nm`
- `native_end_nm`
- `estimated_native_step_nm`
- `citation`
- `license`
- `source_file`

### Class harmonization

Map source-specific labels into a common top-level taxonomy:

- `vegetation`
- `soil`
- `urban`
- `water`
- `snow_ice`
- `rock_mineral`
- `mixed`
- `other`

Keep both:

- original class labels,
- harmonized class labels.

### Measurement filters

Before normalization, reject or route separately:

- emissivity-only records,
- transmittance-only records,
- records with no wavelength axis,
- records with no numeric spectral values,
- records whose units cannot be resolved.

## Normalization Rules

### Wavelength handling

- convert all wavelengths to `nm`,
- sort ascending,
- drop exact duplicate wavelength entries,
- flag non-monotonic or malformed spectra.

### Value handling

- convert reflectance to unitless fraction when source uses percent,
- preserve original values in raw storage,
- do not silently clip out-of-range values,
- instead store QA flags for values outside expected reflectance limits.

### Resampling

For the `400-2500 nm` / `1 nm` core product:

- resample only within the observed native wavelength range,
- do not extrapolate,
- set bands outside native support to `null`,
- write a `coverage_mask` for valid bands,
- write a `bad_band_mask` when the source indicates low-quality regions.

Recommended interpolation method:

- linear interpolation first for robustness,
- optionally evaluate PCHIP later for smoother library-specific builds.

### Partial-range datasets

Some libraries in the catalog cover only part of the core grid, for example
water-focused datasets or short-range urban libraries.

Plan:

- include them in the same normalized schema,
- leave unsupported bands as `null`,
- rely on `coverage_mask` and `valid_fraction`,
- exclude them from analyses that require full VSWIR coverage.

## Database Design

Use `DuckDB + Parquet`.

### Output artifacts

- `catalog.duckdb`
- `sources.parquet`
- `samples.parquet`
- `spectra_raw.parquet`
- `spectra_norm_400_2500_1nm.parquet`
- `artifacts.parquet`

### Core tables

#### `sources`

One row per library / dataset source.

Fields:

- source identifiers
- provider type
- URLs
- license
- fetch status
- parser type
- priority

#### `artifacts`

One row per downloaded file.

Fields:

- artifact path
- source id
- checksum
- file size
- fetch timestamp
- parse status

#### `samples`

One row per spectral record.

Fields:

- source identifiers
- sample identifiers
- taxonomy
- measurement metadata
- wavelength range
- quality summary

#### `spectra_raw`

One row per record with native arrays.

Fields:

- `sample_id`
- `wavelength_nm` list
- `value` list
- `unit`

#### `spectra_norm_400_2500_1nm`

One row per record on the canonical grid.

Fields:

- `sample_id`
- `reflectance` list
- `coverage_mask` list
- `bad_band_mask` list
- `valid_fraction`
- `resampling_method`

## Repository Layout

Recommended layout:

```text
spectral_library/
  comprehensive_spectral_library_catalog.md
  plan.md
  pyproject.toml
  README.md
  manifests/
    sources.csv
  scripts/
    fetch/
    parse/
    normalize/
    build/
  data/
    raw/
    interim/
    normalized/
  db/
  tests/
  docs/
```

## Implementation Phases

### Phase 0: inventory and scoping

- convert the catalog into `manifests/sources.csv`
- assign each source to a provider adapter
- assign each source a tier and initial status
- mark sources that are direct vs derived vs portal

### Phase 1: fetch framework

- implement manifest loader
- implement `static_http` and `zenodo_api` adapters first
- add checksum and fetch logging
- download Tier 1 sources that have direct file access

### Phase 2: parsing framework

- write parsers for common formats: CSV, TSV, XLSX, ZIP bundles, ASCII spectra
- emit intermediate `samples` and `spectra_raw` tables
- validate wavelength and value arrays

### Phase 3: normalization

- implement `400-2500 nm` / `1 nm` grid builder
- write resampling and mask logic
- compute QA metrics
- build `spectra_norm_400_2500_1nm.parquet`

### Phase 4: database assembly

- load Parquet outputs into `catalog.duckdb`
- create standard views for SIAC workflows
- add summary queries and dataset counts

### Phase 5: portal and auth-backed sources

- implement `ecosis_package`, `pangaea`, and `earthdata_lpdaac`
- add manual review flow for portals and archives
- ingest selected Tier 2 and Tier 3 datasets after URL and license checks

## Quality Checks

Every build should check:

- file checksum stability
- parser success rate per source
- wavelength monotonicity
- missing wavelength axis
- non-numeric spectra
- reflectance range anomalies
- number of valid bands after normalization
- duplicate sample ids
- duplicate spectra by hash

## Risks and Constraints

- The catalog contains a mix of direct libraries, portals, archives, and derived
  products. These must not be treated as equivalent.
- Exact `300-2500 nm` normalization across all sources is not realistic from the
  listed inventory; `400-2500 nm` is the more stable common grid.
- Some sources are partial-range only; null-padding plus masks is required.
- Some sources may require authentication, manual download, or license review.
- Image-derived spectral libraries should be tagged clearly so they are not
  confused with laboratory or field spectra.

## First Deliverables

The first concrete outputs should be:

1. `manifests/sources.csv` generated from the catalog
2. fetch adapters for `static_http` and `zenodo_api`
3. a raw cache layout under `data/raw/`
4. a parser that emits `samples.parquet` and `spectra_raw.parquet`
5. a normalizer that builds `spectra_norm_400_2500_1nm.parquet`
6. `catalog.duckdb` with queryable source and sample tables

## Immediate Next Step

Start by converting the markdown catalog into a structured source manifest,
because every later step depends on a clean inventory of:

- source name,
- category,
- provider,
- URL,
- format,
- access mode,
- priority,
- normalization eligibility.
