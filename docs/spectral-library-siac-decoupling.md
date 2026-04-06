# Spectral-Library Additions Required To Remove Sensor Definition And Runtime Bridging From SIAC

## Goal

`spectral-library` should become the source of truth for custom and canonical
sensor definitions used during spectral mapping. SIAC should stop owning:

- per-band RSRF lookup and realization
- custom sensor-definition assembly
- `vnir` / `swir` segment assignment for mapping runtimes
- temporary `srf_root` JSON emission
- temporary `siac_root` export formatting

After this change, SIAC should only pass source observations, target-band
requests, and a hyperspectral library into `spectral-library`, then consume the
mapping outputs.

## Problem Statement

SIAC still contains a spectral-response and runtime-bridge layer that should
belong upstream.

Current SIAC-owned seams:

- `python/siac/adapters/rsrf.py` loads per-band response definitions from
  `rsrf` and rebuilds `SensorConfig` objects with attached sampled curves.
- `python/siac/adapters/satellite/sentinel2.py` depends on that adapter and
  still falls back to SIAC-local built-in sensor metadata when upstream lookup
  fails.
- `python/siac/algorithms/surface/spectral_mapping.py` still:
  - realizes source and target band responses
  - assigns runtime band ids
  - assigns `vnir` / `swir`
  - writes custom `rsrf_sensor_definition` JSON to `srf_root`
  - exports a SIAC-formatted hyperspectral library to `siac_root`

This split is wrong because the mapping package already depends on `rsrf` and
already validates sensor definitions internally. SIAC should not need a second
bridge that reconstructs the same concepts.

## Current Boundary

| Area | SIAC today | spectral-library today | Why the split is wrong |
| --- | --- | --- | --- |
| Canonical sensor metadata | Maintains local `SensorConfig` and `SensorBand` definitions, plus sensor-specific preprocessor attachment | Knows canonical mapping sensors and their mapping-specific band layout | Canonical sensor ownership is duplicated across packages |
| Canonical RSRF lookup | Calls `rsrf` directly through `python/siac/adapters/rsrf.py` | Also calls `rsrf` internally when building runtimes | SIAC should not be a second `rsrf` client for the same mapping path |
| Custom sensor definitions | Builds `rsrf_sensor_definition` payloads in `python/siac/algorithms/surface/spectral_mapping.py` | Requires those payloads, but only after SIAC serializes them to disk | SIAC should not have to write `srf_root` JSON |
| Target sensor registration | Stages both source and target definitions into the runtime build area so the target is present in `sensor_schema.json` | Precomputes only `source_sensors` during build and resolves `target_sensor` later from packaged schemas | SIAC should not need a second staging path just to make custom targets visible at runtime |
| Segment assignment | Assigns segment from band center wavelength and clips realized support before serialization | Validates segment support, but does not own the caller-facing segment policy | Segment ownership is split across packages |
| Hyperspectral library export | Writes SIAC tabular CSV files to a temporary `siac_root` | Requires `siac_root` on disk for runtime build | SIAC should be able to pass an in-memory library |
| Runtime identity and caching | Hashes custom sensor payloads and upstream versions to decide cache reuse | Builds prepared runtimes from on-disk inputs | The caller should not need to know the runtime cache artifact format |

## Target Boundary

The desired ownership rule is:

- `rsrf` owns response-definition semantics and realization.
- `spectral-library` owns sensor-definition coercion, segment policy, runtime
  build inputs, and prepared-runtime caching.
- SIAC owns scene data, `xarray` integration, source reflectance arrays, target
  band requests, and output packaging.

In the target state, SIAC should be able to say:

1. Here is the hyperspectral library.
2. Here is the source sensor or source band set.
3. Here is the target sensor or target band set.
4. Map these reflectance rows.

Everything below that boundary should belong to `spectral-library`.

## Required Additions In spectral-library

| Capability | What spectral-library must provide | What SIAC can delete |
| --- | --- | --- |
| In-memory hyperspectral library input | Accept canonical wavelengths, spectra, and sample ids directly, without requiring caller-written `siac_root` tabular files | `_export_hyperspectral_library_root`, `_ensure_siac_library_root`, and SIAC-owned temporary library export handling |
| In-memory sensor input | Accept canonical sensor ids, `rsrf_sensor_definition` mappings, or neutral custom sensor inputs directly, without requiring caller-written `srf_root` JSON | `_write_sensor_schema` and SIAC-owned temporary sensor-schema emission |
| Custom band input model | Accept per-band input as one of: canonical `rsrf` band reference, sampled `response_definition`, or center/FWHM `gaussian` / `band_spec` input | SIAC-owned response-definition assembly and per-band bridge code |
| Segment policy ownership | Own the rule for `vnir` / `swir` assignment when the caller does not provide a segment. Default policy should be center-wavelength-based. | SIAC `segment_for_band(...)` as a mapping-runtime concern |
| Internal segment clipping | When `rsrf` realizes a `gaussian` / `band_spec`, clip the realized positive support to the declared segment before runtime validation | SIAC `segmentize_curve(...)` as a runtime-serialization concern |
| Stable sensor and band identity | Preserve caller band ids when provided, generate deterministic ids when omitted, preserve caller order, and return output columns in the same logical order | SIAC band-id rewriting, runtime band-id bookkeeping, and output remapping glue |
| Internal custom-sensor caching | Hash normalized custom sensor inputs and cache the prepared runtime internally | SIAC-owned runtime signature logic for custom sensor payloads |
| Direct runtime build API | Build a runtime from in-memory library + sensor inputs and return a reusable runtime object | Caller knowledge of `siac_root`, `srf_root`, and prepared-runtime file staging |
| Target-sensor availability in runtime | Accept target sensors during runtime build, or register them lazily through the same public input API used for source sensors | SIAC-owned target schema staging solely to satisfy runtime lookup |
| Clear error model | Distinguish invalid sensor input, missing canonical sensor, missing band, invalid segment policy, and realization failure | SIAC-side generic fallback paths and broad warning-based recovery |
| Optional prepared-runtime persistence | Allow callers to persist the runtime to disk when they want distribution or offline reuse, but do not require filesystem staging for normal use | SIAC staging directories as a mandatory integration detail |

## Required Integration Contract For SIAC

SIAC should consume a narrow upstream API with the following behavior:

### Sensor input

`spectral-library` should accept any of:

- canonical sensor id string such as `"sentinel-2a_msi"`
- full `rsrf_sensor_definition` mapping
- neutral custom `SensorInput` object

The neutral custom `SensorInput` contract should support:

- `sensor_id: str | None`
- `bands: Sequence[BandInput]`
- `band_id_policy: "preserve" | "deterministic"`
- `segment_policy: "explicit" | "center_wavelength"`

If `segment_policy="center_wavelength"`, the package shall assign:

- `vnir` for center wavelength `< 1000 nm`
- `swir` for center wavelength `>= 1000 nm`

Each `BandInput` should support:

- `band_id: str | None`
- `center_wavelength_nm: float | None`
- `fwhm_nm: float | None`
- `response_definition: Mapping[str, Any] | None`
- `rsrf_sensor_id: str | None`
- `rsrf_band_id: str | None`
- `rsrf_representation_variant: str | None`
- `segment: str | None`

### Library input

`spectral-library` should accept a neutral in-memory hyperspectral library
contract:

- `wavelengths_nm`
- `spectra`
- `sample_ids`
- optional provenance metadata

The caller should not need to write SIAC-formatted CSV files first.

### Runtime build

The runtime build surface should look like:

```python
from spectral_library import build_mapping_runtime

runtime = build_mapping_runtime(
    library=library_input,
    source_sensors=[source_sensor_input],
    target_sensors=[target_sensor_input],
    cache_root=Path("..."),   # optional
    output_root=Path("..."),  # optional
)
```

Required behavior:

- If `output_root` is omitted, runtime build may use an internal cache or temp
  directory.
- If `output_root` is provided, the runtime is persisted in the existing
  prepared-runtime format.
- The returned runtime object exposes normalized sensor ids and band ids.
- Both source and target sensors are accepted through the same public input
  model. The caller does not need to stage target schemas separately just so
  the runtime can resolve them later.

### Mapping

The mapping surface should accept a runtime object plus reflectance rows in the
exact source-band order declared by the caller's normalized source sensor.

The returned target columns must preserve the normalized target-band order
without requiring caller-side padding or canonical subset expansion.

## Proposed Design

### 1. Add neutral input models to spectral-library

Add public types that are independent of SIAC internals:

- `BandInput`
- `SensorInput`
- `HyperspectralLibraryInput`
- `PreparedRuntime`

These types must be neutral. SIAC should not have to pass a SIAC-specific
`SensorBand` or `SensorConfig` object into `spectral-library`.

### 2. Add a sensor coercion layer above the current schema loader

Add a public coercion entry point:

```python
from spectral_library import coerce_sensor_input

sensor = coerce_sensor_input(
    sensor_input,
    segment_policy="center_wavelength",
)
```

This layer should:

- resolve canonical sensor ids through `rsrf`
- accept already-normalized `rsrf_sensor_definition` payloads
- accept band-spec and gaussian inputs and normalize them through `rsrf`
- assign missing band ids deterministically
- assign missing segments from center wavelength when allowed
- clip realized support to segment bounds before validation
- return one normalized runtime sensor object

### 3. Move segment ownership into spectral-library

The current split is:

- SIAC assigns segment from center wavelength
- `spectral-library` validates support stays inside the segment

This is the wrong boundary for FWHM-only inputs. The new rule should be:

1. If the caller gives an explicit segment, use it.
2. Otherwise assign segment from center wavelength.
3. Realize any `gaussian` / `band_spec` through `rsrf`.
4. Clip the realized curve to the segment range.
5. Validate the clipped support and continue.

This keeps `rsrf` responsible for realization semantics and
`spectral-library` responsible for mapping-segment semantics.

It also removes the current split where SIAC chooses the segment while
`spectral-library` only enforces support bounds afterward.

### 4. Add in-memory runtime build

Add a public builder that accepts in-memory library and sensor inputs. The
existing prepared-runtime layout may stay unchanged internally, but the caller
must not need to know about `siac_root` and `srf_root`.

Implementation options:

- build internally in a temp/cache directory and return `PreparedRuntime`
- or add a true in-memory build path and optional persistence

Either option is acceptable for SIAC as long as the on-disk staging details are
fully owned by `spectral-library`.

### 5. Preserve caller order and identity

For custom sensor inputs, `spectral-library` must guarantee:

- band order is preserved
- output columns are returned in normalized target-band order
- stable ids are returned for both source and target sensors

This removes the need for SIAC to:

- generate hashed sensor ids itself
- rename the primary NIR band itself
- pad source subsets into canonical source layouts
- remap target outputs through local band-id bookkeeping

### 6. Keep the low-level file builder as an implementation detail

The current `build_mapping_library(siac_root, srf_root, output_root, ...)`
surface can remain as a lower-level API for maintainers or offline tooling.

The new SIAC-facing API should sit above it and own:

- in-memory input coercion
- temporary staging if needed
- prepared-runtime cache reuse
- normalization and validation

## Responsibilities After The Change

| Concern | Owner after change |
| --- | --- |
| Canonical response-definition semantics | `rsrf` |
| Canonical sensor-definition lookup | `spectral-library` using `rsrf` |
| Custom sensor-definition coercion | `spectral-library` |
| Segment assignment and segment clipping for mapping | `spectral-library` |
| Prepared-runtime caching and persistence | `spectral-library` |
| `xarray` flattening and restoration | SIAC |
| Observation extraction and TOA preprocessing | SIAC |
| Surface-prior orchestration | SIAC |

## SIAC Code Expected To Be Deleted

Once the new `spectral-library` API exists and SIAC upgrades to it, SIAC should
be able to delete or drastically shrink:

- `python/siac/adapters/rsrf.py`
- the RSRF fallback branch in `python/siac/adapters/satellite/sentinel2.py`
- custom sensor-definition and runtime-staging logic in
  `python/siac/algorithms/surface/spectral_mapping.py`

The remaining SIAC spectral-mapping code should be limited to:

- source observation flattening
- target `xarray` reconstruction
- uncertainty propagation
- workflow integration

## Rollout Assumptions

This design does not require SIAC backward compatibility with older
`spectral-library` integration paths.

Assumptions:

- SIAC may raise its minimum `spectral-library` version to the first release
  that includes the new API.
- SIAC may delete the old local bridge directly once that floor is adopted.
- The old low-level file builder may remain upstream, but SIAC does not need to
  keep dual-path support.

## Verification Plan

The upstream change is complete when the following are true:

- [ ] SIAC does not import `python/siac/adapters/rsrf.py` during normal
  spectral mapping or preprocessor sensor-config resolution.
- [ ] SIAC does not write temporary `srf_root` JSON.
- [ ] SIAC does not write temporary `siac_root` CSV exports for mapping.
- [ ] SIAC does not assign mapping segments for runtime build.
- [ ] SIAC does not have to stage target schemas separately from source schemas.
- [ ] A Sentinel-2 preprocessor path can resolve its runtime sensor metadata
  through `spectral-library` alone.
- [ ] A custom target-band set defined only by center wavelength and FWHM can be
  passed directly to `spectral-library`.
- [ ] A mixed-source mapping path preserves source-band order and target output
  order without caller-side padding.
- [ ] Error cases distinguish:
  - missing canonical sensor
  - missing band within a canonical sensor
  - invalid custom band input
  - invalid segment assignment
  - failure to realize a response definition

## Non-Goals

This document does not propose:

- redesigning SIAC runtime payloads broadly
- changing solver or correction science
- moving scene preprocessing into `spectral-library`
- changing the prepared-runtime file format unless needed internally

## Open Questions

1. Should `spectral-library` expose a neutral `SensorInput` type publicly, or
   accept plain mappings and dataclasses interchangeably?
2. Should in-memory runtime build be a true no-files path, or is internal temp
   staging acceptable as long as the caller does not manage it?
3. Should `spectral-library` own preprocessor-ready sensor objects, or only
   mapping-ready sensor/runtime objects?
4. Should deterministic id generation be caller-visible, or treated as an
   internal cache-detail only?
