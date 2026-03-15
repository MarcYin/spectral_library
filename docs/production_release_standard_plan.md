# Production Release Standard Design

## Purpose

This document defines how the spectral mapping design in
`docs/spectral_mapping_usage_plan.md` should be turned into a stable Python
package and a production-ready public release.

The target is a public PyPI package with a stable v1 contract that covers both:

- the public Python and CLI interfaces,
- the prepared runtime and file-format standard used by mapping and
  reconstruction workflows.

This is a production-release design for the remaining standardization work.
Parts of the package surface described here are now implemented in the repo,
but the full release process and compatibility policy are not yet complete.

## Current Repo Facts

The current repository state that matters for release planning is:

- distribution name in `pyproject.toml`:
  `spectral-library-db`
- chosen public distribution name for v1 release:
  `spectral-library`
- version:
  `0.1.0`
- import package:
  `spectral_library`
- CLI entry point:
  `spectral-library`
- tests:
  present under `tests/`
- GitHub Actions workflows:
  one workflow exists for fetch/catalog assembly
- current gaps:
  no release workflow, no changelog file, no top-level license file, and no
  fully documented compatibility policy for the new public API and
  prepared-file standard

This means the repository already has usable code, tests, and packaging
scaffolding, but it is still organized like an internal build system rather
than a public product package. The production release work should therefore
include aligning the package metadata with the chosen public name
`spectral-library`.

## Package Identity Decision

The public distribution name for the first production release is:

- PyPI distribution:
  `spectral-library`

The v1 release keeps the runtime-facing names stable:

- import package:
  `spectral_library`
- CLI:
  `spectral-library`

### Why `spectral-library`

`spectral-library` is clearer, more descriptive, and better aligned with the
release-facing purpose of the package: spectral mapping, reconstruction, and
prepared spectral-library workflows in one public package.

It is preferred over `spectral-library-db` because the old name describes the
original build/catalog scaffold rather than the public product.

### v1 Naming Rule

The first public release should:

- rename the distribution metadata from `spectral-library-db` to
  `spectral-library`,
- keep `spectral_library` as the import package for v1,
- keep `spectral-library` as the CLI for v1,
- document the difference between distribution name, import name, and CLI name
  in install and quickstart documentation.

### Future Rename Rule

If a later major release wants to rename the import package or CLI to match
the distribution name more closely, treat that as a breaking change and ship an
explicit migration path.

## Standard Definition

The project standard for v1 is the combination of four versioned contracts:

1. Python API contract
2. CLI contract
3. prepared runtime and file-format contract
4. semantic-versioning and release contract

These contracts must be versioned, documented, tested, and treated as public
compatibility commitments for 1.x releases.

## Public vs Internal Surface

The v1 public surface should be intentionally narrow.

### Public Stable Surface

The stable public surface for 1.x should include:

- `prepare_mapping_library(...)`
- `SpectralMapper`
- a result type for mapping and reconstruction outputs plus diagnostics
- the sensor SRF schema
- the prepared-library manifest and schema
- the stable output modes:
  - `target_sensor`
  - `vnir_spectrum`
  - `swir_spectrum`
  - `full_spectrum`

The full-spectrum output mode includes the overlap-blending rule already defined
in the mapping design: VNIR and SWIR outputs are merged across `900-1000 nm`
with a linear weighted average.

### Internal Non-Stable Surface

Everything not explicitly listed above should be documented as internal and not
covered by compatibility guarantees, including:

- internal helper modules,
- low-level distance or convolution kernels,
- storage internals beyond the documented prepared-runtime contract,
- intermediate builder-only data structures,
- benchmark internals and development utilities.

The implementation should prefer one public import path and avoid exposing
internal modules as part of the public contract.

## Python API Contract

The Python API should be versioned around a small stable surface.

### Stable Entry Points

The v1 stable Python API should be centered on:

```python
prepare_mapping_library(...)
SpectralMapper(...)
MappingResult(...)
SensorSRFSchema(...)
PreparedLibraryManifest(...)
```

### Contract Rules

- The function and class names above are stable across 1.x.
- Public constructor parameters and core method names are stable across 1.x.
- Public dataclass or model fields are stable unless a major version is cut.
- Additive optional parameters are allowed in minor releases.
- Removals, renames, required-parameter additions, or semantic changes in
  return fields require a major version.

### Stable Behavior

The public API contract must guarantee:

- deterministic numeric output for the same prepared runtime, inputs, and
  package version,
- explicit validation for invalid sensor metadata and invalid reflectance input,
- explicit output-mode selection,
- stable segment behavior for `vnir`, `swir`, and `full_spectrum`,
- stable diagnostic field names for neighbor identities, distances, and
  validity information.

### Type Support

The production package should ship with complete type annotations on the public
API and include `py.typed` in v1. Type hints on public symbols are therefore
part of the supported package experience.

## CLI Contract

The CLI should be treated as a public contract, not just a developer utility.

### Stable Commands

The v1 stable commands are:

- `spectral-library prepare-mapping-library`
- `spectral-library map-reflectance`
- `spectral-library benchmark-mapping`

### Stable Core Flags

The following flags should be guaranteed across 1.x for the commands above:

`prepare-mapping-library`

- `--siac-root`
- `--srf-root`
- `--source-sensor`
- `--output-root`

`map-reflectance`

- `--prepared-root`
- `--source-sensor`
- `--target-sensor`
- `--input`
- `--output-mode`
- `--k`
- `--output`

`benchmark-mapping`

- `--prepared-root`
- `--source-sensor`
- `--target-sensor`
- `--report`

Rules:

- `--target-sensor` is required when `--output-mode target_sensor` is used,
- `--output-mode` must accept only the documented public modes,
- additive optional flags are allowed in minor releases,
- flag removal, rename, or changed meaning requires a major version.

### CLI Error And Logging Standard

The production CLI should provide:

- non-zero exit codes on failure,
- human-readable error messages by default,
- a machine-readable JSON error mode for automation,
- structured logging with a JSON log format option for batch or service use.

The error envelope should carry at least:

- error code,
- message,
- command,
- relevant input context when safe to emit.

## Prepared Runtime And File-Format Contract

The prepared runtime root is a public on-disk artifact and must be documented
as such.

### Required Root Layout

The v1 prepared runtime root should contain:

- `manifest.json`
- `mapping_metadata.parquet`
- `hyperspectral_vnir.f32` or `.npy`
- `hyperspectral_swir.f32` or `.npy`
- `source_<sensor_id>_vnir.f32` or `.npy`
- `source_<sensor_id>_swir.f32` or `.npy`
- `checksums.json`

Additional files are allowed, but the documented files above form the stable
minimum contract.

### Manifest Contract

`manifest.json` should be the primary schema-bearing file and must include:

- `schema_version`
- `package_version`
- `source_siac_root`
- `source_siac_build_id` or equivalent provenance identifier
- `prepared_at`
- `source_sensors`
- `supported_output_modes`
- `row_count`
- `vnir_wavelength_range_nm`
- `swir_wavelength_range_nm`
- `array_dtype`
- `file_checksums`

Optional additive fields are allowed in minor releases.

### Row-Alignment Contract

The prepared runtime standard must guarantee that row `i` refers to the same
library spectrum across:

- `mapping_metadata.parquet`
- hyperspectral arrays
- source-sensor retrieval matrices

Any prepared runtime build that cannot guarantee row alignment is invalid.

### Compatibility Rules

- `schema_version` is part of the public compatibility contract.
- Backward-compatible additive fields may be introduced in minor releases.
- Breaking layout or manifest changes require a schema-version bump and a major
  package release.
- The package must accept prepared artifacts written by the same major schema
  version and reject incompatible versions with a clear error.

### Output-Mode Contract

The prepared runtime manifest must declare the supported public output modes:

- `target_sensor`
- `vnir_spectrum`
- `swir_spectrum`
- `full_spectrum`

For `full_spectrum`, the standard behavior is:

- use VNIR values directly below `900 nm`,
- use SWIR values directly above `1000 nm`,
- blend `900-1000 nm` with the documented linear weighted average.

That overlap rule is part of the public standard and must not change within 1.x.

## Production-Readiness Requirements

### Deterministic Numeric Behavior

The production implementation must return the same numeric outputs for the same:

- prepared runtime artifact,
- package version,
- input reflectance,
- sensor definitions,
- output mode,
- `k` value.

Tolerance-based comparisons are acceptable in tests, but runtime behavior must
be deterministic under the supported environment matrix.

### Input Validation And Failure Modes

The production package must validate:

- missing required sensor fields,
- invalid segment assignments,
- malformed SRF arrays,
- unsupported output modes,
- reflectance values with invalid shapes or unresolvable band mappings,
- insufficient valid-band support for a segment.

Failures must be explicit and machine-readable. Silent fallback behavior is not
acceptable for production release.

### Logging And Errors

The production package should define:

- a stable logger namespace,
- structured log fields for command, sensor pair, output mode, and timing,
- typed Python exceptions rooted in one public base error type,
- JSON CLI error output for automation workflows.

### Reproducibility And Provenance

Every prepared runtime build must record:

- package version,
- schema version,
- source SIAC package path or identifier,
- sensor ids used in preparation,
- preparation timestamp,
- array dtype,
- checksum records for all stable files.

This metadata is required for auditability and release support.

### Memory And Performance

The production design should standardize on:

- `float32` dense arrays,
- memory-mapped numeric assets,
- prepared retrieval matrices instead of repeated wide-table scans,
- Python-first implementation with profiling-driven acceleration only where
  justified.

Performance work is production hardening, but it must not alter the public API
or the prepared-runtime contract.

## Packaging Standard

The package should be upgraded from build scaffolding to a public Python
distribution.

### Required Packaging Metadata

The production release should include complete PEP 621 metadata, including:

- public distribution name `spectral-library`,
- version,
- description,
- readme,
- license metadata,
- authors or maintainers,
- project URLs,
- classifiers,
- supported Python versions.

### Supported Python Versions

For v1, support should be documented and CI-tested for:

- Python `3.9`
- Python `3.10`
- Python `3.11`
- Python `3.12`

The release must not claim support for versions that are not exercised in CI.

### Artifact Policy

Every release should publish:

- one source distribution (`sdist`)
- platform-independent wheels when possible

Release artifacts must pass install and CLI smoke tests before publish.

### Dependency Policy

Dependencies should be separated into:

- runtime dependencies required for the public package,
- optional acceleration dependencies,
- development dependencies for test, lint, type checks, and docs.

The public package must not require development-only tools at runtime.

### Typed Package Marker

The v1 release should include:

- `py.typed`

This makes the public type annotations discoverable and consistent with the
stable API promise.

### Missing Release Assets That Must Be Added

Before the first public release, add:

- a top-level license file,
- a changelog file,
- release notes for the tagged version,
- user-facing install and quickstart documentation for the mapping package.

## Release Engineering Standard

### CI Split

CI should be split into:

1. fast PR checks
2. release checks

Fast PR checks should run on every change and include:

- unit tests for public modules,
- CLI parser and smoke tests,
- lint and formatting checks,
- type checks for the public API,
- docs link and basic render checks.

Release checks should run only for release candidates or tags and include:

- full test suite,
- benchmark suite,
- build of wheel and sdist,
- clean-environment installation tests,
- prepared-runtime compatibility tests,
- publish step after all gates pass.

### Release Workflow

The release workflow should be tag-based and include:

1. create version bump and changelog entry,
2. tag the release,
3. build wheel and sdist,
4. run artifact smoke tests in a clean environment,
5. publish to the package index,
6. publish release notes,
7. run post-release verification.

### Versioning Policy

Use semantic versioning for the public contracts:

- patch:
  bug fixes, performance improvements, internal refactors, documentation
  updates, and non-breaking validation clarifications
- minor:
  additive optional flags, additive manifest fields, additive API helpers, and
  backward-compatible feature additions
- major:
  contract-breaking changes to Python API, CLI, output-mode behavior, or the
  prepared-runtime schema

The chosen distribution name does not affect the internal semantic-versioning
rules, but any later rename of the distribution, import package, or CLI should
be treated as a major release event.

### Post-Release Verification

After each public release, verify in a clean environment:

- `pip install` succeeds,
- import of `spectral_library` succeeds,
- `spectral-library --help` succeeds,
- minimal prepare and map CLI workflows succeed on a small fixture,
- the documented public symbols are importable.

## Test And Benchmark Gate

Production release requires contract testing in addition to algorithmic
correctness testing.

### Contract Tests

The release gate must include:

- Python API contract tests for public imports, constructor signatures, and
  result fields
- CLI contract tests for command presence, core flags, exit-code behavior, and
  JSON error mode
- prepared-runtime tests for required files, manifest fields, schema-version
  handling, and row-alignment validation

### Release Smoke Tests

Every release candidate must:

- build wheel and sdist,
- install them in a clean environment,
- run CLI help,
- execute one minimal prepare workflow,
- execute one minimal map workflow,
- confirm public imports and type metadata are present.

### Compatibility Tests

The package must:

- successfully load prepared artifacts written by the same major schema version,
- reject incompatible schema versions with a clear compatibility error,
- keep stable output-mode semantics across 1.x.

### Functional Algorithm Gate

The existing mapping and reconstruction functional tests remain part of the
production gate:

- convolution sanity,
- identity retrieval,
- segment isolation,
- overlap blending,
- missing-band handling,
- storage-consistency checks.

### Benchmark Gate

Benchmark reports must include:

- per-band RMSE,
- per-band MAE,
- per-band bias

for:

- `target_sensor`
- `vnir_spectrum`
- `swir_spectrum`
- `full_spectrum`

Initial v1 release acceptance thresholds:

- target-sensor mapping:
  mean per-band RMSE must be no worse than `5%` above the regression baseline
  on the committed reference benchmark, and mean absolute per-band bias must be
  `<= 0.01`
- direct hyperspectral reconstruction:
  VNIR RMSE must be `<= 0.02`, SWIR RMSE must be `<= 0.03`, and mean absolute
  bias must be `<= 0.01` on held-out library truth
- full-spectrum overlap quality:
  the blend region must not introduce a discontinuity larger than `0.005`
  reflectance units at the `900/1000 nm` boundaries on the reference test set

These thresholds are release gates for v1 package readiness, not claims about
all possible sensor pairs.

## Recommended v1 Release Shape

The first production release should ship as:

- a public Python package,
- a stable CLI,
- a versioned prepared-runtime standard,
- stable mapping and reconstruction output modes,
- release automation and verification,
- explicit compatibility boundaries between public and internal code.

The release should stay narrow. Production readiness here means a small stable
surface with clear guarantees, not exposing every internal module as public API.
