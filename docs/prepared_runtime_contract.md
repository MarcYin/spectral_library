# Prepared Runtime Contract

The prepared runtime is the stable on-disk contract used by
`SpectralMapper`.

## Required Root Layout

Every prepared runtime root must contain:

- `manifest.json`
- `mapping_metadata.parquet`
- `hyperspectral_vnir.npy`
- `hyperspectral_swir.npy`
- `sensor_schema.json`
- `checksums.json`
- `source_<sensor_id>_vnir.npy`
- `source_<sensor_id>_swir.npy`

The source-array files are required for every source sensor listed in the
manifest.

## Manifest Fields

`manifest.json` contains:

- `schema_version`
- `package_version`
- `source_siac_root`
- `source_siac_build_id`
- `prepared_at`
- `source_sensors`
- `supported_output_modes`
- `row_count`
- `vnir_wavelength_range_nm`
- `swir_wavelength_range_nm`
- `array_dtype`
- `file_checksums`

Additive optional fields are allowed in minor releases. Renames, removals, or
required-field changes require a major version change.

## Sensor Schema File

`sensor_schema.json` contains:

- `schema_version`
- `canonical_wavelength_grid`
- `sensors`

Each sensor entry follows the public `SensorSRFSchema` JSON shape:

```json
{
  "sensor_id": "sentinel2a_msi",
  "bands": [
    {
      "band_id": "blue",
      "segment": "vnir",
      "wavelength_nm": [456.0, 457.0, 458.0],
      "rsr": [0.001, 0.01, 0.05]
    }
  ]
}
```

Band rules:

- `band_id` must be unique within one sensor
- `segment` must be `vnir` or `swir`
- `wavelength_nm` must be strictly increasing
- `rsr` must contain at least one positive value
- positive support must remain inside the declared segment bounds

Segment bounds:

- `vnir`: `400-1000 nm`
- `swir`: `900-2500 nm`

## Checksums

`checksums.json` is part of the required runtime layout.

It stores:

- `schema_version`
- `files`

`files` maps file names to SHA-256 digests. Validation can skip hashing with
`verify_checksums=False` or `--no-verify-checksums`, but the checksum document
itself is still required.

## Validation Rules

`validate_prepared_library(...)` and
`spectral-library validate-prepared-library` check:

- manifest readability and schema compatibility
- presence of all required root files
- prepared-array readability
- sensor-schema readability
- row-index continuity and uniqueness in `mapping_metadata.parquet`
- checksum integrity when hashing is enabled

## Output Modes

Prepared runtimes currently support these stable output modes:

- `target_sensor`
- `vnir_spectrum`
- `swir_spectrum`
- `full_spectrum`

## Compatibility Policy

The stable `1.x` public contract covers:

- the Python entry points documented in the public docs
- the CLI commands documented in the public docs
- the required prepared-runtime root layout
- the prepared-runtime manifest schema version
- the stable output-mode names

Repository contract tests covering these guarantees are in
[`tests/test_release_contracts.py`](../tests/test_release_contracts.py).
