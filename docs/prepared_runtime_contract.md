# Prepared Runtime Contract

The prepared runtime is the stable on-disk artifact consumed by
`SpectralMapper`.

If you are producing, validating, or distributing prepared runtimes, this page
defines the public contract.

## Required Root Layout

Every prepared runtime root must contain:

| File | Purpose |
| --- | --- |
| `manifest.json` | top-level runtime metadata |
| `mapping_metadata.parquet` | row-aligned metadata table |
| `hyperspectral_vnir.npy` | hyperspectral `400-1000 nm` array |
| `hyperspectral_swir.npy` | hyperspectral `800-2500 nm` array |
| `sensor_schema.json` | packaged sensor schema document |
| `checksums.json` | SHA-256 digest manifest |
| `source_<sensor_id>_vnir.npy` | precomputed source-sensor VNIR matrix |
| `source_<sensor_id>_swir.npy` | precomputed source-sensor SWIR matrix |

The `source_<sensor_id>_*` files are required for every source sensor listed in
the manifest.

## `manifest.json`

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | prepared-runtime schema version |
| `package_version` | package version that created the runtime |
| `source_siac_root` | original SIAC root path |
| `source_siac_build_id` | deterministic source-build digest |
| `prepared_at` | timestamp for runtime creation |
| `source_sensors` | source sensor ids precomputed in the runtime |
| `supported_output_modes` | public output modes available to mapping |
| `row_count` | number of row-aligned spectra |
| `vnir_wavelength_range_nm` | VNIR segment bounds |
| `swir_wavelength_range_nm` | SWIR segment bounds |
| `array_dtype` | stored array dtype |
| `file_checksums` | files included in checksum validation |
| `interpolation_summary` | summary counts for repaired SIAC `nm_*` gaps, when any rows needed repair |

Additive optional fields are allowed in minor releases. Renames, removals, or
required-field changes require a major version change.

## `sensor_schema.json`

The packaged sensor schema document contains:

- `schema_version` matching `manifest.json`
- `canonical_wavelength_grid` equal to `start_nm=400`, `end_nm=2500`,
  `step_nm=1`
- `sensors`

Each sensor entry stores an `rsrf`-compatible `response_definition` for every
band:

```json
{
  "sensor_id": "sentinel-2a_msi",
  "bands": [
    {
      "band_id": "blue",
      "segment": "vnir",
      "response_definition": {
        "wavelength_nm": [456.0, 457.0, 458.0],
        "response": [0.001, 0.01, 0.05]
      }
    }
  ]
}
```

Band rules:

| Rule | Meaning |
| --- | --- |
| unique `band_id` | no duplicates within one sensor |
| valid `segment` | must be `vnir` or `swir` |
| valid `response_definition` | must be accepted by `rsrf` |
| increasing sampled wavelengths | realized sampled wavelengths must be strictly increasing |
| positive SRF support | realized SRF must contain at least one positive sample |
| segment-bounded support | positive support must stay within the declared segment |

Segment bounds:

- `vnir`: `400-1000 nm`
- `swir`: `800-2500 nm`

## `checksums.json`

`checksums.json` is part of the required runtime layout.

It stores:

- `schema_version`
- `files`

`files` maps file names to SHA-256 digests.

Validation can skip hashing with:

- `verify_checksums=False`
- `spectral-library validate-prepared-library --no-verify-checksums`

But the checksum document itself is still required.

## Validation Rules

`validate_prepared_library(...)` and
`spectral-library validate-prepared-library` check:

- manifest readability and schema compatibility
- presence of all required root files
- array readability
- sensor-schema readability
- row-index continuity and uniqueness in `mapping_metadata.parquet`
- checksum integrity when hashing is enabled

During `prepare_mapping_library(...)`, blank `nm_*` cells in the SIAC
normalized spectra table are handled conservatively:

- leading and trailing gaps are edge-extrapolated onto the canonical grid
- small internal gaps are linearly interpolated
- rows with fewer than two numeric values still fail
- rows with more than `8` internal missing samples or an internal missing run
  longer than `8` samples fail

Prepared manifests record repair counts in `interpolation_summary` so runtime
artifacts can be audited after build.

## Stable Output Modes

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

Repository contract coverage for this standard lives in the public repo at:

- [tests/test_release_contracts.py](https://github.com/MarcYin/spectral_library/blob/main/tests/test_release_contracts.py)

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Python API Reference](python_api_reference.md)
- [Mathematical Foundations](theory.md)
