# Python API Reference

This page documents the stable public imports for `spectral-library`.

## Stable Imports

```python
from spectral_library import (
    BatchMappingResult,
    MappingInputError,
    MappingResult,
    PreparedLibraryCompatibilityError,
    PreparedLibraryManifest,
    PreparedLibraryValidationError,
    SensorSRFSchema,
    SpectralLibraryError,
    SpectralMapper,
    benchmark_mapping,
    prepare_mapping_library,
    validate_prepared_library,
)
```

## Public Workflow

### 1. `prepare_mapping_library(...)`

Build the prepared runtime used by mapping.

```python
from pathlib import Path

from spectral_library import prepare_mapping_library

manifest = prepare_mapping_library(
    siac_root=Path("build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37"),
    srf_root=Path("examples/official_mapping/srfs"),
    output_root=Path("build/official_mapping_runtime"),
    source_sensors=["modis_terra", "sentinel2a_msi", "landsat8_oli", "landsat9_oli"],
)
```

Returns:

- `PreparedLibraryManifest`

Raises:

- `PreparedLibraryBuildError` via `SpectralLibraryError`
- `SensorSchemaError` via `SpectralLibraryError`

### 2. `validate_prepared_library(...)`

Validate the runtime layout and optionally verify checksums.

```python
from pathlib import Path

from spectral_library import validate_prepared_library

manifest = validate_prepared_library(
    Path("build/official_mapping_runtime"),
    verify_checksums=True,
)
```

Returns:

- `PreparedLibraryManifest`

Raises:

- `PreparedLibraryValidationError`
- `PreparedLibraryCompatibilityError`

### 3. `SpectralMapper`

Load a prepared runtime and serve mapping requests.

```python
from pathlib import Path

from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"), verify_checksums=True)
```

Public methods:

| Method | Purpose |
| --- | --- |
| `get_sensor_schema(sensor_id)` | return the loaded `SensorSRFSchema` |
| `map_reflectance(...)` | map one source observation |
| `map_reflectance_batch(...)` | map many observations in one call |

## Single-Sample Example

```python
from pathlib import Path
import csv

from spectral_library import SpectralMapper

query_path = Path("examples/official_mapping/queries/single/blue_spruce_needles_modis_terra.csv")
reflectance = {}
with query_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        reflectance[row["band_id"]] = float(row["reflectance"])

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
result = mapper.map_reflectance(
    source_sensor="modis_terra",
    reflectance=reflectance,
    output_mode="target_sensor",
    target_sensor="sentinel2a_msi",
)
```

## Batch Example

```python
from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
batch = mapper.map_reflectance_batch(
    source_sensor="landsat8_oli",
    sample_ids=["blue_spruce_needles", "pale_brown_silty_loam", "tap_water", "asphalt_road"],
    reflectance_rows=[
        {"ultra_blue": 0.08565344, "blue": 0.08364366, "green": 0.10364797, "red": 0.06556322, "nir": 0.39777808, "swir1": 0.09562342, "swir2": 0.03500909},
        {"ultra_blue": 0.05807071, "blue": 0.09366218, "green": 0.19350962, "red": 0.28368705, "nir": 0.36777489, "swir1": 0.48421241, "swir2": 0.45429637},
        {"ultra_blue": 0.02863228, "blue": 0.02789226, "green": 0.02699205, "red": 0.02652986, "nir": 0.02617234, "swir1": 0.02100237, "swir2": 0.01889161},
        {"ultra_blue": 0.06766724, "blue": 0.07308879, "green": 0.08826971, "red": 0.10323628, "nir": 0.12662063, "swir1": 0.19511989, "swir2": 0.21389012},
    ],
    output_mode="target_sensor",
    target_sensor="sentinel2a_msi",
)
```

The public official examples in the docs use the external full SIAC library and
exclude exact held-out rows in the CLI workflow with `--exclude-row-id` or the
batch CSV `exclude_row_id` column. The Python examples above show the direct
runtime calls themselves; if you need the exact held-out evaluation behavior,
follow the CLI example inputs in [Official Sensor Examples](official_sensor_examples.md).

## Result Objects

### `MappingResult`

| Field | Meaning |
| --- | --- |
| `target_reflectance` | mapped target reflectance values, when applicable |
| `target_band_ids` | target band ids paired with `target_reflectance` |
| `reconstructed_vnir` | reconstructed `400-1000 nm` segment |
| `reconstructed_swir` | reconstructed `800-2500 nm` segment |
| `reconstructed_full_spectrum` | overlap-blended `400-2500 nm` reconstruction |
| `reconstructed_wavelength_nm` | wavelength grid for the requested spectral output |
| `neighbor_ids_by_segment` | retrieved library row ids for each segment |
| `neighbor_distances_by_segment` | neighbor distances for each segment |
| `segment_outputs` | successful segment reconstructions |
| `segment_valid_band_counts` | number of valid source bands used per segment |
| `diagnostics` | stable diagnostic payload |

Helper:

- `to_summary_dict()`

### `BatchMappingResult`

| Field | Meaning |
| --- | --- |
| `sample_ids` | output sample ids in order |
| `results` | per-sample `MappingResult` values |

Helper:

- `to_summary_dict()`

## Public Error Types

All public errors inherit from `SpectralLibraryError`.

Common subclasses:

- `PreparedLibraryValidationError`
- `PreparedLibraryCompatibilityError`
- `MappingInputError`

Every public error carries:

- `code`
- `message`
- `context`
- `to_dict(...)`

## Public Data Models

### `PreparedLibraryManifest`

Represents the stable prepared-runtime manifest returned by:

- `prepare_mapping_library(...)`
- `validate_prepared_library(...)`

### `SensorSRFSchema`

Represents one public sensor JSON schema in memory.

The runtime format itself is documented in
[Prepared Runtime Contract](prepared_runtime_contract.md).

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Mathematical Foundations](theory.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)
