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
    siac_root=Path("examples/official_mapping/siac"),
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

query_path = Path("examples/official_mapping/queries/single/dense_vegetation_modis_terra.csv")
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
    k=3,
)
```

## Batch Example

```python
from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
batch = mapper.map_reflectance_batch(
    source_sensor="landsat8_oli",
    sample_ids=["dense_vegetation", "bright_soil", "turbid_water", "asphalt"],
    reflectance_rows=[
        {"ultra_blue": 0.03511018, "blue": 0.04232701, "green": 0.08010689, "red": 0.02404218, "nir": 0.56492711, "swir1": 0.45158788, "swir2": 0.28757895},
        {"ultra_blue": 0.16515787, "blue": 0.16991066, "green": 0.17936012, "red": 0.19055300, "nir": 0.21574854, "swir1": 0.33057901, "swir2": 0.34816245},
        {"ultra_blue": 0.08220505, "blue": 0.08545536, "green": 0.09594336, "red": 0.09473721, "nir": 0.02853783, "swir1": 0.0, "swir2": 0.0},
        {"ultra_blue": 0.05254817, "blue": 0.05482323, "green": 0.05978893, "red": 0.06668515, "nir": 0.08302571, "swir1": 0.10971880, "swir2": 0.13942782},
    ],
    output_mode="target_sensor",
    target_sensor="sentinel2a_msi",
    k=3,
)
```

The bundled example runtime keeps the full synthetic catalogue. The published
CLI examples then self-exclude the matching `sample_name` for each query so the
mapper cannot return the identical row as its own neighbor while still keeping
the rest of the catalogue available.

## Result Objects

### `MappingResult`

| Field | Meaning |
| --- | --- |
| `target_reflectance` | mapped target reflectance values, when applicable |
| `target_band_ids` | target band ids paired with `target_reflectance` |
| `reconstructed_vnir` | reconstructed `400-1000 nm` segment |
| `reconstructed_swir` | reconstructed `900-2500 nm` segment |
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
