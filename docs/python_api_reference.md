# Python API Reference

The stable public imports are:

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

## `prepare_mapping_library(...)`

Build the prepared runtime used by retrieval-based mapping.

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

- `PreparedLibraryBuildError` through the base type `SpectralLibraryError`
- `SensorSchemaError` through the base type `SpectralLibraryError`

## `validate_prepared_library(...)`

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

## `SpectralMapper`

Load a prepared runtime and serve mapping requests.

```python
from pathlib import Path

from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"), verify_checksums=True)
```

Public methods:

- `get_sensor_schema(sensor_id)`
  return the loaded `SensorSRFSchema`
- `map_reflectance(...)`
  map one query to a target sensor or spectral output
- `map_reflectance_batch(...)`
  map many samples from one call

### Single-Sample Example

```python
from pathlib import Path
import csv

from spectral_library import SpectralMapper

query_path = Path("examples/official_mapping/queries/single/veg_soil_mix_modis_terra.csv")
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

### Batch Example

```python
from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
batch = mapper.map_reflectance_batch(
    source_sensor="landsat8_oli",
    sample_ids=["veg_soil_mix", "urban_soil_mix", "water_edge_mix"],
    reflectance_rows=[
        {"ultra_blue": 0.07671433, "blue": 0.08305546, "green": 0.11208171, "red": 0.07725103, "nir": 0.45319132, "swir1": 0.41291375, "swir2": 0.30716747},
        {"ultra_blue": 0.13998734, "blue": 0.14528265, "green": 0.15596958, "red": 0.16224651, "nir": 0.16884728, "swir1": 0.22944125, "swir2": 0.25201676},
        {"ultra_blue": 0.08171818, "blue": 0.08769056, "green": 0.10877393, "red": 0.08937566, "nir": 0.16752776, "swir1": 0.12579797, "swir2": 0.08902762},
    ],
    output_mode="target_sensor",
    target_sensor="sentinel2a_msi",
    k=3,
)
```

## Result Objects

### `MappingResult`

Fields:

- `target_reflectance`
- `target_band_ids`
- `reconstructed_vnir`
- `reconstructed_swir`
- `reconstructed_full_spectrum`
- `reconstructed_wavelength_nm`
- `neighbor_ids_by_segment`
- `neighbor_distances_by_segment`
- `segment_outputs`
- `segment_valid_band_counts`
- `diagnostics`

Helper:

- `to_summary_dict()`

### `BatchMappingResult`

Fields:

- `sample_ids`
- `results`

Helper:

- `to_summary_dict()`

## Public Errors

All public mapping failures inherit from `SpectralLibraryError`.

Common subclasses:

- `PreparedLibraryValidationError`
- `PreparedLibraryCompatibilityError`
- `MappingInputError`

Each public error includes:

- `code`
- `message`
- `context`
- `to_dict(...)`

## Sensor Schemas

`SensorSRFSchema` is the public in-memory representation of a sensor JSON
document. The JSON contract and prepared-runtime layout are documented in
[Prepared Runtime Contract](prepared_runtime_contract.md).

For the mathematical model behind retrieval, target-sensor simulation, and the
benchmark baseline, see [Mathematical Foundations](theory.md).
