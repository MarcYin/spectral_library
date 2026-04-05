# Python API Reference

This page documents the stable public imports for `spectral-library`.

## Stable Imports

```python
from spectral_library import (
    BatchMappingArrayResult,
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

### 0. `download_prepared_library(...)`

Download a pre-built prepared runtime from GitHub Releases or a direct URL.

```python
from pathlib import Path

from spectral_library.runtime_download import download_prepared_library

output = download_prepared_library(
    Path("build/mapping_runtime"),
    # tag="v0.2.0",       # optional: pin to a specific release
    # url="https://...",   # optional: direct tarball URL
    # sha256="abc123...",  # optional: expected digest
)
```

Returns:

- `Path` to the extracted runtime directory

Raises:

- `RuntimeDownloadError` when the download, verification, or extraction fails

### 1. `prepare_mapping_library(...)`

Build the prepared runtime used by mapping.

```python
from pathlib import Path

from spectral_library import prepare_mapping_library

manifest = prepare_mapping_library(
    siac_root=Path("build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37"),
    srf_root=None,
    output_root=Path("build/official_mapping_runtime"),
    source_sensors=["terra_modis", "sentinel-2a_msi", "landsat-8_oli", "landsat-9_oli2"],
    knn_index_backends=["faiss"],
)
```

Built-in sensors now use canonical `rsrf` ids directly, such as
`sentinel-2a_msi`, `sentinel-2b_msi`, `sentinel-2c_msi`, `landsat-8_oli`,
`landsat-9_oli2`, `terra_modis`, `snpp_viirs`, `noaa-20_viirs`, and
`noaa-21_viirs`. Pass `srf_root=Path(...)` only when you need extra local
sensor JSON definitions. If your `rsrf` install does not include its registry
data, set `RSRF_ROOT` to an `rsrf` checkout before using the built-in sensor
catalog.

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
| `compile_linear_mapper(...)` | compile a fixed array-to-array mapper for high-throughput inference |
| `map_reflectance(...)` | map one source observation on the slim default path |
| `map_reflectance_debug(...)` | map one source observation and include neighbor/diagnostic payloads |
| `map_reflectance_batch(...)` | map many observations and return dense arrays on the fast default path |
| `map_reflectance_batch_debug(...)` | map many observations and return rich `MappingResult` objects |
| `map_reflectance_batch_arrays(...)` | explicit alias for the dense batch path |

## Single-Sample Example

```python
from pathlib import Path
import csv

from spectral_library import SpectralMapper

query_path = Path("examples/official_mapping/queries/single/blue_spruce_needles_terra_modis.csv")
reflectance = {}
with query_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        reflectance[row["band_id"]] = float(row["reflectance"])

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
result = mapper.map_reflectance(
    source_sensor="terra_modis",
    reflectance=reflectance,
    output_mode="target_sensor",
    target_sensor="sentinel-2a_msi",
    neighbor_estimator="simplex_mixture",
    knn_backend="scipy_ckdtree",
    knn_eps=0.05,
    exclude_row_ids=[
        "usgs_v7:usgs_v7_002183:Blue_Spruce DW92-5 needles    BECKa AREF",
    ],
)
```

`map_reflectance(...)` now defaults to the slim result path. If you need
neighbor ids, per-segment diagnostics, or review payloads, call
`map_reflectance_debug(...)` instead.

## Batch Example

```python
from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
batch = mapper.map_reflectance_batch(
    source_sensor="landsat-8_oli",
    sample_ids=["blue_spruce_needles", "pale_brown_silty_loam", "tap_water", "asphalt_road"],
    reflectance_rows=[
        {"ultra_blue": 0.08565344, "blue": 0.08364366, "green": 0.10364797, "red": 0.06556322, "nir": 0.39777808, "swir1": 0.09562342, "swir2": 0.03500909},
        {"ultra_blue": 0.05807071, "blue": 0.09366218, "green": 0.19350962, "red": 0.28368705, "nir": 0.36777489, "swir1": 0.48421241, "swir2": 0.45429637},
        {"ultra_blue": 0.02863228, "blue": 0.02789226, "green": 0.02699205, "red": 0.02652986, "nir": 0.02617234, "swir1": 0.02100237, "swir2": 0.01889161},
        {"ultra_blue": 0.06766724, "blue": 0.07308879, "green": 0.08826971, "red": 0.10323628, "nir": 0.12662063, "swir1": 0.19511989, "swir2": 0.21389012},
    ],
    output_mode="target_sensor",
    target_sensor="sentinel-2a_msi",
    neighbor_estimator="simplex_mixture",
    knn_backend="scipy_ckdtree",
    exclude_row_ids_per_sample=[
        "usgs_v7:usgs_v7_002183:Blue_Spruce DW92-5 needles    BECKa AREF",
        "ecostress_v1:ecostress_v1_002334:Pale brown silty loam",
        "ecostress_v1:ecostress_v1_003451:Tap water",
        "usgs_v7:usgs_v7_000004:Asphalt GDS376 Blck_Road old ASDFRa AREF",
    ],
)
```

`map_reflectance_batch(...)` returns `BatchMappingArrayResult` by default, so
the main outputs are `batch.reflectance`, `batch.output_columns`, and
`batch.source_fit_rmse`. If you need per-sample `MappingResult` objects and
diagnostics, call `map_reflectance_batch_debug(...)`.

Supported exclusion controls:

- `neighbor_estimator="mean"`, `"distance_weighted_mean"`, or `"simplex_mixture"`
- `knn_backend="numpy"`, `"scipy_ckdtree"`, `"faiss"`, `"pynndescent"`, or `"scann"`
- `knn_eps > 0` to relax supported ANN searches
- `map_reflectance(..., exclude_row_ids=..., exclude_sample_names=...)`
- `map_reflectance_batch(..., exclude_row_ids=..., exclude_sample_names=...)`
- `map_reflectance_batch(..., exclude_row_ids_per_sample=..., self_exclude_sample_id=True)`

That means the public Python API can now reproduce the same held-out
self-exclusion workflow shown in [Official Sensor Examples](official_sensor_examples.md).

Optional backend install extras:

```bash
python3 -m pip install "spectral-library[knn]"
python3 -m pip install "spectral-library[knn-faiss]"
python3 -m pip install "spectral-library[knn-pynndescent]"
python3 -m pip install "spectral-library[knn-scann]"
```

## High-Throughput Linear Mapper

If you need to map millions of pixels and can trade retrieval-style diagnostics
for a fixed fast model, compile the prepared runtime into one linear mapper and
then apply it to dense arrays in chunks.

```python
import numpy as np
from spectral_library import SpectralMapper

mapper = SpectralMapper(Path("build/official_mapping_runtime"))
linear_mapper = mapper.compile_linear_mapper(
    source_sensor="landsat-8_oli",
    target_sensor="sentinel-2a_msi",
    output_mode="target_sensor",
    dtype="float32",
)

pixel_block = np.asarray(
    [
        [0.08565344, 0.08364366, 0.10364797, 0.06556322, 0.39777808, 0.09562342, 0.03500909],
        [0.05807071, 0.09366218, 0.19350962, 0.28368705, 0.36777489, 0.48421241, 0.45429637],
    ],
    dtype=np.float32,
)
mapped = linear_mapper.map_array(pixel_block, chunk_size=65536)
```

This fast path is intentionally narrow:

- input must be a dense `numpy.ndarray` in source-band order
- it returns a plain `numpy.ndarray` instead of `MappingResult` objects
- it is best suited for raster/tile processing with reusable `out=` buffers or
  `numpy.memmap` outputs

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
| `neighbor_ids_by_segment` | retrieved library row ids for each segment; empty on the slim default path |
| `neighbor_distances_by_segment` | neighbor distances for each segment; empty on the slim default path |
| `segment_outputs` | successful segment reconstructions; empty on the slim default path |
| `segment_valid_band_counts` | number of valid source bands used per segment |
| `diagnostics` | stable debug payload, including per-segment query values when requested; empty on the slim default path |

Helper:

- `to_summary_dict()`

Use `map_reflectance_debug(...)` when you need the populated diagnostic fields.

### `BatchMappingResult`

Returned by `map_reflectance_batch_debug(...)`.

| Field | Meaning |
| --- | --- |
| `sample_ids` | output sample ids in order |
| `results` | per-sample `MappingResult` values |

Helper:

- `to_summary_dict()`

### `BatchMappingArrayResult`

Returned by `map_reflectance_batch(...)` and `map_reflectance_batch_arrays(...)`.

| Field | Meaning |
| --- | --- |
| `sample_ids` | output sample ids in order |
| `reflectance` | dense `(n_samples, n_outputs)` output matrix |
| `source_fit_rmse` | one scalar source-sensor fit RMSE per sample; lower is better |
| `output_columns` | output column labels aligned to `reflectance[:, j]` |
| `wavelength_nm` | wavelength axis for spectral modes, otherwise `None` |

## Public Error Types

All public errors inherit from `SpectralLibraryError` and carry structured
fields for programmatic handling.

```python
from pathlib import Path

from spectral_library import (
    MappingInputError,
    PreparedLibraryCompatibilityError,
    PreparedLibraryValidationError,
    SpectralLibraryError,
    SpectralMapper,
    validate_prepared_library,
)

# Catch all spectral-library errors
try:
    manifest = validate_prepared_library(Path("build/mapping_runtime"))
except PreparedLibraryCompatibilityError as exc:
    print(f"Runtime was built with an incompatible schema: {exc.message}")
    print(f"Context: {exc.context}")
except PreparedLibraryValidationError as exc:
    print(f"Runtime validation failed: {exc.message}")
except SpectralLibraryError as exc:
    print(f"[{exc.code}] {exc.message}")

# Handle mapping-specific errors
mapper = SpectralMapper(Path("build/mapping_runtime"))
try:
    result = mapper.map_reflectance(
        source_sensor="terra_modis",
        reflectance={"blue": 0.08, "nir": 0.34},
        output_mode="target_sensor",
        target_sensor="sentinel-2a_msi",
    )
except MappingInputError as exc:
    # Structured error for logging or API responses
    error_dict = exc.to_dict(command="map_reflectance")
    print(error_dict)
    # {"error_code": "invalid_mapping_input", "message": "...", "command": "map_reflectance", "context": {...}}
```

### Public error types

| Class | `error_code` | Raised when |
| --- | --- | --- |
| `SpectralLibraryError` | *(varies)* | Base class for all package errors |
| `SensorSchemaError` | `invalid_sensor_schema` | Sensor schema data from `rsrf` or local JSON is malformed or unavailable |
| `PreparedLibraryBuildError` | `prepare_failed` | Runtime build fails |
| `PreparedLibraryValidationError` | `invalid_prepared_library` | Runtime layout or checksums are invalid |
| `PreparedLibraryCompatibilityError` | `prepared_library_incompatible` | Schema version mismatch |
| `MappingInputError` | `invalid_mapping_input` | Invalid query input |

Every public error carries:

- `code` - machine-readable error code string
- `message` - human-readable description
- `context` - dict with additional diagnostic fields
- `to_dict(command=...)` - serialize to a structured dict

## Public Data Models

### `PreparedLibraryManifest`

Represents the stable prepared-runtime manifest returned by:

- `prepare_mapping_library(...)`
- `validate_prepared_library(...)`

Current prepared runtimes also expose `interpolation_summary` so you can see how
many SIAC rows required gap repair during prepare. When requested at prepare
time, they also expose `knn_index_artifacts` so you can see which persisted ANN
indexes were written into the runtime.

Debug diagnostics expose a heuristic `confidence_score` at the overall mapping
level and per segment. This is not a calibrated uncertainty model; it is a
compact ranking signal derived from distance, source fit, weight
concentration, and valid-band coverage. The diagnostics also include
`confidence_policy`, which currently maps the score to:

- `high` / `accept` for scores `>= 0.85`
- `medium` / `manual_review` for scores `>= 0.60` and `< 0.85`
- `low` / `reject` for scores `< 0.60`

`benchmark_mapping(...)` also accepts `max_test_rows` so large prepared
libraries can run bounded held-out evaluations without scoring every row in the
test split.

### `SensorSRFSchema`

Represents one public sensor JSON schema in memory.

The runtime format itself is documented in
[Prepared Runtime Contract](prepared_runtime_contract.md).

## Related Docs

- [Getting Started](mapping_quickstart.md)
- [Troubleshooting](troubleshooting.md)
- [Mathematical Foundations](theory.md)
- [Prepared Runtime Contract](prepared_runtime_contract.md)
