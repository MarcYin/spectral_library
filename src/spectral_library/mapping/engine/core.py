"""Core models, constants, and shared helpers for spectral mapping."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import duckdb
import numpy as np

from ... import _rustaccel
from ..._version import __version__

PREPARED_SCHEMA_VERSION = "3.0.0"
SUPPORTED_OUTPUT_MODES = (
    "target_sensor",
    "vnir_spectrum",
    "swir_spectrum",
    "full_spectrum",
)
SUPPORTED_NEIGHBOR_ESTIMATORS = (
    "mean",
    "distance_weighted_mean",
    "simplex_mixture",
)
SUPPORTED_KNN_BACKENDS = (
    "numpy",
    "scipy_ckdtree",
    "faiss",
    "pynndescent",
    "scann",
)
SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS = (
    "faiss",
    "pynndescent",
    "scann",
)
SCANN_MIN_AH_TRAINING_SAMPLE_SIZE = 16
KNN_BACKEND_INSTALL_HINTS = {
    "scipy_ckdtree": 'pip install "spectral-library[knn]"',
    "faiss": 'pip install "spectral-library[knn-faiss]"',
    "pynndescent": 'pip install "spectral-library[knn-pynndescent]"',
    "scann": 'pip install "spectral-library[knn-scann]"',
}
ZARR_INSTALL_HINT = 'pip install "spectral-library[zarr]"'
CONFIDENCE_POLICY_VERSION = "1"
CONFIDENCE_REVIEW_THRESHOLD = 0.60
CONFIDENCE_ACCEPT_THRESHOLD = 0.85
SEGMENTS = ("vnir", "swir")
CANONICAL_START_NM = 400
CANONICAL_END_NM = 2500
VNIR_START_NM = 400
VNIR_END_NM = 1000
SWIR_START_NM = 800
SWIR_END_NM = 2500
FULL_BLEND_START_NM = 800
FULL_BLEND_END_NM = 1000
MAX_INTERNAL_INTERPOLATED_GAP_COUNT = 8
MAX_INTERNAL_INTERPOLATED_RUN_COUNT = 8

CANONICAL_WAVELENGTHS = np.arange(CANONICAL_START_NM, CANONICAL_END_NM + 1, dtype=np.int32)
VNIR_WAVELENGTHS = np.arange(VNIR_START_NM, VNIR_END_NM + 1, dtype=np.int32)
SWIR_WAVELENGTHS = np.arange(SWIR_START_NM, SWIR_END_NM + 1, dtype=np.int32)
SEGMENT_RANGES = {
    "vnir": (VNIR_START_NM, VNIR_END_NM),
    "swir": (SWIR_START_NM, SWIR_END_NM),
}
SEGMENT_WAVELENGTHS = {
    "vnir": VNIR_WAVELENGTHS,
    "swir": SWIR_WAVELENGTHS,
}
SEGMENT_FILE_NAMES = {
    "vnir": "hyperspectral_vnir.npy",
    "swir": "hyperspectral_swir.npy",
}
FULL_WAVELENGTH_COUNT = int(CANONICAL_WAVELENGTHS.size)
RSRF_ROOT_ENV_VAR = "RSRF_ROOT"
RSRF_CACHE_DIR_ENV_VAR = "RSRF_CACHE_DIR"
RSRF_REGISTRY_RELATIVE_PATH = Path("data") / "registry" / "sensors.parquet"
RSRF_INSTALL_HINT = 'pip install "rsrf>=0.3.1"'
RSRF_REPRESENTATION_VARIANT = "band_average"


@dataclass(frozen=True)
class _RsrfBandSelection:
    band_id: str
    rsrf_band_id: str
    segment: str


RSRF_SENSOR_BAND_SELECTIONS: dict[str, tuple[_RsrfBandSelection, ...]] = {
    "sentinel-2a_msi": (
        _RsrfBandSelection("ultra_blue", "B01", "vnir"),
        _RsrfBandSelection("blue", "B02", "vnir"),
        _RsrfBandSelection("green", "B03", "vnir"),
        _RsrfBandSelection("red", "B04", "vnir"),
        _RsrfBandSelection("nir", "B8A", "vnir"),
        _RsrfBandSelection("swir1", "B11", "swir"),
        _RsrfBandSelection("swir2", "B12", "swir"),
    ),
    "sentinel-2b_msi": (
        _RsrfBandSelection("ultra_blue", "B01", "vnir"),
        _RsrfBandSelection("blue", "B02", "vnir"),
        _RsrfBandSelection("green", "B03", "vnir"),
        _RsrfBandSelection("red", "B04", "vnir"),
        _RsrfBandSelection("nir", "B8A", "vnir"),
        _RsrfBandSelection("swir1", "B11", "swir"),
        _RsrfBandSelection("swir2", "B12", "swir"),
    ),
    "sentinel-2c_msi": (
        _RsrfBandSelection("ultra_blue", "B01", "vnir"),
        _RsrfBandSelection("blue", "B02", "vnir"),
        _RsrfBandSelection("green", "B03", "vnir"),
        _RsrfBandSelection("red", "B04", "vnir"),
        _RsrfBandSelection("nir", "B8A", "vnir"),
        _RsrfBandSelection("swir1", "B11", "swir"),
        _RsrfBandSelection("swir2", "B12", "swir"),
    ),
    "landsat-8_oli": (
        _RsrfBandSelection("ultra_blue", "B1", "vnir"),
        _RsrfBandSelection("blue", "B2", "vnir"),
        _RsrfBandSelection("green", "B3", "vnir"),
        _RsrfBandSelection("red", "B4", "vnir"),
        _RsrfBandSelection("nir", "B5", "vnir"),
        _RsrfBandSelection("swir1", "B6", "swir"),
        _RsrfBandSelection("swir2", "B7", "swir"),
    ),
    "landsat-9_oli2": (
        _RsrfBandSelection("ultra_blue", "B1", "vnir"),
        _RsrfBandSelection("blue", "B2", "vnir"),
        _RsrfBandSelection("green", "B3", "vnir"),
        _RsrfBandSelection("red", "B4", "vnir"),
        _RsrfBandSelection("nir", "B5", "vnir"),
        _RsrfBandSelection("swir1", "B6", "swir"),
        _RsrfBandSelection("swir2", "B7", "swir"),
    ),
    "terra_modis": (
        _RsrfBandSelection("blue", "B3", "vnir"),
        _RsrfBandSelection("green", "B4", "vnir"),
        _RsrfBandSelection("red", "B1", "vnir"),
        _RsrfBandSelection("nir", "B2", "vnir"),
        _RsrfBandSelection("swir1", "B6", "swir"),
        _RsrfBandSelection("swir2", "B7", "swir"),
    ),
    "snpp_viirs": (
        _RsrfBandSelection("blue", "M2", "vnir"),
        _RsrfBandSelection("green", "M4", "vnir"),
        _RsrfBandSelection("red", "M5", "vnir"),
        _RsrfBandSelection("nir", "M7", "vnir"),
        _RsrfBandSelection("swir1", "M10", "swir"),
        _RsrfBandSelection("swir2", "M11", "swir"),
    ),
    "noaa-20_viirs": (
        _RsrfBandSelection("blue", "M2", "vnir"),
        _RsrfBandSelection("green", "M4", "vnir"),
        _RsrfBandSelection("red", "M5", "vnir"),
        _RsrfBandSelection("nir", "M7", "vnir"),
        _RsrfBandSelection("swir1", "M10", "swir"),
        _RsrfBandSelection("swir2", "M11", "swir"),
    ),
    "noaa-21_viirs": (
        _RsrfBandSelection("blue", "M2", "vnir"),
        _RsrfBandSelection("green", "M4", "vnir"),
        _RsrfBandSelection("red", "M5", "vnir"),
        _RsrfBandSelection("nir", "M7", "vnir"),
        _RsrfBandSelection("swir1", "M10", "swir"),
        _RsrfBandSelection("swir2", "M11", "swir"),
    ),
}


class SpectralLibraryError(Exception):
    """Base error type used by the mapping workflow."""

    def __init__(self, code: str, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})

    def to_dict(self, *, command: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "error_code": self.code,
            "message": self.message,
        }
        if command:
            payload["command"] = command
        if self.context:
            payload["context"] = self.context
        return payload

    def __str__(self) -> str:
        return self.message

class SensorSchemaError(SpectralLibraryError):
    """Raised when an SRF schema is malformed or internally inconsistent."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_sensor_schema", message, context=context)

class PreparedLibraryBuildError(SpectralLibraryError):
    """Raised while building a prepared runtime bundle."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("prepare_failed", message, context=context)

class PreparedLibraryValidationError(SpectralLibraryError):
    """Raised when a prepared runtime bundle fails validation."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_prepared_library", message, context=context)

class PreparedLibraryCompatibilityError(SpectralLibraryError):
    """Raised when a prepared runtime uses an incompatible schema version."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("prepared_library_incompatible", message, context=context)

class MappingInputError(SpectralLibraryError):
    """Raised when a mapping request cannot be executed as provided."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_mapping_input", message, context=context)

@dataclass
class SensorBandDefinition:
    """Single-band spectral response definition for a multispectral sensor."""

    band_id: str
    segment: str
    wavelength_nm: tuple[float, ...]
    rsr: tuple[float, ...]
    center_nm: float | None = None
    fwhm_nm: float | None = None
    support_min_nm: float | None = None
    support_max_nm: float | None = None

    def __post_init__(self) -> None:
        if not self.band_id:
            raise SensorSchemaError("Sensor band_id must be non-empty.")
        if self.segment not in SEGMENTS:
            raise SensorSchemaError(
                "Sensor segment must be either 'vnir' or 'swir'.",
                context={"band_id": self.band_id, "segment": self.segment},
            )
        if len(self.wavelength_nm) != len(self.rsr):
            raise SensorSchemaError(
                "Sensor wavelength_nm and rsr arrays must have the same length.",
                context={"band_id": self.band_id},
            )
        if len(self.wavelength_nm) == 0:
            raise SensorSchemaError("Sensor bands must define at least one SRF sample.", context={"band_id": self.band_id})

        wavelengths = np.asarray(self.wavelength_nm, dtype=np.float64)
        rsr = np.asarray(self.rsr, dtype=np.float64)
        if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(rsr)):
            raise SensorSchemaError("Sensor SRF values must be finite.", context={"band_id": self.band_id})
        if np.any(np.diff(wavelengths) <= 0):
            raise SensorSchemaError("Sensor wavelengths must be strictly increasing.", context={"band_id": self.band_id})
        if np.all(rsr <= 0):
            raise SensorSchemaError(
                "Sensor SRF values must include at least one positive support sample.",
                context={"band_id": self.band_id},
            )

        support_min = self.support_min_nm if self.support_min_nm is not None else float(wavelengths[rsr > 0].min())
        support_max = self.support_max_nm if self.support_max_nm is not None else float(wavelengths[rsr > 0].max())
        segment_min, segment_max = SEGMENT_RANGES[self.segment]
        tolerance = 1e-6
        if support_min < segment_min - tolerance or support_max > segment_max + tolerance:
            raise SensorSchemaError(
                "Sensor band support must stay inside its declared segment range.",
                context={
                    "band_id": self.band_id,
                    "segment": self.segment,
                    "support_min_nm": support_min,
                    "support_max_nm": support_max,
                },
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SensorBandDefinition":
        from ..adapters.sensors import _coerce_sensor_band_response_definition, _sensor_band_segment_from_payload

        if "band_id" not in payload:
            raise SensorSchemaError("Sensor band definition is missing band_id.")
        if "response_definition" not in payload:
            raise SensorSchemaError(
                "Sensor band definition must include an rsrf-compatible response definition.",
                context={"band_id": payload.get("band_id")},
            )
        segment = _sensor_band_segment_from_payload(payload)
        ordered_wavelengths, ordered_rsr, center_nm, fwhm_nm, support_min_nm, support_max_nm = (
            _coerce_sensor_band_response_definition(payload)
        )

        return cls(
            band_id=str(payload["band_id"]),
            segment=segment,
            wavelength_nm=ordered_wavelengths,
            rsr=ordered_rsr,
            center_nm=center_nm,
            fwhm_nm=fwhm_nm,
            support_min_nm=support_min_nm,
            support_max_nm=support_max_nm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "band_id": self.band_id,
            "response_definition": {
                "kind": "sampled",
                "wavelength_nm": list(self.wavelength_nm),
                "response": list(self.rsr),
            },
            "extensions": {
                "spectral_library": {
                    "segment": self.segment,
                }
            },
        }

@dataclass
class SensorSRFSchema:
    """Collection of band response definitions for a sensor."""

    sensor_id: str
    bands: tuple[SensorBandDefinition, ...]

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise SensorSchemaError("Sensor schema sensor_id must be non-empty.")
        if not self.bands:
            raise SensorSchemaError("Sensor schema must include at least one band.", context={"sensor_id": self.sensor_id})
        band_ids = [band.band_id for band in self.bands]
        if len(set(band_ids)) != len(band_ids):
            raise SensorSchemaError("Sensor band_id values must be unique.", context={"sensor_id": self.sensor_id})

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SensorSRFSchema":
        from ..adapters.sensors import _coerce_rsrf_sensor_definition_payload, _sensor_schema_from_rsrf_definition

        sensor_definition = _coerce_rsrf_sensor_definition_payload(payload)
        return _sensor_schema_from_rsrf_definition(sensor_definition)

    def to_dict(self) -> dict[str, object]:
        from ..adapters.sensors import _sensor_schema_payload_for_rsrf

        return _sensor_schema_payload_for_rsrf(self)

    def band_ids(self) -> tuple[str, ...]:
        return tuple(band.band_id for band in self.bands)

    def bands_for_segment(self, segment: str) -> tuple[SensorBandDefinition, ...]:
        if segment not in SEGMENTS:
            raise SensorSchemaError("Unknown sensor segment.", context={"segment": segment, "sensor_id": self.sensor_id})
        return tuple(band for band in self.bands if band.segment == segment)

@dataclass
class PreparedLibraryManifest:
    """Metadata contract for a prepared runtime bundle."""

    schema_version: str
    package_version: str
    source_siac_root: str
    source_siac_build_id: str
    prepared_at: str
    source_sensors: tuple[str, ...]
    supported_output_modes: tuple[str, ...]
    row_count: int
    vnir_wavelength_range_nm: tuple[int, int]
    swir_wavelength_range_nm: tuple[int, int]
    array_dtype: str
    file_checksums: dict[str, str] = field(default_factory=dict)
    knn_index_artifacts: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    interpolation_summary: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PreparedLibraryManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            package_version=str(payload["package_version"]),
            source_siac_root=str(payload["source_siac_root"]),
            source_siac_build_id=str(payload["source_siac_build_id"]),
            prepared_at=str(payload["prepared_at"]),
            source_sensors=tuple(str(value) for value in payload["source_sensors"]),  # type: ignore[index]
            supported_output_modes=tuple(str(value) for value in payload["supported_output_modes"]),  # type: ignore[index]
            row_count=int(payload["row_count"]),
            vnir_wavelength_range_nm=tuple(int(value) for value in payload["vnir_wavelength_range_nm"]),  # type: ignore[index]
            swir_wavelength_range_nm=tuple(int(value) for value in payload["swir_wavelength_range_nm"]),  # type: ignore[index]
            array_dtype=str(payload["array_dtype"]),
            file_checksums={str(key): str(value) for key, value in dict(payload["file_checksums"]).items()},  # type: ignore[arg-type]
            knn_index_artifacts={
                str(backend): {
                    str(sensor_id): {
                        str(segment): str(path)
                        for segment, path in dict(sensor_payload).items()
                    }
                    for sensor_id, sensor_payload in dict(backend_payload).items()
                }
                for backend, backend_payload in dict(payload.get("knn_index_artifacts") or {}).items()  # type: ignore[arg-type]
            },
            interpolation_summary={
                str(key): int(value)
                for key, value in dict(payload.get("interpolation_summary") or {}).items()  # type: ignore[arg-type]
            },
        )

    @classmethod
    def from_json(cls, path: Path) -> "PreparedLibraryManifest":
        payload = _read_json_document(
            path,
            error_factory=PreparedLibraryValidationError,
            document_name=path.name,
        )
        if not isinstance(payload, dict):
            raise PreparedLibraryValidationError(
                f"{path.name} must contain a JSON object.",
                context={"path": str(path)},
            )
        try:
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise PreparedLibraryValidationError(
                f"{path.name} is missing required fields or contains invalid values.",
                context={"path": str(path)},
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_version": self.package_version,
            "source_siac_root": self.source_siac_root,
            "source_siac_build_id": self.source_siac_build_id,
            "prepared_at": self.prepared_at,
            "source_sensors": list(self.source_sensors),
            "supported_output_modes": list(self.supported_output_modes),
            "row_count": self.row_count,
            "vnir_wavelength_range_nm": list(self.vnir_wavelength_range_nm),
            "swir_wavelength_range_nm": list(self.swir_wavelength_range_nm),
            "array_dtype": self.array_dtype,
            "file_checksums": dict(self.file_checksums),
            "knn_index_artifacts": {
                backend: {
                    sensor_id: dict(sensor_payload)
                    for sensor_id, sensor_payload in backend_payload.items()
                }
                for backend, backend_payload in self.knn_index_artifacts.items()
            },
            **(
                {"interpolation_summary": dict(self.interpolation_summary)}
                if self.interpolation_summary
                else {}
            ),
        }

@dataclass
class MappingResult:
    """Outputs and diagnostics for a single mapping request."""

    target_reflectance: np.ndarray | None = None
    target_band_ids: tuple[str, ...] = ()
    reconstructed_vnir: np.ndarray | None = None
    reconstructed_swir: np.ndarray | None = None
    reconstructed_full_spectrum: np.ndarray | None = None
    reconstructed_wavelength_nm: np.ndarray | None = None
    neighbor_ids_by_segment: dict[str, tuple[str, ...]] = field(default_factory=dict)
    neighbor_distances_by_segment: dict[str, np.ndarray] = field(default_factory=dict)
    segment_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    segment_valid_band_counts: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "target_band_ids": list(self.target_band_ids),
            "segment_valid_band_counts": dict(self.segment_valid_band_counts),
            "neighbor_ids_by_segment": {
                segment: list(neighbor_ids) for segment, neighbor_ids in self.neighbor_ids_by_segment.items()
            },
            "neighbor_distances_by_segment": {
                segment: [float(value) for value in distances]
                for segment, distances in self.neighbor_distances_by_segment.items()
            },
            "diagnostics": self.diagnostics,
        }

@dataclass
class BatchMappingResult:
    """Mapping results for a batch of samples."""

    sample_ids: tuple[str, ...]
    results: tuple[MappingResult, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "sample_ids": list(self.sample_ids),
            "results": [
                {"sample_id": sample_id, **result.to_summary_dict()}
                for sample_id, result in zip(self.sample_ids, self.results)
            ],
        }

@dataclass
class BatchMappingArrayResult:
    """Dense array outputs for a batch of mapped samples."""

    sample_ids: tuple[str, ...]
    reflectance: np.ndarray
    source_fit_rmse: np.ndarray
    output_columns: tuple[str, ...]
    wavelength_nm: np.ndarray | None = None

@dataclass
class LinearSpectralMapper:
    """Fixed linear mapper compiled from a prepared runtime."""

    source_sensor: str
    output_mode: str
    weights: np.ndarray
    bias: np.ndarray
    dtype: np.dtype[Any]
    target_sensor: str | None = None
    source_band_ids: tuple[str, ...] = ()
    output_band_ids: tuple[str, ...] = ()
    output_wavelength_nm: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype.kind != "f":
            raise MappingInputError("LinearSpectralMapper requires a floating-point dtype.", context={"dtype": self.dtype.name})

        self.weights = np.ascontiguousarray(self.weights, dtype=self.dtype)
        self.bias = np.ascontiguousarray(self.bias, dtype=self.dtype)
        if self.weights.ndim != 2:
            raise MappingInputError("LinearSpectralMapper weights must be two-dimensional.")
        if self.bias.ndim != 1:
            raise MappingInputError("LinearSpectralMapper bias must be one-dimensional.")
        if self.weights.shape[1] != self.bias.shape[0]:
            raise MappingInputError(
                "LinearSpectralMapper weights and bias shapes must agree.",
                context={
                    "weights_shape": list(self.weights.shape),
                    "bias_shape": list(self.bias.shape),
                },
            )
        if self.source_band_ids and self.weights.shape[0] != len(self.source_band_ids):
            raise MappingInputError(
                "LinearSpectralMapper weights do not match the declared source band ids.",
                context={
                    "weights_shape": list(self.weights.shape),
                    "source_band_count": len(self.source_band_ids),
                },
            )
        if self.output_band_ids and self.bias.shape[0] != len(self.output_band_ids):
            raise MappingInputError(
                "LinearSpectralMapper bias does not match the declared target band ids.",
                context={
                    "bias_shape": list(self.bias.shape),
                    "target_band_count": len(self.output_band_ids),
                },
            )
        if self.output_wavelength_nm is not None:
            self.output_wavelength_nm = np.asarray(self.output_wavelength_nm, dtype=np.float64)
            if self.output_wavelength_nm.ndim != 1 or self.output_wavelength_nm.shape[0] != self.bias.shape[0]:
                raise MappingInputError(
                    "LinearSpectralMapper output_wavelength_nm must match the output width.",
                    context={
                        "bias_shape": list(self.bias.shape),
                        "wavelength_shape": list(self.output_wavelength_nm.shape),
                    },
                )

    @property
    def output_count(self) -> int:
        """Return the number of emitted output columns."""

        return int(self.bias.shape[0])

    def map_array(
        self,
        reflectance_rows: np.ndarray,
        *,
        out: np.ndarray | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Map dense source-sensor rows with one matrix multiply per chunk."""

        inputs = np.asarray(reflectance_rows)
        if inputs.ndim != 2:
            raise MappingInputError(
                "LinearSpectralMapper.map_array requires a two-dimensional array.",
                context={"shape": list(inputs.shape)},
            )
        if inputs.shape[1] != self.weights.shape[0]:
            raise MappingInputError(
                "reflectance_rows must match the compiled source band count.",
                context={
                    "shape": list(inputs.shape),
                    "expected_source_band_count": int(self.weights.shape[0]),
                },
            )

        output_shape = (int(inputs.shape[0]), int(self.output_count))
        if out is None:
            output = np.empty(output_shape, dtype=self.dtype)
        else:
            output = np.asarray(out)
            if output.shape != output_shape:
                raise MappingInputError(
                    "out must have the same row count as reflectance_rows and the compiled output width.",
                    context={"expected_shape": list(output_shape), "actual_shape": list(output.shape)},
                )
            if output.dtype.kind != "f":
                raise MappingInputError("out must use a floating-point dtype.", context={"dtype": output.dtype.name})
            if not output.flags.writeable:
                raise MappingInputError("out must be writeable.")

        rows_per_chunk = _normalized_linear_mapper_chunk_size(
            chunk_size,
            input_width=int(self.weights.shape[0]),
            output_width=int(self.output_count),
            dtype=self.dtype,
        )
        use_direct_out = output.dtype == self.dtype

        for start in range(0, inputs.shape[0], rows_per_chunk):
            stop = min(inputs.shape[0], start + rows_per_chunk)
            input_chunk = np.ascontiguousarray(inputs[start:stop], dtype=self.dtype)
            if use_direct_out:
                out_chunk = output[start:stop]
                np.matmul(input_chunk, self.weights, out=out_chunk)
                out_chunk += self.bias
            else:
                mapped_chunk = input_chunk @ self.weights
                mapped_chunk += self.bias
                output[start:stop] = mapped_chunk
        return output

def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)

def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text

def _default_sample_id(index: int) -> str:
    return f"sample_{index + 1:06d}"

def _attach_sample_context(error: SpectralLibraryError, *, sample_id: str, sample_index: int) -> SpectralLibraryError:
    context = {"sample_id": sample_id, "sample_index": sample_index, **error.context}
    if type(error) is SpectralLibraryError:
        return SpectralLibraryError(error.code, error.message, context=context)
    return error.__class__(error.message, context=context)

def _normalized_sample_ids(sample_ids: Sequence[str] | None, *, sample_count: int) -> tuple[str, ...]:
    """Normalize batch sample identifiers and enforce one id per row."""

    if sample_count < 1:
        raise MappingInputError("Batch mapping requires at least one reflectance sample.")
    if sample_ids is None:
        return tuple(_default_sample_id(index) for index in range(sample_count))

    if len(sample_ids) != sample_count:
        raise MappingInputError(
            "sample_ids must have the same length as the reflectance batch.",
            context={"sample_count": sample_count, "sample_id_count": len(sample_ids)},
        )

    normalized: list[str] = []
    for index, sample_id in enumerate(sample_ids):
        text = str(sample_id).strip()
        if not text:
            raise MappingInputError("sample_ids must be non-empty.", context={"sample_index": index})
        normalized.append(text)

    if len(set(normalized)) != len(normalized):
        raise MappingInputError("sample_ids must be unique within a batch.")
    return tuple(normalized)

def _segment_slice(segment: str) -> slice:
    start_nm, end_nm = SEGMENT_RANGES[segment]
    start_index = start_nm - CANONICAL_START_NM
    stop_index = end_nm - CANONICAL_START_NM + 1
    return slice(start_index, stop_index)

def _blend_overlap(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Blend the 800-1000 nm overlap so full spectra transition smoothly."""

    overlap_wavelengths = np.arange(FULL_BLEND_START_NM, FULL_BLEND_END_NM + 1, dtype=np.float64)
    weights = (FULL_BLEND_END_NM - overlap_wavelengths) / (FULL_BLEND_END_NM - FULL_BLEND_START_NM)
    vnir_overlap_start = FULL_BLEND_START_NM - VNIR_START_NM
    swir_overlap_start = FULL_BLEND_START_NM - SWIR_START_NM
    overlap_length = overlap_wavelengths.size
    return (
        weights * vnir[vnir_overlap_start : vnir_overlap_start + overlap_length]
        + (1.0 - weights) * swir[swir_overlap_start : swir_overlap_start + overlap_length]
    )

def _assemble_full_spectrum(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Merge VNIR and SWIR reconstructions onto the canonical wavelength grid."""

    full = np.empty(FULL_WAVELENGTH_COUNT, dtype=np.float64)
    blend_start = FULL_BLEND_START_NM - CANONICAL_START_NM
    blend_stop = FULL_BLEND_END_NM - CANONICAL_START_NM + 1
    swir_overlap_stop = FULL_BLEND_END_NM - SWIR_START_NM + 1
    full[:blend_start] = vnir[:blend_start]
    full[blend_start:blend_stop] = _blend_overlap(vnir, swir)
    full[blend_stop:] = swir[swir_overlap_stop:]
    return full

def _assemble_full_spectrum_batch(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Rust-backed batch variant of :func:`_assemble_full_spectrum`."""

    return np.asarray(
        _rustaccel.assemble_full_spectrum_batch(
            vnir=np.asarray(vnir, dtype=np.float64),
            swir=np.asarray(swir, dtype=np.float64),
        ),
        dtype=np.float64,
    )

def _normalized_linear_mapper_dtype(dtype: str | np.dtype[Any]) -> np.dtype[Any]:
    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        raise MappingInputError("Linear mapper dtype must be floating-point.", context={"dtype": resolved.name})
    return resolved

def _normalized_linear_mapper_chunk_size(
    chunk_size: int | None,
    *,
    input_width: int,
    output_width: int,
    dtype: np.dtype[Any],
) -> int:
    if chunk_size is not None:
        if int(chunk_size) < 1:
            raise MappingInputError("chunk_size must be at least 1.", context={"chunk_size": int(chunk_size)})
        return int(chunk_size)

    bytes_per_row = max(1, (int(input_width) + int(output_width)) * int(np.dtype(dtype).itemsize))
    return max(1, int((64 * 1024 * 1024) // bytes_per_row))

def _estimated_dense_array_bytes(*, row_count: int, column_count: int, dtype: np.dtype[Any]) -> int:
    return int(int(row_count) * int(column_count) * int(np.dtype(dtype).itemsize))

def _fit_linear_map(
    source_matrix: np.ndarray,
    *,
    output_width: int,
    output_loader: Callable[[int, int], np.ndarray],
    compile_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one linear projection with chunked normal-equation accumulation."""

    row_count = int(source_matrix.shape[0])
    input_width = int(source_matrix.shape[1])
    design_width = input_width + 1
    xtx = np.zeros((design_width, design_width), dtype=np.float64)
    xty = np.zeros((design_width, int(output_width)), dtype=np.float64)

    for start in range(0, row_count, int(compile_chunk_size)):
        stop = min(row_count, start + int(compile_chunk_size))
        x_chunk = np.asarray(source_matrix[start:stop], dtype=np.float64)
        y_chunk = np.asarray(output_loader(start, stop), dtype=np.float64)
        expected_shape = (stop - start, int(output_width))
        if y_chunk.shape != expected_shape:
            raise PreparedLibraryValidationError(
                "Linear mapper compile output block shape did not match the prepared runtime.",
                context={"expected_shape": list(expected_shape), "actual_shape": list(y_chunk.shape)},
            )

        design = np.empty((stop - start, design_width), dtype=np.float64)
        design[:, 0] = 1.0
        design[:, 1:] = x_chunk
        xtx += design.T @ design
        xty += design.T @ y_chunk

    try:
        coefficients = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(xtx) @ xty
    return coefficients[0], coefficients[1:]

def _ensure_supported_output_mode(output_mode: str) -> None:
    if output_mode not in SUPPORTED_OUTPUT_MODES:
        raise MappingInputError(
            "Unsupported output_mode.",
            context={"output_mode": output_mode, "supported_output_modes": list(SUPPORTED_OUTPUT_MODES)},
        )

def _validate_mapping_request(
    output_mode: str,
    *,
    k: int,
    min_valid_bands: int,
    neighbor_estimator: str = "mean",
    knn_backend: str = "numpy",
    knn_eps: float = 0.0,
) -> None:
    """Validate a mapping configuration before any retrieval work starts."""

    _ensure_supported_output_mode(output_mode)
    if k < 1:
        raise MappingInputError("k must be at least 1.", context={"k": k})
    if min_valid_bands < 1:
        raise MappingInputError("min_valid_bands must be at least 1.", context={"min_valid_bands": min_valid_bands})
    if neighbor_estimator not in SUPPORTED_NEIGHBOR_ESTIMATORS:
        raise MappingInputError(
            "neighbor_estimator is not supported.",
            context={
                "neighbor_estimator": neighbor_estimator,
                "supported_neighbor_estimators": list(SUPPORTED_NEIGHBOR_ESTIMATORS),
            },
        )
    if knn_backend not in SUPPORTED_KNN_BACKENDS:
        raise MappingInputError(
            "knn_backend is not supported.",
            context={
                "knn_backend": knn_backend,
                "supported_knn_backends": list(SUPPORTED_KNN_BACKENDS),
            },
        )
    if knn_eps < 0:
        raise MappingInputError("knn_eps must be non-negative.", context={"knn_eps": knn_eps})

def _normalized_source_sensors(source_sensors: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for sensor_id in source_sensors:
        text = str(sensor_id).strip()
        if text and text not in normalized:
            normalized.append(text)
    if not normalized:
        raise PreparedLibraryBuildError("At least one source sensor must be provided.")
    return normalized

def _read_json_document(
    path: Path,
    *,
    error_factory: type[SpectralLibraryError],
    document_name: str,
) -> object:
    """Read a JSON document and translate I/O/parsing errors into domain errors."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error_factory(
            f"Could not read {document_name}.",
            context={"path": str(path)},
        ) from exc
    try:
        return json.loads(text)
    except JSONDecodeError as exc:
        raise error_factory(
            f"{document_name} is not valid JSON.",
            context={"path": str(path)},
        ) from exc
