"""Utilities for preparing and querying spectral mapping runtimes.

This module covers two related workflows:

1. Preparing a runtime bundle from canonical SIAC spectra plus sensor response
   definitions.
2. Retrieving nearest-neighbor spectra from a prepared bundle and projecting
   them into hyperspectral or target-sensor space.

The implementation favors explicit validation because prepared runtimes are
treated as stable artifacts that may be reused across commands and machines.
"""

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

from . import _rustaccel
from ._version import __version__


PREPARED_SCHEMA_VERSION = "1.2.0"
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


@dataclass(frozen=True)
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
        if "band_id" not in payload:
            raise SensorSchemaError("Sensor band definition is missing band_id.")
        if "segment" not in payload:
            raise SensorSchemaError("Sensor band definition is missing segment.", context={"band_id": payload.get("band_id")})
        if "wavelength_nm" not in payload or "rsr" not in payload:
            raise SensorSchemaError(
                "Sensor band definition must include wavelength_nm and rsr arrays.",
                context={"band_id": payload.get("band_id")},
            )

        wavelengths = tuple(float(value) for value in payload["wavelength_nm"])  # type: ignore[index]
        rsr = tuple(float(value) for value in payload["rsr"])  # type: ignore[index]
        ordered_pairs = tuple(sorted(zip(wavelengths, rsr), key=lambda item: item[0]))
        ordered_wavelengths = tuple(pair[0] for pair in ordered_pairs)
        ordered_rsr = tuple(pair[1] for pair in ordered_pairs)
        positive_support = [wavelength for wavelength, weight in ordered_pairs if weight > 0]

        return cls(
            band_id=str(payload["band_id"]),
            segment=str(payload["segment"]),
            wavelength_nm=ordered_wavelengths,
            rsr=ordered_rsr,
            center_nm=_optional_float(payload.get("center_nm")),
            fwhm_nm=_optional_float(payload.get("fwhm_nm")),
            support_min_nm=(
                _optional_float(payload.get("support_min_nm"))
                if _optional_float(payload.get("support_min_nm")) is not None
                else (min(positive_support) if positive_support else None)
            ),
            support_max_nm=(
                _optional_float(payload.get("support_max_nm"))
                if _optional_float(payload.get("support_max_nm")) is not None
                else (max(positive_support) if positive_support else None)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "band_id": self.band_id,
            "segment": self.segment,
            "wavelength_nm": list(self.wavelength_nm),
            "rsr": list(self.rsr),
        }
        if self.center_nm is not None:
            payload["center_nm"] = self.center_nm
        if self.fwhm_nm is not None:
            payload["fwhm_nm"] = self.fwhm_nm
        if self.support_min_nm is not None:
            payload["support_min_nm"] = self.support_min_nm
        if self.support_max_nm is not None:
            payload["support_max_nm"] = self.support_max_nm
        return payload


@dataclass(frozen=True)
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
        if "sensor_id" not in payload:
            raise SensorSchemaError("Sensor schema is missing sensor_id.")
        if "bands" not in payload:
            raise SensorSchemaError("Sensor schema is missing bands.", context={"sensor_id": payload.get("sensor_id")})
        bands = tuple(SensorBandDefinition.from_dict(band_payload) for band_payload in payload["bands"])  # type: ignore[index]
        return cls(sensor_id=str(payload["sensor_id"]), bands=bands)

    def to_dict(self) -> dict[str, object]:
        return {
            "sensor_id": self.sensor_id,
            "bands": [band.to_dict() for band in self.bands],
        }

    def band_ids(self) -> tuple[str, ...]:
        return tuple(band.band_id for band in self.bands)

    def bands_for_segment(self, segment: str) -> tuple[SensorBandDefinition, ...]:
        if segment not in SEGMENTS:
            raise SensorSchemaError("Unknown sensor segment.", context={"segment": segment, "sensor_id": self.sensor_id})
        return tuple(band for band in self.bands if band.segment == segment)


@dataclass(frozen=True)
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
class _ZarrBatchExport:
    root: Any
    reflectance_dataset: Any
    source_fit_rmse_dataset: Any
    sample_id_dataset: Any
    output_columns: tuple[str, ...]
    output_width: int
    chunk_size: int


@dataclass
class _ScipyCkdtreeCacheEntry:
    data: np.ndarray
    index: Any


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


@dataclass
class _SegmentRetrieval:
    """Internal container for one segment-level neighbor retrieval."""

    segment: str
    valid_band_count: int
    query_band_ids: tuple[str, ...]
    success: bool
    query_band_values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    query_valid_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    reconstructed: np.ndarray | None = None
    neighbor_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    neighbor_ids: tuple[str, ...] = ()
    neighbor_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    neighbor_weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    neighbor_band_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    source_fit_rmse: float | None = None
    confidence_score: float | None = None
    confidence_components: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class _RichSegmentMetadata:
    valid_band_count: int
    success: bool
    neighbor_ids: tuple[str, ...] = ()
    neighbor_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    confidence_score: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class _DenseSegmentOutputBatch:
    success: np.ndarray
    reconstructed: np.ndarray
    confidence_score: np.ndarray
    source_fit_rmse: np.ndarray
    valid_band_count: np.ndarray
    target_output_indices: np.ndarray | None = None


@dataclass
class _RichSegmentBatch:
    dense_output: _DenseSegmentOutputBatch
    metadata: tuple[_RichSegmentMetadata, ...]


def _empty_dense_segment_output_batch(
    *,
    segment: str,
    batch_size: int,
    output_width: int | None = None,
    target_output_indices: np.ndarray | None = None,
) -> _DenseSegmentOutputBatch:
    resolved_output_width = int(SEGMENT_WAVELENGTHS[segment].size) if output_width is None else int(output_width)
    return _DenseSegmentOutputBatch(
        success=np.zeros(batch_size, dtype=bool),
        reconstructed=np.zeros((batch_size, resolved_output_width), dtype=np.float64),
        confidence_score=np.zeros(batch_size, dtype=np.float64),
        source_fit_rmse=np.zeros(batch_size, dtype=np.float64),
        valid_band_count=np.zeros(batch_size, dtype=np.int32),
        target_output_indices=(
            None if target_output_indices is None else np.asarray(target_output_indices, dtype=np.int64)
        ),
    )


def _assign_dense_segment_output_batch_rows(
    *,
    target: _DenseSegmentOutputBatch,
    sample_indices: np.ndarray,
    source: _DenseSegmentOutputBatch,
) -> None:
    target.success[sample_indices] = source.success
    target.reconstructed[sample_indices] = source.reconstructed
    target.confidence_score[sample_indices] = source.confidence_score
    target.source_fit_rmse[sample_indices] = source.source_fit_rmse
    target.valid_band_count[sample_indices] = source.valid_band_count
    if source.target_output_indices is not None:
        if target.target_output_indices is None:
            target.target_output_indices = np.asarray(source.target_output_indices, dtype=np.int64)
        elif not np.array_equal(target.target_output_indices, source.target_output_indices):
            raise PreparedLibraryValidationError("Dense segment batches disagree on target output indices.")


@dataclass
class _TargetSensorProjection:
    output_width: int
    vnir_response_matrix: np.ndarray
    swir_response_matrix: np.ndarray
    vnir_output_indices: np.ndarray
    swir_output_indices: np.ndarray
    vnir_support_indices: np.ndarray
    swir_support_indices: np.ndarray
    vnir_hyperspectral_rows: np.ndarray
    swir_hyperspectral_rows: np.ndarray
    vnir_target_rows: np.ndarray
    swir_target_rows: np.ndarray


@dataclass
class _CandidateBatchGroup:
    sample_indices: np.ndarray
    candidate_rows: np.ndarray


@dataclass
class _BatchedResultMaterialization:
    vnir_batch: _DenseSegmentOutputBatch
    swir_batch: _DenseSegmentOutputBatch
    confidence_scores: np.ndarray
    reconstructed_full_batch: np.ndarray | None
    target_rows: np.ndarray | None
    target_status_codes: np.ndarray | None
    target_band_ids: tuple[str, ...]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


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


def _normalized_knn_index_backends(knn_index_backends: Sequence[str] | None) -> list[str]:
    if not knn_index_backends:
        return []
    normalized: list[str] = []
    for backend in knn_index_backends:
        text = str(backend).strip()
        if not text:
            continue
        if text not in SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS:
            raise PreparedLibraryBuildError(
                "Requested knn_index_backend is not supported for persisted indexes.",
                context={
                    "knn_index_backend": text,
                    "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS),
                },
            )
        if text not in normalized:
            normalized.append(text)
    return normalized


def _update_digest_from_file(digest: Any, path: Path) -> None:
    """Feed file contents into an existing SHA256 digest."""

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest_from_file(digest, path)
    return digest.hexdigest()


def _sha256_runtime_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            _update_digest_from_file(digest, child)
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _sha256_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        _update_digest_from_file(digest, path)
    return digest.hexdigest()


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if bool(value):
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest)


def _summarize_missing_gap_counts(spectrum: np.ndarray) -> dict[str, int]:
    missing_mask = np.isnan(spectrum)
    if not missing_mask.any():
        return {
            "missing_count": 0,
            "leading_gap_count": 0,
            "trailing_gap_count": 0,
            "internal_gap_count": 0,
            "max_internal_gap_run_count": 0,
        }

    leading_gap_count = 0
    while leading_gap_count < missing_mask.size and bool(missing_mask[leading_gap_count]):
        leading_gap_count += 1

    trailing_gap_count = 0
    while trailing_gap_count < missing_mask.size and bool(missing_mask[missing_mask.size - 1 - trailing_gap_count]):
        trailing_gap_count += 1

    internal_start = leading_gap_count
    internal_end = missing_mask.size - trailing_gap_count
    internal_mask = np.asarray(missing_mask[internal_start:internal_end], dtype=bool)
    return {
        "missing_count": int(missing_mask.sum()),
        "leading_gap_count": int(leading_gap_count),
        "trailing_gap_count": int(trailing_gap_count),
        "internal_gap_count": int(internal_mask.sum()),
        "max_internal_gap_run_count": _longest_true_run(internal_mask),
    }


def _empty_interpolation_summary() -> dict[str, int]:
    return {
        "interpolated_row_count": 0,
        "rows_with_leading_gaps": 0,
        "rows_with_trailing_gaps": 0,
        "rows_with_internal_gaps": 0,
        "max_missing_count": 0,
        "max_leading_gap_count": 0,
        "max_trailing_gap_count": 0,
        "max_internal_gap_count": 0,
        "max_internal_gap_run_count": 0,
    }


def _update_interpolation_summary(summary: dict[str, int], gap_counts: Mapping[str, int]) -> None:
    summary["interpolated_row_count"] += 1
    if int(gap_counts["leading_gap_count"]) > 0:
        summary["rows_with_leading_gaps"] += 1
    if int(gap_counts["trailing_gap_count"]) > 0:
        summary["rows_with_trailing_gaps"] += 1
    if int(gap_counts["internal_gap_count"]) > 0:
        summary["rows_with_internal_gaps"] += 1
    summary["max_missing_count"] = max(summary["max_missing_count"], int(gap_counts["missing_count"]))
    summary["max_leading_gap_count"] = max(summary["max_leading_gap_count"], int(gap_counts["leading_gap_count"]))
    summary["max_trailing_gap_count"] = max(summary["max_trailing_gap_count"], int(gap_counts["trailing_gap_count"]))
    summary["max_internal_gap_count"] = max(summary["max_internal_gap_count"], int(gap_counts["internal_gap_count"]))
    summary["max_internal_gap_run_count"] = max(
        summary["max_internal_gap_run_count"],
        int(gap_counts["max_internal_gap_run_count"]),
    )


def _ordered_neighbor_rows(
    distances: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest candidate rows ordered deterministically by distance."""

    neighbor_count = min(int(k), int(candidate_row_indices.size))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})
    if neighbor_count == candidate_row_indices.size:
        local_top = np.arange(candidate_row_indices.size)
    else:
        local_top = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    ordered_local = local_top[np.lexsort((candidate_row_indices[local_top], distances[local_top]))]
    return candidate_row_indices[ordered_local], np.asarray(distances[ordered_local], dtype=np.float64)


def _ordered_neighbor_rows_from_local_distances(
    local_candidate_indices: np.ndarray,
    local_distances: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    query_width: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve exact backend distances without rescoring the candidate matrix."""

    if query_width <= 0:
        raise MappingInputError("At least one valid source band is required for mapping.")
    local_indices = np.asarray(local_candidate_indices, dtype=np.int64)
    resolved_distances = np.asarray(local_distances, dtype=np.float64)
    valid = (local_indices >= 0) & (local_indices < int(candidate_row_indices.size)) & np.isfinite(resolved_distances)
    local_indices = local_indices[valid]
    resolved_distances = resolved_distances[valid]
    if local_indices.size == 0:
        raise PreparedLibraryValidationError("Neighbor search backend returned no candidate rows.")
    unique_local_indices, unique_positions = np.unique(local_indices, return_index=True)
    exact_rmse = resolved_distances[unique_positions] / math.sqrt(float(query_width))
    return _ordered_neighbor_rows(exact_rmse, candidate_row_indices[unique_local_indices], k=k)


def _ordered_neighbor_rows_batch_from_local_distances(
    local_candidate_indices: np.ndarray,
    local_distance_matrix: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    query_width: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch variant of ``_ordered_neighbor_rows_from_local_distances``."""

    local_indices = np.asarray(local_candidate_indices, dtype=np.int64)
    local_distances = np.asarray(local_distance_matrix, dtype=np.float64)
    if local_indices.shape != local_distances.shape:
        raise MappingInputError(
            "Exact local neighbor indices and distances must share the same shape.",
            context={
                "local_index_shape": list(local_indices.shape),
                "local_distance_shape": list(local_distances.shape),
            },
        )
    if local_indices.ndim != 2:
        raise MappingInputError(
            "Exact local neighbor indices must be two-dimensional for batched mapping.",
            context={"local_index_shape": list(local_indices.shape)},
        )
    neighbor_indices_rows: list[np.ndarray] = []
    neighbor_distance_rows: list[np.ndarray] = []
    for row_indices, row_distances in zip(local_indices, local_distances):
        resolved_indices, resolved_distances = _ordered_neighbor_rows_from_local_distances(
            row_indices,
            row_distances,
            candidate_row_indices,
            query_width=query_width,
            k=k,
        )
        neighbor_indices_rows.append(np.asarray(resolved_indices, dtype=np.int64))
        neighbor_distance_rows.append(np.asarray(resolved_distances, dtype=np.float64))
    return (
        np.asarray(neighbor_indices_rows, dtype=np.int64),
        np.asarray(neighbor_distance_rows, dtype=np.float64),
    )


def _load_ckdtree_class() -> type[Any]:
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'scipy_ckdtree backend requires scipy. Install it with `{KNN_BACKEND_INSTALL_HINTS["scipy_ckdtree"]}`.',
            context={"knn_backend": "scipy_ckdtree"},
        ) from exc
    return cKDTree


def _scipy_ckdtree_workers() -> int:
    raw_value = (os.environ.get("SPECTRAL_LIBRARY_SCIPY_WORKERS") or "").strip()
    if not raw_value:
        return 1
    try:
        workers = int(raw_value)
    except ValueError as exc:
        raise MappingInputError(
            "SPECTRAL_LIBRARY_SCIPY_WORKERS must be an integer when set.",
            context={"env_var": "SPECTRAL_LIBRARY_SCIPY_WORKERS", "value": raw_value},
        ) from exc
    if workers == 0 or workers < -1:
        raise MappingInputError(
            "SPECTRAL_LIBRARY_SCIPY_WORKERS must be -1 or a positive integer.",
            context={"env_var": "SPECTRAL_LIBRARY_SCIPY_WORKERS", "value": raw_value},
        )
    return workers


def _load_faiss_module() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'faiss backend requires faiss-cpu. Install it with `{KNN_BACKEND_INSTALL_HINTS["faiss"]}`.',
            context={"knn_backend": "faiss"},
        ) from exc
    return faiss


def _load_pynndescent_class() -> type[Any]:
    try:
        from pynndescent import NNDescent  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'pynndescent backend requires pynndescent. Install it with `{KNN_BACKEND_INSTALL_HINTS["pynndescent"]}`.',
            context={"knn_backend": "pynndescent"},
        ) from exc
    return NNDescent


def _load_scann_ops() -> Any:
    try:
        import scann  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'scann backend requires ScaNN. Install it with `{KNN_BACKEND_INSTALL_HINTS["scann"]}`.',
            context={"knn_backend": "scann"},
        ) from exc
    return scann.scann_ops_pybind


def _load_zarr_module() -> Any:
    try:
        import zarr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires zarr. Install it with `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return zarr


def _default_zarr_compressor() -> Any | None:
    try:
        from numcodecs import Blosc  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _load_zarr_vlen_utf8_codec() -> Any:
    try:
        from numcodecs import VLenUTF8  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires numcodecs via `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return VLenUTF8()


def _zarr_utf8_codec() -> Any:
    try:
        from numcodecs import VLenUTF8  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise MappingInputError(
            f'Zarr batch export requires numcodecs. Install it with `{ZARR_INSTALL_HINT}`.',
            context={"output_format": "zarr"},
        ) from exc
    return VLenUTF8()


def _remove_output_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _temporary_output_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp-{uuid4().hex}"


def _finalize_output_path(temp_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if output_path.exists():
        backup_path = output_path.parent / f".{output_path.name}.bak-{uuid4().hex}"
        output_path.replace(backup_path)
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    else:
        if backup_path is not None:
            _remove_output_path(backup_path)


def _persist_faiss_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    faiss = _load_faiss_module()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexHNSWFlat(int(candidate_matrix.shape[1]), 32)
    if hasattr(index, "hnsw"):
        index.hnsw.efConstruction = max(40, min(int(candidate_matrix.shape[0]), 320))
    index.add(np.asarray(candidate_matrix, dtype=np.float32))
    faiss.write_index(index, str(output_path))


def _load_persisted_faiss_index(path: Path) -> Any:
    faiss = _load_faiss_module()
    return faiss.read_index(str(path))


def _persist_pynndescent_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    NNDescent = _load_pynndescent_class()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = NNDescent(np.asarray(candidate_matrix, dtype=np.float32), metric="euclidean")
    if hasattr(index, "prepare"):
        index.prepare()
    with output_path.open("wb") as handle:
        pickle.dump(index, handle)


def _load_persisted_pynndescent_index(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _persist_scann_index(candidate_matrix: np.ndarray, output_path: Path) -> None:
    searcher = _build_scann_searcher(
        np.asarray(candidate_matrix, dtype=np.float32),
        neighbor_count=min(int(candidate_matrix.shape[0]), 64),
        knn_eps=0.0,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    if hasattr(searcher, "serialize"):
        searcher.serialize(str(output_path))
        return
    raise PreparedLibraryBuildError(
        "scann searcher does not support serialization in this environment.",
        context={"knn_backend": "scann", "path": str(output_path)},
    )


def _load_persisted_scann_index(path: Path) -> Any:
    scann_ops = _load_scann_ops()
    if hasattr(scann_ops, "load_searcher"):
        return scann_ops.load_searcher(str(path))
    raise MappingInputError(
        "scann backend does not support persisted searcher loading in this environment.",
        context={"knn_backend": "scann", "path": str(path)},
    )


def _persist_knn_index(candidate_matrix: np.ndarray, *, backend: str, output_path: Path) -> None:
    handlers = {
        "faiss": _persist_faiss_index,
        "pynndescent": _persist_pynndescent_index,
        "scann": _persist_scann_index,
    }
    handler = handlers.get(backend)
    if handler is not None:
        handler(candidate_matrix, output_path)
        return
    raise PreparedLibraryBuildError(
        "KNN index persistence is not supported for the requested backend.",
        context={"knn_backend": backend, "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS)},
    )


def _load_persisted_knn_index(path: Path, *, backend: str) -> Any:
    handlers = {
        "faiss": _load_persisted_faiss_index,
        "pynndescent": _load_persisted_pynndescent_index,
        "scann": _load_persisted_scann_index,
    }
    handler = handlers.get(backend)
    if handler is not None:
        return handler(path)
    raise MappingInputError(
        "KNN index persistence is not supported for the requested backend.",
        context={"knn_backend": backend, "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS)},
    )


def _normalize_query_matrix(query_values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Normalize single-row and batched queries to a 2D matrix."""

    query_matrix = np.asarray(query_values, dtype=dtype)
    if query_matrix.ndim == 1:
        return query_matrix.reshape(1, -1)
    if query_matrix.ndim != 2:
        raise MappingInputError(
            "Query values must be one-dimensional or two-dimensional.",
            context={"query_shape": list(query_matrix.shape)},
        )
    return query_matrix


def _normalize_local_indices(local_indices: np.ndarray, *, query_count: int) -> np.ndarray:
    """Normalize backend neighbor indices to ``(query_count, k)``."""

    normalized = np.asarray(local_indices, dtype=np.int64)
    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    if normalized.ndim == 1:
        return normalized.reshape(query_count, -1)
    if normalized.ndim != 2:
        raise MappingInputError(
            "Neighbor search backend returned indices with an unsupported shape.",
            context={"local_index_shape": list(normalized.shape)},
        )
    return normalized


def _normalize_local_distances(local_distances: np.ndarray, *, query_count: int) -> np.ndarray:
    """Normalize backend neighbor distances to ``(query_count, k)``."""

    normalized = np.asarray(local_distances, dtype=np.float64)
    if normalized.ndim == 0:
        return normalized.reshape(1, 1)
    if normalized.ndim == 1:
        return normalized.reshape(query_count, -1)
    if normalized.ndim != 2:
        raise MappingInputError(
            "Neighbor search backend returned distances with an unsupported shape.",
            context={"local_distance_shape": list(normalized.shape)},
        )
    return normalized


def _build_scann_searcher(
    candidate_matrix: np.ndarray,
    *,
    neighbor_count: int,
    knn_eps: float,
) -> Any:
    """Build a ScaNN searcher tuned for the current candidate set size."""

    scann_ops = _load_scann_ops()
    candidate_count = int(candidate_matrix.shape[0])
    num_leaves = max(1, min(candidate_count, int(round(math.sqrt(candidate_count)))))
    base_leaves_to_search = max(1, min(num_leaves, int(round(math.sqrt(num_leaves))) or 1))
    leaves_to_search = max(
        1,
        min(
            num_leaves,
            int(math.ceil(base_leaves_to_search / (1.0 + max(float(knn_eps), 0.0) * 4.0))),
        ),
    )
    training_sample_size = min(candidate_count, max(32, num_leaves * 10))
    builder = scann_ops.builder(np.asarray(candidate_matrix, dtype=np.float32), neighbor_count, "squared_l2")
    builder = builder.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=leaves_to_search,
        training_sample_size=training_sample_size,
    )
    # Tiny candidate sets cannot train ScaNN's asymmetric hashing path.
    # Fall back to brute-force scoring so the builder still satisfies ScaNN's
    # requirement that exactly one scoring mode is configured.
    if hasattr(builder, "score_ah") and training_sample_size >= SCANN_MIN_AH_TRAINING_SAMPLE_SIZE:
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
    elif hasattr(builder, "score_brute_force"):
        builder = builder.score_brute_force()
    if hasattr(builder, "reorder"):
        builder = builder.reorder(neighbor_count)
    return builder.build()


def _query_faiss_index(index: Any, query_values: np.ndarray, *, k: int) -> np.ndarray:
    _, local_indices = index.search(_normalize_query_matrix(query_values, dtype=np.float32), int(k))
    return np.asarray(local_indices, dtype=np.int64)


def _query_pynndescent_index(index: Any, query_values: np.ndarray, *, k: int, knn_eps: float) -> np.ndarray:
    local_indices, _ = index.query(
        _normalize_query_matrix(query_values, dtype=np.float32),
        k=int(k),
        epsilon=float(knn_eps),
    )
    return np.asarray(local_indices, dtype=np.int64)


def _query_scann_index(index: Any, query_values: np.ndarray, *, k: int) -> np.ndarray:
    query_matrix = _normalize_query_matrix(query_values, dtype=np.float32)
    try:
        search_result = index.search_batched(query_matrix, final_num_neighbors=int(k))
    except TypeError:
        search_result = index.search_batched(query_matrix)
    local_indices = search_result[0] if isinstance(search_result, tuple) else search_result
    return np.asarray(local_indices, dtype=np.int64)


def _query_knn_index(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray:
    """Query a backend index and normalize its output shape."""

    query_count = int(_normalize_query_matrix(query_values, dtype=np.float64).shape[0])
    if knn_backend == "scipy_ckdtree":
        _, local_indices = index.query(
            _normalize_query_matrix(query_values, dtype=np.float64),
            k=int(k),
            eps=float(knn_eps),
            workers=_scipy_ckdtree_workers(),
        )
    elif knn_backend == "faiss":
        local_indices = _query_faiss_index(index, query_values, k=k)
    elif knn_backend == "pynndescent":
        local_indices = _query_pynndescent_index(index, query_values, k=k, knn_eps=knn_eps)
    elif knn_backend == "scann":
        local_indices = _query_scann_index(index, query_values, k=k)
    else:
        raise MappingInputError(
            "knn_backend is not supported.",
            context={
                "knn_backend": knn_backend,
                "supported_knn_backends": list(SUPPORTED_KNN_BACKENDS),
            },
        )
    return _normalize_local_indices(np.asarray(local_indices, dtype=np.int64), query_count=query_count)


def _query_knn_index_with_distances(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Query a backend index and normalize both indices and distances."""

    query_count = int(_normalize_query_matrix(query_values, dtype=np.float64).shape[0])
    if knn_backend != "scipy_ckdtree":
        raise MappingInputError(
            "Distance-returning KNN queries are only supported for scipy_ckdtree.",
            context={"knn_backend": knn_backend},
        )
    local_distances, local_indices = index.query(
        _normalize_query_matrix(query_values, dtype=np.float64),
        k=int(k),
        eps=float(knn_eps),
        workers=_scipy_ckdtree_workers(),
    )
    return (
        _normalize_local_indices(np.asarray(local_indices, dtype=np.int64), query_count=query_count),
        _normalize_local_distances(np.asarray(local_distances, dtype=np.float64), query_count=query_count),
    )


def _query_local_scipy_ckdtree_results(
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    *,
    k: int,
    knn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a temporary SciPy cKDTree and return local indices with distances."""

    neighbor_count = min(int(k), int(candidate_matrix.shape[0]))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})
    cKDTree = _load_ckdtree_class()
    index = cKDTree(np.asarray(candidate_matrix, dtype=np.float64))
    return _query_knn_index_with_distances(
        index,
        query_values,
        k=neighbor_count,
        knn_backend="scipy_ckdtree",
        knn_eps=knn_eps,
    )


def _search_local_neighbor_indices(
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray | None:
    """Return approximate local neighbor indices for one or more query rows.

    Approximate backends are only used to generate a candidate shortlist. The
    final row order is still re-ranked with exact RMSE distances.
    """

    if knn_backend == "numpy":
        return None

    neighbor_count = min(int(k), int(candidate_matrix.shape[0]))
    if neighbor_count <= 0:
        raise MappingInputError("k must be at least 1.", context={"k": k})

    if knn_backend == "scipy_ckdtree":
        cKDTree = _load_ckdtree_class()
        index = cKDTree(np.asarray(candidate_matrix, dtype=np.float64))
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "faiss":
        faiss = _load_faiss_module()
        vector_dim = int(candidate_matrix.shape[1])
        index = faiss.IndexHNSWFlat(vector_dim, 32)
        if hasattr(index, "hnsw"):
            index.hnsw.efConstruction = max(40, neighbor_count * 8)
            index.hnsw.efSearch = max(
                neighbor_count,
                int(math.ceil(max(32, neighbor_count * 8) / (1.0 + max(float(knn_eps), 0.0) * 4.0))),
            )
        index.add(np.asarray(candidate_matrix, dtype=np.float32))
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "pynndescent":
        NNDescent = _load_pynndescent_class()
        index = NNDescent(np.asarray(candidate_matrix, dtype=np.float32), metric="euclidean")
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    if knn_backend == "scann":
        index = _build_scann_searcher(candidate_matrix, neighbor_count=neighbor_count, knn_eps=knn_eps)
        return _query_knn_index(index, query_values, k=neighbor_count, knn_backend=knn_backend, knn_eps=knn_eps)

    raise MappingInputError(
        "knn_backend is not supported.",
        context={
            "knn_backend": knn_backend,
            "supported_knn_backends": list(SUPPORTED_KNN_BACKENDS),
        },
    )


def _query_persisted_knn_index(
    index: Any,
    query_values: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> np.ndarray:
    if knn_backend not in SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS:
        raise MappingInputError(
            "Persisted KNN querying is not supported for the requested backend.",
            context={"knn_backend": knn_backend},
        )
    return _query_knn_index(index, query_values, k=k, knn_backend=knn_backend, knn_eps=knn_eps)


def _refine_neighbor_rows(
    candidate_matrix: np.ndarray,
    query_vector: np.ndarray,
    candidate_row_indices: np.ndarray,
    local_candidate_indices: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_indices = np.asarray(local_candidate_indices, dtype=np.int64)
    local_indices = local_indices[(local_indices >= 0) & (local_indices < int(candidate_row_indices.size))]
    local_indices = np.unique(local_indices)
    if local_indices.size == 0:
        raise PreparedLibraryValidationError("Neighbor search backend returned no candidate rows.")
    found_row_indices = candidate_row_indices[local_indices]
    exact_distances = np.sqrt(np.mean((candidate_matrix[local_indices] - query_vector) ** 2, axis=1))
    return _ordered_neighbor_rows(exact_distances, found_row_indices, k=k)


def _refine_neighbor_rows_batch_accel(
    *,
    candidate_matrix: np.ndarray,
    query_values: np.ndarray,
    candidate_row_indices: np.ndarray,
    local_candidate_indices: np.ndarray | None,
    local_candidate_distances: np.ndarray | None,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_indices, neighbor_distances = _rustaccel.refine_neighbor_rows_batch(
        candidate_matrix=candidate_matrix,
        query_values=query_values,
        candidate_row_indices=candidate_row_indices,
        local_candidate_indices=local_candidate_indices,
        local_candidate_distances=local_candidate_distances,
        k=k,
    )
    return (
        np.asarray(neighbor_indices, dtype=np.int64),
        np.asarray(neighbor_distances, dtype=np.float64),
    )


def _search_neighbor_rows(
    candidate_matrix: np.ndarray,
    query_vector: np.ndarray,
    candidate_row_indices: np.ndarray,
    *,
    k: int,
    knn_backend: str,
    knn_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if knn_backend == "numpy":
        distances = np.sqrt(np.mean((candidate_matrix - query_vector) ** 2, axis=1))
        return _ordered_neighbor_rows(distances, candidate_row_indices, k=k)

    local_indices = _search_local_neighbor_indices(
        candidate_matrix,
        query_vector,
        k=k,
        knn_backend=knn_backend,
        knn_eps=knn_eps,
    )
    if local_indices is None:
        raise PreparedLibraryValidationError("Neighbor search backend returned no candidate rows.")
    return _refine_neighbor_rows(candidate_matrix, query_vector, candidate_row_indices, local_indices[0], k=k)


def _combine_neighbor_spectra_batch_accel(
    *,
    source_matrix: np.ndarray,
    hyperspectral_rows: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    query_values: np.ndarray,
    valid_indices: np.ndarray | None,
    neighbor_estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reconstructed, neighbor_weights, source_fit_rmse = _rustaccel.combine_neighbor_spectra_batch(
        source_matrix=source_matrix,
        hyperspectral_rows=hyperspectral_rows,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        query_values=query_values,
        valid_indices=valid_indices,
        neighbor_estimator=neighbor_estimator,
    )
    return (
        np.asarray(reconstructed, dtype=np.float64),
        np.asarray(neighbor_weights, dtype=np.float64),
        np.asarray(source_fit_rmse, dtype=np.float64),
    )


def _refine_and_combine_neighbor_spectra_batch_accel(
    *,
    candidate_matrix: np.ndarray,
    candidate_row_indices: np.ndarray,
    source_matrix: np.ndarray,
    hyperspectral_rows: np.ndarray,
    query_values: np.ndarray,
    local_candidate_indices: np.ndarray | None,
    local_candidate_distances: np.ndarray | None,
    valid_indices: np.ndarray | None,
    k: int,
    neighbor_estimator: str,
    out_reconstructed: np.ndarray | None = None,
    out_source_fit_rmse: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed, source_fit_rmse = _rustaccel.refine_and_combine_neighbor_spectra_batch(
        candidate_matrix=candidate_matrix,
        candidate_row_indices=candidate_row_indices,
        source_matrix=source_matrix,
        hyperspectral_rows=hyperspectral_rows,
        query_values=query_values,
        local_candidate_indices=local_candidate_indices,
        local_candidate_distances=local_candidate_distances,
        valid_indices=valid_indices,
        k=k,
        neighbor_estimator=neighbor_estimator,
        out_reconstructed=out_reconstructed,
        out_source_fit_rmse=out_source_fit_rmse,
    )
    return np.asarray(reconstructed, dtype=np.float64), np.asarray(source_fit_rmse, dtype=np.float64)


def _combine_neighbor_spectra_batch(
    *,
    hyperspectral_rows: np.ndarray,
    source_matrix: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    query_vectors: np.ndarray,
    valid_indices: slice | np.ndarray,
    neighbor_estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_valid_indices = None if isinstance(valid_indices, slice) else np.asarray(valid_indices, dtype=np.int64)
    return _combine_neighbor_spectra_batch_accel(
        source_matrix=source_matrix,
        hyperspectral_rows=hyperspectral_rows,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        query_values=query_vectors,
        valid_indices=resolved_valid_indices,
        neighbor_estimator=neighbor_estimator,
    )


def _finalize_target_sensor_batch_accel(
    *,
    vnir_reconstructed: np.ndarray,
    swir_reconstructed: np.ndarray,
    vnir_success: np.ndarray,
    swir_success: np.ndarray,
    vnir_response_matrix: np.ndarray,
    swir_response_matrix: np.ndarray,
    vnir_output_indices: np.ndarray,
    swir_output_indices: np.ndarray,
    output_width: int,
    out_rows: np.ndarray | None = None,
    out_status_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_rows, status_codes = _rustaccel.finalize_target_sensor_batch(
        vnir_reconstructed=vnir_reconstructed,
        swir_reconstructed=swir_reconstructed,
        vnir_success=vnir_success,
        swir_success=swir_success,
        vnir_response_matrix=vnir_response_matrix,
        swir_response_matrix=swir_response_matrix,
        vnir_output_indices=vnir_output_indices,
        swir_output_indices=swir_output_indices,
        output_width=output_width,
        out_output_rows=out_rows,
        out_status_codes=out_status_codes,
    )
    return np.asarray(output_rows, dtype=np.float64), np.asarray(status_codes, dtype=np.int32)


def _merge_target_sensor_segments_batch_accel(
    *,
    vnir_rows: np.ndarray,
    swir_rows: np.ndarray,
    vnir_success: np.ndarray,
    swir_success: np.ndarray,
    vnir_output_indices: np.ndarray,
    swir_output_indices: np.ndarray,
    output_width: int,
    out_rows: np.ndarray | None = None,
    out_status_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_rows, status_codes = _rustaccel.merge_target_sensor_segments_batch(
        vnir_rows=vnir_rows,
        swir_rows=swir_rows,
        vnir_success=vnir_success,
        swir_success=swir_success,
        vnir_output_indices=vnir_output_indices,
        swir_output_indices=swir_output_indices,
        output_width=output_width,
        out_output_rows=out_rows,
        out_status_codes=out_status_codes,
    )
    return np.asarray(output_rows, dtype=np.float64), np.asarray(status_codes, dtype=np.int32)


def _stitch_target_sensor_segment_rows(
    *,
    vnir_rows: np.ndarray,
    swir_rows: np.ndarray,
    vnir_success: np.ndarray,
    swir_success: np.ndarray,
    vnir_output_indices: np.ndarray,
    swir_output_indices: np.ndarray,
    output_width: int,
    out_rows: np.ndarray | None = None,
    out_status_codes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_rows, status_codes = _merge_target_sensor_segments_batch_accel(
        vnir_rows=vnir_rows,
        swir_rows=swir_rows,
        vnir_success=vnir_success,
        swir_success=swir_success,
        vnir_output_indices=vnir_output_indices,
        swir_output_indices=swir_output_indices,
        output_width=output_width,
        out_rows=out_rows,
        out_status_codes=out_status_codes,
    )
    return np.asarray(output_rows, dtype=np.float64), np.asarray(status_codes, dtype=np.int32)


def _segment_confidence_payload(
    *,
    query_vector: np.ndarray,
    valid_band_count: int,
    total_band_count: int,
    neighbor_distances: np.ndarray,
    neighbor_weights: np.ndarray,
    source_fit_rmse: float | None,
) -> tuple[float, dict[str, float]]:
    """Score the quality of one segment reconstruction on a 0-1 scale."""

    if valid_band_count <= 0 or total_band_count <= 0 or neighbor_distances.size == 0:
        return 0.0, {
            "coverage": 0.0,
            "distance": 0.0,
            "fit": 0.0,
            "weight_concentration": 0.0,
        }
    query_scale = float(np.mean(np.abs(np.asarray(query_vector, dtype=np.float64)))) if query_vector.size else 0.0
    if not np.isfinite(query_scale) or query_scale <= 1e-12:
        query_scale = 1.0
    coverage = float(valid_band_count) / float(total_band_count)
    distance_score = math.exp(-float(np.mean(np.asarray(neighbor_distances, dtype=np.float64))) / query_scale)
    fit_score = math.exp(-float(source_fit_rmse or 0.0) / query_scale)
    weight_concentration = float(np.max(np.asarray(neighbor_weights, dtype=np.float64))) if neighbor_weights.size else 0.0
    confidence_score = max(
        0.0,
        min(
            1.0,
            0.25 * coverage + 0.35 * distance_score + 0.25 * fit_score + 0.15 * weight_concentration,
        ),
    )
    return confidence_score, {
        "coverage": coverage,
        "distance": distance_score,
        "fit": fit_score,
        "weight_concentration": weight_concentration,
    }


def _segment_confidence_payload_batch(
    *,
    query_matrix: np.ndarray,
    valid_band_count: int,
    total_band_count: int,
    neighbor_distance_matrix: np.ndarray,
    neighbor_weight_matrix: np.ndarray,
    source_fit_rmse: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, float], ...]]:
    """Vectorized confidence scoring for one successful segment batch."""

    batch_size = int(np.asarray(query_matrix).shape[0])
    if valid_band_count <= 0 or total_band_count <= 0 or batch_size == 0:
        empty_scores = np.zeros(batch_size, dtype=np.float64)
        empty_components = tuple(
            {
                "coverage": 0.0,
                "distance": 0.0,
                "fit": 0.0,
                "weight_concentration": 0.0,
            }
            for _ in range(batch_size)
        )
        return empty_scores, empty_components

    query_matrix_f64 = np.asarray(query_matrix, dtype=np.float64)
    neighbor_distance_matrix_f64 = np.asarray(neighbor_distance_matrix, dtype=np.float64)
    neighbor_weight_matrix_f64 = np.asarray(neighbor_weight_matrix, dtype=np.float64)
    source_fit_rmse_f64 = np.asarray(source_fit_rmse, dtype=np.float64)

    if neighbor_distance_matrix_f64.ndim != 2 or neighbor_weight_matrix_f64.ndim != 2:
        raise MappingInputError("Batch confidence inputs must be two-dimensional.")
    if neighbor_distance_matrix_f64.shape[0] != batch_size or neighbor_weight_matrix_f64.shape[0] != batch_size:
        raise MappingInputError("Batch confidence inputs must align on sample count.")
    if source_fit_rmse_f64.shape != (batch_size,):
        raise MappingInputError("Batch confidence source_fit_rmse must align on sample count.")

    if query_matrix_f64.shape[1] == 0 or neighbor_distance_matrix_f64.shape[1] == 0:
        empty_scores = np.zeros(batch_size, dtype=np.float64)
        empty_components = tuple(
            {
                "coverage": 0.0,
                "distance": 0.0,
                "fit": 0.0,
                "weight_concentration": 0.0,
            }
            for _ in range(batch_size)
        )
        return empty_scores, empty_components

    query_scale = np.mean(np.abs(query_matrix_f64), axis=1)
    invalid_scale = ~np.isfinite(query_scale) | (query_scale <= 1e-12)
    if np.any(invalid_scale):
        query_scale = np.asarray(query_scale, dtype=np.float64)
        query_scale[invalid_scale] = 1.0

    coverage = float(valid_band_count) / float(total_band_count)
    distance_score = np.exp(-np.mean(neighbor_distance_matrix_f64, axis=1) / query_scale)
    fit_score = np.exp(-source_fit_rmse_f64 / query_scale)
    weight_concentration = (
        np.max(neighbor_weight_matrix_f64, axis=1)
        if neighbor_weight_matrix_f64.shape[1] > 0
        else np.zeros(batch_size, dtype=np.float64)
    )
    confidence_scores = np.clip(
        0.25 * coverage + 0.35 * distance_score + 0.25 * fit_score + 0.15 * weight_concentration,
        0.0,
        1.0,
    ).astype(np.float64, copy=False)
    confidence_components = tuple(
        {
            "coverage": coverage,
            "distance": float(distance_score[row_index]),
            "fit": float(fit_score[row_index]),
            "weight_concentration": float(weight_concentration[row_index]),
        }
        for row_index in range(batch_size)
    )
    return confidence_scores, confidence_components


def _confidence_policy_payload(confidence_score: float | None) -> dict[str, object]:
    """Translate a confidence score into the public review/accept policy."""

    if confidence_score is None:
        return {
            "version": CONFIDENCE_POLICY_VERSION,
            "band": "unavailable",
            "recommended_action": "reject",
            "review_threshold": CONFIDENCE_REVIEW_THRESHOLD,
            "accept_threshold": CONFIDENCE_ACCEPT_THRESHOLD,
        }
    score = max(0.0, min(1.0, float(confidence_score)))
    if score >= CONFIDENCE_ACCEPT_THRESHOLD:
        band = "high"
        action = "accept"
    elif score >= CONFIDENCE_REVIEW_THRESHOLD:
        band = "medium"
        action = "manual_review"
    else:
        band = "low"
        action = "reject"
    return {
        "version": CONFIDENCE_POLICY_VERSION,
        "band": band,
        "recommended_action": action,
        "review_threshold": CONFIDENCE_REVIEW_THRESHOLD,
        "accept_threshold": CONFIDENCE_ACCEPT_THRESHOLD,
    }


def _source_retrieval_bands(schema: SensorSRFSchema, segment: str) -> tuple[SensorBandDefinition, ...]:
    if segment == "swir":
        return tuple(
            band
            for band in schema.bands
            if band.segment == "swir" or band.band_id == "nir"
        )
    return schema.bands_for_segment(segment)


def _source_retrieval_band_indices(schema: SensorSRFSchema, segment: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, band in enumerate(schema.bands)
        if band.segment == segment or (segment == "swir" and band.band_id == "nir")
    )


def _source_retrieval_band_ids(schema: SensorSRFSchema, segment: str) -> tuple[str, ...]:
    """Return source band ids in the exact order used for retrieval."""

    return tuple(band.band_id for band in _source_retrieval_bands(schema, segment))


def _resample_band_response(band: SensorBandDefinition, *, segment_only: bool) -> np.ndarray:
    canonical = np.interp(
        CANONICAL_WAVELENGTHS.astype(np.float64),
        np.asarray(band.wavelength_nm, dtype=np.float64),
        np.asarray(band.rsr, dtype=np.float64),
        left=0.0,
        right=0.0,
    )
    if segment_only:
        return canonical[_segment_slice(band.segment)]
    return canonical


def _response_weighted_average(
    values: np.ndarray,
    response: np.ndarray,
    *,
    error_message: str,
    error_context: Mapping[str, object],
) -> np.ndarray:
    """Apply a normalized spectral response to one vector or a row matrix."""

    denominator = float(np.sum(response))
    if denominator <= 0:
        raise SensorSchemaError(error_message, context=error_context)
    return np.asarray(values @ response / denominator)


def _simulate_response_matrix(
    input_matrix: np.ndarray,
    bands: Sequence[SensorBandDefinition],
    *,
    dtype: np.dtype[Any],
    segment_only: bool,
) -> np.ndarray:
    """Simulate multispectral band values from hyperspectral inputs."""

    matrix = np.empty((input_matrix.shape[0], len(bands)), dtype=dtype)
    for index, band in enumerate(bands):
        response = _resample_band_response(band, segment_only=segment_only)
        matrix[:, index] = _response_weighted_average(
            input_matrix,
            response,
            error_message="Resampled SRF support must remain positive.",
            error_context={"band_id": band.band_id, "segment": band.segment},
        ).astype(dtype, copy=False)
    return matrix


def _simulate_segment_matrix(
    hyperspectral_segment: np.ndarray,
    bands: Sequence[SensorBandDefinition],
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Test-visible helper that simulates segment-local band responses."""

    return _simulate_response_matrix(
        hyperspectral_segment,
        bands,
        dtype=dtype,
        segment_only=True,
    )


def _simulate_source_retrieval_matrix(
    hyperspectral_full: np.ndarray,
    bands: Sequence[SensorBandDefinition],
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Simulate source-sensor retrieval inputs from full hyperspectral rows."""

    return _simulate_response_matrix(
        hyperspectral_full,
        bands,
        dtype=dtype,
        segment_only=False,
    )


def _load_sensor_payloads(path: Path) -> list[Mapping[str, object]]:
    payload = _read_json_document(
        path,
        error_factory=SensorSchemaError,
        document_name=path.name,
    )
    if isinstance(payload, dict) and "sensors" in payload:
        sensors = payload["sensors"]
        if not isinstance(sensors, list):
            raise SensorSchemaError("sensor_schema.json sensors entry must be a list.", context={"path": str(path)})
        return [sensor_payload for sensor_payload in sensors if isinstance(sensor_payload, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise SensorSchemaError("Sensor schema JSON must be an object.", context={"path": str(path)})


def load_sensor_schemas(srf_root: Path) -> dict[str, SensorSRFSchema]:
    """Load every sensor SRF schema found under ``srf_root``."""

    if not srf_root.exists():
        raise SensorSchemaError("SRF root does not exist.", context={"srf_root": str(srf_root)})

    schemas: dict[str, SensorSRFSchema] = {}
    json_paths = sorted(path for path in srf_root.glob("*.json") if path.is_file())
    if not json_paths:
        raise SensorSchemaError("No sensor schema JSON files were found.", context={"srf_root": str(srf_root)})

    for path in json_paths:
        for payload in _load_sensor_payloads(path):
            schema = SensorSRFSchema.from_dict(payload)
            if schema.sensor_id in schemas:
                raise SensorSchemaError(
                    "Duplicate sensor_id encountered while loading SRF definitions.",
                    context={"sensor_id": schema.sensor_id, "path": str(path)},
                )
            schemas[schema.sensor_id] = schema
    return schemas


def _load_siac_rows(
    metadata_path: Path,
    spectra_path: Path,
    *,
    dtype: np.dtype[Any],
) -> tuple[list[str], list[dict[str, object]], np.ndarray, dict[str, int]]:
    """Load and align SIAC metadata rows with canonical hyperspectral spectra."""

    if not metadata_path.exists() or not spectra_path.exists():
        raise PreparedLibraryBuildError(
            "SIAC tabular inputs are missing required files.",
            context={"metadata_path": str(metadata_path), "spectra_path": str(spectra_path)},
        )

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_reader = csv.DictReader(handle)
        metadata_fieldnames = list(metadata_reader.fieldnames or [])
        metadata_rows = [dict(row) for row in metadata_reader]

    if not metadata_rows:
        raise PreparedLibraryBuildError("SIAC spectra metadata is empty.", context={"metadata_path": str(metadata_path)})

    required_keys = ("source_id", "spectrum_id", "sample_name")
    missing_keys = [key for key in required_keys if key not in metadata_fieldnames]
    if missing_keys:
        raise PreparedLibraryBuildError(
            "SIAC spectra metadata is missing required identifier columns.",
            context={"metadata_path": str(metadata_path), "missing_columns": missing_keys},
        )

    with spectra_path.open("r", encoding="utf-8", newline="") as handle:
        spectra_reader = csv.DictReader(handle)
        spectra_fieldnames = list(spectra_reader.fieldnames or [])
        nm_columns = [column for column in spectra_fieldnames if column.startswith("nm_")]
        if not nm_columns:
            raise PreparedLibraryBuildError(
                "SIAC normalized spectra is missing nm_* columns.",
                context={"spectra_path": str(spectra_path)},
            )

        wavelengths = [int(column.split("_", 1)[1]) for column in nm_columns]
        expected_wavelengths = list(range(CANONICAL_START_NM, CANONICAL_END_NM + 1))
        if wavelengths != expected_wavelengths:
            raise PreparedLibraryBuildError(
                "SIAC normalized spectra must be on the canonical 400-2500 nm grid.",
                context={
                    "spectra_path": str(spectra_path),
                    "expected_start_nm": CANONICAL_START_NM,
                    "expected_end_nm": CANONICAL_END_NM,
                },
            )

        spectra_by_key: dict[tuple[str, str, str], np.ndarray] = {}
        interpolation_summary = _empty_interpolation_summary()
        canonical_wavelengths = np.asarray(expected_wavelengths, dtype=np.float64)
        for row in spectra_reader:
            key = (row["source_id"], row["spectrum_id"], row["sample_name"])
            if key in spectra_by_key:
                raise PreparedLibraryBuildError(
                    "Duplicate spectra rows were found while preparing the mapping library.",
                    context={"source_id": key[0], "spectrum_id": key[1], "sample_name": key[2]},
                )
            values: list[float] = []
            for column in nm_columns:
                cell = row.get(column)
                if cell is None or str(cell).strip() == "":
                    values.append(float("nan"))
                    continue
                try:
                    values.append(float(cell))
                except (TypeError, ValueError) as exc:
                    raise PreparedLibraryBuildError(
                        "SIAC normalized spectra nm_* values must be numeric when present.",
                        context={
                            "spectra_path": str(spectra_path),
                            "source_id": key[0],
                            "spectrum_id": key[1],
                            "sample_name": key[2],
                            "column": column,
                            "value": cell,
                        },
                    ) from exc
            spectrum = np.asarray(values, dtype=np.float64)
            if np.isnan(spectrum).any():
                valid = np.isfinite(spectrum)
                gap_counts = _summarize_missing_gap_counts(spectrum)
                if int(valid.sum()) < 2:
                    raise PreparedLibraryBuildError(
                        "SIAC normalized spectra rows with missing nm_* cells must contain at least two numeric values.",
                        context={
                            "spectra_path": str(spectra_path),
                            "source_id": key[0],
                            "spectrum_id": key[1],
                            "sample_name": key[2],
                            "missing_value_count": int(gap_counts["missing_count"]),
                        },
                    )
                if int(gap_counts["internal_gap_count"]) > MAX_INTERNAL_INTERPOLATED_GAP_COUNT:
                    raise PreparedLibraryBuildError(
                        "SIAC normalized spectra rows may only interpolate a small number of internal nm_* gaps.",
                        context={
                            "spectra_path": str(spectra_path),
                            "source_id": key[0],
                            "spectrum_id": key[1],
                            "sample_name": key[2],
                            "internal_gap_count": int(gap_counts["internal_gap_count"]),
                            "max_allowed_internal_gap_count": MAX_INTERNAL_INTERPOLATED_GAP_COUNT,
                        },
                    )
                if int(gap_counts["max_internal_gap_run_count"]) > MAX_INTERNAL_INTERPOLATED_RUN_COUNT:
                    raise PreparedLibraryBuildError(
                        "SIAC normalized spectra rows may only interpolate short contiguous internal nm_* gaps.",
                        context={
                            "spectra_path": str(spectra_path),
                            "source_id": key[0],
                            "spectrum_id": key[1],
                            "sample_name": key[2],
                            "max_internal_gap_run_count": int(gap_counts["max_internal_gap_run_count"]),
                            "max_allowed_internal_gap_run_count": MAX_INTERNAL_INTERPOLATED_RUN_COUNT,
                        },
                    )
                spectrum = np.interp(canonical_wavelengths, canonical_wavelengths[valid], spectrum[valid])
                _update_interpolation_summary(interpolation_summary, gap_counts)
            spectra_by_key[key] = np.asarray(spectrum, dtype=dtype)

    aligned_metadata_rows: list[dict[str, object]] = []
    hyperspectral = np.empty((len(metadata_rows), FULL_WAVELENGTH_COUNT), dtype=dtype)
    metadata_keys: set[tuple[str, str, str]] = set()
    for row_index, row in enumerate(metadata_rows):
        key = (str(row["source_id"]), str(row["spectrum_id"]), str(row["sample_name"]))
        metadata_keys.add(key)
        if key not in spectra_by_key:
            raise PreparedLibraryBuildError(
                "SIAC spectra metadata and normalized spectra are not row-complete.",
                context={"source_id": key[0], "spectrum_id": key[1], "sample_name": key[2]},
            )
        hyperspectral[row_index] = spectra_by_key[key]
        aligned_metadata_rows.append({"row_index": row_index, **row})

    extra_keys = set(spectra_by_key) - metadata_keys
    if extra_keys:
        source_id, spectrum_id, sample_name = sorted(extra_keys)[0]
        raise PreparedLibraryBuildError(
            "SIAC normalized spectra contains rows that are missing from spectra metadata.",
            context={"source_id": source_id, "spectrum_id": spectrum_id, "sample_name": sample_name},
        )

    return ["row_index", *metadata_fieldnames], aligned_metadata_rows, hyperspectral, interpolation_summary


def _write_mapping_metadata_parquet(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_csv = Path(tmpdir) / "mapping_metadata.csv"
        with temp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        connection = duckdb.connect()
        try:
            temp_csv_sql = str(temp_csv).replace("'", "''")
            output_sql = str(path).replace("'", "''")
            connection.execute(
                f"""
                COPY (
                  SELECT * FROM read_csv_auto('{temp_csv_sql}', HEADER=TRUE, SAMPLE_SIZE=-1)
                ) TO '{output_sql}' (FORMAT PARQUET)
                """
            )
        finally:
            connection.close()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_runtime_name(path: Path, *, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _artifact_name_for_knn_index(backend: str, sensor_id: str, segment: str) -> str:
    suffix = {
        "faiss": ".faiss",
        "pynndescent": ".pkl",
        "scann": "",
    }.get(backend)
    if suffix is None:
        raise PreparedLibraryBuildError(
            "KNN index persistence is not supported for the requested backend.",
            context={"knn_backend": backend, "supported_knn_index_backends": list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS)},
        )
    return f"knn_indexes/{backend}_{sensor_id}_{segment}{suffix}"


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


def _flatten_knn_index_artifact_names(knn_index_artifacts: Mapping[str, Mapping[str, Mapping[str, str]]]) -> tuple[str, ...]:
    file_names: list[str] = []
    for backend in sorted(knn_index_artifacts):
        backend_payload = knn_index_artifacts[backend]
        for sensor_id in sorted(backend_payload):
            for segment in sorted(backend_payload[sensor_id]):
                file_name = str(backend_payload[sensor_id][segment]).strip()
                if file_name:
                    file_names.append(file_name)
    return tuple(file_names)


def _required_runtime_file_names(manifest: PreparedLibraryManifest) -> tuple[str, ...]:
    stable_files = [
        "manifest.json",
        "mapping_metadata.parquet",
        SEGMENT_FILE_NAMES["vnir"],
        SEGMENT_FILE_NAMES["swir"],
        "sensor_schema.json",
        "checksums.json",
    ]
    stable_files.extend(f"source_{sensor_id}_{segment}.npy" for sensor_id in manifest.source_sensors for segment in SEGMENTS)
    stable_files.extend(_flatten_knn_index_artifact_names(manifest.knn_index_artifacts))
    return tuple(stable_files)


def _validate_manifest_compatibility(manifest: PreparedLibraryManifest) -> None:
    current_major = PREPARED_SCHEMA_VERSION.split(".", 1)[0]
    prepared_major = manifest.schema_version.split(".", 1)[0]
    if current_major != prepared_major:
        raise PreparedLibraryCompatibilityError(
            "Prepared mapping runtime schema major version is incompatible with this package.",
            context={
                "prepared_schema_version": manifest.schema_version,
                "expected_schema_version": PREPARED_SCHEMA_VERSION,
            },
        )


def _load_prepared_manifest(prepared_root: Path) -> PreparedLibraryManifest:
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.exists():
        raise PreparedLibraryValidationError(
            "Prepared mapping runtime is missing manifest.json.",
            context={"prepared_root": str(prepared_root)},
        )

    payload = _read_json_document(
        manifest_path,
        error_factory=PreparedLibraryValidationError,
        document_name="manifest.json",
    )
    if not isinstance(payload, dict):
        raise PreparedLibraryValidationError(
            "manifest.json must contain a JSON object.",
            context={"path": str(manifest_path)},
        )

    try:
        manifest = PreparedLibraryManifest.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparedLibraryValidationError(
            "manifest.json is missing required fields or contains invalid values.",
            context={"path": str(manifest_path)},
        ) from exc

    if not manifest.source_sensors:
        raise PreparedLibraryValidationError(
            "manifest.json must declare at least one source sensor.",
            context={"path": str(manifest_path)},
        )
    if len(set(manifest.source_sensors)) != len(manifest.source_sensors):
        raise PreparedLibraryValidationError(
            "manifest.json source_sensors must be unique.",
            context={"path": str(manifest_path)},
        )
    if manifest.row_count < 1:
        raise PreparedLibraryValidationError(
            "manifest.json row_count must be at least 1.",
            context={"path": str(manifest_path), "row_count": manifest.row_count},
        )
    unsupported_modes = [mode for mode in manifest.supported_output_modes if mode not in SUPPORTED_OUTPUT_MODES]
    if unsupported_modes:
        raise PreparedLibraryValidationError(
            "manifest.json contains unsupported output modes.",
            context={"path": str(manifest_path), "unsupported_output_modes": unsupported_modes},
        )
    if len(manifest.vnir_wavelength_range_nm) != 2 or len(manifest.swir_wavelength_range_nm) != 2:
        raise PreparedLibraryValidationError(
            "manifest.json wavelength ranges must contain exactly two values.",
            context={"path": str(manifest_path)},
        )
    try:
        dtype_np = np.dtype(manifest.array_dtype)
    except TypeError as exc:
        raise PreparedLibraryValidationError(
            "manifest.json array_dtype is not a valid NumPy dtype.",
            context={"path": str(manifest_path), "array_dtype": manifest.array_dtype},
        ) from exc
    if dtype_np.kind != "f":
        raise PreparedLibraryValidationError(
            "manifest.json array_dtype must be floating-point.",
            context={"path": str(manifest_path), "array_dtype": manifest.array_dtype},
        )
    unsupported_knn_backends = [
        backend for backend in manifest.knn_index_artifacts if backend not in SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS
    ]
    if unsupported_knn_backends:
        raise PreparedLibraryValidationError(
            "manifest.json contains unsupported knn_index_artifacts backends.",
            context={"path": str(manifest_path), "unsupported_knn_index_backends": unsupported_knn_backends},
        )
    for backend, backend_payload in manifest.knn_index_artifacts.items():
        for sensor_id, sensor_payload in backend_payload.items():
            if sensor_id not in manifest.source_sensors:
                raise PreparedLibraryValidationError(
                    "manifest.json knn_index_artifacts references a sensor that is not in source_sensors.",
                    context={"path": str(manifest_path), "knn_backend": backend, "sensor_id": sensor_id},
                )
            invalid_segments = [segment for segment in sensor_payload if segment not in SEGMENTS]
            if invalid_segments:
                raise PreparedLibraryValidationError(
                    "manifest.json knn_index_artifacts contains invalid segment keys.",
                    context={
                        "path": str(manifest_path),
                        "knn_backend": backend,
                        "sensor_id": sensor_id,
                        "invalid_segments": invalid_segments,
                    },
                )

    required_checksum_files = {
        file_name for file_name in _required_runtime_file_names(manifest) if file_name not in {"manifest.json", "checksums.json"}
    }
    missing_checksum_files = sorted(required_checksum_files - set(manifest.file_checksums))
    if missing_checksum_files:
        raise PreparedLibraryValidationError(
            "manifest.json file_checksums is missing required runtime files.",
            context={"path": str(manifest_path), "missing_files": missing_checksum_files},
        )
    return manifest


def _load_checksums_payload(prepared_root: Path) -> dict[str, object]:
    checksums_path = prepared_root / "checksums.json"
    if not checksums_path.exists():
        raise PreparedLibraryValidationError(
            "Prepared mapping runtime is missing checksums.json.",
            context={"prepared_root": str(prepared_root)},
        )
    payload = _read_json_document(
        checksums_path,
        error_factory=PreparedLibraryValidationError,
        document_name="checksums.json",
    )
    if not isinstance(payload, dict):
        raise PreparedLibraryValidationError(
            "checksums.json must contain a JSON object.",
            context={"path": str(checksums_path)},
        )
    files_payload = payload.get("files")
    if not isinstance(files_payload, dict):
        raise PreparedLibraryValidationError(
            "checksums.json must contain a files object.",
            context={"path": str(checksums_path)},
        )
    return payload


def _expected_runtime_checksums(manifest: PreparedLibraryManifest, manifest_path: Path) -> dict[str, str]:
    return {
        **manifest.file_checksums,
        manifest_path.name: _sha256_runtime_path(manifest_path),
    }


def prepare_mapping_library(
    siac_root: Path,
    srf_root: Path,
    output_root: Path,
    source_sensors: Sequence[str],
    *,
    dtype: str = "float32",
    knn_index_backends: Sequence[str] | None = None,
) -> PreparedLibraryManifest:
    """Prepare a reusable runtime bundle from SIAC spectra and SRF definitions."""

    dtype_np = np.dtype(dtype)
    if dtype_np.kind != "f":
        raise PreparedLibraryBuildError("prepare_mapping_library only supports floating-point dtypes.", context={"dtype": dtype})

    source_sensor_ids = _normalized_source_sensors(source_sensors)
    persisted_knn_backends = _normalized_knn_index_backends(knn_index_backends)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(siac_root) / "tabular" / "siac_spectra_metadata.csv"
    spectra_path = Path(siac_root) / "tabular" / "siac_normalized_spectra.csv"
    metadata_fieldnames, metadata_rows, hyperspectral, interpolation_summary = _load_siac_rows(
        metadata_path,
        spectra_path,
        dtype=dtype_np,
    )

    sensors = load_sensor_schemas(Path(srf_root))
    missing_source_sensors = [sensor_id for sensor_id in source_sensor_ids if sensor_id not in sensors]
    if missing_source_sensors:
        raise SensorSchemaError(
            "Requested source sensors are missing from the SRF root.",
            context={"missing_source_sensors": missing_source_sensors, "srf_root": str(srf_root)},
        )

    hyperspectral_vnir = hyperspectral[:, _segment_slice("vnir")]
    hyperspectral_swir = hyperspectral[:, _segment_slice("swir")]
    np.save(output_root / SEGMENT_FILE_NAMES["vnir"], hyperspectral_vnir)
    np.save(output_root / SEGMENT_FILE_NAMES["swir"], hyperspectral_swir)

    for sensor_id in source_sensor_ids:
        schema = sensors[sensor_id]
        for segment in SEGMENTS:
            bands = _source_retrieval_bands(schema, segment)
            segment_matrix = _simulate_source_retrieval_matrix(
                hyperspectral,
                bands,
                dtype=dtype_np,
            )
            np.save(output_root / f"source_{sensor_id}_{segment}.npy", segment_matrix)

    knn_index_artifacts: dict[str, dict[str, dict[str, str]]] = {}
    for backend in persisted_knn_backends:
        backend_payload: dict[str, dict[str, str]] = {}
        for sensor_id in source_sensor_ids:
            sensor_payload: dict[str, str] = {}
            for segment in SEGMENTS:
                matrix_path = output_root / f"source_{sensor_id}_{segment}.npy"
                candidate_matrix = np.asarray(np.load(matrix_path), dtype=np.float32)
                artifact_relative_name = _artifact_name_for_knn_index(backend, sensor_id, segment)
                artifact_path = output_root / artifact_relative_name
                _persist_knn_index(candidate_matrix, backend=backend, output_path=artifact_path)
                sensor_payload[segment] = artifact_relative_name
            backend_payload[sensor_id] = sensor_payload
        knn_index_artifacts[backend] = backend_payload

    mapping_metadata_path = output_root / "mapping_metadata.parquet"
    _write_mapping_metadata_parquet(mapping_metadata_path, metadata_fieldnames, metadata_rows)

    sensor_schema_path = output_root / "sensor_schema.json"
    _write_json(
        sensor_schema_path,
        {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "canonical_wavelength_grid": {
                "start_nm": CANONICAL_START_NM,
                "end_nm": CANONICAL_END_NM,
                "step_nm": 1,
            },
            "sensors": [schema.to_dict() for _, schema in sorted(sensors.items())],
        },
    )

    stable_files = [
        mapping_metadata_path,
        output_root / SEGMENT_FILE_NAMES["vnir"],
        output_root / SEGMENT_FILE_NAMES["swir"],
        sensor_schema_path,
    ]
    stable_files.extend(output_root / f"source_{sensor_id}_{segment}.npy" for sensor_id in source_sensor_ids for segment in SEGMENTS)
    stable_files.extend(output_root / file_name for file_name in _flatten_knn_index_artifact_names(knn_index_artifacts))
    file_checksums = {
        _relative_runtime_name(path, root=output_root): _sha256_runtime_path(path)
        for path in stable_files
    }

    manifest = PreparedLibraryManifest(
        schema_version=PREPARED_SCHEMA_VERSION,
        package_version=__version__,
        source_siac_root=str(Path(siac_root)),
        source_siac_build_id=_sha256_paths([metadata_path, spectra_path]),
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source_sensors=tuple(source_sensor_ids),
        supported_output_modes=SUPPORTED_OUTPUT_MODES,
        row_count=int(hyperspectral.shape[0]),
        vnir_wavelength_range_nm=(VNIR_START_NM, VNIR_END_NM),
        swir_wavelength_range_nm=(SWIR_START_NM, SWIR_END_NM),
        array_dtype=dtype_np.name,
        file_checksums=file_checksums,
        knn_index_artifacts=knn_index_artifacts,
        interpolation_summary=interpolation_summary,
    )
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    checksums_payload = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "files": {
            **file_checksums,
            "manifest.json": _sha256_runtime_path(manifest_path),
        },
    }
    _write_json(output_root / "checksums.json", checksums_payload)
    return manifest


def validate_prepared_library(
    prepared_root: Path,
    *,
    verify_checksums: bool = True,
) -> PreparedLibraryManifest:
    """Validate a prepared runtime bundle and return its manifest."""

    prepared_root = Path(prepared_root)
    mapper = SpectralMapper(prepared_root, verify_checksums=verify_checksums)
    return mapper.manifest


def _normalized_text_values(values: Sequence[str] | None) -> tuple[str, ...]:
    """Strip empty strings from a sequence while preserving order."""

    return tuple(text for text in (str(value).strip() for value in (values or ())) if text)


def _normalized_batch_rows(
    reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
) -> list[Sequence[float] | Mapping[str, float]]:
    """Normalize batch reflectance inputs into a mutable row list."""

    if isinstance(reflectance_rows, Mapping):
        raise MappingInputError("map_reflectance_batch requires a batch of samples, not a single mapping.")
    if isinstance(reflectance_rows, np.ndarray):
        if reflectance_rows.ndim != 2 or reflectance_rows.shape[0] == 0:
            raise MappingInputError(
                "reflectance_rows arrays must be two-dimensional with at least one sample row.",
                context={"shape": list(reflectance_rows.shape)},
            )
        return [np.asarray(row) for row in reflectance_rows]
    if isinstance(reflectance_rows, Sequence) and not isinstance(reflectance_rows, (str, bytes)):
        rows = list(reflectance_rows)
        if not rows:
            raise MappingInputError("Batch mapping requires at least one reflectance sample.")
        return rows
    raise MappingInputError("reflectance_rows must be a two-dimensional array or a sequence of per-sample inputs.")


def _normalized_batch_valid_masks(
    valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None,
    *,
    sample_count: int,
) -> list[Sequence[bool] | Mapping[str, bool] | None]:
    """Normalize per-sample valid-band masks for batch mapping."""

    if isinstance(valid_mask_rows, Mapping):
        raise MappingInputError("map_reflectance_batch requires valid_mask_rows to be batched when provided.")
    if valid_mask_rows is None:
        return [None] * sample_count
    if isinstance(valid_mask_rows, np.ndarray):
        if valid_mask_rows.ndim != 2 or valid_mask_rows.shape[0] != sample_count:
            raise MappingInputError(
                "valid_mask_rows arrays must be two-dimensional and aligned to reflectance_rows.",
                context={
                    "reflectance_sample_count": sample_count,
                    "valid_mask_shape": list(valid_mask_rows.shape),
                },
            )
        return [np.asarray(row, dtype=bool) for row in valid_mask_rows]
    if isinstance(valid_mask_rows, Sequence) and not isinstance(valid_mask_rows, (str, bytes)):
        masks = list(valid_mask_rows)
        if len(masks) != sample_count:
            raise MappingInputError(
                "valid_mask_rows must have the same length as reflectance_rows.",
                context={"sample_count": sample_count, "valid_mask_count": len(masks)},
            )
        return masks
    raise MappingInputError("valid_mask_rows must be a two-dimensional array or a sequence of per-sample masks.")


def _failed_segment_retrieval(
    *,
    segment: str,
    valid_band_count: int,
    query_band_ids: tuple[str, ...],
    query_values: np.ndarray,
    query_valid_mask: np.ndarray,
    reason: str,
    include_diagnostics: bool = True,
) -> _SegmentRetrieval:
    """Build a failed retrieval payload with normalized query state."""

    if not include_diagnostics:
        return _SegmentRetrieval(
            segment=segment,
            valid_band_count=valid_band_count,
            query_band_ids=query_band_ids,
            success=False,
            reason=reason,
        )

    return _SegmentRetrieval(
        segment=segment,
        valid_band_count=valid_band_count,
        query_band_ids=query_band_ids,
        query_band_values=np.asarray(query_values, dtype=np.float64),
        query_valid_mask=np.asarray(query_valid_mask, dtype=bool),
        success=False,
        reason=reason,
    )


def _segment_diagnostics_payload_from_fields(
    *,
    success: bool,
    reason: str | None,
    valid_band_count: int,
    query_band_ids: tuple[str, ...],
    query_band_values: np.ndarray,
    query_valid_mask: np.ndarray,
    neighbor_ids: tuple[str, ...],
    neighbor_distances: np.ndarray,
    neighbor_weights: np.ndarray,
    source_fit_rmse: float | None,
    confidence_score: float | None,
    confidence_components: Mapping[str, float],
    neighbor_band_values: np.ndarray,
) -> dict[str, object]:
    normalized_query_values = np.asarray(query_band_values, dtype=np.float64)
    normalized_query_mask = np.asarray(query_valid_mask, dtype=bool)
    normalized_neighbor_distances = np.asarray(neighbor_distances, dtype=np.float64)
    normalized_neighbor_weights = np.asarray(neighbor_weights, dtype=np.float64)
    normalized_neighbor_band_values = np.asarray(neighbor_band_values, dtype=np.float64)
    return {
        "status": "ok" if success else reason,
        "valid_band_count": valid_band_count,
        "query_band_ids": list(query_band_ids),
        "query_band_values": [
            float(value) if bool(is_valid) and np.isfinite(value) else None
            for value, is_valid in zip(normalized_query_values, normalized_query_mask)
        ],
        "query_valid_mask": normalized_query_mask.astype(bool, copy=False).tolist(),
        "neighbor_ids": list(neighbor_ids),
        "neighbor_distances": normalized_neighbor_distances.tolist(),
        "neighbor_weights": normalized_neighbor_weights.tolist(),
        "source_fit_rmse": None if source_fit_rmse is None else float(source_fit_rmse),
        "confidence_score": None if confidence_score is None else float(confidence_score),
        "confidence_components": dict(confidence_components),
        "confidence_policy": _confidence_policy_payload(confidence_score),
        "neighbor_band_values": normalized_neighbor_band_values.tolist(),
    }


def _failed_segment_metadata(
    *,
    valid_band_count: int,
    query_band_ids: tuple[str, ...],
    query_values: np.ndarray,
    query_valid_mask: np.ndarray,
    reason: str,
) -> _RichSegmentMetadata:
    diagnostics = _segment_diagnostics_payload_from_fields(
        success=False,
        reason=reason,
        valid_band_count=valid_band_count,
        query_band_ids=query_band_ids,
        query_band_values=query_values,
        query_valid_mask=query_valid_mask,
        neighbor_ids=(),
        neighbor_distances=np.empty(0, dtype=np.float64),
        neighbor_weights=np.empty(0, dtype=np.float64),
        source_fit_rmse=None,
        confidence_score=None,
        confidence_components={},
        neighbor_band_values=np.empty((0, 0), dtype=np.float64),
    )
    return _RichSegmentMetadata(
        valid_band_count=valid_band_count,
        success=False,
        confidence_score=None,
        diagnostics=diagnostics,
        reason=reason,
    )


def _successful_segment_metadata(
    *,
    valid_band_count: int,
    query_band_ids: tuple[str, ...],
    query_values: np.ndarray,
    query_valid_mask: np.ndarray,
    query_vector: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_weights: np.ndarray,
    neighbor_band_values: np.ndarray,
    source_fit_rmse: float | None,
    row_ids: Sequence[str],
    include_confidence: bool = True,
    confidence_score: float | None = None,
    confidence_components: Mapping[str, float] | None = None,
) -> _RichSegmentMetadata:
    normalized_neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    normalized_neighbor_distances = np.asarray(neighbor_distances, dtype=np.float64)
    normalized_neighbor_weights = np.asarray(neighbor_weights, dtype=np.float64)
    resolved_confidence_score = confidence_score
    resolved_confidence_components = dict(confidence_components or {})
    if include_confidence and confidence_components is None:
        resolved_confidence_score, resolved_confidence_components = _segment_confidence_payload(
            query_vector=np.asarray(query_vector, dtype=np.float64),
            valid_band_count=valid_band_count,
            total_band_count=len(query_band_ids),
            neighbor_distances=normalized_neighbor_distances,
            neighbor_weights=normalized_neighbor_weights,
            source_fit_rmse=source_fit_rmse,
        )
    neighbor_ids = tuple(row_ids[index] for index in normalized_neighbor_indices)
    diagnostics = _segment_diagnostics_payload_from_fields(
        success=True,
        reason=None,
        valid_band_count=valid_band_count,
        query_band_ids=query_band_ids,
        query_band_values=query_values,
        query_valid_mask=query_valid_mask,
        neighbor_ids=neighbor_ids,
        neighbor_distances=normalized_neighbor_distances,
        neighbor_weights=normalized_neighbor_weights,
        source_fit_rmse=source_fit_rmse,
        confidence_score=resolved_confidence_score,
        confidence_components=resolved_confidence_components,
        neighbor_band_values=neighbor_band_values,
    )
    return _RichSegmentMetadata(
        valid_band_count=valid_band_count,
        success=True,
        neighbor_ids=neighbor_ids,
        neighbor_distances=normalized_neighbor_distances,
        confidence_score=resolved_confidence_score,
        diagnostics=diagnostics,
    )


def _successful_segment_retrieval(
    *,
    segment: str,
    valid_band_count: int,
    query_band_ids: tuple[str, ...],
    query_values: np.ndarray,
    query_valid_mask: np.ndarray,
    query_vector: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_weights: np.ndarray,
    neighbor_band_values: np.ndarray,
    source_fit_rmse: float | None,
    reconstructed: np.ndarray,
    row_ids: Sequence[str],
    include_confidence: bool = True,
    include_diagnostics: bool = True,
) -> _SegmentRetrieval:
    """Build a successful retrieval payload including confidence diagnostics."""

    if not include_diagnostics:
        return _SegmentRetrieval(
            segment=segment,
            valid_band_count=valid_band_count,
            query_band_ids=query_band_ids,
            success=True,
            reconstructed=np.asarray(reconstructed, dtype=np.float64),
            source_fit_rmse=source_fit_rmse,
        )

    confidence_score: float | None = None
    confidence_components: dict[str, float] = {}
    if include_confidence:
        confidence_score, confidence_components = _segment_confidence_payload(
            query_vector=np.asarray(query_vector, dtype=np.float64),
            valid_band_count=valid_band_count,
            total_band_count=len(query_band_ids),
            neighbor_distances=np.asarray(neighbor_distances, dtype=np.float64),
            neighbor_weights=np.asarray(neighbor_weights, dtype=np.float64),
            source_fit_rmse=source_fit_rmse,
        )
    return _SegmentRetrieval(
        segment=segment,
        valid_band_count=valid_band_count,
        query_band_ids=query_band_ids,
        query_band_values=np.asarray(query_values, dtype=np.float64),
        query_valid_mask=np.asarray(query_valid_mask, dtype=bool),
        success=True,
        reconstructed=np.asarray(reconstructed, dtype=np.float64),
        neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
        neighbor_ids=tuple(row_ids[index] for index in np.asarray(neighbor_indices, dtype=np.int64)),
        neighbor_distances=np.asarray(neighbor_distances, dtype=np.float64),
        neighbor_weights=np.asarray(neighbor_weights, dtype=np.float64),
        neighbor_band_values=np.asarray(neighbor_band_values, dtype=np.float64),
        source_fit_rmse=source_fit_rmse,
        confidence_score=confidence_score,
        confidence_components=confidence_components,
    )


def _segment_diagnostics_payload(retrieval: _SegmentRetrieval) -> dict[str, object]:
    """Serialize one segment retrieval into a JSON-friendly diagnostic payload."""

    return _segment_diagnostics_payload_from_fields(
        success=retrieval.success,
        reason=retrieval.reason,
        valid_band_count=retrieval.valid_band_count,
        query_band_ids=retrieval.query_band_ids,
        query_band_values=retrieval.query_band_values,
        query_valid_mask=retrieval.query_valid_mask,
        neighbor_ids=retrieval.neighbor_ids,
        neighbor_distances=retrieval.neighbor_distances,
        neighbor_weights=retrieval.neighbor_weights,
        source_fit_rmse=retrieval.source_fit_rmse,
        confidence_score=retrieval.confidence_score,
        confidence_components=retrieval.confidence_components,
        neighbor_band_values=retrieval.neighbor_band_values,
    )


class SpectralMapper:
    """Runtime interface for nearest-neighbor spectral mapping."""

    def __init__(self, prepared_root: Path, *, verify_checksums: bool = False) -> None:
        self.prepared_root = Path(prepared_root)
        self.manifest = _load_prepared_manifest(self.prepared_root)
        _validate_manifest_compatibility(self.manifest)
        self._validate_required_runtime_files()
        self._validate_checksums(verify_checksums=verify_checksums)

        self._sensor_schemas = self._load_prepared_sensor_schemas()
        self._row_ids, self._row_sample_names = self._load_row_metadata()
        self._row_index_by_id = {row_id: index for index, row_id in enumerate(self._row_ids)}
        row_indices_by_sample_name: dict[str, list[int]] = {}
        for row_index, sample_name in enumerate(self._row_sample_names):
            row_indices_by_sample_name.setdefault(sample_name, []).append(row_index)
        self._row_indices_by_sample_name = {
            sample_name: tuple(row_indices)
            for sample_name, row_indices in row_indices_by_sample_name.items()
        }
        self._all_row_indices = np.arange(self.manifest.row_count, dtype=np.int64)
        self._hyperspectral_cache: dict[str, np.ndarray] = {}
        self._source_matrix_cache: dict[tuple[str, str], np.ndarray] = {}
        self._response_cache: dict[tuple[str, str, bool], np.ndarray] = {}
        self._target_sensor_projection_cache: dict[str, _TargetSensorProjection] = {}
        self._source_query_cache: dict[str, np.ndarray] = {}
        self._knn_index_cache: dict[tuple[str, str, str], Any] = {}
        self._scipy_ckdtree_cache: dict[tuple[str, str], _ScipyCkdtreeCacheEntry] = {}

        self._validate_prepared_layout()

    def _validate_required_runtime_files(self) -> None:
        for file_name in _required_runtime_file_names(self.manifest):
            if not (self.prepared_root / file_name).exists():
                raise PreparedLibraryValidationError(
                    "Prepared mapping runtime is missing a required file.",
                    context={"prepared_root": str(self.prepared_root), "file_name": file_name},
                )

    def _validate_checksums(self, *, verify_checksums: bool) -> None:
        checksums_payload = _load_checksums_payload(self.prepared_root)
        recorded_checksums = checksums_payload["files"]
        manifest_path = self.prepared_root / "manifest.json"
        expected_checksums = _expected_runtime_checksums(self.manifest, manifest_path)
        for file_name, expected_checksum in expected_checksums.items():
            if recorded_checksums.get(file_name) != expected_checksum:
                raise PreparedLibraryValidationError(
                    "checksums.json does not match the manifest checksum records.",
                    context={"prepared_root": str(self.prepared_root), "file_name": file_name},
                )
            if verify_checksums:
                actual_checksum = _sha256_runtime_path(self.prepared_root / file_name)
                if actual_checksum != expected_checksum:
                    raise PreparedLibraryValidationError(
                        "Prepared mapping runtime checksum verification failed.",
                        context={"prepared_root": str(self.prepared_root), "file_name": file_name},
                    )

    def _load_prepared_sensor_schemas(self) -> dict[str, SensorSRFSchema]:
        sensor_schema_path = self.prepared_root / "sensor_schema.json"
        schemas: dict[str, SensorSRFSchema] = {}
        for payload in _load_sensor_payloads(sensor_schema_path):
            schema = SensorSRFSchema.from_dict(payload)
            if schema.sensor_id in schemas:
                raise PreparedLibraryValidationError(
                    "sensor_schema.json contains duplicate sensor_id values.",
                    context={"prepared_root": str(self.prepared_root), "sensor_id": schema.sensor_id},
                )
            schemas[schema.sensor_id] = schema
        return schemas

    def _load_row_metadata(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        metadata_path = self.prepared_root / "mapping_metadata.parquet"
        connection = duckdb.connect()
        try:
            try:
                rows = connection.execute(
                    """
                    SELECT row_index, source_id, spectrum_id, sample_name
                    FROM read_parquet(?)
                    ORDER BY row_index
                    """,
                    [str(metadata_path)],
                ).fetchall()
            except duckdb.Error as exc:
                raise PreparedLibraryValidationError(
                    "mapping_metadata.parquet is missing required row-alignment columns.",
                    context={"prepared_root": str(self.prepared_root), "path": str(metadata_path)},
                ) from exc
        finally:
            connection.close()

        if len(rows) != self.manifest.row_count:
            raise PreparedLibraryValidationError(
                "Prepared mapping metadata row count does not match manifest row_count.",
                context={"metadata_rows": len(rows), "manifest_row_count": self.manifest.row_count},
            )
        try:
            row_indices = [int(row[0]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise PreparedLibraryValidationError(
                "mapping_metadata.parquet row_index values must be integers.",
                context={"prepared_root": str(self.prepared_root), "path": str(metadata_path)},
            ) from exc
        expected_row_indices = list(range(self.manifest.row_count))
        if row_indices != expected_row_indices:
            raise PreparedLibraryValidationError(
                "mapping_metadata.parquet row_index values must be unique and contiguous from 0 to row_count - 1.",
                context={"prepared_root": str(self.prepared_root), "path": str(metadata_path)},
            )

        row_ids = tuple(f"{row[1]}:{row[2]}:{row[3]}" for row in rows)
        if len(set(row_ids)) != len(row_ids):
            raise PreparedLibraryValidationError(
                "mapping_metadata.parquet contains duplicate prepared row identities.",
                context={"prepared_root": str(self.prepared_root), "path": str(metadata_path)},
            )
        sample_names = tuple(str(row[3]) for row in rows)
        return row_ids, sample_names

    def _validate_prepared_layout(self) -> None:
        for segment in SEGMENTS:
            array = self._load_hyperspectral(segment)
            expected_width = len(SEGMENT_WAVELENGTHS[segment])
            if array.shape != (self.manifest.row_count, expected_width):
                raise PreparedLibraryValidationError(
                    "Prepared hyperspectral array shape does not match the manifest.",
                    context={
                        "segment": segment,
                        "expected_shape": [self.manifest.row_count, expected_width],
                        "actual_shape": list(array.shape),
                    },
                )

        for sensor_id in self.manifest.source_sensors:
            schema = self.get_sensor_schema(sensor_id)
            for segment in SEGMENTS:
                matrix = self._load_source_matrix(sensor_id, segment)
                expected_shape = (self.manifest.row_count, len(_source_retrieval_bands(schema, segment)))
                if matrix.shape != expected_shape:
                    raise PreparedLibraryValidationError(
                        "Prepared source retrieval matrix shape does not match the sensor schema.",
                        context={
                            "sensor_id": sensor_id,
                            "segment": segment,
                            "expected_shape": list(expected_shape),
                            "actual_shape": list(matrix.shape),
                        },
                    )

    def get_sensor_schema(self, sensor_id: str) -> SensorSRFSchema:
        """Return the prepared sensor schema for ``sensor_id``."""

        if sensor_id not in self._sensor_schemas:
            raise SensorSchemaError(
                "Requested sensor_id is not present in the prepared sensor schema.",
                context={"sensor_id": sensor_id},
            )
        return self._sensor_schemas[sensor_id]

    def _load_hyperspectral(self, segment: str) -> np.ndarray:
        if segment not in self._hyperspectral_cache:
            path = self.prepared_root / SEGMENT_FILE_NAMES[segment]
            try:
                self._hyperspectral_cache[segment] = np.load(path, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise PreparedLibraryValidationError(
                    "Prepared hyperspectral array could not be loaded.",
                    context={"prepared_root": str(self.prepared_root), "path": str(path), "segment": segment},
                ) from exc
        return self._hyperspectral_cache[segment]

    def _load_source_matrix(self, source_sensor: str, segment: str) -> np.ndarray:
        key = (source_sensor, segment)
        if key not in self._source_matrix_cache:
            path = self.prepared_root / f"source_{source_sensor}_{segment}.npy"
            try:
                self._source_matrix_cache[key] = np.load(path, mmap_mode="r")
            except (OSError, ValueError) as exc:
                raise PreparedLibraryValidationError(
                    "Prepared source retrieval matrix could not be loaded.",
                    context={"path": str(path), "source_sensor": source_sensor, "segment": segment},
                ) from exc
        return self._source_matrix_cache[key]

    def _persisted_knn_index_path(self, backend: str, source_sensor: str, segment: str) -> Path | None:
        relative_path = (
            self.manifest.knn_index_artifacts.get(backend, {})
            .get(source_sensor, {})
            .get(segment)
        )
        if not relative_path:
            return None
        return self.prepared_root / relative_path

    def _load_persisted_knn_index(self, backend: str, source_sensor: str, segment: str) -> Any | None:
        path = self._persisted_knn_index_path(backend, source_sensor, segment)
        if path is None:
            return None
        key = (backend, source_sensor, segment)
        if key not in self._knn_index_cache:
            try:
                self._knn_index_cache[key] = _load_persisted_knn_index(path, backend=backend)
            except SpectralLibraryError:
                raise
            except Exception as exc:
                raise PreparedLibraryValidationError(
                    "Persisted KNN index could not be loaded.",
                    context={
                        "prepared_root": str(self.prepared_root),
                        "knn_backend": backend,
                        "source_sensor": source_sensor,
                        "segment": segment,
                        "path": str(path),
                    },
                ) from exc
        return self._knn_index_cache[key]

    def _can_use_persisted_knn_index(
        self,
        *,
        backend: str,
        source_sensor: str,
        segment: str,
        valid_mask: np.ndarray,
        candidate_row_indices: np.ndarray,
    ) -> bool:
        if backend not in SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS:
            return False
        if not np.all(valid_mask):
            return False
        if not self._uses_all_candidate_rows(candidate_row_indices):
            return False
        return self._persisted_knn_index_path(backend, source_sensor, segment) is not None

    def _query_persisted_knn_local_indices(
        self,
        *,
        backend: str,
        source_sensor: str,
        segment: str,
        query_values: np.ndarray,
        k: int,
        knn_eps: float,
    ) -> np.ndarray | None:
        index = self._load_persisted_knn_index(backend, source_sensor, segment)
        if index is None:
            return None
        return _query_persisted_knn_index(
            index,
            query_values,
            k=k,
            knn_backend=backend,
            knn_eps=knn_eps,
        )

    def _band_response(self, sensor_id: str, band: SensorBandDefinition, *, segment_only: bool) -> np.ndarray:
        cache_key = (sensor_id, band.band_id, segment_only)
        if cache_key not in self._response_cache:
            self._response_cache[cache_key] = _resample_band_response(band, segment_only=segment_only)
        return self._response_cache[cache_key]

    def _target_sensor_projection(self, target_sensor: str) -> _TargetSensorProjection:
        if target_sensor not in self._target_sensor_projection_cache:
            schema = self.get_sensor_schema(target_sensor)
            response_columns: dict[str, list[np.ndarray]] = {segment: [] for segment in SEGMENTS}
            output_indices: dict[str, list[int]] = {segment: [] for segment in SEGMENTS}
            for output_index, band in enumerate(schema.bands):
                response = np.asarray(self._band_response(target_sensor, band, segment_only=True), dtype=np.float64)
                denominator = float(np.sum(response))
                if denominator <= 0:
                    raise SensorSchemaError(
                        "Resampled target SRF support must remain positive.",
                        context={"target_sensor": target_sensor, "band_id": band.band_id},
                    )
                response_columns[band.segment].append(response / denominator)
                output_indices[band.segment].append(output_index)

            def _compressed_projection(segment: str) -> tuple[np.ndarray, np.ndarray]:
                if not response_columns[segment]:
                    return (
                        np.empty((0, 0), dtype=np.float64),
                        np.empty(0, dtype=np.int64),
                    )
                full_projection = np.ascontiguousarray(np.column_stack(response_columns[segment]), dtype=np.float64)
                support_mask = np.any(full_projection != 0.0, axis=1)
                support_indices = np.flatnonzero(support_mask).astype(np.int64)
                if support_indices.size == 0:
                    return (
                        np.empty((0, full_projection.shape[1]), dtype=np.float64),
                        support_indices,
                    )
                return (
                    np.ascontiguousarray(full_projection[support_indices], dtype=np.float64),
                    support_indices,
                )

            vnir_response_matrix, vnir_support_indices = _compressed_projection("vnir")
            swir_response_matrix, swir_support_indices = _compressed_projection("swir")
            vnir_hyperspectral_rows = np.ascontiguousarray(
                self._load_hyperspectral("vnir")[:, vnir_support_indices],
                dtype=np.float64,
            )
            swir_hyperspectral_rows = np.ascontiguousarray(
                self._load_hyperspectral("swir")[:, swir_support_indices],
                dtype=np.float64,
            )
            vnir_target_rows = np.ascontiguousarray(vnir_hyperspectral_rows @ vnir_response_matrix, dtype=np.float64)
            swir_target_rows = np.ascontiguousarray(swir_hyperspectral_rows @ swir_response_matrix, dtype=np.float64)

            self._target_sensor_projection_cache[target_sensor] = _TargetSensorProjection(
                output_width=len(schema.bands),
                vnir_response_matrix=vnir_response_matrix,
                swir_response_matrix=swir_response_matrix,
                vnir_output_indices=np.asarray(output_indices["vnir"], dtype=np.int64),
                swir_output_indices=np.asarray(output_indices["swir"], dtype=np.int64),
                vnir_support_indices=vnir_support_indices,
                swir_support_indices=swir_support_indices,
                vnir_hyperspectral_rows=vnir_hyperspectral_rows,
                swir_hyperspectral_rows=swir_hyperspectral_rows,
                vnir_target_rows=vnir_target_rows,
                swir_target_rows=swir_target_rows,
            )
        return self._target_sensor_projection_cache[target_sensor]

    def _coerce_query(
        self,
        schema: SensorSRFSchema,
        reflectance: Sequence[float] | Mapping[str, float],
        valid_mask: Sequence[bool] | Mapping[str, bool] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(reflectance, Mapping):
            reflectance_values = np.asarray(
                [float(reflectance.get(band.band_id, np.nan)) for band in schema.bands],
                dtype=np.float64,
            )
            if isinstance(valid_mask, Mapping):
                valid_values = np.asarray([bool(valid_mask.get(band.band_id, True)) for band in schema.bands], dtype=bool)
            elif valid_mask is None:
                valid_values = np.isfinite(reflectance_values)
            else:
                raise MappingInputError(
                    "Mapping reflectance dictionaries require valid_mask to be a dictionary when provided.",
                    context={"sensor_id": schema.sensor_id},
                )
        else:
            reflectance_values = np.asarray(reflectance, dtype=np.float64)
            if reflectance_values.ndim != 1 or reflectance_values.shape[0] != len(schema.bands):
                raise MappingInputError(
                    "Reflectance arrays must be one-dimensional and aligned to the source sensor band order.",
                    context={"sensor_id": schema.sensor_id, "expected_length": len(schema.bands)},
                )
            if valid_mask is None:
                valid_values = np.isfinite(reflectance_values)
            else:
                valid_values = np.asarray(valid_mask, dtype=bool)
                if valid_values.shape != reflectance_values.shape:
                    raise MappingInputError(
                        "valid_mask must have the same shape as reflectance.",
                        context={"sensor_id": schema.sensor_id, "expected_length": len(schema.bands)},
                    )

        valid_values = valid_values & np.isfinite(reflectance_values)
        if not np.any(valid_values):
            raise MappingInputError("At least one valid source reflectance band is required for mapping.")
        return reflectance_values, valid_values

    def _validated_source_matrix(self, source_sensor: str, segment: str, query_band_ids: tuple[str, ...]) -> np.ndarray:
        """Load a prepared source matrix and verify its width against the schema."""

        source_matrix = self._load_source_matrix(source_sensor, segment)
        if source_matrix.shape[1] != len(query_band_ids):
            raise PreparedLibraryValidationError(
                "Prepared source matrix width does not match the source sensor schema.",
                context={"source_sensor": source_sensor, "segment": segment},
            )
        return source_matrix

    def _uses_all_candidate_rows(self, candidate_row_indices: np.ndarray) -> bool:
        return candidate_row_indices is self._all_row_indices or (
            candidate_row_indices.shape == self._all_row_indices.shape
            and np.array_equal(candidate_row_indices, self._all_row_indices)
        )

    def _candidate_source_matrix_view(
        self,
        source_matrix: np.ndarray,
        candidate_row_indices: np.ndarray,
    ) -> np.ndarray:
        if self._uses_all_candidate_rows(candidate_row_indices):
            return source_matrix
        return source_matrix[candidate_row_indices]

    def _load_scipy_ckdtree(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_band_ids: tuple[str, ...],
    ) -> Any:
        cache_key = (source_sensor, segment)
        if cache_key not in self._scipy_ckdtree_cache:
            source_matrix = self._validated_source_matrix(source_sensor, segment, query_band_ids)
            tree_data = np.asarray(source_matrix, dtype=np.float64)
            cKDTree = _load_ckdtree_class()
            self._scipy_ckdtree_cache[cache_key] = _ScipyCkdtreeCacheEntry(
                data=tree_data,
                index=cKDTree(tree_data),
            )
        return self._scipy_ckdtree_cache[cache_key].index

    def _query_cached_scipy_ckdtree_local_indices(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_band_ids: tuple[str, ...],
        query_values: np.ndarray,
        k: int,
        knn_eps: float,
    ) -> np.ndarray:
        index = self._load_scipy_ckdtree(
            source_sensor=source_sensor,
            segment=segment,
            query_band_ids=query_band_ids,
        )
        return _query_knn_index(
            index,
            query_values,
            k=k,
            knn_backend="scipy_ckdtree",
            knn_eps=knn_eps,
        )

    def _query_cached_scipy_ckdtree_local_results(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_band_ids: tuple[str, ...],
        query_values: np.ndarray,
        k: int,
        knn_eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self._load_scipy_ckdtree(
            source_sensor=source_sensor,
            segment=segment,
            query_band_ids=query_band_ids,
        )
        return _query_knn_index_with_distances(
            index,
            query_values,
            k=k,
            knn_backend="scipy_ckdtree",
            knn_eps=knn_eps,
        )

    def _retrieve_segment(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_values: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        candidate_row_indices: np.ndarray,
        include_diagnostics: bool = True,
    ) -> _SegmentRetrieval:
        source_schema = self.get_sensor_schema(source_sensor)
        query_band_ids = _source_retrieval_band_ids(source_schema, segment)
        valid_band_count = int(valid_mask.sum())
        if valid_band_count < min_valid_bands:
            return _failed_segment_retrieval(
                segment=segment,
                valid_band_count=valid_band_count,
                query_band_ids=query_band_ids,
                query_values=query_values,
                query_valid_mask=valid_mask,
                reason="insufficient_valid_bands",
                include_diagnostics=include_diagnostics,
            )

        source_matrix = self._validated_source_matrix(source_sensor, segment, query_band_ids)
        candidate_source_matrix = self._candidate_source_matrix_view(source_matrix, candidate_row_indices)
        if np.all(valid_mask):
            valid_indices: slice | np.ndarray = slice(None)
            candidate_matrix = candidate_source_matrix
            query_vector = query_values
        else:
            valid_indices = np.flatnonzero(valid_mask)
            candidate_matrix = candidate_source_matrix[:, valid_indices]
            query_vector = query_values[valid_indices]
        if knn_backend == "scipy_ckdtree" and float(knn_eps) == 0.0:
            if self._uses_all_candidate_rows(candidate_row_indices) and np.all(valid_mask):
                local_indices, local_distances = self._query_cached_scipy_ckdtree_local_results(
                    source_sensor=source_sensor,
                    segment=segment,
                    query_band_ids=query_band_ids,
                    query_values=query_values,
                    k=min(int(k), int(candidate_row_indices.size)),
                    knn_eps=knn_eps,
                )
            else:
                local_indices, local_distances = _query_local_scipy_ckdtree_results(
                    candidate_matrix,
                    np.asarray(query_vector, dtype=np.float64),
                    k=min(int(k), int(candidate_row_indices.size)),
                    knn_eps=knn_eps,
                )
            neighbor_indices, neighbor_distances = _ordered_neighbor_rows_from_local_distances(
                local_indices[0],
                local_distances[0],
                candidate_row_indices,
                query_width=int(query_vector.shape[0]),
                k=k,
            )
        elif knn_backend == "scipy_ckdtree" and self._uses_all_candidate_rows(candidate_row_indices) and np.all(valid_mask):
            local_indices = self._query_cached_scipy_ckdtree_local_indices(
                source_sensor=source_sensor,
                segment=segment,
                query_band_ids=query_band_ids,
                query_values=query_values,
                k=min(int(k), int(candidate_row_indices.size)),
                knn_eps=knn_eps,
            )
            neighbor_indices, neighbor_distances = _refine_neighbor_rows(
                candidate_source_matrix,
                query_values,
                candidate_row_indices,
                local_indices[0],
                k=k,
            )
        elif self._can_use_persisted_knn_index(
            backend=knn_backend,
            source_sensor=source_sensor,
            segment=segment,
            valid_mask=np.asarray(valid_mask, dtype=bool),
            candidate_row_indices=candidate_row_indices,
        ):
            local_indices = self._query_persisted_knn_local_indices(
                backend=knn_backend,
                source_sensor=source_sensor,
                segment=segment,
                query_values=query_values,
                k=min(int(k), int(candidate_row_indices.size)),
                knn_eps=knn_eps,
            )
            if local_indices is None:
                raise MappingInputError(
                    f"Persisted KNN index query returned no results for {source_sensor}/{segment}.",
                    context={"source_sensor": source_sensor, "segment": segment, "knn_backend": knn_backend},
                )
            neighbor_indices, neighbor_distances = _refine_neighbor_rows(
                candidate_source_matrix,
                query_values,
                candidate_row_indices,
                local_indices[0],
                k=k,
            )
        else:
            neighbor_indices, neighbor_distances = _search_neighbor_rows(
                candidate_matrix,
                query_vector,
                candidate_row_indices,
                k=k,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
            )

        resolved_valid_indices = None if isinstance(valid_indices, slice) else np.asarray(valid_indices, dtype=np.int64)
        neighbor_index_matrix = np.asarray(neighbor_indices, dtype=np.int64)[np.newaxis, :]
        neighbor_distance_matrix = np.asarray(neighbor_distances, dtype=np.float64)[np.newaxis, :]
        reconstructed_batch, neighbor_weight_batch, source_fit_rmse_batch = _combine_neighbor_spectra_batch_accel(
            hyperspectral_rows=self._load_hyperspectral(segment),
            source_matrix=source_matrix,
            neighbor_indices=neighbor_index_matrix,
            neighbor_distances=neighbor_distance_matrix,
            query_values=np.asarray(query_vector, dtype=np.float64)[np.newaxis, :],
            valid_indices=resolved_valid_indices,
            neighbor_estimator=neighbor_estimator,
        )
        neighbor_band_values = (
            np.asarray(source_matrix[neighbor_indices], dtype=np.float64)
            if include_diagnostics
            else np.empty((0, 0), dtype=np.float64)
        )
        reconstructed = reconstructed_batch[0]
        neighbor_weights = neighbor_weight_batch[0]
        source_fit_rmse = float(source_fit_rmse_batch[0])
        return _successful_segment_retrieval(
            segment=segment,
            valid_band_count=valid_band_count,
            query_band_ids=query_band_ids,
            query_values=query_values,
            query_valid_mask=valid_mask,
            query_vector=query_vector,
            neighbor_indices=neighbor_indices,
            neighbor_distances=np.asarray(neighbor_distances, dtype=np.float64),
            neighbor_weights=neighbor_weights,
            neighbor_band_values=neighbor_band_values,
            source_fit_rmse=source_fit_rmse,
            reconstructed=reconstructed,
            row_ids=self._row_ids,
            include_diagnostics=include_diagnostics,
        )

    def _retrieve_segment_batch(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_values: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        candidate_row_indices: np.ndarray,
        include_confidence: bool = True,
    ) -> _RichSegmentBatch:
        source_schema = self.get_sensor_schema(source_sensor)
        query_band_ids = _source_retrieval_band_ids(source_schema, segment)
        source_matrix = self._validated_source_matrix(source_sensor, segment, query_band_ids)
        batch_size = int(query_values.shape[0])
        dense_output = _empty_dense_segment_output_batch(segment=segment, batch_size=batch_size)
        metadata_rows: list[_RichSegmentMetadata | None] = [None] * batch_size
        candidate_matrix = self._candidate_source_matrix_view(source_matrix, candidate_row_indices)
        segment_hyperspectral = self._load_hyperspectral(segment)
        valid_patterns, inverse = np.unique(valid_mask, axis=0, return_inverse=True)

        for pattern_index, pattern in enumerate(valid_patterns):
            batch_indices = np.flatnonzero(inverse == pattern_index)
            valid_band_count = int(pattern.sum())
            dense_output.valid_band_count[batch_indices] = valid_band_count
            if valid_band_count < min_valid_bands:
                for batch_index in batch_indices:
                    metadata_rows[int(batch_index)] = _failed_segment_metadata(
                        valid_band_count=valid_band_count,
                        query_band_ids=query_band_ids,
                        query_values=query_values[batch_index],
                        query_valid_mask=valid_mask[batch_index],
                        reason="insufficient_valid_bands",
                    )
                continue

            if np.all(pattern):
                valid_indices = slice(None)
                candidate_valid = candidate_matrix
                query_group = query_values[batch_indices]
            else:
                valid_indices = np.flatnonzero(pattern)
                candidate_valid = candidate_matrix[:, valid_indices]
                query_group = query_values[batch_indices][:, valid_indices]
            neighbor_index_matrix, neighbor_distance_matrix = self._refine_neighbor_rows_batch(
                source_sensor=source_sensor,
                segment=segment,
                query_band_ids=query_band_ids,
                query_values=np.asarray(query_values[batch_indices], dtype=np.float64),
                query_group=np.asarray(query_group, dtype=np.float64),
                valid_pattern=np.asarray(pattern, dtype=bool),
                candidate_row_indices=candidate_row_indices,
                candidate_valid=candidate_valid,
                k=k,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
            )
            resolved_valid_indices = None if isinstance(valid_indices, slice) else np.asarray(valid_indices, dtype=np.int64)
            reconstructed_batch, neighbor_weight_batch, source_fit_rmse_batch = _combine_neighbor_spectra_batch_accel(
                hyperspectral_rows=segment_hyperspectral,
                source_matrix=source_matrix,
                neighbor_indices=neighbor_index_matrix,
                neighbor_distances=neighbor_distance_matrix,
                query_values=np.asarray(query_group, dtype=np.float64),
                valid_indices=resolved_valid_indices,
                neighbor_estimator=neighbor_estimator,
            )
            neighbor_band_values_batch = np.asarray(source_matrix[neighbor_index_matrix], dtype=np.float64)
            if include_confidence:
                confidence_scores_batch, confidence_components_batch = _segment_confidence_payload_batch(
                    query_matrix=np.asarray(query_group, dtype=np.float64),
                    valid_band_count=valid_band_count,
                    total_band_count=len(query_band_ids),
                    neighbor_distance_matrix=neighbor_distance_matrix,
                    neighbor_weight_matrix=np.asarray(neighbor_weight_batch, dtype=np.float64),
                    source_fit_rmse=np.asarray(source_fit_rmse_batch, dtype=np.float64),
                )
            else:
                confidence_scores_batch = np.zeros(batch_indices.size, dtype=np.float64)
                confidence_components_batch = tuple({} for _ in range(batch_indices.size))
            dense_output.success[batch_indices] = True
            dense_output.reconstructed[batch_indices] = np.asarray(reconstructed_batch, dtype=np.float64)
            dense_output.source_fit_rmse[batch_indices] = np.asarray(source_fit_rmse_batch, dtype=np.float64)

            for local_index, batch_index in enumerate(batch_indices):
                metadata = _successful_segment_metadata(
                    valid_band_count=valid_band_count,
                    query_band_ids=query_band_ids,
                    query_values=query_values[batch_index],
                    query_valid_mask=valid_mask[batch_index],
                    query_vector=query_group[local_index],
                    neighbor_indices=neighbor_index_matrix[local_index],
                    neighbor_distances=neighbor_distance_matrix[local_index],
                    neighbor_weights=neighbor_weight_batch[local_index],
                    neighbor_band_values=neighbor_band_values_batch[local_index],
                    source_fit_rmse=float(source_fit_rmse_batch[local_index]),
                    row_ids=self._row_ids,
                    include_confidence=include_confidence,
                    confidence_score=None if not include_confidence else float(confidence_scores_batch[local_index]),
                    confidence_components=confidence_components_batch[local_index],
                )
                metadata_rows[int(batch_index)] = metadata
                if metadata.confidence_score is not None:
                    dense_output.confidence_score[int(batch_index)] = float(metadata.confidence_score)

        return _RichSegmentBatch(
            dense_output=dense_output,
            metadata=tuple(
                metadata
                if metadata is not None
                else _failed_segment_metadata(
                    valid_band_count=0,
                    query_band_ids=query_band_ids,
                    query_values=np.empty(0, dtype=np.float64),
                    query_valid_mask=np.empty(0, dtype=bool),
                    reason="insufficient_valid_bands",
                )
                for metadata in metadata_rows
            ),
        )

    def _simulate_target_sensor(
        self,
        target_sensor: str,
        segment_outputs: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        target_schema = self.get_sensor_schema(target_sensor)
        values: list[float] = []
        band_ids: list[str] = []
        for band in target_schema.bands:
            if band.segment not in segment_outputs:
                continue
            reconstructed = segment_outputs[band.segment]
            response = self._band_response(target_sensor, band, segment_only=True)
            values.append(
                float(
                    _response_weighted_average(
                        reconstructed,
                        response,
                        error_message="Resampled target SRF support must remain positive.",
                        error_context={"target_sensor": target_sensor, "band_id": band.band_id},
                    )
                )
            )
            band_ids.append(band.band_id)
        if not values:
            return None, ()
        return np.asarray(values, dtype=np.float64), tuple(band_ids)

    def compile_linear_mapper(
        self,
        *,
        source_sensor: str,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        dtype: str | np.dtype[Any] = np.float32,
        compile_chunk_size: int | None = None,
    ) -> LinearSpectralMapper:
        """Compile a dense array-to-array mapper for high-throughput inference."""

        _ensure_supported_output_mode(output_mode)
        runtime_dtype = _normalized_linear_mapper_dtype(dtype)
        source_schema = self.get_sensor_schema(source_sensor)
        source_queries = self._source_queries(source_sensor)

        output_band_ids: tuple[str, ...] = ()
        output_wavelength_nm: np.ndarray | None = None
        if output_mode == "target_sensor":
            if not target_sensor:
                raise MappingInputError("target_sensor is required when output_mode is target_sensor.")
            target_matrix, output_band_ids = self._simulate_full_sensor_matrix(target_sensor)
            output_width = int(target_matrix.shape[1])

            def output_loader(start: int, stop: int) -> np.ndarray:
                return np.asarray(target_matrix[start:stop], dtype=np.float64)

        elif output_mode == "vnir_spectrum":
            vnir = self._load_hyperspectral("vnir")
            output_width = int(vnir.shape[1])
            output_wavelength_nm = VNIR_WAVELENGTHS.astype(np.float64)

            def output_loader(start: int, stop: int) -> np.ndarray:
                return np.asarray(vnir[start:stop], dtype=np.float64)

        elif output_mode == "swir_spectrum":
            swir = self._load_hyperspectral("swir")
            output_width = int(swir.shape[1])
            output_wavelength_nm = SWIR_WAVELENGTHS.astype(np.float64)

            def output_loader(start: int, stop: int) -> np.ndarray:
                return np.asarray(swir[start:stop], dtype=np.float64)

        else:
            vnir = self._load_hyperspectral("vnir")
            swir = self._load_hyperspectral("swir")
            output_width = FULL_WAVELENGTH_COUNT
            output_wavelength_nm = CANONICAL_WAVELENGTHS.astype(np.float64)

            def output_loader(start: int, stop: int) -> np.ndarray:
                return _assemble_full_spectrum_batch(
                    np.asarray(vnir[start:stop], dtype=np.float64),
                    np.asarray(swir[start:stop], dtype=np.float64),
                )

        resolved_compile_chunk_size = _normalized_linear_mapper_chunk_size(
            compile_chunk_size,
            input_width=int(source_queries.shape[1]),
            output_width=int(output_width),
            dtype=np.dtype(np.float64),
        )
        bias, weights = _fit_linear_map(
            source_queries,
            output_width=output_width,
            output_loader=output_loader,
            compile_chunk_size=resolved_compile_chunk_size,
        )
        return LinearSpectralMapper(
            source_sensor=source_sensor,
            output_mode=output_mode,
            target_sensor=target_sensor,
            source_band_ids=source_schema.band_ids(),
            output_band_ids=output_band_ids,
            output_wavelength_nm=output_wavelength_nm,
            weights=np.ascontiguousarray(weights, dtype=runtime_dtype),
            bias=np.ascontiguousarray(bias, dtype=runtime_dtype),
            dtype=runtime_dtype,
        )

    def _candidate_rows(
        self,
        candidate_row_indices: Sequence[int] | None,
        *,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        if candidate_row_indices is None:
            candidate = self._all_row_indices
        else:
            candidate = np.asarray(candidate_row_indices, dtype=np.int64)
            if candidate.ndim != 1 or candidate.size == 0:
                raise MappingInputError("candidate_row_indices must be a non-empty one-dimensional sequence.")
            if np.any(candidate < 0) or np.any(candidate >= self.manifest.row_count):
                raise MappingInputError(
                    "candidate_row_indices must refer to valid prepared-library rows.",
                    context={"row_count": self.manifest.row_count},
                )
            candidate = np.unique(candidate)

        if not exclude_row_ids and not exclude_sample_names:
            return candidate

        excluded = self._excluded_candidate_indices(
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
        )
        if excluded.size:
            candidate = candidate[~np.isin(candidate, excluded)]
        if candidate.size == 0:
            raise MappingInputError(
                "Candidate exclusions removed every prepared-library row.",
                context={
                    "row_count": self.manifest.row_count,
                    "excluded_row_id_count": len(tuple(exclude_row_ids or ())),
                    "excluded_sample_name_count": len(tuple(exclude_sample_names or ())),
                },
            )
        return candidate

    def _excluded_candidate_indices(
        self,
        *,
        exclude_row_ids: Sequence[str] | None,
        exclude_sample_names: Sequence[str] | None,
    ) -> np.ndarray:
        normalized_exclude_row_ids = _normalized_text_values(exclude_row_ids)
        normalized_exclude_sample_names = _normalized_text_values(exclude_sample_names)

        excluded_indices: set[int] = set()
        if normalized_exclude_row_ids:
            unknown_row_ids = sorted(
                row_id for row_id in normalized_exclude_row_ids if row_id not in self._row_index_by_id
            )
            if unknown_row_ids:
                raise MappingInputError(
                    "exclude_row_ids must refer to existing prepared row ids.",
                    context={"unknown_row_ids": unknown_row_ids},
                )
            excluded_indices.update(self._row_index_by_id[row_id] for row_id in normalized_exclude_row_ids)

        if normalized_exclude_sample_names:
            unknown_sample_names = sorted(
                sample_name
                for sample_name in normalized_exclude_sample_names
                if sample_name not in self._row_indices_by_sample_name
            )
            if unknown_sample_names:
                raise MappingInputError(
                    "exclude_sample_names must refer to existing prepared sample_name values.",
                    context={"unknown_sample_names": unknown_sample_names},
                )
            for sample_name in normalized_exclude_sample_names:
                excluded_indices.update(self._row_indices_by_sample_name[sample_name])

        return np.asarray(sorted(excluded_indices), dtype=np.int64)

    def candidate_row_indices(
        self,
        candidate_row_indices: Sequence[int] | None = None,
        *,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return prepared row indices after optional inclusion/exclusion filters."""

        return self._candidate_rows(
            candidate_row_indices,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
        )

    def has_prepared_sample_name(self, sample_name: str) -> bool:
        """Return whether ``sample_name`` is present in the prepared metadata."""

        return str(sample_name).strip() in self._row_indices_by_sample_name

    def _normalized_batch_exclude_row_ids(
        self,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None,
        *,
        sample_ids: Sequence[str],
    ) -> tuple[str | None, ...]:
        if exclude_row_ids_per_sample is None:
            return (None,) * len(sample_ids)
        if isinstance(exclude_row_ids_per_sample, Mapping):
            sample_id_set = {str(sample_id) for sample_id in sample_ids}
            unknown_sample_ids = sorted(
                str(sample_id) for sample_id in exclude_row_ids_per_sample if str(sample_id) not in sample_id_set
            )
            if unknown_sample_ids:
                raise MappingInputError(
                    "exclude_row_ids_per_sample mapping keys must match sample_ids.",
                    context={"unknown_sample_ids": unknown_sample_ids},
                )
            return tuple(
                (
                    text
                    if (text := str(exclude_row_ids_per_sample.get(sample_id) or "").strip())
                    else None
                )
                for sample_id in sample_ids
            )
        if isinstance(exclude_row_ids_per_sample, Sequence) and not isinstance(exclude_row_ids_per_sample, (str, bytes)):
            values = list(exclude_row_ids_per_sample)
            if len(values) != len(sample_ids):
                raise MappingInputError(
                    "exclude_row_ids_per_sample must have the same length as reflectance_rows.",
                    context={"sample_count": len(sample_ids), "exclude_row_id_count": len(values)},
                )
            return tuple((text if (text := str(value or "").strip()) else None) for value in values)
        raise MappingInputError(
            "exclude_row_ids_per_sample must be a sequence aligned to reflectance_rows or a mapping keyed by sample_id."
        )

    def _batch_candidate_groups(
        self,
        *,
        sample_ids: Sequence[str],
        exclude_row_ids: Sequence[str] | None,
        exclude_sample_names: Sequence[str] | None,
        exclude_row_ids_per_sample: Sequence[str | None],
        self_exclude_sample_id: bool,
    ) -> tuple[_CandidateBatchGroup, ...]:
        base_candidate_rows = self._candidate_rows(
            None,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
        )
        grouped_indices: dict[tuple[int, ...], list[int]] = {}
        grouped_candidates: dict[tuple[int, ...], np.ndarray] = {}

        for sample_index, (sample_id, sample_exclude_row_id) in enumerate(zip(sample_ids, exclude_row_ids_per_sample)):
            candidate_rows = base_candidate_rows
            effective_excluded: list[np.ndarray] = []

            if sample_exclude_row_id:
                if sample_exclude_row_id not in self._row_index_by_id:
                    raise _attach_sample_context(
                        MappingInputError(
                            "exclude_row_ids must refer to existing prepared row ids.",
                            context={"unknown_row_ids": [sample_exclude_row_id]},
                        ),
                        sample_id=sample_id,
                        sample_index=sample_index,
                    )
                excluded_row_index = self._row_index_by_id[sample_exclude_row_id]
                if np.any(candidate_rows == excluded_row_index):
                    effective_excluded.append(np.asarray([excluded_row_index], dtype=np.int64))
                    candidate_rows = candidate_rows[candidate_rows != excluded_row_index]
                if candidate_rows.size == 0:
                    raise _attach_sample_context(
                        MappingInputError(
                            "Candidate exclusions removed every prepared-library row.",
                            context={
                                "row_count": self.manifest.row_count,
                                "excluded_row_id_count": 1,
                                "excluded_sample_name_count": 0,
                            },
                        ),
                        sample_id=sample_id,
                        sample_index=sample_index,
                    )

            if self_exclude_sample_id and self.has_prepared_sample_name(sample_id):
                excluded_sample_rows = np.asarray(self._row_indices_by_sample_name[sample_id], dtype=np.int64)
                if excluded_sample_rows.size:
                    effective_rows = excluded_sample_rows[np.isin(excluded_sample_rows, candidate_rows, assume_unique=False)]
                    if effective_rows.size:
                        effective_excluded.append(effective_rows)
                        candidate_rows = candidate_rows[~np.isin(candidate_rows, effective_rows, assume_unique=False)]
                if candidate_rows.size == 0:
                    raise _attach_sample_context(
                        MappingInputError(
                            "Candidate exclusions removed every prepared-library row.",
                            context={
                                "row_count": self.manifest.row_count,
                                "excluded_row_id_count": 0,
                                "excluded_sample_name_count": 1,
                            },
                        ),
                        sample_id=sample_id,
                        sample_index=sample_index,
                    )

            if effective_excluded:
                group_key = tuple(
                    int(value)
                    for value in np.unique(np.concatenate(effective_excluded))
                )
            else:
                group_key = ()
            grouped_indices.setdefault(group_key, []).append(sample_index)
            grouped_candidates.setdefault(group_key, candidate_rows)

        return tuple(
            _CandidateBatchGroup(
                sample_indices=np.asarray(sample_indices, dtype=np.int64),
                candidate_rows=np.asarray(grouped_candidates[group_key], dtype=np.int64),
            )
            for group_key, sample_indices in grouped_indices.items()
        )

    def _coerced_batch_query_arrays(
        self,
        *,
        source_sensor: str,
        batch_rows: Sequence[Sequence[float] | Mapping[str, float]],
        batch_valid_masks: Sequence[Sequence[bool] | Mapping[str, bool] | None],
        sample_ids: Sequence[str],
    ) -> tuple[SensorSRFSchema, np.ndarray, np.ndarray]:
        source_schema = self.get_sensor_schema(source_sensor)
        query_values_batch = np.empty((len(batch_rows), len(source_schema.bands)), dtype=np.float64)
        query_valid_mask_batch = np.empty((len(batch_rows), len(source_schema.bands)), dtype=bool)
        for sample_index, (sample_id, reflectance, valid_mask) in enumerate(
            zip(sample_ids, batch_rows, batch_valid_masks)
        ):
            try:
                query_values, query_valid_mask = self._coerce_query(source_schema, reflectance, valid_mask)
            except SpectralLibraryError as error:
                raise _attach_sample_context(error, sample_id=sample_id, sample_index=sample_index) from error
            query_values_batch[sample_index] = query_values
            query_valid_mask_batch[sample_index] = query_valid_mask
        return source_schema, query_values_batch, query_valid_mask_batch

    def _coerced_ndarray_batch_query_arrays(
        self,
        *,
        source_sensor: str,
        reflectance_rows: np.ndarray,
        valid_mask_rows: np.ndarray | None,
    ) -> tuple[SensorSRFSchema, np.ndarray, np.ndarray]:
        source_schema = self.get_sensor_schema(source_sensor)
        query_values_batch = np.asarray(reflectance_rows, dtype=np.float64)
        if query_values_batch.ndim != 2 or query_values_batch.shape[0] == 0:
            raise MappingInputError(
                "reflectance_rows arrays must be two-dimensional with at least one sample row.",
                context={"shape": list(query_values_batch.shape)},
            )
        expected_width = len(source_schema.bands)
        if query_values_batch.shape[1] != expected_width:
            raise MappingInputError(
                "Reflectance arrays must align to the source sensor band order.",
                context={"sensor_id": source_sensor, "expected_length": expected_width},
            )
        if valid_mask_rows is None:
            query_valid_mask_batch = np.isfinite(query_values_batch)
        else:
            query_valid_mask_batch = np.asarray(valid_mask_rows, dtype=bool)
            if query_valid_mask_batch.shape != query_values_batch.shape:
                raise MappingInputError(
                    "valid_mask_rows arrays must have the same shape as reflectance_rows.",
                    context={
                        "reflectance_shape": list(query_values_batch.shape),
                        "valid_mask_shape": list(query_valid_mask_batch.shape),
                    },
                )
            query_valid_mask_batch = query_valid_mask_batch & np.isfinite(query_values_batch)
        if not np.all(np.any(query_valid_mask_batch, axis=1)):
            failing_row = int(np.flatnonzero(~np.any(query_valid_mask_batch, axis=1))[0])
            raise MappingInputError(
                "At least one valid source reflectance band is required for mapping.",
                context={"sample_index": failing_row},
            )
        return source_schema, query_values_batch, query_valid_mask_batch

    def _grouped_segment_retrievals(
        self,
        *,
        source_sensor: str,
        sample_ids: Sequence[str],
        query_values_batch: np.ndarray,
        query_valid_mask_batch: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        exclude_row_ids: Sequence[str] | None,
        exclude_sample_names: Sequence[str] | None,
        exclude_row_ids_per_sample: Sequence[str | None],
        self_exclude_sample_id: bool,
        include_confidence: bool = True,
    ) -> dict[str, _RichSegmentBatch]:
        source_schema = self.get_sensor_schema(source_sensor)
        segment_indices = {
            segment: list(_source_retrieval_band_indices(source_schema, segment))
            for segment in SEGMENTS
        }
        candidate_groups = self._batch_candidate_groups(
            sample_ids=sample_ids,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
        )
        dense_outputs_by_segment = {
            segment: _empty_dense_segment_output_batch(
                segment=segment,
                batch_size=int(query_values_batch.shape[0]),
            )
            for segment in SEGMENTS
        }
        metadata_by_segment: dict[str, list[_RichSegmentMetadata | None]] = {
            segment: [None] * int(query_values_batch.shape[0]) for segment in SEGMENTS
        }
        for candidate_group in candidate_groups:
            anchor_index = int(candidate_group.sample_indices[0])
            anchor_sample_id = str(sample_ids[anchor_index])
            try:
                group_batches_by_segment = self._segment_retrievals_by_segment_for_candidate_rows(
                    source_sensor=source_sensor,
                    segment_indices_by_segment=segment_indices,
                    query_values_batch=query_values_batch,
                    query_valid_mask_batch=query_valid_mask_batch,
                    sample_indices=candidate_group.sample_indices,
                    k=k,
                    min_valid_bands=min_valid_bands,
                    neighbor_estimator=neighbor_estimator,
                    knn_backend=knn_backend,
                    knn_eps=knn_eps,
                    candidate_row_indices=candidate_group.candidate_rows,
                    include_confidence=include_confidence,
                )
            except SpectralLibraryError as error:
                raise _attach_sample_context(error, sample_id=anchor_sample_id, sample_index=anchor_index) from error
            for segment in SEGMENTS:
                group_batch = group_batches_by_segment[segment]
                _assign_dense_segment_output_batch_rows(
                    target=dense_outputs_by_segment[segment],
                    sample_indices=candidate_group.sample_indices,
                    source=group_batch.dense_output,
                )
                for local_index, batch_index in enumerate(candidate_group.sample_indices):
                    metadata_by_segment[segment][int(batch_index)] = group_batch.metadata[local_index]

        return {
            segment: _RichSegmentBatch(
                dense_output=dense_outputs_by_segment[segment],
                metadata=tuple(
                    metadata
                    if metadata is not None
                    else _failed_segment_metadata(
                        valid_band_count=0,
                        query_band_ids=_source_retrieval_band_ids(source_schema, segment),
                        query_values=np.empty(0, dtype=np.float64),
                        query_valid_mask=np.empty(0, dtype=bool),
                        reason="unassigned_retrieval",
                    )
                    for metadata in metadata_by_segment[segment]
                )
            )
            for segment in SEGMENTS
        }

    def _build_mapping_result(
        self,
        *,
        source_sensor: str,
        target_sensor: str | None,
        output_mode: str,
        k: int,
        neighbor_estimator: str,
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        segment_retrievals: Mapping[str, _SegmentRetrieval],
        include_diagnostics: bool = True,
    ) -> MappingResult:
        successful_segment_outputs: dict[str, np.ndarray] = {}
        segment_valid_band_counts: dict[str, int] = {}
        neighbor_ids_by_segment: dict[str, tuple[str, ...]] = {}
        neighbor_distances_by_segment: dict[str, np.ndarray] = {}

        for segment, retrieval in segment_retrievals.items():
            segment_valid_band_counts[segment] = retrieval.valid_band_count
            if retrieval.success and retrieval.reconstructed is not None:
                successful_segment_outputs[segment] = retrieval.reconstructed
            if include_diagnostics:
                neighbor_ids_by_segment[segment] = retrieval.neighbor_ids
                neighbor_distances_by_segment[segment] = retrieval.neighbor_distances

        diagnostics: dict[str, object] = {}
        if include_diagnostics:
            diagnostics = {
                "source_sensor": source_sensor,
                "target_sensor": target_sensor,
                "output_mode": output_mode,
                "k": k,
                "neighbor_estimator": neighbor_estimator,
                "knn_backend": knn_backend,
                "knn_eps": float(knn_eps),
                "segments": {},
            }
            diagnostics_segments: dict[str, object] = {}
            confidence_weights: list[int] = []
            confidence_scores: list[float] = []
            for segment, retrieval in segment_retrievals.items():
                if retrieval.success and retrieval.confidence_score is not None:
                    confidence_weights.append(max(retrieval.valid_band_count, 1))
                    confidence_scores.append(float(retrieval.confidence_score))
                diagnostics_segments[segment] = _segment_diagnostics_payload(retrieval)
            diagnostics["segments"] = diagnostics_segments
            if confidence_scores:
                diagnostics["confidence_score"] = float(
                    np.average(np.asarray(confidence_scores), weights=np.asarray(confidence_weights))
                )
            else:
                diagnostics["confidence_score"] = 0.0
            diagnostics["confidence_policy"] = _confidence_policy_payload(float(diagnostics["confidence_score"]))

        reconstructed_vnir = successful_segment_outputs.get("vnir")
        reconstructed_swir = successful_segment_outputs.get("swir")
        reconstructed_full: np.ndarray | None = None
        if reconstructed_vnir is not None and reconstructed_swir is not None:
            reconstructed_full = _assemble_full_spectrum(reconstructed_vnir, reconstructed_swir)

        if output_mode == "target_sensor" and not target_sensor:
            raise MappingInputError("target_sensor is required when output_mode is target_sensor.")
        if output_mode == "vnir_spectrum" and reconstructed_vnir is None:
            raise MappingInputError(
                "VNIR reconstruction requires enough valid VNIR source bands.",
                context={"diagnostics": diagnostics},
            )
        if output_mode == "swir_spectrum" and reconstructed_swir is None:
            raise MappingInputError(
                "SWIR reconstruction requires enough valid SWIR source bands.",
                context={"diagnostics": diagnostics},
            )
        if output_mode == "full_spectrum" and reconstructed_full is None:
            raise MappingInputError(
                "Full-spectrum reconstruction requires both VNIR and SWIR segment retrievals.",
                context={"diagnostics": diagnostics},
            )
        

        target_reflectance: np.ndarray | None = None
        target_band_ids: tuple[str, ...] = ()
        if target_sensor:
            target_reflectance, target_band_ids = self._simulate_target_sensor(target_sensor, successful_segment_outputs)
            if output_mode == "target_sensor" and target_reflectance is None:
                raise MappingInputError(
                    "Target-sensor mapping could not produce any bands because no target segments were retrievable.",
                    context={"diagnostics": diagnostics, "target_sensor": target_sensor},
                )

        reconstructed_wavelength_nm: np.ndarray | None = None
        if output_mode == "vnir_spectrum":
            reconstructed_wavelength_nm = VNIR_WAVELENGTHS.astype(np.float64)
        elif output_mode == "swir_spectrum":
            reconstructed_wavelength_nm = SWIR_WAVELENGTHS.astype(np.float64)
        elif output_mode == "full_spectrum":
            reconstructed_wavelength_nm = CANONICAL_WAVELENGTHS.astype(np.float64)

        return MappingResult(
            target_reflectance=target_reflectance,
            target_band_ids=target_band_ids,
            reconstructed_vnir=reconstructed_vnir,
            reconstructed_swir=reconstructed_swir,
            reconstructed_full_spectrum=reconstructed_full,
            reconstructed_wavelength_nm=reconstructed_wavelength_nm,
            neighbor_ids_by_segment=neighbor_ids_by_segment,
            neighbor_distances_by_segment=neighbor_distances_by_segment,
            segment_outputs=successful_segment_outputs if include_diagnostics else {},
            segment_valid_band_counts=segment_valid_band_counts,
            diagnostics=diagnostics,
        )

    def _segment_retrievals_by_segment_for_candidate_rows(
        self,
        *,
        source_sensor: str,
        segment_indices_by_segment: Mapping[str, Sequence[int]],
        query_values_batch: np.ndarray,
        query_valid_mask_batch: np.ndarray,
        sample_indices: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        candidate_row_indices: np.ndarray,
        include_confidence: bool = True,
    ) -> dict[str, _RichSegmentBatch]:
        query_values_group = np.asarray(query_values_batch[sample_indices], dtype=np.float64)
        query_valid_mask_group = np.asarray(query_valid_mask_batch[sample_indices], dtype=bool)
        segment_retrievals_by_segment: dict[str, _RichSegmentBatch] = {}
        for segment, segment_indices in segment_indices_by_segment.items():
            segment_retrievals_by_segment[segment] = self._retrieve_segment_batch(
                source_sensor=source_sensor,
                segment=segment,
                query_values=query_values_group[:, segment_indices],
                valid_mask=query_valid_mask_group[:, segment_indices],
                k=k,
                min_valid_bands=min_valid_bands,
                neighbor_estimator=neighbor_estimator,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
                candidate_row_indices=candidate_row_indices,
                include_confidence=include_confidence,
            )
        return segment_retrievals_by_segment

    def _dense_segment_outputs_by_segment_for_candidate_rows(
        self,
        *,
        source_sensor: str,
        segment_indices_by_segment: Mapping[str, Sequence[int]],
        query_values_batch: np.ndarray,
        query_valid_mask_batch: np.ndarray,
        sample_indices: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        candidate_row_indices: np.ndarray,
        hyperspectral_rows_by_segment: Mapping[str, np.ndarray] | None = None,
        target_output_indices_by_segment: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, _DenseSegmentOutputBatch]:
        query_values_group = np.asarray(query_values_batch[sample_indices], dtype=np.float64)
        query_valid_mask_group = np.asarray(query_valid_mask_batch[sample_indices], dtype=bool)
        dense_outputs_by_segment: dict[str, _DenseSegmentOutputBatch] = {}
        for segment, segment_indices in segment_indices_by_segment.items():
            dense_outputs_by_segment[segment] = self._retrieve_segment_dense_batch(
                source_sensor=source_sensor,
                segment=segment,
                query_values=query_values_group[:, segment_indices],
                valid_mask=query_valid_mask_group[:, segment_indices],
                k=k,
                min_valid_bands=min_valid_bands,
                neighbor_estimator=neighbor_estimator,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
                candidate_row_indices=candidate_row_indices,
                hyperspectral_rows=None if hyperspectral_rows_by_segment is None else hyperspectral_rows_by_segment.get(segment),
                target_output_indices=(
                    None if target_output_indices_by_segment is None else target_output_indices_by_segment.get(segment)
                ),
            )
        return dense_outputs_by_segment

    def _grouped_dense_segment_outputs(
        self,
        *,
        source_sensor: str,
        sample_ids: Sequence[str],
        query_values_batch: np.ndarray,
        query_valid_mask_batch: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        exclude_row_ids: Sequence[str] | None,
        exclude_sample_names: Sequence[str] | None,
        exclude_row_ids_per_sample: Sequence[str | None],
        self_exclude_sample_id: bool,
        hyperspectral_rows_by_segment: Mapping[str, np.ndarray] | None = None,
        target_output_indices_by_segment: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, _DenseSegmentOutputBatch]:
        source_schema = self.get_sensor_schema(source_sensor)
        segment_indices = {
            segment: list(_source_retrieval_band_indices(source_schema, segment))
            for segment in SEGMENTS
        }
        candidate_groups = self._batch_candidate_groups(
            sample_ids=sample_ids,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
        )
        dense_outputs_by_segment = {
            segment: _empty_dense_segment_output_batch(
                segment=segment,
                batch_size=int(query_values_batch.shape[0]),
                output_width=(
                    None if hyperspectral_rows_by_segment is None else int(hyperspectral_rows_by_segment[segment].shape[1])
                ),
                target_output_indices=(
                    None if target_output_indices_by_segment is None else target_output_indices_by_segment.get(segment)
                ),
            )
            for segment in SEGMENTS
        }
        for candidate_group in candidate_groups:
            anchor_index = int(candidate_group.sample_indices[0])
            anchor_sample_id = str(sample_ids[anchor_index])
            try:
                group_dense_outputs_by_segment = self._dense_segment_outputs_by_segment_for_candidate_rows(
                    source_sensor=source_sensor,
                    segment_indices_by_segment=segment_indices,
                    query_values_batch=query_values_batch,
                    query_valid_mask_batch=query_valid_mask_batch,
                    sample_indices=candidate_group.sample_indices,
                    k=k,
                    min_valid_bands=min_valid_bands,
                    neighbor_estimator=neighbor_estimator,
                    knn_backend=knn_backend,
                    knn_eps=knn_eps,
                    candidate_row_indices=candidate_group.candidate_rows,
                    hyperspectral_rows_by_segment=hyperspectral_rows_by_segment,
                    target_output_indices_by_segment=target_output_indices_by_segment,
                )
            except SpectralLibraryError as error:
                raise _attach_sample_context(error, sample_id=anchor_sample_id, sample_index=anchor_index) from error
            for segment in SEGMENTS:
                _assign_dense_segment_output_batch_rows(
                    target=dense_outputs_by_segment[segment],
                    sample_indices=candidate_group.sample_indices,
                    source=group_dense_outputs_by_segment[segment],
                )

        return dense_outputs_by_segment

    def _retrieve_segment_dense_batch(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_values: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        candidate_row_indices: np.ndarray,
        hyperspectral_rows: np.ndarray | None = None,
        target_output_indices: np.ndarray | None = None,
    ) -> _DenseSegmentOutputBatch:
        source_schema = self.get_sensor_schema(source_sensor)
        query_band_ids = _source_retrieval_band_ids(source_schema, segment)
        source_matrix = self._validated_source_matrix(source_sensor, segment, query_band_ids)
        segment_hyperspectral = (
            self._load_hyperspectral(segment)
            if hyperspectral_rows is None
            else np.ascontiguousarray(hyperspectral_rows, dtype=np.float64)
        )
        output_width = int(segment_hyperspectral.shape[1])
        batch_size = int(query_values.shape[0])
        success = np.zeros(batch_size, dtype=bool)
        reconstructed = np.zeros((batch_size, output_width), dtype=np.float64)
        source_fit_rmse = np.zeros(batch_size, dtype=np.float64)
        valid_band_count = np.zeros(batch_size, dtype=np.int32)
        reconstructed_scratch = np.empty((batch_size, output_width), dtype=np.float64)
        source_fit_rmse_scratch = np.empty(batch_size, dtype=np.float64)
        candidate_matrix = self._candidate_source_matrix_view(source_matrix, candidate_row_indices)
        valid_patterns, inverse = np.unique(valid_mask, axis=0, return_inverse=True)

        for pattern_index, pattern in enumerate(valid_patterns):
            batch_indices = np.flatnonzero(inverse == pattern_index)
            pattern_valid_band_count = int(pattern.sum())
            valid_band_count[batch_indices] = pattern_valid_band_count
            if pattern_valid_band_count < min_valid_bands:
                continue

            if np.all(pattern):
                valid_indices = slice(None)
                candidate_valid = candidate_matrix
                query_group = query_values[batch_indices]
            else:
                valid_indices = np.flatnonzero(pattern)
                candidate_valid = candidate_matrix[:, valid_indices]
                query_group = query_values[batch_indices][:, valid_indices]
            query_group = np.asarray(query_group, dtype=np.float64)
            resolved_valid_indices = None if isinstance(valid_indices, slice) else np.asarray(valid_indices, dtype=np.int64)
            group_size = int(batch_indices.size)
            reconstructed_out = reconstructed_scratch[:group_size]
            source_fit_rmse_out = source_fit_rmse_scratch[:group_size]

            if knn_backend == "scipy_ckdtree" and float(knn_eps) == 0.0:
                if self._uses_all_candidate_rows(candidate_row_indices) and np.all(pattern):
                    group_local_indices, group_local_distances = self._query_cached_scipy_ckdtree_local_results(
                        source_sensor=source_sensor,
                        segment=segment,
                        query_band_ids=query_band_ids,
                        query_values=np.asarray(query_values[batch_indices], dtype=np.float64),
                        k=min(int(k), int(candidate_row_indices.size)),
                        knn_eps=knn_eps,
                    )
                else:
                    group_local_indices, group_local_distances = _query_local_scipy_ckdtree_results(
                        candidate_valid,
                        query_group,
                        k=min(int(k), int(candidate_row_indices.size)),
                        knn_eps=knn_eps,
                    )
                reconstructed_batch, source_fit_rmse_batch = _refine_and_combine_neighbor_spectra_batch_accel(
                    candidate_matrix=candidate_valid,
                    candidate_row_indices=candidate_row_indices,
                    source_matrix=source_matrix,
                    hyperspectral_rows=segment_hyperspectral,
                    query_values=query_group,
                    local_candidate_indices=np.asarray(group_local_indices, dtype=np.int64),
                    local_candidate_distances=np.asarray(group_local_distances, dtype=np.float64),
                    valid_indices=resolved_valid_indices,
                    k=k,
                    neighbor_estimator=neighbor_estimator,
                    out_reconstructed=reconstructed_out,
                    out_source_fit_rmse=source_fit_rmse_out,
                )
            else:
                if knn_backend == "scipy_ckdtree" and self._uses_all_candidate_rows(candidate_row_indices) and np.all(pattern):
                    group_local_indices = self._query_cached_scipy_ckdtree_local_indices(
                        source_sensor=source_sensor,
                        segment=segment,
                        query_band_ids=query_band_ids,
                        query_values=np.asarray(query_values[batch_indices], dtype=np.float64),
                        k=min(int(k), int(candidate_row_indices.size)),
                        knn_eps=knn_eps,
                    )
                elif self._can_use_persisted_knn_index(
                    backend=knn_backend,
                    source_sensor=source_sensor,
                    segment=segment,
                    valid_mask=np.asarray(pattern, dtype=bool),
                    candidate_row_indices=candidate_row_indices,
                ):
                    group_local_indices = self._query_persisted_knn_local_indices(
                        backend=knn_backend,
                        source_sensor=source_sensor,
                        segment=segment,
                        query_values=np.asarray(query_values[batch_indices], dtype=np.float64),
                        k=min(int(k), int(candidate_row_indices.size)),
                        knn_eps=knn_eps,
                    )
                else:
                    group_local_indices = _search_local_neighbor_indices(
                        candidate_valid,
                        query_group,
                        k=k,
                        knn_backend=knn_backend,
                        knn_eps=knn_eps,
                    )
                reconstructed_batch, source_fit_rmse_batch = _refine_and_combine_neighbor_spectra_batch_accel(
                    candidate_matrix=candidate_valid,
                    candidate_row_indices=candidate_row_indices,
                    source_matrix=source_matrix,
                    hyperspectral_rows=segment_hyperspectral,
                    query_values=query_group,
                    local_candidate_indices=None
                    if group_local_indices is None
                    else np.asarray(group_local_indices, dtype=np.int64),
                    local_candidate_distances=None,
                    valid_indices=resolved_valid_indices,
                    k=k,
                    neighbor_estimator=neighbor_estimator,
                    out_reconstructed=reconstructed_out,
                    out_source_fit_rmse=source_fit_rmse_out,
                )
            success[batch_indices] = True
            reconstructed[batch_indices] = np.asarray(reconstructed_batch, dtype=np.float64)
            source_fit_rmse[batch_indices] = np.asarray(source_fit_rmse_batch, dtype=np.float64)

        return _DenseSegmentOutputBatch(
            success=success,
            reconstructed=reconstructed,
            confidence_score=np.zeros(batch_size, dtype=np.float64),
            source_fit_rmse=source_fit_rmse,
            valid_band_count=valid_band_count,
            target_output_indices=(
                None if target_output_indices is None else np.asarray(target_output_indices, dtype=np.int64)
            ),
        )

    def _refine_neighbor_rows_batch(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_band_ids: tuple[str, ...],
        query_values: np.ndarray,
        query_group: np.ndarray,
        valid_pattern: np.ndarray,
        candidate_row_indices: np.ndarray,
        candidate_valid: np.ndarray,
        k: int,
        knn_backend: str,
        knn_eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        group_local_distances: np.ndarray | None = None
        if knn_backend == "scipy_ckdtree" and float(knn_eps) == 0.0:
            if self._uses_all_candidate_rows(candidate_row_indices) and np.all(valid_pattern):
                group_local_indices, group_local_distances = self._query_cached_scipy_ckdtree_local_results(
                    source_sensor=source_sensor,
                    segment=segment,
                    query_band_ids=query_band_ids,
                    query_values=query_values,
                    k=min(int(k), int(candidate_row_indices.size)),
                    knn_eps=knn_eps,
                )
            else:
                group_local_indices, group_local_distances = _query_local_scipy_ckdtree_results(
                    candidate_valid,
                    query_group,
                    k=min(int(k), int(candidate_row_indices.size)),
                    knn_eps=knn_eps,
                )
        elif self._can_use_persisted_knn_index(
            backend=knn_backend,
            source_sensor=source_sensor,
            segment=segment,
            valid_mask=np.asarray(valid_pattern, dtype=bool),
            candidate_row_indices=candidate_row_indices,
        ):
            group_local_indices = self._query_persisted_knn_local_indices(
                backend=knn_backend,
                source_sensor=source_sensor,
                segment=segment,
                query_values=query_values,
                k=min(int(k), int(candidate_row_indices.size)),
                knn_eps=knn_eps,
            )
        else:
            group_local_indices = _search_local_neighbor_indices(
                candidate_valid,
                query_group,
                k=k,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
            )

        return _refine_neighbor_rows_batch_accel(
            candidate_matrix=candidate_valid,
            query_values=query_group,
            candidate_row_indices=candidate_row_indices,
            local_candidate_indices=None
            if group_local_indices is None
            else np.asarray(group_local_indices, dtype=np.int64),
            local_candidate_distances=None
            if group_local_distances is None
            else np.asarray(group_local_distances, dtype=np.float64),
            k=k,
        )

    def _batched_result_materialization(
        self,
        *,
        target_sensor: str | None,
        segment_batches_by_segment: Mapping[str, _RichSegmentBatch],
    ) -> _BatchedResultMaterialization:
        vnir_batch = segment_batches_by_segment["vnir"].dense_output
        swir_batch = segment_batches_by_segment["swir"].dense_output
        confidence_scores = self._segment_confidence_scores_from_dense_batches(
            vnir_batch=vnir_batch,
            swir_batch=swir_batch,
        )

        reconstructed_full_batch: np.ndarray | None = None
        full_success = np.asarray(vnir_batch.success & swir_batch.success, dtype=bool)
        if np.any(full_success):
            reconstructed_full_batch = np.empty((vnir_batch.reconstructed.shape[0], FULL_WAVELENGTH_COUNT), dtype=np.float64)
            reconstructed_full_batch[full_success] = _assemble_full_spectrum_batch(
                vnir_batch.reconstructed[full_success],
                swir_batch.reconstructed[full_success],
            )

        target_rows: np.ndarray | None = None
        target_status_codes: np.ndarray | None = None
        target_band_ids: tuple[str, ...] = ()
        if target_sensor:
            projection = self._target_sensor_projection(target_sensor)
            target_rows, target_status_codes = _finalize_target_sensor_batch_accel(
                vnir_reconstructed=vnir_batch.reconstructed[:, projection.vnir_support_indices],
                swir_reconstructed=swir_batch.reconstructed[:, projection.swir_support_indices],
                vnir_success=vnir_batch.success,
                swir_success=swir_batch.success,
                vnir_response_matrix=projection.vnir_response_matrix,
                swir_response_matrix=projection.swir_response_matrix,
                vnir_output_indices=projection.vnir_output_indices,
                swir_output_indices=projection.swir_output_indices,
                output_width=projection.output_width,
            )
            target_band_ids = self.get_sensor_schema(target_sensor).band_ids()

        return _BatchedResultMaterialization(
            vnir_batch=vnir_batch,
            swir_batch=swir_batch,
            confidence_scores=np.asarray(confidence_scores, dtype=np.float64),
            reconstructed_full_batch=reconstructed_full_batch,
            target_rows=None if target_rows is None else np.asarray(target_rows, dtype=np.float64),
            target_status_codes=None if target_status_codes is None else np.asarray(target_status_codes, dtype=np.int32),
            target_band_ids=target_band_ids,
        )

    def _mapping_results_from_segment_retrievals_batch(
        self,
        *,
        sample_ids: Sequence[str],
        source_sensor: str,
        target_sensor: str | None,
        output_mode: str,
        k: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        segment_batches_by_segment: Mapping[str, _RichSegmentBatch],
    ) -> tuple[MappingResult, ...]:
        materialized = self._batched_result_materialization(
            target_sensor=target_sensor,
            segment_batches_by_segment=segment_batches_by_segment,
        )
        reconstructed_wavelength_nm: np.ndarray | None = None
        if output_mode == "vnir_spectrum":
            reconstructed_wavelength_nm = VNIR_WAVELENGTHS.astype(np.float64)
        elif output_mode == "swir_spectrum":
            reconstructed_wavelength_nm = SWIR_WAVELENGTHS.astype(np.float64)
        elif output_mode == "full_spectrum":
            reconstructed_wavelength_nm = CANONICAL_WAVELENGTHS.astype(np.float64)

        results: list[MappingResult] = []
        for sample_index, sample_id in enumerate(sample_ids):
            try:
                sample_metadata = {
                    segment: segment_batches_by_segment[segment].metadata[sample_index] for segment in SEGMENTS
                }
                segment_outputs: dict[str, np.ndarray] = {}
                segment_valid_band_counts: dict[str, int] = {}
                neighbor_ids_by_segment: dict[str, tuple[str, ...]] = {}
                neighbor_distances_by_segment: dict[str, np.ndarray] = {}
                diagnostics_segments: dict[str, object] = {}
                for segment, metadata in sample_metadata.items():
                    dense_output = segment_batches_by_segment[segment].dense_output
                    segment_valid_band_counts[segment] = metadata.valid_band_count
                    neighbor_ids_by_segment[segment] = metadata.neighbor_ids
                    neighbor_distances_by_segment[segment] = metadata.neighbor_distances
                    diagnostics_segments[segment] = metadata.diagnostics
                    if metadata.success:
                        segment_outputs[segment] = dense_output.reconstructed[sample_index]

                diagnostics: dict[str, object] = {
                    "source_sensor": source_sensor,
                    "target_sensor": target_sensor,
                    "output_mode": output_mode,
                    "k": k,
                    "neighbor_estimator": neighbor_estimator,
                    "knn_backend": knn_backend,
                    "knn_eps": float(knn_eps),
                    "segments": diagnostics_segments,
                    "confidence_score": float(materialized.confidence_scores[sample_index]),
                }
                diagnostics["confidence_policy"] = _confidence_policy_payload(float(diagnostics["confidence_score"]))

                reconstructed_vnir = (
                    materialized.vnir_batch.reconstructed[sample_index]
                    if materialized.vnir_batch.success[sample_index]
                    else None
                )
                reconstructed_swir = (
                    materialized.swir_batch.reconstructed[sample_index]
                    if materialized.swir_batch.success[sample_index]
                    else None
                )
                reconstructed_full = (
                    None
                    if materialized.reconstructed_full_batch is None
                    or not (materialized.vnir_batch.success[sample_index] and materialized.swir_batch.success[sample_index])
                    else materialized.reconstructed_full_batch[sample_index]
                )

                if output_mode == "target_sensor" and not target_sensor:
                    raise MappingInputError("target_sensor is required when output_mode is target_sensor.")
                if output_mode == "vnir_spectrum" and reconstructed_vnir is None:
                    raise MappingInputError(
                        "VNIR reconstruction requires enough valid VNIR source bands.",
                        context={"diagnostics": diagnostics},
                    )
                if output_mode == "swir_spectrum" and reconstructed_swir is None:
                    raise MappingInputError(
                        "SWIR reconstruction requires enough valid SWIR source bands.",
                        context={"diagnostics": diagnostics},
                    )
                if output_mode == "full_spectrum" and reconstructed_full is None:
                    raise MappingInputError(
                        "Full-spectrum reconstruction requires both VNIR and SWIR segment retrievals.",
                        context={"diagnostics": diagnostics},
                    )

                target_reflectance: np.ndarray | None = None
                target_band_ids: tuple[str, ...] = ()
                if target_sensor:
                    assert materialized.target_rows is not None
                    assert materialized.target_status_codes is not None
                    target_row = np.asarray(materialized.target_rows[sample_index], dtype=np.float64)
                    available_mask = np.isfinite(target_row)
                    if np.any(available_mask):
                        target_reflectance = np.asarray(target_row[available_mask], dtype=np.float64)
                        target_band_ids = tuple(
                            band_id
                            for band_id, available in zip(materialized.target_band_ids, available_mask)
                            if bool(available)
                        )
                    if output_mode == "target_sensor" and int(materialized.target_status_codes[sample_index]) != 0:
                        raise MappingInputError(
                            "Target-sensor mapping could not produce any bands because no target segments were retrievable.",
                            context={"diagnostics": diagnostics, "target_sensor": target_sensor},
                        )

                results.append(
                    MappingResult(
                        target_reflectance=target_reflectance,
                        target_band_ids=target_band_ids,
                        reconstructed_vnir=reconstructed_vnir,
                        reconstructed_swir=reconstructed_swir,
                        reconstructed_full_spectrum=reconstructed_full,
                        reconstructed_wavelength_nm=reconstructed_wavelength_nm,
                        neighbor_ids_by_segment=neighbor_ids_by_segment,
                        neighbor_distances_by_segment=neighbor_distances_by_segment,
                        segment_outputs=segment_outputs,
                        segment_valid_band_counts=segment_valid_band_counts,
                        diagnostics=diagnostics,
                    )
                )
            except SpectralLibraryError as error:
                raise _attach_sample_context(error, sample_id=sample_id, sample_index=sample_index) from error
        return tuple(results)

    def _map_reflectance_internal(
        self,
        *,
        source_sensor: str,
        reflectance: Sequence[float] | Mapping[str, float],
        valid_mask: Sequence[bool] | Mapping[str, bool] | None,
        output_mode: str,
        target_sensor: str | None,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        candidate_row_indices: Sequence[int] | None,
        include_diagnostics: bool = True,
    ) -> MappingResult:
        _validate_mapping_request(
            output_mode,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
        )

        source_schema = self.get_sensor_schema(source_sensor)
        query_values, query_valid_mask = self._coerce_query(source_schema, reflectance, valid_mask)
        candidate_rows = self._candidate_rows(candidate_row_indices)

        segment_retrievals: dict[str, _SegmentRetrieval] = {}
        for segment in SEGMENTS:
            segment_indices = list(_source_retrieval_band_indices(source_schema, segment))
            segment_values = query_values[segment_indices]
            segment_valid = query_valid_mask[segment_indices]
            retrieval = self._retrieve_segment(
                source_sensor=source_sensor,
                segment=segment,
                query_values=segment_values,
                valid_mask=segment_valid,
                k=k,
                min_valid_bands=min_valid_bands,
                neighbor_estimator=neighbor_estimator,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
                candidate_row_indices=candidate_rows,
                include_diagnostics=include_diagnostics,
            )
            segment_retrievals[segment] = retrieval
        return self._build_mapping_result(
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            output_mode=output_mode,
            k=k,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            segment_retrievals=segment_retrievals,
            include_diagnostics=include_diagnostics,
        )

    def map_reflectance(
        self,
        *,
        source_sensor: str,
        reflectance: Sequence[float] | Mapping[str, float],
        valid_mask: Sequence[bool] | Mapping[str, bool] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
    ) -> MappingResult:
        """Map one source-sensor sample into target or hyperspectral space."""

        return self._map_reflectance_internal(
            source_sensor=source_sensor,
            reflectance=reflectance,
            valid_mask=valid_mask,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            candidate_row_indices=self._candidate_rows(
                None,
                exclude_row_ids=exclude_row_ids,
                exclude_sample_names=exclude_sample_names,
            ),
            include_diagnostics=False,
        )

    def map_reflectance_debug(
        self,
        *,
        source_sensor: str,
        reflectance: Sequence[float] | Mapping[str, float],
        valid_mask: Sequence[bool] | Mapping[str, bool] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
    ) -> MappingResult:
        """Map one sample and include neighbor/diagnostic payloads for inspection."""

        return self._map_reflectance_internal(
            source_sensor=source_sensor,
            reflectance=reflectance,
            valid_mask=valid_mask,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            candidate_row_indices=self._candidate_rows(
                None,
                exclude_row_ids=exclude_row_ids,
                exclude_sample_names=exclude_sample_names,
            ),
            include_diagnostics=True,
        )

    def map_reflectance_batch(
        self,
        *,
        source_sensor: str,
        reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
        valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
    ) -> BatchMappingArrayResult:
        """Map a batch of samples into dense arrays for the default fast path."""

        return self.map_reflectance_batch_arrays(
            source_sensor=source_sensor,
            reflectance_rows=reflectance_rows,
            valid_mask_rows=valid_mask_rows,
            sample_ids=sample_ids,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
        )

    def map_reflectance_batch_debug(
        self,
        *,
        source_sensor: str,
        reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
        valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
    ) -> BatchMappingResult:
        """Map a batch of samples and return rich per-sample debug payloads."""

        _validate_mapping_request(
            output_mode,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
        )
        batch_rows = _normalized_batch_rows(reflectance_rows)
        normalized_sample_ids = _normalized_sample_ids(sample_ids, sample_count=len(batch_rows))
        batch_valid_masks = _normalized_batch_valid_masks(valid_mask_rows, sample_count=len(batch_rows))
        normalized_exclude_row_ids_per_sample = self._normalized_batch_exclude_row_ids(
            exclude_row_ids_per_sample,
            sample_ids=normalized_sample_ids,
        )
        _, query_values_batch, query_valid_mask_batch = self._coerced_batch_query_arrays(
            source_sensor=source_sensor,
            batch_rows=batch_rows,
            batch_valid_masks=batch_valid_masks,
            sample_ids=normalized_sample_ids,
        )
        segment_batches_by_segment = self._grouped_segment_retrievals(
            source_sensor=source_sensor,
            sample_ids=normalized_sample_ids,
            query_values_batch=query_values_batch,
            query_valid_mask_batch=query_valid_mask_batch,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=normalized_exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            include_confidence=True,
        )
        return BatchMappingResult(
            sample_ids=normalized_sample_ids,
            results=self._mapping_results_from_segment_retrievals_batch(
                sample_ids=normalized_sample_ids,
                source_sensor=source_sensor,
                target_sensor=target_sensor,
                output_mode=output_mode,
                k=k,
                neighbor_estimator=neighbor_estimator,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
                segment_batches_by_segment=segment_batches_by_segment,
            ),
        )

    def map_reflectance_batch_arrays(
        self,
        *,
        source_sensor: str,
        reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
        valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> BatchMappingArrayResult:
        """Map a batch of samples into dense arrays without per-sample result objects."""

        _, output_axis_name, output_axis_values = self._batch_output_layout(
            output_mode=output_mode,
            target_sensor=target_sensor,
        )
        normalized_sample_ids, output_rows, source_fit_rmse, output_columns = self._map_reflectance_batch_output_arrays(
            source_sensor=source_sensor,
            reflectance_rows=reflectance_rows,
            valid_mask_rows=valid_mask_rows,
            sample_ids=sample_ids,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            out=out,
            source_fit_rmse_out=source_fit_rmse_out,
        )
        resolved_out = self._validated_output_buffer(
            buffer=out,
            expected_shape=tuple(output_rows.shape),
            name="out",
        )
        if resolved_out is not None and output_rows is not resolved_out:
            np.copyto(resolved_out, output_rows)
            output_rows = resolved_out
        resolved_source_fit_out = self._validated_output_buffer(
            buffer=source_fit_rmse_out,
            expected_shape=tuple(source_fit_rmse.shape),
            name="source_fit_rmse_out",
        )
        if resolved_source_fit_out is not None and source_fit_rmse is not resolved_source_fit_out:
            np.copyto(resolved_source_fit_out, source_fit_rmse)
            source_fit_rmse = resolved_source_fit_out
        wavelength_nm = None
        if output_axis_name == "wavelength_nm":
            wavelength_nm = np.array(output_axis_values, dtype=np.int32)
        return BatchMappingArrayResult(
            sample_ids=normalized_sample_ids,
            reflectance=np.asarray(output_rows, dtype=np.float64),
            source_fit_rmse=np.asarray(source_fit_rmse, dtype=np.float64),
            output_columns=output_columns,
            wavelength_nm=wavelength_nm,
        )

    def map_reflectance_batch_ndarray(
        self,
        *,
        source_sensor: str,
        reflectance_rows: np.ndarray,
        valid_mask_rows: np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> BatchMappingArrayResult:
        """Map a strict ndarray batch in source-band order, optionally reusing output buffers."""

        return self.map_reflectance_batch_arrays_ndarray(
            source_sensor=source_sensor,
            reflectance_rows=reflectance_rows,
            valid_mask_rows=valid_mask_rows,
            sample_ids=sample_ids,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            out=out,
            source_fit_rmse_out=source_fit_rmse_out,
        )

    def map_reflectance_batch_arrays_ndarray(
        self,
        *,
        source_sensor: str,
        reflectance_rows: np.ndarray,
        valid_mask_rows: np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> BatchMappingArrayResult:
        """Map a strict ndarray batch in source-band order, optionally reusing output buffers."""

        _validate_mapping_request(
            output_mode,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
        )
        _, query_values_batch, query_valid_mask_batch = self._coerced_ndarray_batch_query_arrays(
            source_sensor=source_sensor,
            reflectance_rows=reflectance_rows,
            valid_mask_rows=valid_mask_rows,
        )
        normalized_sample_ids = _normalized_sample_ids(sample_ids, sample_count=int(query_values_batch.shape[0]))
        normalized_exclude_row_ids_per_sample = self._normalized_batch_exclude_row_ids(
            exclude_row_ids_per_sample,
            sample_ids=normalized_sample_ids,
        )
        _, output_axis_name, output_axis_values = self._batch_output_layout(
            output_mode=output_mode,
            target_sensor=target_sensor,
        )
        output_rows, source_fit_rmse, output_columns = self._map_reflectance_batch_output_arrays_from_coerced(
            source_sensor=source_sensor,
            normalized_sample_ids=normalized_sample_ids,
            query_values_batch=query_values_batch,
            query_valid_mask_batch=query_valid_mask_batch,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=normalized_exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            out=out,
            source_fit_rmse_out=source_fit_rmse_out,
        )
        resolved_out = self._validated_output_buffer(
            buffer=out,
            expected_shape=tuple(output_rows.shape),
            name="out",
        )
        if resolved_out is not None and output_rows is not resolved_out:
            np.copyto(resolved_out, output_rows)
            output_rows = resolved_out
        resolved_source_fit_out = self._validated_output_buffer(
            buffer=source_fit_rmse_out,
            expected_shape=tuple(source_fit_rmse.shape),
            name="source_fit_rmse_out",
        )
        if resolved_source_fit_out is not None and source_fit_rmse is not resolved_source_fit_out:
            np.copyto(resolved_source_fit_out, source_fit_rmse)
            source_fit_rmse = resolved_source_fit_out
        wavelength_nm = None
        if output_axis_name == "wavelength_nm":
            wavelength_nm = np.array(output_axis_values, dtype=np.int32)
        return BatchMappingArrayResult(
            sample_ids=normalized_sample_ids,
            reflectance=output_rows,
            source_fit_rmse=source_fit_rmse,
            output_columns=output_columns,
            wavelength_nm=wavelength_nm,
        )

    def _batch_output_layout(
        self,
        *,
        output_mode: str,
        target_sensor: str | None,
    ) -> tuple[tuple[str, ...], str, np.ndarray]:
        if output_mode == "target_sensor":
            if not target_sensor:
                raise MappingInputError("target_sensor is required when output_mode is target_sensor.")
            band_ids = self.get_sensor_schema(target_sensor).band_ids()
            axis_values = np.asarray(band_ids, dtype=object)
            return tuple(band_ids), "band_id", axis_values
        if output_mode == "vnir_spectrum":
            return (
                tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in VNIR_WAVELENGTHS),
                "wavelength_nm",
                np.asarray(VNIR_WAVELENGTHS, dtype=np.int32),
            )
        if output_mode == "swir_spectrum":
            return (
                tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in SWIR_WAVELENGTHS),
                "wavelength_nm",
                np.asarray(SWIR_WAVELENGTHS, dtype=np.int32),
            )
        if output_mode == "full_spectrum":
            return (
                tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in CANONICAL_WAVELENGTHS),
                "wavelength_nm",
                np.asarray(CANONICAL_WAVELENGTHS, dtype=np.int32),
            )
        _ensure_supported_output_mode(output_mode)
        raise MappingInputError(
            "Unsupported output_mode.",
            context={"output_mode": output_mode, "supported_output_modes": list(SUPPORTED_OUTPUT_MODES)},
        )

    def _resolved_batch_output_chunk_size(
        self,
        *,
        source_sensor: str,
        output_width: int,
        chunk_size: int | None,
    ) -> int:
        return _normalized_linear_mapper_chunk_size(
            chunk_size,
            input_width=len(self.get_sensor_schema(source_sensor).bands),
            output_width=output_width,
            dtype=np.dtype(np.float64),
        )

    def _open_batch_zarr_export(
        self,
        *,
        zarr_path: Path,
        source_sensor: str,
        output_mode: str,
        target_sensor: str | None,
        chunk_size: int,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
    ) -> _ZarrBatchExport:
        output_columns, axis_name, axis_values = self._batch_output_layout(
            output_mode=output_mode,
            target_sensor=target_sensor,
        )
        output_width = int(axis_values.shape[0])
        zarr = _load_zarr_module()
        compressor = _default_zarr_compressor()
        utf8_codec = _zarr_utf8_codec()
        output_store = Path(zarr_path)
        output_store.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(output_store), mode="w")
        reflectance_dataset = root.create_dataset(
            "reflectance",
            shape=(0, output_width),
            chunks=(int(chunk_size), output_width),
            dtype="f8",
            overwrite=True,
            compressor=compressor,
        )
        source_fit_dataset = root.create_dataset(
            "source_fit_rmse",
            shape=(0,),
            chunks=(int(chunk_size),),
            dtype="f8",
            overwrite=True,
            compressor=compressor,
        )
        sample_id_dataset = root.create_dataset(
            "sample_id",
            shape=(0,),
            chunks=(int(chunk_size),),
            dtype=object,
            object_codec=utf8_codec,
            overwrite=True,
            compressor=compressor,
        )
        if axis_name == "band_id":
            root.create_dataset(
                axis_name,
                data=np.asarray(axis_values, dtype=object),
                dtype=object,
                object_codec=utf8_codec,
                overwrite=True,
                compressor=compressor,
            )
        else:
            root.create_dataset(
                axis_name,
                data=np.asarray(axis_values),
                overwrite=True,
                compressor=compressor,
            )
        root.attrs.update(
            {
                "format": "spectral_library.batch_mapping.zarr",
                "source_sensor": source_sensor,
                "target_sensor": target_sensor,
                "output_mode": output_mode,
                "sample_count": 0,
                "output_width": output_width,
                "output_columns": list(output_columns),
                "chunk_size": int(chunk_size),
                "neighbor_estimator": neighbor_estimator,
                "knn_backend": knn_backend,
                "knn_eps": float(knn_eps),
                "k": int(k),
                "min_valid_bands": int(min_valid_bands),
                "sample_id_encoding": "utf-8",
            }
        )
        return _ZarrBatchExport(
            root=root,
            reflectance_dataset=reflectance_dataset,
            source_fit_rmse_dataset=source_fit_dataset,
            sample_id_dataset=sample_id_dataset,
            output_columns=output_columns,
            output_width=output_width,
            chunk_size=int(chunk_size),
        )

    def _append_batch_output_arrays_to_zarr(
        self,
        export: _ZarrBatchExport,
        *,
        sample_ids: Sequence[str],
        output_chunk: np.ndarray,
        source_fit_chunk: np.ndarray,
    ) -> int:
        current_count = int(export.reflectance_dataset.shape[0])
        append_count = int(len(sample_ids))
        next_count = current_count + append_count
        normalized_output_chunk = np.asarray(output_chunk, dtype=np.float64)
        normalized_source_fit_chunk = np.asarray(source_fit_chunk, dtype=np.float64)
        if normalized_output_chunk.shape != (append_count, export.output_width):
            raise MappingInputError(
                "output_chunk shape must match the append count and configured output width.",
                context={"expected_shape": [append_count, export.output_width], "actual_shape": list(normalized_output_chunk.shape)},
            )
        if normalized_source_fit_chunk.shape != (append_count,):
            raise MappingInputError(
                "source_fit_chunk must be one-dimensional and aligned to sample_ids.",
                context={"expected_shape": [append_count], "actual_shape": list(normalized_source_fit_chunk.shape)},
            )
        export.reflectance_dataset.resize((next_count, export.output_width))
        export.source_fit_rmse_dataset.resize((next_count,))
        export.sample_id_dataset.resize((next_count,))
        export.reflectance_dataset[current_count:next_count, :] = normalized_output_chunk
        export.source_fit_rmse_dataset[current_count:next_count] = normalized_source_fit_chunk
        export.sample_id_dataset[current_count:next_count] = np.asarray(sample_ids, dtype=object)
        export.root.attrs["sample_count"] = next_count
        return next_count

    def _mapping_result_output_row(
        self,
        result: MappingResult,
        *,
        output_mode: str,
        target_band_index: Mapping[str, int],
        output_width: int,
    ) -> np.ndarray:
        row = np.full(int(output_width), np.nan, dtype=np.float64)
        if output_mode == "target_sensor":
            if result.target_reflectance is None:
                raise SpectralLibraryError("invalid_result", "Batch mapping result is missing target_reflectance.")
            for band_id, reflectance in zip(result.target_band_ids, result.target_reflectance):
                if band_id not in target_band_index:
                    raise SpectralLibraryError(
                        "invalid_result",
                        "Batch mapping result emitted an unexpected target band.",
                        context={"band_id": band_id},
                    )
                row[int(target_band_index[band_id])] = float(reflectance)
            return row

        if output_mode == "vnir_spectrum":
            reflectance = result.reconstructed_vnir
        elif output_mode == "swir_spectrum":
            reflectance = result.reconstructed_swir
        else:
            reflectance = result.reconstructed_full_spectrum
        if reflectance is None:
            raise SpectralLibraryError(
                "invalid_result",
                "Batch mapping result is missing reconstructed spectral output.",
                context={"output_mode": output_mode},
            )
        if len(reflectance) != int(output_width):
            raise SpectralLibraryError(
                "invalid_result",
                "Batch mapping result emitted a spectral output with the wrong width.",
                context={"output_mode": output_mode, "expected_length": int(output_width), "actual_length": len(reflectance)},
            )
        row[:] = np.asarray(reflectance, dtype=np.float64)
        return row

    def _dense_segment_output_batch(
        self,
        *,
        segment: str,
        retrievals: Sequence[_SegmentRetrieval],
    ) -> _DenseSegmentOutputBatch:
        output_width = int(SEGMENT_WAVELENGTHS[segment].size)
        batch_size = len(retrievals)
        success = np.zeros(batch_size, dtype=bool)
        reconstructed = np.zeros((batch_size, output_width), dtype=np.float64)
        confidence_score = np.zeros(batch_size, dtype=np.float64)
        source_fit_rmse = np.zeros(batch_size, dtype=np.float64)
        valid_band_count = np.zeros(batch_size, dtype=np.int32)

        for batch_index, retrieval in enumerate(retrievals):
            valid_band_count[batch_index] = int(retrieval.valid_band_count)
            if not retrieval.success or retrieval.reconstructed is None:
                continue
            row = np.asarray(retrieval.reconstructed, dtype=np.float64)
            if row.shape != (output_width,):
                raise SpectralLibraryError(
                    "invalid_result",
                    "Batch mapping result emitted a spectral output with the wrong width.",
                    context={
                        "segment": segment,
                        "expected_length": output_width,
                        "actual_length": int(row.shape[0]),
                    },
                )
            success[batch_index] = True
            reconstructed[batch_index] = row
            if retrieval.confidence_score is not None:
                confidence_score[batch_index] = float(retrieval.confidence_score)
            if retrieval.source_fit_rmse is not None:
                source_fit_rmse[batch_index] = float(retrieval.source_fit_rmse)

        return _DenseSegmentOutputBatch(
            success=success,
            reconstructed=reconstructed,
            confidence_score=confidence_score,
            source_fit_rmse=source_fit_rmse,
            valid_band_count=valid_band_count,
        )

    def _segment_confidence_scores_from_dense_batches(
        self,
        *,
        vnir_batch: _DenseSegmentOutputBatch,
        swir_batch: _DenseSegmentOutputBatch,
    ) -> np.ndarray:
        vnir_weights = np.where(vnir_batch.success, np.maximum(vnir_batch.valid_band_count, 1), 0).astype(np.float64)
        swir_weights = np.where(swir_batch.success, np.maximum(swir_batch.valid_band_count, 1), 0).astype(np.float64)
        total_weights = vnir_weights + swir_weights
        numerator = vnir_weights * vnir_batch.confidence_score + swir_weights * swir_batch.confidence_score
        confidence_scores = np.zeros(vnir_weights.shape[0], dtype=np.float64)
        np.divide(numerator, total_weights, out=confidence_scores, where=total_weights > 0)
        return confidence_scores

    def _segment_source_fit_rmse_from_dense_batches(
        self,
        *,
        vnir_batch: _DenseSegmentOutputBatch,
        swir_batch: _DenseSegmentOutputBatch,
    ) -> np.ndarray:
        vnir_weights = np.where(vnir_batch.success, np.maximum(vnir_batch.valid_band_count, 1), 0).astype(np.float64)
        swir_weights = np.where(swir_batch.success, np.maximum(swir_batch.valid_band_count, 1), 0).astype(np.float64)
        total_weights = vnir_weights + swir_weights
        numerator = vnir_weights * vnir_batch.source_fit_rmse + swir_weights * swir_batch.source_fit_rmse
        source_fit_rmse = np.zeros(vnir_weights.shape[0], dtype=np.float64)
        np.divide(numerator, total_weights, out=source_fit_rmse, where=total_weights > 0)
        return source_fit_rmse

    def _finalize_dense_batch_outputs(
        self,
        *,
        dense_outputs_by_segment: Mapping[str, _DenseSegmentOutputBatch],
        output_mode: str,
        target_sensor: str | None,
        sample_ids: Sequence[str],
        sample_indices: Sequence[int],
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        vnir_batch = dense_outputs_by_segment["vnir"]
        swir_batch = dense_outputs_by_segment["swir"]
        source_fit_rmse = self._segment_source_fit_rmse_from_dense_batches(
            vnir_batch=vnir_batch,
            swir_batch=swir_batch,
        )
        resolved_source_fit_rmse_out = self._reusable_output_buffer(
            buffer=source_fit_rmse_out,
            expected_shape=tuple(source_fit_rmse.shape),
            name="source_fit_rmse_out",
        )
        if resolved_source_fit_rmse_out is not None:
            np.copyto(resolved_source_fit_rmse_out, source_fit_rmse)
            source_fit_rmse = resolved_source_fit_rmse_out

        def _raise_first_failure(failure_mask: np.ndarray, *, status_codes: np.ndarray | None = None) -> None:
            if not np.any(failure_mask):
                return
            local_index = int(np.flatnonzero(failure_mask)[0])
            self._raise_batch_output_error(
                output_mode=output_mode,
                sample_id=str(sample_ids[local_index]),
                sample_index=int(sample_indices[local_index]),
                target_sensor=target_sensor,
                status_code=None if status_codes is None else int(status_codes[local_index]),
            )

        if output_mode == "vnir_spectrum":
            _raise_first_failure(~vnir_batch.success)
            return np.asarray(vnir_batch.reconstructed, dtype=np.float64), source_fit_rmse
        if output_mode == "swir_spectrum":
            _raise_first_failure(~swir_batch.success)
            return np.asarray(swir_batch.reconstructed, dtype=np.float64), source_fit_rmse
        if output_mode == "full_spectrum":
            _raise_first_failure(~(vnir_batch.success & swir_batch.success))
            return _assemble_full_spectrum_batch(vnir_batch.reconstructed, swir_batch.reconstructed), source_fit_rmse

        if not target_sensor:
            raise MappingInputError("target_sensor is required when output_mode is target_sensor.")
        projection = self._target_sensor_projection(target_sensor)
        if vnir_batch.target_output_indices is not None and swir_batch.target_output_indices is not None:
            resolved_out = self._reusable_output_buffer(
                buffer=out,
                expected_shape=(int(vnir_batch.reconstructed.shape[0]), int(projection.output_width)),
                name="out",
            )
            status_codes_out = None if resolved_out is None else np.empty(vnir_batch.reconstructed.shape[0], dtype=np.int32)
            output_rows, status_codes = _stitch_target_sensor_segment_rows(
                vnir_rows=vnir_batch.reconstructed,
                swir_rows=swir_batch.reconstructed,
                vnir_success=vnir_batch.success,
                swir_success=swir_batch.success,
                vnir_output_indices=vnir_batch.target_output_indices,
                swir_output_indices=swir_batch.target_output_indices,
                output_width=projection.output_width,
                out_rows=resolved_out,
                out_status_codes=status_codes_out,
            )
            _raise_first_failure(status_codes != 0, status_codes=status_codes)
            return np.asarray(output_rows, dtype=np.float64), source_fit_rmse
        vnir_reconstructed = (
            vnir_batch.reconstructed
            if vnir_batch.reconstructed.shape[1] == projection.vnir_response_matrix.shape[0]
            else vnir_batch.reconstructed[:, projection.vnir_support_indices]
        )
        swir_reconstructed = (
            swir_batch.reconstructed
            if swir_batch.reconstructed.shape[1] == projection.swir_response_matrix.shape[0]
            else swir_batch.reconstructed[:, projection.swir_support_indices]
        )
        output_rows, status_codes = _finalize_target_sensor_batch_accel(
            vnir_reconstructed=vnir_reconstructed,
            swir_reconstructed=swir_reconstructed,
            vnir_success=vnir_batch.success,
            swir_success=swir_batch.success,
            vnir_response_matrix=projection.vnir_response_matrix,
            swir_response_matrix=projection.swir_response_matrix,
            vnir_output_indices=projection.vnir_output_indices,
            swir_output_indices=projection.swir_output_indices,
            output_width=projection.output_width,
        )
        status_codes = np.asarray(status_codes, dtype=np.int32)
        _raise_first_failure(status_codes != 0, status_codes=status_codes)
        return np.asarray(output_rows, dtype=np.float64), source_fit_rmse

    def _raise_batch_output_error(
        self,
        *,
        output_mode: str,
        sample_id: str,
        sample_index: int,
        target_sensor: str | None,
        status_code: int | None = None,
    ) -> None:
        if output_mode == "target_sensor":
            if status_code is None or int(status_code) == 1:
                error = MappingInputError(
                    "Target-sensor mapping could not produce any bands because no target segments were retrievable.",
                    context={"target_sensor": target_sensor},
                )
            else:
                error = MappingInputError(
                    "Target-sensor mapping emitted an unexpected batch finalization status.",
                    context={"target_sensor": target_sensor, "status_code": int(status_code)},
                )
        elif output_mode == "vnir_spectrum":
            error = MappingInputError("VNIR reconstruction requires enough valid VNIR source bands.")
        elif output_mode == "swir_spectrum":
            error = MappingInputError("SWIR reconstruction requires enough valid SWIR source bands.")
        else:
            error = MappingInputError("Full-spectrum reconstruction requires both VNIR and SWIR segment retrievals.")
        raise _attach_sample_context(error, sample_id=sample_id, sample_index=sample_index)

    def _map_reflectance_batch_output_arrays_from_coerced(
        self,
        *,
        source_sensor: str,
        normalized_sample_ids: tuple[str, ...],
        query_values_batch: np.ndarray,
        query_valid_mask_batch: np.ndarray,
        output_mode: str,
        target_sensor: str | None,
        k: int,
        min_valid_bands: int,
        neighbor_estimator: str,
        knn_backend: str,
        knn_eps: float,
        exclude_row_ids: Sequence[str] | None,
        exclude_sample_names: Sequence[str] | None,
        exclude_row_ids_per_sample: Sequence[str | None],
        self_exclude_sample_id: bool,
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        output_columns, _, _ = self._batch_output_layout(
            output_mode=output_mode,
            target_sensor=target_sensor,
        )
        projection = None if output_mode != "target_sensor" else self._target_sensor_projection(str(target_sensor))
        hyperspectral_rows_by_segment = None
        target_output_indices_by_segment = None
        if projection is not None:
            hyperspectral_rows_by_segment = {
                "vnir": projection.vnir_target_rows,
                "swir": projection.swir_target_rows,
            }
            target_output_indices_by_segment = {
                "vnir": projection.vnir_output_indices,
                "swir": projection.swir_output_indices,
            }
        dense_outputs_by_segment = self._grouped_dense_segment_outputs(
            source_sensor=source_sensor,
            sample_ids=normalized_sample_ids,
            query_values_batch=query_values_batch,
            query_valid_mask_batch=query_valid_mask_batch,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            hyperspectral_rows_by_segment=hyperspectral_rows_by_segment,
            target_output_indices_by_segment=target_output_indices_by_segment,
        )
        output_rows, source_fit_rmse = self._finalize_dense_batch_outputs(
            dense_outputs_by_segment=dense_outputs_by_segment,
            output_mode=output_mode,
            target_sensor=target_sensor,
            sample_ids=normalized_sample_ids,
            sample_indices=tuple(range(len(normalized_sample_ids))),
            out=out,
            source_fit_rmse_out=source_fit_rmse_out,
        )
        return np.asarray(output_rows, dtype=np.float64), np.asarray(source_fit_rmse, dtype=np.float64), output_columns

    def _validated_output_buffer(
        self,
        *,
        buffer: np.ndarray | None,
        expected_shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray | None:
        if buffer is None:
            return None
        if not isinstance(buffer, np.ndarray):
            raise MappingInputError(f"{name} must be a NumPy array when provided.")
        if buffer.dtype != np.float64:
            raise MappingInputError(
                f"{name} must have dtype float64.",
                context={"name": name, "dtype": str(buffer.dtype)},
            )
        if buffer.shape != expected_shape:
            raise MappingInputError(
                f"{name} must match the requested batch output shape.",
                context={"name": name, "expected_shape": list(expected_shape), "actual_shape": list(buffer.shape)},
            )
        if not buffer.flags.writeable:
            raise MappingInputError(f"{name} must be writeable.")
        return buffer

    def _reusable_output_buffer(
        self,
        *,
        buffer: np.ndarray | None,
        expected_shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray | None:
        resolved = self._validated_output_buffer(
            buffer=buffer,
            expected_shape=expected_shape,
            name=name,
        )
        if resolved is None or not resolved.flags.c_contiguous:
            return None
        return resolved

    def _map_reflectance_batch_output_arrays(
        self,
        *,
        source_sensor: str,
        reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
        valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
        out: np.ndarray | None = None,
        source_fit_rmse_out: np.ndarray | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
        _validate_mapping_request(
            output_mode,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
        )

        if isinstance(reflectance_rows, np.ndarray) and (valid_mask_rows is None or isinstance(valid_mask_rows, np.ndarray)):
            _, query_values_batch, query_valid_mask_batch = self._coerced_ndarray_batch_query_arrays(
                source_sensor=source_sensor,
                reflectance_rows=reflectance_rows,
                valid_mask_rows=None if valid_mask_rows is None else np.asarray(valid_mask_rows, dtype=bool),
            )
            normalized_sample_ids = _normalized_sample_ids(sample_ids, sample_count=int(query_values_batch.shape[0]))
        else:
            batch_rows = _normalized_batch_rows(reflectance_rows)
            normalized_sample_ids = _normalized_sample_ids(sample_ids, sample_count=len(batch_rows))
            batch_valid_masks = _normalized_batch_valid_masks(valid_mask_rows, sample_count=len(batch_rows))
            _, query_values_batch, query_valid_mask_batch = self._coerced_batch_query_arrays(
                source_sensor=source_sensor,
                batch_rows=batch_rows,
                batch_valid_masks=batch_valid_masks,
                sample_ids=normalized_sample_ids,
            )
        normalized_exclude_row_ids_per_sample = self._normalized_batch_exclude_row_ids(
            exclude_row_ids_per_sample,
            sample_ids=normalized_sample_ids,
        )
        output_rows, source_fit_rmse, output_columns = self._map_reflectance_batch_output_arrays_from_coerced(
            source_sensor=source_sensor,
            normalized_sample_ids=normalized_sample_ids,
            query_values_batch=query_values_batch,
            query_valid_mask_batch=query_valid_mask_batch,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            exclude_row_ids=exclude_row_ids,
            exclude_sample_names=exclude_sample_names,
            exclude_row_ids_per_sample=normalized_exclude_row_ids_per_sample,
            self_exclude_sample_id=self_exclude_sample_id,
            out=out,
            source_fit_rmse_out=source_fit_rmse_out,
        )
        return normalized_sample_ids, output_rows, source_fit_rmse, output_columns

    def map_reflectance_batch_to_zarr(
        self,
        *,
        zarr_path: Path,
        source_sensor: str,
        reflectance_rows: Sequence[Sequence[float] | Mapping[str, float]] | np.ndarray,
        valid_mask_rows: Sequence[Sequence[bool] | Mapping[str, bool] | None] | np.ndarray | None = None,
        sample_ids: Sequence[str] | None = None,
        output_mode: str = "target_sensor",
        target_sensor: str | None = None,
        k: int = 10,
        min_valid_bands: int = 1,
        neighbor_estimator: str = "mean",
        knn_backend: str = "numpy",
        knn_eps: float = 0.0,
        exclude_row_ids: Sequence[str] | None = None,
        exclude_sample_names: Sequence[str] | None = None,
        exclude_row_ids_per_sample: Sequence[str | None] | Mapping[str, str | None] | None = None,
        self_exclude_sample_id: bool = False,
        chunk_size: int | None = None,
    ) -> dict[str, object]:
        """Map a batch in chunks and persist dense outputs into a Zarr store."""

        _validate_mapping_request(
            output_mode,
            k=k,
            min_valid_bands=min_valid_bands,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
        )

        batch_rows = _normalized_batch_rows(reflectance_rows)
        normalized_sample_ids = _normalized_sample_ids(sample_ids, sample_count=len(batch_rows))
        batch_valid_masks = _normalized_batch_valid_masks(valid_mask_rows, sample_count=len(batch_rows))
        normalized_exclude_row_ids_per_sample = self._normalized_batch_exclude_row_ids(
            exclude_row_ids_per_sample,
            sample_ids=normalized_sample_ids,
        )
        sample_count = int(len(normalized_sample_ids))
        output_columns, _, axis_values = self._batch_output_layout(
            output_mode=output_mode,
            target_sensor=target_sensor,
        )
        resolved_chunk_size = min(
            sample_count,
            self._resolved_batch_output_chunk_size(
                source_sensor=source_sensor,
                output_width=int(axis_values.shape[0]),
                chunk_size=chunk_size,
            ),
        )
        temp_zarr_path = _temporary_output_path(Path(zarr_path))
        _remove_output_path(temp_zarr_path)
        try:
            export = self._open_batch_zarr_export(
                zarr_path=temp_zarr_path,
                source_sensor=source_sensor,
                output_mode=output_mode,
                target_sensor=target_sensor,
                chunk_size=resolved_chunk_size,
                k=k,
                min_valid_bands=min_valid_bands,
                neighbor_estimator=neighbor_estimator,
                knn_backend=knn_backend,
                knn_eps=knn_eps,
            )
            for start in range(0, sample_count, resolved_chunk_size):
                stop = min(sample_count, start + resolved_chunk_size)
                chunk_sample_ids, output_chunk, source_fit_chunk, _ = self._map_reflectance_batch_output_arrays(
                    source_sensor=source_sensor,
                    reflectance_rows=batch_rows[start:stop],
                    valid_mask_rows=batch_valid_masks[start:stop],
                    sample_ids=normalized_sample_ids[start:stop],
                    output_mode=output_mode,
                    target_sensor=target_sensor,
                    k=k,
                    min_valid_bands=min_valid_bands,
                    neighbor_estimator=neighbor_estimator,
                    knn_backend=knn_backend,
                    knn_eps=knn_eps,
                    exclude_row_ids=exclude_row_ids,
                    exclude_sample_names=exclude_sample_names,
                    exclude_row_ids_per_sample=normalized_exclude_row_ids_per_sample[start:stop],
                    self_exclude_sample_id=self_exclude_sample_id,
                )
                self._append_batch_output_arrays_to_zarr(
                    export,
                    sample_ids=chunk_sample_ids,
                    output_chunk=output_chunk,
                    source_fit_chunk=source_fit_chunk,
                )
            _finalize_output_path(temp_zarr_path, Path(zarr_path))
        except Exception:
            _remove_output_path(temp_zarr_path)
            raise
        return {
            "path": str(Path(zarr_path)),
            "sample_count": sample_count,
            "output_columns": output_columns,
            "chunk_size": resolved_chunk_size,
            "estimated_output_bytes": _estimated_dense_array_bytes(
                row_count=sample_count,
                column_count=int(axis_values.shape[0]),
                dtype=np.dtype(np.float64),
            ),
        }

    def _source_queries(self, source_sensor: str) -> np.ndarray:
        if source_sensor not in self._source_query_cache:
            query_matrix, band_ids = self._simulate_full_sensor_matrix(source_sensor)
            schema = self.get_sensor_schema(source_sensor)
            if band_ids != schema.band_ids():
                raise PreparedLibraryValidationError(
                    "Prepared source query matrix band order does not match the sensor schema.",
                    context={"source_sensor": source_sensor},
                )
            self._source_query_cache[source_sensor] = query_matrix
        return self._source_query_cache[source_sensor]

    def _simulate_full_sensor_matrix(self, sensor_id: str) -> tuple[np.ndarray, tuple[str, ...]]:
        schema = self.get_sensor_schema(sensor_id)
        values: list[np.ndarray] = []
        band_ids: list[str] = []
        segment_hyperspectral = {
            segment: np.asarray(self._load_hyperspectral(segment), dtype=np.float64)
            for segment in SEGMENTS
        }
        for band in schema.bands:
            response = self._band_response(sensor_id, band, segment_only=True)
            values.append(
                _response_weighted_average(
                    segment_hyperspectral[band.segment],
                    response,
                    error_message="Resampled SRF support must remain positive.",
                    error_context={"sensor_id": sensor_id, "band_id": band.band_id},
                )
            )
            band_ids.append(band.band_id)
        if not values:
            raise SensorSchemaError("Sensor schema must include at least one band.", context={"sensor_id": sensor_id})
        return np.column_stack(values), tuple(band_ids)


def _metric_report(predicted: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    """Compute compact reconstruction metrics for benchmarking output."""

    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    bias = np.mean(residual, axis=0)
    return {
        "mean_rmse": float(np.mean(rmse)),
        "mean_mae": float(np.mean(mae)),
        "mean_abs_bias": float(np.mean(np.abs(bias))),
        "per_band_rmse": [float(value) for value in rmse],
        "per_band_mae": [float(value) for value in mae],
        "per_band_bias": [float(value) for value in bias],
    }


def benchmark_mapping(
    prepared_root: Path,
    source_sensor: str,
    target_sensor: str,
    *,
    k: int = 10,
    test_fraction: float = 0.2,
    max_test_rows: int | None = None,
    random_seed: int = 0,
    neighbor_estimator: str = "mean",
    knn_backend: str = "numpy",
    knn_eps: float = 0.0,
) -> dict[str, object]:
    """Benchmark retrieval mapping against held-out prepared-library rows."""

    _validate_mapping_request(
        "target_sensor",
        k=k,
        min_valid_bands=1,
        neighbor_estimator=neighbor_estimator,
        knn_backend=knn_backend,
        knn_eps=knn_eps,
    )
    if not 0 < test_fraction < 1:
        raise MappingInputError("test_fraction must be between 0 and 1.", context={"test_fraction": test_fraction})
    if max_test_rows is not None and int(max_test_rows) < 1:
        raise MappingInputError("max_test_rows must be at least 1 when provided.", context={"max_test_rows": max_test_rows})

    mapper = SpectralMapper(prepared_root)
    source_queries = mapper._source_queries(source_sensor)
    target_truth, target_band_ids = mapper._simulate_full_sensor_matrix(target_sensor)
    vnir_truth = np.asarray(mapper._load_hyperspectral("vnir"), dtype=np.float64)
    swir_truth = np.asarray(mapper._load_hyperspectral("swir"), dtype=np.float64)
    full_truth = _assemble_full_spectrum_batch(vnir_truth, swir_truth)

    row_count = source_queries.shape[0]
    if row_count < 2:
        raise MappingInputError("benchmark_mapping requires at least two prepared-library rows.", context={"row_count": row_count})

    rng = np.random.default_rng(random_seed)
    permutation = rng.permutation(row_count)
    test_count = max(1, int(math.ceil(row_count * test_fraction)))
    if max_test_rows is not None:
        test_count = min(test_count, int(max_test_rows))
    if test_count >= row_count:
        test_count = row_count - 1
    test_indices = permutation[:test_count]
    train_indices = permutation[test_count:]

    design_train = np.column_stack([np.ones(train_indices.size, dtype=np.float64), source_queries[train_indices]])
    coefficients, _, _, _ = np.linalg.lstsq(design_train, target_truth[train_indices], rcond=None)
    design_test = np.column_stack([np.ones(test_indices.size, dtype=np.float64), source_queries[test_indices]])
    regression_predictions = design_test @ coefficients

    retrieval_target = np.empty((test_indices.size, len(target_band_ids)), dtype=np.float64)
    retrieval_vnir = np.empty((test_indices.size, len(VNIR_WAVELENGTHS)), dtype=np.float64)
    retrieval_swir = np.empty((test_indices.size, len(SWIR_WAVELENGTHS)), dtype=np.float64)
    retrieval_full = np.empty((test_indices.size, FULL_WAVELENGTH_COUNT), dtype=np.float64)

    for output_index, row_index in enumerate(test_indices):
        result = mapper._map_reflectance_internal(
            source_sensor=source_sensor,
            reflectance=source_queries[row_index],
            valid_mask=None,
            output_mode="full_spectrum",
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=1,
            neighbor_estimator=neighbor_estimator,
            knn_backend=knn_backend,
            knn_eps=knn_eps,
            candidate_row_indices=train_indices,
        )
        if result.target_reflectance is None or result.reconstructed_vnir is None or result.reconstructed_swir is None:
            raise PreparedLibraryValidationError(
                "benchmark_mapping expected full mapping outputs for every held-out query.",
                context={"row_index": int(row_index)},
            )
        retrieval_target[output_index] = result.target_reflectance
        retrieval_vnir[output_index] = result.reconstructed_vnir
        retrieval_swir[output_index] = result.reconstructed_swir
        if result.reconstructed_full_spectrum is None:
            raise PreparedLibraryValidationError(
                "benchmark_mapping expected a full reconstructed spectrum for every held-out query.",
                context={"row_index": int(row_index)},
            )
        retrieval_full[output_index] = result.reconstructed_full_spectrum

    return {
        "source_sensor_id": source_sensor,
        "target_sensor_id": target_sensor,
        "k": int(k),
        "neighbor_estimator": neighbor_estimator,
        "knn_backend": knn_backend,
        "knn_eps": float(knn_eps),
        "max_test_rows": None if max_test_rows is None else int(max_test_rows),
        "random_seed": int(random_seed),
        "train_rows": int(train_indices.size),
        "test_rows": int(test_indices.size),
        "target_sensor": {
            "band_ids": list(target_band_ids),
            "retrieval": _metric_report(retrieval_target, target_truth[test_indices]),
            "regression_baseline": _metric_report(regression_predictions, target_truth[test_indices]),
        },
        "vnir_spectrum": {
            "wavelength_nm": [int(value) for value in VNIR_WAVELENGTHS],
            "retrieval": _metric_report(retrieval_vnir, vnir_truth[test_indices]),
        },
        "swir_spectrum": {
            "wavelength_nm": [int(value) for value in SWIR_WAVELENGTHS],
            "retrieval": _metric_report(retrieval_swir, swir_truth[test_indices]),
        },
        "full_spectrum": {
            "wavelength_nm": [int(value) for value in CANONICAL_WAVELENGTHS],
            "retrieval": _metric_report(retrieval_full, full_truth[test_indices]),
        },
    }
