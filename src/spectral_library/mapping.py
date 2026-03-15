from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np

from ._version import __version__


PREPARED_SCHEMA_VERSION = "1.0.0"
SUPPORTED_OUTPUT_MODES = (
    "target_sensor",
    "vnir_spectrum",
    "swir_spectrum",
    "full_spectrum",
)
SEGMENTS = ("vnir", "swir")
CANONICAL_START_NM = 400
CANONICAL_END_NM = 2500
VNIR_START_NM = 400
VNIR_END_NM = 1000
SWIR_START_NM = 900
SWIR_END_NM = 2500
FULL_BLEND_START_NM = 900
FULL_BLEND_END_NM = 1000

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
    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_sensor_schema", message, context=context)


class PreparedLibraryBuildError(SpectralLibraryError):
    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("prepare_failed", message, context=context)


class PreparedLibraryValidationError(SpectralLibraryError):
    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_prepared_library", message, context=context)


class PreparedLibraryCompatibilityError(SpectralLibraryError):
    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("prepared_library_incompatible", message, context=context)


class MappingInputError(SpectralLibraryError):
    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__("invalid_mapping_input", message, context=context)


@dataclass(frozen=True)
class SensorBandDefinition:
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
        }


@dataclass
class MappingResult:
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
class _SegmentRetrieval:
    segment: str
    valid_band_count: int
    query_band_ids: tuple[str, ...]
    success: bool
    reconstructed: np.ndarray | None = None
    neighbor_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    neighbor_ids: tuple[str, ...] = ()
    neighbor_distances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    reason: str | None = None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _segment_slice(segment: str) -> slice:
    start_nm, end_nm = SEGMENT_RANGES[segment]
    start_index = start_nm - CANONICAL_START_NM
    stop_index = end_nm - CANONICAL_START_NM + 1
    return slice(start_index, stop_index)


def _segment_wavelengths(segment: str) -> np.ndarray:
    return SEGMENT_WAVELENGTHS[segment]


def _blend_overlap(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    overlap_wavelengths = np.arange(FULL_BLEND_START_NM, FULL_BLEND_END_NM + 1, dtype=np.float64)
    weights = (FULL_BLEND_END_NM - overlap_wavelengths) / (FULL_BLEND_END_NM - FULL_BLEND_START_NM)
    return weights * vnir[500:601] + (1.0 - weights) * swir[:101]


def _assemble_full_spectrum(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    full = np.empty(FULL_WAVELENGTH_COUNT, dtype=np.float64)
    full[:500] = vnir[:500]
    full[500:601] = _blend_overlap(vnir, swir)
    full[601:] = swir[101:]
    return full


def _assemble_full_spectrum_batch(vnir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    full = np.empty((vnir.shape[0], FULL_WAVELENGTH_COUNT), dtype=np.float64)
    full[:, :500] = vnir[:, :500]
    overlap_wavelengths = np.arange(FULL_BLEND_START_NM, FULL_BLEND_END_NM + 1, dtype=np.float64)
    weights = (FULL_BLEND_END_NM - overlap_wavelengths) / (FULL_BLEND_END_NM - FULL_BLEND_START_NM)
    full[:, 500:601] = weights * vnir[:, 500:601] + (1.0 - weights) * swir[:, :101]
    full[:, 601:] = swir[:, 101:]
    return full


def _ensure_supported_output_mode(output_mode: str) -> None:
    if output_mode not in SUPPORTED_OUTPUT_MODES:
        raise MappingInputError(
            "Unsupported output_mode.",
            context={"output_mode": output_mode, "supported_output_modes": list(SUPPORTED_OUTPUT_MODES)},
        )


def _normalized_source_sensors(source_sensors: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for sensor_id in source_sensors:
        text = str(sensor_id).strip()
        if text and text not in normalized:
            normalized.append(text)
    if not normalized:
        raise PreparedLibraryBuildError("At least one source sensor must be provided.")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


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


def _simulate_segment_matrix(
    hyperspectral_segment: np.ndarray,
    bands: Sequence[SensorBandDefinition],
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    matrix = np.empty((hyperspectral_segment.shape[0], len(bands)), dtype=dtype)
    for index, band in enumerate(bands):
        response = _resample_band_response(band, segment_only=True)
        denominator = float(response.sum())
        if denominator <= 0:
            raise SensorSchemaError(
                "Resampled SRF support must remain positive.",
                context={"band_id": band.band_id, "segment": band.segment},
            )
        matrix[:, index] = (hyperspectral_segment @ response / denominator).astype(dtype, copy=False)
    return matrix


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
) -> tuple[list[str], list[dict[str, object]], np.ndarray]:
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
        for row in spectra_reader:
            key = (row["source_id"], row["spectrum_id"], row["sample_name"])
            if key in spectra_by_key:
                raise PreparedLibraryBuildError(
                    "Duplicate spectra rows were found while preparing the mapping library.",
                    context={"source_id": key[0], "spectrum_id": key[1], "sample_name": key[2]},
                )
            spectra_by_key[key] = np.asarray([float(row[column]) for column in nm_columns], dtype=dtype)

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

    return ["row_index", *metadata_fieldnames], aligned_metadata_rows, hyperspectral


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


def _read_json_document(
    path: Path,
    *,
    error_factory: type[SpectralLibraryError],
    document_name: str,
) -> object:
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
        manifest_path.name: _sha256_file(manifest_path),
    }


def prepare_mapping_library(
    siac_root: Path,
    srf_root: Path,
    output_root: Path,
    source_sensors: Sequence[str],
    *,
    dtype: str = "float32",
) -> PreparedLibraryManifest:
    dtype_np = np.dtype(dtype)
    if dtype_np.kind != "f":
        raise PreparedLibraryBuildError("prepare_mapping_library only supports floating-point dtypes.", context={"dtype": dtype})

    source_sensor_ids = _normalized_source_sensors(source_sensors)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(siac_root) / "tabular" / "siac_spectra_metadata.csv"
    spectra_path = Path(siac_root) / "tabular" / "siac_normalized_spectra.csv"
    metadata_fieldnames, metadata_rows, hyperspectral = _load_siac_rows(metadata_path, spectra_path, dtype=dtype_np)

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
            bands = schema.bands_for_segment(segment)
            segment_matrix = _simulate_segment_matrix(
                hyperspectral_vnir if segment == "vnir" else hyperspectral_swir,
                bands,
                dtype=dtype_np,
            )
            np.save(output_root / f"source_{sensor_id}_{segment}.npy", segment_matrix)

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
    file_checksums = {path.name: _sha256_file(path) for path in stable_files}

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
    )
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest.to_dict())

    checksums_payload = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "files": {
            **file_checksums,
            "manifest.json": _sha256_file(manifest_path),
        },
    }
    _write_json(output_root / "checksums.json", checksums_payload)
    return manifest


def validate_prepared_library(
    prepared_root: Path,
    *,
    verify_checksums: bool = True,
) -> PreparedLibraryManifest:
    prepared_root = Path(prepared_root)
    mapper = SpectralMapper(prepared_root, verify_checksums=verify_checksums)
    return mapper.manifest


class SpectralMapper:
    def __init__(self, prepared_root: Path, *, verify_checksums: bool = False) -> None:
        self.prepared_root = Path(prepared_root)
        self.manifest = _load_prepared_manifest(self.prepared_root)
        _validate_manifest_compatibility(self.manifest)
        self._validate_required_runtime_files()
        self._validate_checksums(verify_checksums=verify_checksums)

        self._sensor_schemas = self._load_prepared_sensor_schemas()
        self._row_ids = self._load_row_ids()
        self._hyperspectral_cache: dict[str, np.ndarray] = {}
        self._source_matrix_cache: dict[tuple[str, str], np.ndarray] = {}
        self._response_cache: dict[tuple[str, str, bool], np.ndarray] = {}
        self._source_query_cache: dict[str, np.ndarray] = {}

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
                actual_checksum = _sha256_file(self.prepared_root / file_name)
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

    def _load_row_ids(self) -> tuple[str, ...]:
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
        return row_ids

    def _validate_prepared_layout(self) -> None:
        for segment in SEGMENTS:
            array = self._load_hyperspectral(segment)
            expected_width = len(_segment_wavelengths(segment))
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
                expected_shape = (self.manifest.row_count, len(schema.bands_for_segment(segment)))
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

    def _band_response(self, sensor_id: str, band: SensorBandDefinition, *, segment_only: bool) -> np.ndarray:
        cache_key = (sensor_id, band.band_id, segment_only)
        if cache_key not in self._response_cache:
            self._response_cache[cache_key] = _resample_band_response(band, segment_only=segment_only)
        return self._response_cache[cache_key]

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

    def _retrieve_segment(
        self,
        *,
        source_sensor: str,
        segment: str,
        query_values: np.ndarray,
        valid_mask: np.ndarray,
        k: int,
        min_valid_bands: int,
        candidate_row_indices: np.ndarray,
    ) -> _SegmentRetrieval:
        source_schema = self.get_sensor_schema(source_sensor)
        query_band_ids = tuple(band.band_id for band in source_schema.bands_for_segment(segment))
        valid_band_count = int(valid_mask.sum())
        if valid_band_count < min_valid_bands:
            return _SegmentRetrieval(
                segment=segment,
                valid_band_count=valid_band_count,
                query_band_ids=query_band_ids,
                success=False,
                reason="insufficient_valid_bands",
            )

        source_matrix = self._load_source_matrix(source_sensor, segment)
        if source_matrix.shape[1] != len(query_band_ids):
            raise PreparedLibraryValidationError(
                "Prepared source matrix width does not match the source sensor schema.",
                context={"source_sensor": source_sensor, "segment": segment},
            )

        valid_indices = np.flatnonzero(valid_mask)
        candidate_matrix = np.asarray(source_matrix[candidate_row_indices][:, valid_indices], dtype=np.float64)
        query_vector = query_values[valid_indices]
        distances = np.sqrt(np.mean((candidate_matrix - query_vector) ** 2, axis=1))

        neighbor_count = min(int(k), int(candidate_row_indices.size))
        if neighbor_count <= 0:
            raise MappingInputError("k must be at least 1.", context={"k": k})
        if neighbor_count == candidate_row_indices.size:
            local_top = np.arange(candidate_row_indices.size)
        else:
            local_top = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
        ordered_local = local_top[np.lexsort((candidate_row_indices[local_top], distances[local_top]))]
        neighbor_indices = candidate_row_indices[ordered_local]
        neighbor_distances = distances[ordered_local]

        reconstructed = np.asarray(
            np.mean(self._load_hyperspectral(segment)[neighbor_indices], axis=0),
            dtype=np.float64,
        )
        return _SegmentRetrieval(
            segment=segment,
            valid_band_count=valid_band_count,
            query_band_ids=query_band_ids,
            success=True,
            reconstructed=reconstructed,
            neighbor_indices=neighbor_indices,
            neighbor_ids=tuple(self._row_ids[index] for index in neighbor_indices),
            neighbor_distances=np.asarray(neighbor_distances, dtype=np.float64),
        )

    def _simulate_target_sensor(
        self,
        target_sensor: str,
        segment_outputs: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        target_schema = self.get_sensor_schema(target_sensor)
        values: list[float] = []
        band_ids: list[str] = []
        for segment in SEGMENTS:
            if segment not in segment_outputs:
                continue
            reconstructed = segment_outputs[segment]
            for band in target_schema.bands_for_segment(segment):
                response = self._band_response(target_sensor, band, segment_only=True)
                denominator = float(response.sum())
                if denominator <= 0:
                    raise SensorSchemaError(
                        "Resampled target SRF support must remain positive.",
                        context={"target_sensor": target_sensor, "band_id": band.band_id},
                    )
                values.append(float(np.dot(reconstructed, response) / denominator))
                band_ids.append(band.band_id)
        if not values:
            return None, ()
        return np.asarray(values, dtype=np.float64), tuple(band_ids)

    def _candidate_rows(self, candidate_row_indices: Sequence[int] | None) -> np.ndarray:
        if candidate_row_indices is None:
            return np.arange(self.manifest.row_count, dtype=np.int64)
        candidate = np.asarray(candidate_row_indices, dtype=np.int64)
        if candidate.ndim != 1 or candidate.size == 0:
            raise MappingInputError("candidate_row_indices must be a non-empty one-dimensional sequence.")
        if np.any(candidate < 0) or np.any(candidate >= self.manifest.row_count):
            raise MappingInputError(
                "candidate_row_indices must refer to valid prepared-library rows.",
                context={"row_count": self.manifest.row_count},
            )
        return np.unique(candidate)

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
        candidate_row_indices: Sequence[int] | None,
    ) -> MappingResult:
        _ensure_supported_output_mode(output_mode)
        if k < 1:
            raise MappingInputError("k must be at least 1.", context={"k": k})
        if min_valid_bands < 1:
            raise MappingInputError("min_valid_bands must be at least 1.", context={"min_valid_bands": min_valid_bands})

        source_schema = self.get_sensor_schema(source_sensor)
        query_values, query_valid_mask = self._coerce_query(source_schema, reflectance, valid_mask)
        candidate_rows = self._candidate_rows(candidate_row_indices)

        segment_retrievals: dict[str, _SegmentRetrieval] = {}
        segment_outputs: dict[str, np.ndarray] = {}
        segment_valid_band_counts: dict[str, int] = {}
        neighbor_ids_by_segment: dict[str, tuple[str, ...]] = {}
        neighbor_distances_by_segment: dict[str, np.ndarray] = {}

        for segment in SEGMENTS:
            segment_indices = [index for index, band in enumerate(source_schema.bands) if band.segment == segment]
            segment_values = query_values[segment_indices]
            segment_valid = query_valid_mask[segment_indices]
            retrieval = self._retrieve_segment(
                source_sensor=source_sensor,
                segment=segment,
                query_values=segment_values,
                valid_mask=segment_valid,
                k=k,
                min_valid_bands=min_valid_bands,
                candidate_row_indices=candidate_rows,
            )
            segment_retrievals[segment] = retrieval
            segment_valid_band_counts[segment] = retrieval.valid_band_count
            neighbor_ids_by_segment[segment] = retrieval.neighbor_ids
            neighbor_distances_by_segment[segment] = retrieval.neighbor_distances
            if retrieval.success and retrieval.reconstructed is not None:
                segment_outputs[segment] = retrieval.reconstructed

        diagnostics: dict[str, object] = {
            "source_sensor": source_sensor,
            "target_sensor": target_sensor,
            "output_mode": output_mode,
            "k": k,
            "segments": {
                segment: {
                    "status": "ok" if retrieval.success else retrieval.reason,
                    "valid_band_count": retrieval.valid_band_count,
                    "query_band_ids": list(retrieval.query_band_ids),
                    "neighbor_ids": list(retrieval.neighbor_ids),
                    "neighbor_distances": [float(value) for value in retrieval.neighbor_distances],
                }
                for segment, retrieval in segment_retrievals.items()
            },
        }

        reconstructed_vnir = segment_outputs.get("vnir")
        reconstructed_swir = segment_outputs.get("swir")
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
            target_reflectance, target_band_ids = self._simulate_target_sensor(target_sensor, segment_outputs)
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
            segment_outputs=segment_outputs,
            segment_valid_band_counts=segment_valid_band_counts,
            diagnostics=diagnostics,
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
    ) -> MappingResult:
        return self._map_reflectance_internal(
            source_sensor=source_sensor,
            reflectance=reflectance,
            valid_mask=valid_mask,
            output_mode=output_mode,
            target_sensor=target_sensor,
            k=k,
            min_valid_bands=min_valid_bands,
            candidate_row_indices=None,
        )

    def _source_queries(self, source_sensor: str) -> np.ndarray:
        if source_sensor not in self._source_query_cache:
            schema = self.get_sensor_schema(source_sensor)
            query_matrix = np.empty((self.manifest.row_count, len(schema.bands)), dtype=np.float64)
            segment_offsets = {segment: 0 for segment in SEGMENTS}
            for band_index, band in enumerate(schema.bands):
                segment_matrix = self._load_source_matrix(source_sensor, band.segment)
                query_matrix[:, band_index] = segment_matrix[:, segment_offsets[band.segment]]
                segment_offsets[band.segment] += 1
            self._source_query_cache[source_sensor] = query_matrix
        return self._source_query_cache[source_sensor]

    def _simulate_full_sensor_matrix(self, sensor_id: str) -> tuple[np.ndarray, tuple[str, ...]]:
        schema = self.get_sensor_schema(sensor_id)
        values: list[np.ndarray] = []
        band_ids: list[str] = []
        for segment in SEGMENTS:
            segment_hyperspectral = np.asarray(self._load_hyperspectral(segment), dtype=np.float64)
            for band in schema.bands_for_segment(segment):
                response = self._band_response(sensor_id, band, segment_only=True)
                denominator = float(response.sum())
                values.append(segment_hyperspectral @ response / denominator)
                band_ids.append(band.band_id)
        if not values:
            raise SensorSchemaError("Sensor schema must include at least one band.", context={"sensor_id": sensor_id})
        return np.column_stack(values), tuple(band_ids)


def _metric_report(predicted: np.ndarray, truth: np.ndarray) -> dict[str, object]:
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
    random_seed: int = 0,
) -> dict[str, object]:
    if not 0 < test_fraction < 1:
        raise MappingInputError("test_fraction must be between 0 and 1.", context={"test_fraction": test_fraction})

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
