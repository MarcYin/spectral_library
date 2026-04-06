"""Prepared-library build workflow for spectral mapping."""

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

from ..adapters import backends as _backends, sensors as _sensors
from ..engine import core as _core

for _module in (_core, _sensors):
    globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})

del _module, _core, _sensors

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
    if manifest.schema_version != PREPARED_SCHEMA_VERSION:
        raise PreparedLibraryCompatibilityError(
            "Prepared mapping runtime schema version is incompatible with this package.",
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

def build_mapping_library(
    siac_root: Path,
    srf_root: Path | None,
    output_root: Path,
    source_sensors: Sequence[str],
    *,
    dtype: str = "float32",
    knn_index_backends: Sequence[str] | None = None,
) -> PreparedLibraryManifest:
    """Prepare a reusable runtime bundle from SIAC spectra and SRF definitions."""

    dtype_np = np.dtype(dtype)
    if dtype_np.kind != "f":
        raise PreparedLibraryBuildError(
            "build_mapping_library only supports floating-point dtypes.",
            context={"dtype": dtype},
        )

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

    sensors = load_sensor_schemas(srf_root, required_sensor_ids=source_sensor_ids)
    missing_source_sensors = [sensor_id for sensor_id in source_sensor_ids if sensor_id not in sensors]
    if missing_source_sensors:
        raise SensorSchemaError(
            "Requested source sensors could not be resolved.",
            context={"missing_source_sensors": missing_source_sensors, "srf_root": str(srf_root) if srf_root is not None else None},
        )

    hyperspectral_vnir = hyperspectral[:, _segment_slice("vnir")]
    hyperspectral_swir = hyperspectral[:, _segment_slice("swir")]
    np.save(output_root / SEGMENT_FILE_NAMES["vnir"], hyperspectral_vnir)
    np.save(output_root / SEGMENT_FILE_NAMES["swir"], hyperspectral_swir)

    for sensor_id in source_sensor_ids:
        schema = sensors[sensor_id]
        for segment in SEGMENTS:
            bands = _source_retrieval_bands(schema, segment)
            segment_matrix = _simulate_source_retrieval_matrix_from_segments(
                hyperspectral_vnir,
                hyperspectral_swir,
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
                _backends._persist_knn_index(candidate_matrix, backend=backend, output_path=artifact_path)
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
            "sensors": [_prepared_runtime_sensor_payload(schema) for _, schema in sorted(sensors.items())],
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

    from ..engine.runtime import SpectralMapper

    prepared_root = Path(prepared_root)
    mapper = SpectralMapper(prepared_root, verify_checksums=verify_checksums)
    return mapper.manifest


prepare_mapping_library = build_mapping_library
