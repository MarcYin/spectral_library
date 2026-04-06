"""High-level in-memory runtime build surface for spectral mapping."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np

from .adapters import backends as _backends, sensors as _sensors
from .build import library as _build
from .engine import core as _core
from .engine.runtime import SpectralMapper
from .inputs import HyperspectralLibraryInput, SensorInput

SensorInputLike = Union[str, SensorInput, _core.SensorSRFSchema, Mapping[str, object]]
LibraryInputLike = Union[HyperspectralLibraryInput, Mapping[str, object]]


@dataclass
class PreparedRuntime:
    """Reusable prepared runtime plus its normalized sensor identities."""

    prepared_root: Path
    mapper: SpectralMapper
    manifest: _core.PreparedLibraryManifest
    source_sensor_ids: tuple[str, ...]
    target_sensor_ids: tuple[str, ...]
    source_band_ids: dict[str, tuple[str, ...]]
    target_band_ids: dict[str, tuple[str, ...]]
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False, compare=False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mapper, name)

    def close(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> "PreparedRuntime":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def get_sensor_schema(self, sensor_id: str) -> _core.SensorSRFSchema:
        return self.mapper.get_sensor_schema(sensor_id)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _rsrf_version() -> str | None:
    try:
        module = _sensors._load_rsrf_module()
    except _core.SensorSchemaError:
        return None
    return _core._optional_string(getattr(module, "__version__", None))


def _library_input_mapping(library: LibraryInputLike) -> dict[str, object]:
    if isinstance(library, HyperspectralLibraryInput):
        metadata_rows = library.metadata_rows
        if metadata_rows is None:
            metadata_rows = library.provenance_metadata
        return {
            "wavelengths_nm": np.asarray(library.wavelengths_nm, dtype=np.float64),
            "spectra": np.asarray(library.spectra, dtype=np.float64),
            "sample_ids": list(library.sample_ids),
            "metadata_rows": None if metadata_rows is None else [dict(row) for row in metadata_rows],
            "source_id": library.source_id,
        }
    if isinstance(library, Mapping):
        payload = dict(library)
        if "metadata_rows" not in payload and "provenance_metadata" in payload:
            payload["metadata_rows"] = payload["provenance_metadata"]
        if "metadata_rows" not in payload and "sample_metadata" in payload:
            payload["metadata_rows"] = payload["sample_metadata"]
        return payload
    raise _core.PreparedLibraryBuildError(
        "library must be HyperspectralLibraryInput or a mapping.",
        context={"library_input_type": type(library).__name__},
    )


def _coerce_library_input(
    library: LibraryInputLike,
) -> tuple[list[str], list[dict[str, object]], np.ndarray, str, str]:
    values = _library_input_mapping(library)
    wavelengths = np.asarray(values.get("wavelengths_nm"), dtype=np.float64)
    expected_wavelengths = _core.CANONICAL_WAVELENGTHS.astype(np.float64)
    if wavelengths.ndim != 1 or wavelengths.shape[0] != expected_wavelengths.shape[0]:
        raise _core.PreparedLibraryBuildError(
            "wavelengths_nm must define the canonical 400-2500 nm grid.",
            context={
                "shape": list(wavelengths.shape),
                "expected_wavelength_count": int(expected_wavelengths.shape[0]),
            },
        )
    if not np.allclose(wavelengths, expected_wavelengths, atol=0.0, rtol=0.0):
        raise _core.PreparedLibraryBuildError(
            "wavelengths_nm must exactly match the canonical 400-2500 nm grid.",
            context={"expected_start_nm": _core.CANONICAL_START_NM, "expected_end_nm": _core.CANONICAL_END_NM},
        )

    spectra = np.asarray(values.get("spectra"), dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[1] != expected_wavelengths.shape[0]:
        raise _core.PreparedLibraryBuildError(
            "spectra must be a two-dimensional array aligned to wavelengths_nm.",
            context={"shape": list(spectra.shape)},
        )

    raw_sample_ids = values.get("sample_ids")
    if not isinstance(raw_sample_ids, Sequence) or isinstance(raw_sample_ids, (str, bytes, bytearray)):
        raise _core.PreparedLibraryBuildError("sample_ids must be a sequence of strings.")
    sample_ids = [str(sample_id).strip() for sample_id in raw_sample_ids]
    if len(sample_ids) != spectra.shape[0]:
        raise _core.PreparedLibraryBuildError(
            "sample_ids must have the same length as spectra.",
            context={"sample_id_count": len(sample_ids), "row_count": int(spectra.shape[0])},
        )
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise _core.PreparedLibraryBuildError("sample_ids must be non-empty.")
    if len(set(sample_ids)) != len(sample_ids):
        raise _core.PreparedLibraryBuildError("sample_ids must be unique.")

    source_id = _core._optional_string(values.get("source_id")) or "in_memory"
    metadata_rows_input = values.get("metadata_rows")
    metadata_rows: list[Mapping[str, object]]
    if metadata_rows_input is None:
        metadata_rows = [{} for _ in sample_ids]
    elif isinstance(metadata_rows_input, Sequence) and not isinstance(metadata_rows_input, (str, bytes, bytearray)):
        metadata_rows = [dict(row) for row in metadata_rows_input]
    else:
        raise _core.PreparedLibraryBuildError("metadata_rows must be a sequence of mappings when provided.")
    if len(metadata_rows) != len(sample_ids):
        raise _core.PreparedLibraryBuildError(
            "metadata_rows must have the same length as spectra.",
            context={"metadata_row_count": len(metadata_rows), "row_count": int(spectra.shape[0])},
        )

    reserved_keys = {"row_index"}
    extra_fieldnames: list[str] = []
    normalized_rows: list[dict[str, object]] = []
    for sample_id, metadata_row in zip(sample_ids, metadata_rows):
        duplicate_reserved = reserved_keys & set(metadata_row)
        if duplicate_reserved:
            raise _core.PreparedLibraryBuildError(
                "metadata_rows may not redefine reserved identifier columns.",
                context={"reserved_columns": sorted(duplicate_reserved)},
            )
        row_source_id = _core._optional_string(metadata_row.get("source_id")) or source_id
        row_spectrum_id = _core._optional_string(metadata_row.get("spectrum_id")) or sample_id
        row_sample_name = _core._optional_string(metadata_row.get("sample_name")) or sample_id
        normalized_row = {
            "source_id": row_source_id,
            "spectrum_id": row_spectrum_id,
            "sample_name": row_sample_name,
        }
        for key, value in metadata_row.items():
            text = str(key).strip()
            if not text:
                raise _core.PreparedLibraryBuildError("metadata_rows keys must be non-empty strings.")
            if text in {"source_id", "spectrum_id", "sample_name"}:
                continue
            if text not in extra_fieldnames:
                extra_fieldnames.append(text)
            normalized_row[text] = value
        normalized_rows.append(normalized_row)

    metadata_fieldnames = ["row_index", "source_id", "spectrum_id", "sample_name", *extra_fieldnames]
    aligned_rows = [
        {"row_index": row_index, **row}
        for row_index, row in enumerate(normalized_rows)
    ]

    library_identity = hashlib.sha256(
        _stable_json_dumps(
            {
                "sample_ids": sample_ids,
                "metadata_rows": normalized_rows,
                "wavelengths_nm": wavelengths,
                "spectra": spectra,
            }
        ).encode("utf-8")
    ).hexdigest()
    return metadata_fieldnames, aligned_rows, spectra, f"in_memory://library/{library_identity[:12]}", library_identity


def _runtime_identity(
    *,
    library_build_id: str,
    source_schemas: Sequence[_core.SensorSRFSchema],
    target_schemas: Sequence[_core.SensorSRFSchema],
    dtype: str,
    knn_index_backends: Sequence[str] | None,
) -> str:
    payload = {
        "library_build_id": library_build_id,
        "source_sensors": [schema.to_dict() for schema in source_schemas],
        "target_sensors": [schema.to_dict() for schema in target_schemas],
        "dtype": str(np.dtype(dtype).name),
        "knn_index_backends": list(knn_index_backends or ()),
        "spectral_library_version": _core.__version__,
        "rsrf_version": _rsrf_version(),
    }
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _prepared_runtime_wrapper(
    *,
    prepared_root: Path,
    source_schemas: Sequence[_core.SensorSRFSchema],
    target_schemas: Sequence[_core.SensorSRFSchema],
    verify_checksums: bool,
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None,
) -> PreparedRuntime:
    mapper = SpectralMapper(prepared_root, verify_checksums=verify_checksums)
    return PreparedRuntime(
        prepared_root=Path(prepared_root),
        mapper=mapper,
        manifest=mapper.manifest,
        source_sensor_ids=tuple(schema.sensor_id for schema in source_schemas),
        target_sensor_ids=tuple(schema.sensor_id for schema in target_schemas),
        source_band_ids={schema.sensor_id: schema.band_ids() for schema in source_schemas},
        target_band_ids={schema.sensor_id: schema.band_ids() for schema in target_schemas},
        _temporary_directory=temporary_directory,
    )


def _normalized_sensor_inputs(
    sensor_inputs: Sequence[SensorInputLike] | SensorInputLike | None,
    *,
    kind: str,
) -> tuple[_core.SensorSRFSchema, ...]:
    if sensor_inputs is None:
        return ()
    if isinstance(sensor_inputs, (str, SensorInput, _core.SensorSRFSchema, Mapping)):
        raw_inputs = [sensor_inputs]
    else:
        raw_inputs = list(sensor_inputs)
    if kind == "source" and not raw_inputs:
        raise _core.PreparedLibraryBuildError("At least one source sensor must be provided.")
    return tuple(_sensors.coerce_sensor_input(sensor_input) for sensor_input in raw_inputs)


def build_mapping_runtime(
    library: LibraryInputLike,
    source_sensors: Sequence[SensorInputLike] | SensorInputLike,
    target_sensors: Sequence[SensorInputLike] | SensorInputLike | None = None,
    *,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    dtype: str = "float32",
    knn_index_backends: Sequence[str] | None = None,
    verify_checksums: bool = False,
) -> PreparedRuntime:
    """Build a reusable mapping runtime from in-memory library and sensor inputs."""

    metadata_fieldnames, metadata_rows, hyperspectral, source_identity_root, source_identity_build_id = _coerce_library_input(library)
    source_schemas = _normalized_sensor_inputs(source_sensors, kind="source")
    target_schema_list = _normalized_sensor_inputs(target_sensors, kind="target")

    runtime_identity = _runtime_identity(
        library_build_id=source_identity_build_id,
        source_schemas=source_schemas,
        target_schemas=target_schema_list,
        dtype=dtype,
        knn_index_backends=knn_index_backends,
    )

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    final_root: Path
    temp_root: Path | None = None
    use_finalize = False

    if output_root is not None:
        final_root = Path(output_root)
        temp_root = _backends._temporary_output_path(final_root)
        use_finalize = True
    elif cache_root is not None:
        final_root = Path(cache_root) / runtime_identity
        if (final_root / "manifest.json").exists():
            try:
                return _prepared_runtime_wrapper(
                    prepared_root=final_root,
                    source_schemas=source_schemas,
                    target_schemas=target_schema_list,
                    verify_checksums=verify_checksums,
                )
            except _core.SpectralLibraryError:
                pass
        temp_root = _backends._temporary_output_path(final_root)
        use_finalize = True
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="spectral-library-runtime-")
        final_root = Path(temporary_directory.name)

    combined_schemas = {
        schema.sensor_id: schema
        for schema in (*source_schemas, *target_schema_list)
    }
    source_sensor_ids = [schema.sensor_id for schema in source_schemas]
    build_root = final_root if temp_root is None else temp_root

    try:
        _build._build_mapping_library_from_inputs(
            output_root=build_root,
            metadata_fieldnames=metadata_fieldnames,
            metadata_rows=metadata_rows,
            hyperspectral=hyperspectral,
            sensors=combined_schemas,
            source_sensor_ids=source_sensor_ids,
            source_identity_root=source_identity_root,
            source_identity_build_id=source_identity_build_id,
            dtype=np.dtype(dtype),
            knn_index_backends=knn_index_backends,
            interpolation_summary={},
        )
        if use_finalize and temp_root is not None:
            _backends._finalize_output_path(temp_root, final_root)
    except Exception:
        if temp_root is not None and temp_root.exists():
            _backends._remove_output_path(temp_root)
        if temporary_directory is not None:
            temporary_directory.cleanup()
        raise

    return _prepared_runtime_wrapper(
        prepared_root=final_root,
        source_schemas=source_schemas,
        target_schemas=target_schema_list,
        verify_checksums=verify_checksums,
        temporary_directory=temporary_directory,
    )
