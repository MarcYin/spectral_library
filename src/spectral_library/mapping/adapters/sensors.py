"""Sensor-definition adapters for the mapping package."""

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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
from uuid import uuid4

import duckdb
import numpy as np

from ... import _rustaccel
from ..._version import __version__

from ..engine import core as _core
from ..inputs import BandInput, SensorInput

globals().update({name: getattr(_core, name) for name in dir(_core) if not name.startswith("__")})

del _core

if TYPE_CHECKING:
    from ..inputs import BandInput as PublicBandInput
    from ..inputs import SensorInput as PublicSensorInput

def _sensor_band_segment_from_payload(payload: Mapping[str, object]) -> str:
    band_id = payload.get("band_id")
    extensions = payload.get("extensions")
    if not isinstance(extensions, Mapping):
        raise SensorSchemaError(
            "Sensor band definition must include extensions.spectral_library.segment.",
            context={"band_id": band_id},
        )
    spectral_library = extensions.get("spectral_library")
    if not isinstance(spectral_library, Mapping):
        raise SensorSchemaError(
            "Sensor band definition must include extensions.spectral_library.segment.",
            context={"band_id": band_id},
        )
    segment = _optional_string(spectral_library.get("segment"))
    if segment is None:
        raise SensorSchemaError(
            "Sensor band definition must include extensions.spectral_library.segment.",
            context={"band_id": band_id},
        )
    return segment

def _segment_from_center_wavelength(center_wavelength_nm: float) -> str:
    return "vnir" if float(center_wavelength_nm) < 1000.0 else "swir"

def _center_wavelength_from_response_definition(response_definition: object) -> float | None:
    if not isinstance(response_definition, Mapping):
        return None
    center = _optional_float(response_definition.get("center_wavelength_nm") or response_definition.get("center_nm"))
    if center is not None:
        return center
    wavelengths = response_definition.get("wavelength_nm")
    response = response_definition.get("response")
    if not isinstance(wavelengths, Sequence) or isinstance(wavelengths, (str, bytes, bytearray)):
        return None
    if not isinstance(response, Sequence) or isinstance(response, (str, bytes, bytearray)):
        return None
    wavelength_values = np.asarray(list(wavelengths), dtype=np.float64)
    response_values = np.asarray(list(response), dtype=np.float64)
    if wavelength_values.shape != response_values.shape or wavelength_values.size == 0:
        return None
    positive = response_values > 0
    if not np.any(positive):
        return float(np.mean(wavelength_values))
    positive_wavelengths = wavelength_values[positive]
    positive_weights = response_values[positive]
    weight_sum = float(np.sum(positive_weights))
    if weight_sum <= 0:
        return float(np.mean(positive_wavelengths))
    return float(np.average(positive_wavelengths, weights=positive_weights))

def _normalized_band_input(
    band_input: BandInput | Mapping[str, Any],
) -> BandInput:
    if isinstance(band_input, BandInput):
        return band_input
    if isinstance(band_input, Mapping):
        extensions = band_input.get("extensions")
        segment = _optional_string(band_input.get("segment"))
        if segment is None and isinstance(extensions, Mapping):
            spectral_library = extensions.get("spectral_library")
            if isinstance(spectral_library, Mapping):
                segment = _optional_string(spectral_library.get("segment"))
        return BandInput(
            band_id=_optional_string(band_input.get("band_id")),
            center_wavelength_nm=_optional_float(
                band_input.get("center_wavelength_nm") or band_input.get("center_nm")
            ),
            fwhm_nm=_optional_float(band_input.get("fwhm_nm")),
            response_definition=(
                dict(band_input["response_definition"])
                if isinstance(band_input.get("response_definition"), Mapping)
                else band_input.get("response_definition")
            ),
            rsrf_sensor_id=_optional_string(band_input.get("rsrf_sensor_id")),
            rsrf_band_id=_optional_string(band_input.get("rsrf_band_id")),
            rsrf_representation_variant=_optional_string(band_input.get("rsrf_representation_variant")),
            segment=segment,
        )
    raise SensorSchemaError(
        "SensorInput bands must be BandInput objects or mappings.",
        context={"band_input_type": type(band_input).__name__},
    )

def _clip_curve_to_segment(
    *,
    wavelengths: Sequence[float],
    response: Sequence[float],
    segment: str,
    support_min_nm: float | None,
    support_max_nm: float | None,
    band_id: str,
) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
    wavelengths_array = np.asarray(wavelengths, dtype=np.float64)
    response_array = np.asarray(response, dtype=np.float64)
    clipped_min, clipped_max = _rsrf_band_support_bounds(
        wavelengths_array,
        response_array,
        native_support_min_nm=support_min_nm,
        native_support_max_nm=support_max_nm,
        segment=segment,
    )
    mask = (wavelengths_array >= clipped_min) & (wavelengths_array <= clipped_max)
    clipped_wavelengths = wavelengths_array[mask]
    clipped_response = response_array[mask]
    if clipped_wavelengths.size == 0 or np.all(clipped_response <= 0):
        raise SensorSchemaError(
            "Sensor band support does not overlap its declared segment after clipping.",
            context={"band_id": band_id, "segment": segment},
        )
    return (
        tuple(float(value) for value in clipped_wavelengths),
        tuple(float(value) for value in clipped_response),
        float(clipped_min),
        float(clipped_max),
    )

def _coerce_band_input(
    band_input: BandInput | Mapping[str, Any],
    *,
    index: int,
    band_id_policy: str,
    segment_policy: str,
) -> SensorBandDefinition:
    normalized = _normalized_band_input(band_input)
    if band_id_policy not in {"preserve", "deterministic"}:
        raise SensorSchemaError(
            "SensorInput band_id_policy must be 'preserve' or 'deterministic'.",
            context={"band_id_policy": band_id_policy},
        )
    if segment_policy not in {"explicit", "center_wavelength"}:
        raise SensorSchemaError(
            "SensorInput segment_policy must be 'explicit' or 'center_wavelength'.",
            context={"segment_policy": segment_policy},
        )
    band_id = normalized.band_id if band_id_policy == "preserve" else None
    if band_id is None:
        band_id = f"band_{index + 1}"

    if normalized.rsrf_sensor_id and normalized.rsrf_band_id:
        variant = normalized.rsrf_representation_variant or RSRF_REPRESENTATION_VARIANT
        rsrf_module = _load_rsrf_module()
        rsrf_root = _resolve_rsrf_root()
        band_rows = rsrf_module.list_bands(
            normalized.rsrf_sensor_id,
            representation_variant=variant,
            root=rsrf_root,
        )
        band_row = next((row for row in band_rows if str(row["band_id"]) == normalized.rsrf_band_id), None)
        if band_row is None:
            raise SensorSchemaError(
                "Requested rsrf band could not be resolved for the custom sensor input.",
                context={"sensor_id": normalized.rsrf_sensor_id, "band_id": normalized.rsrf_band_id},
            )
        center_nm = _optional_float(band_row.get("center_wavelength_nm")) or normalized.center_wavelength_nm
        segment = normalized.segment
        if segment is None:
            if segment_policy != "center_wavelength" or center_nm is None:
                raise SensorSchemaError(
                    "Custom rsrf band inputs must declare segment or provide a center wavelength for automatic assignment.",
                    context={"band_id": band_id},
                )
            segment = _segment_from_center_wavelength(center_nm)
        selection = _RsrfBandSelection(
            band_id=band_id,
            rsrf_band_id=normalized.rsrf_band_id,
            segment=segment,
        )
        schema_band = _sensor_band_definition_from_rsrf(rsrf_module, rsrf_root, normalized.rsrf_sensor_id, selection)
        return SensorBandDefinition(
            band_id=band_id,
            segment=segment,
            wavelength_nm=schema_band.wavelength_nm,
            rsr=schema_band.rsr,
            center_nm=center_nm,
            fwhm_nm=_optional_float(band_row.get("fwhm_nm")) or normalized.fwhm_nm,
            support_min_nm=schema_band.support_min_nm,
            support_max_nm=schema_band.support_max_nm,
        )

    response_definition: object | None = None
    if normalized.response_definition is not None:
        response_definition = (
            dict(normalized.response_definition)
            if isinstance(normalized.response_definition, Mapping)
            else normalized.response_definition
        )
    elif normalized.center_wavelength_nm is not None and normalized.fwhm_nm is not None:
        response_definition = {
            "kind": "band_spec",
            "center_wavelength_nm": float(normalized.center_wavelength_nm),
            "fwhm_nm": float(normalized.fwhm_nm),
        }
    if response_definition is None:
        raise SensorSchemaError(
            "Custom band inputs must provide either response_definition, rsrf band reference, or center_wavelength_nm plus fwhm_nm.",
            context={"band_id": band_id},
        )
    resolved_center_nm = normalized.center_wavelength_nm
    if resolved_center_nm is None:
        resolved_center_nm = _center_wavelength_from_response_definition(response_definition)

    payload: dict[str, object] = {
        "band_id": band_id,
        "response_definition": response_definition,
    }
    provisional_segment = normalized.segment
    if provisional_segment is None and resolved_center_nm is not None and segment_policy == "center_wavelength":
        provisional_segment = _segment_from_center_wavelength(resolved_center_nm)
    if provisional_segment is not None:
        payload["extensions"] = {"spectral_library": {"segment": provisional_segment}}
    if resolved_center_nm is not None:
        payload["center_wavelength_nm"] = float(resolved_center_nm)
        payload["center_nm"] = float(resolved_center_nm)
    if normalized.fwhm_nm is not None:
        payload["fwhm_nm"] = float(normalized.fwhm_nm)

    ordered_wavelengths, ordered_rsr, center_nm, fwhm_nm, support_min_nm, support_max_nm = (
        _coerce_sensor_band_response_definition(payload)
    )
    segment = normalized.segment
    resolved_center_nm = resolved_center_nm if resolved_center_nm is not None else center_nm
    if segment is None:
        if segment_policy != "center_wavelength" or resolved_center_nm is None:
            raise SensorSchemaError(
                "Custom band inputs must declare segment or provide center_wavelength_nm when segment_policy is explicit.",
                context={"band_id": band_id},
            )
        segment = _segment_from_center_wavelength(resolved_center_nm)

    clipped_wavelengths, clipped_rsr, clipped_min, clipped_max = _clip_curve_to_segment(
        wavelengths=ordered_wavelengths,
        response=ordered_rsr,
        segment=segment,
        support_min_nm=support_min_nm,
        support_max_nm=support_max_nm,
        band_id=band_id,
    )
    return SensorBandDefinition(
        band_id=band_id,
        segment=segment,
        wavelength_nm=clipped_wavelengths,
        rsr=clipped_rsr,
        center_nm=resolved_center_nm,
        fwhm_nm=normalized.fwhm_nm if normalized.fwhm_nm is not None else fwhm_nm,
        support_min_nm=clipped_min,
        support_max_nm=clipped_max,
    )

def coerce_sensor_input(
    sensor_input: str | SensorInput | Mapping[str, Any] | SensorSRFSchema,
    *,
    segment_policy: str | None = None,
) -> SensorSRFSchema:
    """Normalize canonical, rsrf, or neutral custom sensor input into a schema."""

    if isinstance(sensor_input, SensorSRFSchema):
        return sensor_input
    if isinstance(sensor_input, str):
        sensor_id = sensor_input.strip()
        if not sensor_id:
            raise SensorSchemaError("Canonical sensor_id must be non-empty.")
        return _load_rsrf_sensor_schema(sensor_id)
    if isinstance(sensor_input, Mapping):
        bands = sensor_input.get("bands")
        if not isinstance(bands, Sequence) or isinstance(bands, (str, bytes, bytearray)):
            raise SensorSchemaError(
                "Mapping sensor inputs must include a bands sequence.",
                context={"sensor_input_type": type(sensor_input).__name__},
            )
        return coerce_sensor_input(
            SensorInput(
                sensor_id=_optional_string(sensor_input.get("sensor_id")),
                bands=tuple(bands),
                band_id_policy=_optional_string(sensor_input.get("band_id_policy")) or "preserve",
                segment_policy=_optional_string(sensor_input.get("segment_policy")) or "center_wavelength",
            ),
            segment_policy=segment_policy,
        )
    if not isinstance(sensor_input, SensorInput):
        raise SensorSchemaError(
            "sensor_input must be a canonical sensor id, SensorInput, rsrf sensor-definition mapping, or SensorSRFSchema.",
            context={"sensor_input_type": type(sensor_input).__name__},
        )

    resolved_segment_policy = segment_policy or sensor_input.segment_policy
    bands = tuple(
        _coerce_band_input(
            band,
            index=index,
            band_id_policy=sensor_input.band_id_policy,
            segment_policy=resolved_segment_policy,
        )
        for index, band in enumerate(sensor_input.bands)
    )
    if not bands:
        raise SensorSchemaError("SensorInput must include at least one band.")

    sensor_id = (sensor_input.sensor_id or "").strip()
    if not sensor_id:
        digest = hashlib.sha256()
        digest.update(json.dumps([band.to_dict() for band in bands], sort_keys=True).encode("utf-8"))
        sensor_id = f"custom_{digest.hexdigest()[:12]}"
    return SensorSRFSchema(sensor_id=sensor_id, bands=bands)

def _ensure_rsrf_sensor_definition_api(rsrf_module: Any, *names: str) -> None:
    missing_api = [name for name in names if not hasattr(rsrf_module, name)]
    if missing_api:
        raise SensorSchemaError(
            "rsrf does not expose the sensor-definition API required by this package.",
            context={"missing_attributes": missing_api},
        )

def _coerce_rsrf_sensor_definition_payload(payload: Mapping[str, object]) -> object:
    rsrf_module = _load_rsrf_module()
    _ensure_rsrf_sensor_definition_api(rsrf_module, "sensor_definition_from_dict")
    try:
        return rsrf_module.sensor_definition_from_dict(dict(payload))
    except Exception as exc:
        raise SensorSchemaError(
            "Sensor schema is not a valid rsrf sensor definition.",
            context={
                "sensor_id": payload.get("sensor_id"),
                "errors": list(getattr(exc, "errors", [])),
            },
        ) from exc

def _sensor_schema_from_rsrf_definition(sensor_definition: object) -> SensorSRFSchema:
    rsrf_module = _load_rsrf_module()
    _ensure_rsrf_sensor_definition_api(rsrf_module, "sensor_definition_to_dict")
    try:
        payload = rsrf_module.sensor_definition_to_dict(sensor_definition)
    except Exception as exc:
        raise SensorSchemaError(
            "rsrf could not serialize the sensor definition for spectral-library use.",
            context={"errors": list(getattr(exc, "errors", []))},
        ) from exc

    bands_payload = payload.get("bands")
    if not isinstance(bands_payload, Sequence) or isinstance(bands_payload, (str, bytes, bytearray)):
        raise SensorSchemaError(
            "rsrf returned an invalid sensor definition payload.",
            context={"sensor_id": payload.get("sensor_id")},
        )
    bands = tuple(
        SensorBandDefinition.from_dict(band_payload)
        for band_payload in bands_payload
        if isinstance(band_payload, Mapping)
    )
    return SensorSRFSchema(sensor_id=str(payload["sensor_id"]), bands=bands)

def _sensor_schema_payload_for_rsrf(schema: SensorSRFSchema) -> dict[str, object]:
    rsrf_module = _load_rsrf_module()
    _ensure_rsrf_sensor_definition_api(rsrf_module, "sensor_definition_from_dict", "sensor_definition_to_dict")
    payload = {
        "schema_type": "rsrf_sensor_definition",
        "schema_version": "1.0.0",
        "sensor_id": schema.sensor_id,
        "bands": [band.to_dict() for band in schema.bands],
        "extensions": {},
    }
    try:
        sensor_definition = rsrf_module.sensor_definition_from_dict(payload)
        return rsrf_module.sensor_definition_to_dict(sensor_definition)
    except Exception as exc:
        raise SensorSchemaError(
            "Sensor schema could not be serialized through rsrf.",
            context={"sensor_id": schema.sensor_id, "errors": list(getattr(exc, "errors", []))},
        ) from exc

def _coerce_sensor_band_response_definition(
    payload: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...], float | None, float | None, float | None, float | None]:
    rsrf_module = _load_rsrf_module()
    missing_api = [
        name
        for name in ("coerce_response_definition", "realize_curve")
        if not hasattr(rsrf_module, name)
    ]
    if missing_api:
        raise SensorSchemaError(
            "rsrf does not expose the custom response-definition API required for sensor schema loading.",
            context={"missing_attributes": missing_api},
        )
    response_definition = _response_definition_input_from_payload(payload)

    try:
        resolved_definition = rsrf_module.coerce_response_definition(
            response_definition,
            band_id=str(payload["band_id"]),
            source_variant=_optional_string(payload.get("source_variant")) or "custom",
        )
    except (TypeError, ValueError) as exc:
        raise SensorSchemaError(
            "Sensor band definition could not be normalized through rsrf.",
            context={"band_id": payload.get("band_id")},
        ) from exc

    resolved_support_min_nm: float | None = None
    resolved_support_max_nm: float | None = None

    if hasattr(resolved_definition, "wavelength_nm") and hasattr(resolved_definition, "response"):
        wavelengths = np.asarray(resolved_definition.wavelength_nm, dtype=np.float64)
        response = np.asarray(resolved_definition.response, dtype=np.float64)
        center_nm = _optional_float(payload.get("center_nm")) or _optional_float(payload.get("center_wavelength_nm"))
        fwhm_nm = _optional_float(payload.get("fwhm_nm"))
    else:
        try:
            realized_curve = rsrf_module.realize_curve(resolved_definition)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise SensorSchemaError(
                "Sensor band definition could not be realized through rsrf.",
                context={"band_id": payload.get("band_id")},
            ) from exc
        wavelengths = np.asarray(realized_curve.wavelength_nm, dtype=np.float64)
        response = np.asarray(realized_curve.response, dtype=np.float64)
        center_nm = _optional_float(payload.get("center_nm"))
        if center_nm is None:
            center_nm = float(resolved_definition.center_wavelength_nm)
        fwhm_nm = _optional_float(payload.get("fwhm_nm"))
        if fwhm_nm is None:
            fwhm_nm = float(resolved_definition.fwhm_nm)
        segment = _sensor_band_segment_from_payload(payload)
        resolved_support_min_nm, resolved_support_max_nm = _rsrf_band_support_bounds(
            wavelengths,
            response,
            native_support_min_nm=_optional_float(payload.get("support_min_nm")),
            native_support_max_nm=_optional_float(payload.get("support_max_nm")),
            segment=segment,
        )
        mask = (wavelengths >= resolved_support_min_nm) & (wavelengths <= resolved_support_max_nm)
        wavelengths = wavelengths[mask]
        response = response[mask]

    ordered_pairs = sorted(zip(wavelengths.tolist(), response.tolist()), key=lambda item: item[0])
    ordered_wavelengths = tuple(float(wavelength) for wavelength, _ in ordered_pairs)
    ordered_rsr = tuple(float(weight) for _, weight in ordered_pairs)
    positive_support = [wavelength for wavelength, weight in ordered_pairs if weight > 0]

    support_min_nm = resolved_support_min_nm if resolved_support_min_nm is not None else _optional_float(payload.get("support_min_nm"))
    if support_min_nm is None and positive_support:
        support_min_nm = float(min(positive_support))

    support_max_nm = resolved_support_max_nm if resolved_support_max_nm is not None else _optional_float(payload.get("support_max_nm"))
    if support_max_nm is None and positive_support:
        support_max_nm = float(max(positive_support))

    return ordered_wavelengths, ordered_rsr, center_nm, fwhm_nm, support_min_nm, support_max_nm

def _response_definition_input_from_payload(payload: Mapping[str, object]) -> object:
    response_definition = payload["response_definition"]
    if isinstance(response_definition, Mapping):
        return dict(response_definition)
    return response_definition

def _load_rsrf_module() -> Any:
    try:
        import rsrf  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SensorSchemaError(
            "rsrf is required to resolve built-in sensor schemas and normalize custom sensor definitions. "
            f"Install it with `{RSRF_INSTALL_HINT}`.",
        ) from exc
    return rsrf

def _candidate_rsrf_roots(rsrf_module: Any) -> tuple[Path, ...]:
    candidates: list[Path] = []
    env_value = (os.environ.get(RSRF_ROOT_ENV_VAR) or "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())

    package_root = Path(getattr(rsrf_module, "PACKAGE_ROOT", Path(rsrf_module.__file__).resolve().parent)).resolve()
    candidates.extend((package_root.parent.parent, package_root.parent))

    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique_candidates:
            unique_candidates.append(resolved)
    return tuple(unique_candidates)

def _resolve_rsrf_root() -> Path:
    rsrf_module = _load_rsrf_module()
    for candidate in _candidate_rsrf_roots(rsrf_module):
        if (candidate / RSRF_REGISTRY_RELATIVE_PATH).exists():
            return candidate
    raise SensorSchemaError(
        "rsrf is installed but its registry data could not be located.",
        context={
            "searched_roots": [str(candidate) for candidate in _candidate_rsrf_roots(rsrf_module)],
            "env_var": RSRF_ROOT_ENV_VAR,
        },
    )

def _rsrf_band_support_bounds(
    curve_wavelengths: np.ndarray,
    curve_response: np.ndarray,
    *,
    native_support_min_nm: float | None,
    native_support_max_nm: float | None,
    segment: str,
) -> tuple[float, float]:
    positive_mask = np.asarray(curve_response, dtype=np.float64) > 0
    if not np.any(positive_mask):
        raise SensorSchemaError("rsrf curve has no positive support.")

    support_min = float(curve_wavelengths[positive_mask].min()) if native_support_min_nm is None else float(native_support_min_nm)
    support_max = float(curve_wavelengths[positive_mask].max()) if native_support_max_nm is None else float(native_support_max_nm)
    segment_min, segment_max = SEGMENT_RANGES[segment]
    clipped_min = max(float(CANONICAL_START_NM), float(segment_min), support_min)
    clipped_max = min(float(CANONICAL_END_NM), float(segment_max), support_max)
    if clipped_min > clipped_max:
        raise SensorSchemaError(
            "rsrf curve support does not overlap the canonical segment range.",
            context={
                "segment": segment,
                "support_min_nm": support_min,
                "support_max_nm": support_max,
            },
        )
    return clipped_min, clipped_max

def _sensor_band_definition_from_rsrf(
    rsrf_module: Any,
    rsrf_root: Path,
    sensor_id: str,
    selection: _RsrfBandSelection,
) -> SensorBandDefinition:
    band_rows = rsrf_module.list_bands(
        sensor_id,
        representation_variant=RSRF_REPRESENTATION_VARIANT,
        root=rsrf_root,
    )
    band_row = next((row for row in band_rows if str(row["band_id"]) == selection.rsrf_band_id), None)
    if band_row is None:
        raise SensorSchemaError(
            "rsrf does not provide the requested band for the selected sensor.",
            context={"sensor_id": sensor_id, "band_id": selection.rsrf_band_id},
        )

    curve = rsrf_module.load_curve(
        sensor_id,
        selection.rsrf_band_id,
        representation_variant=RSRF_REPRESENTATION_VARIANT,
        root=rsrf_root,
    )
    wavelengths = np.asarray(curve.wavelength_nm, dtype=np.float64)
    response = np.asarray(curve.response, dtype=np.float64)
    support_min_nm, support_max_nm = _rsrf_band_support_bounds(
        wavelengths,
        response,
        native_support_min_nm=_optional_float(band_row.get("native_support_min_nm")),
        native_support_max_nm=_optional_float(band_row.get("native_support_max_nm")),
        segment=selection.segment,
    )
    mask = (wavelengths >= support_min_nm) & (wavelengths <= support_max_nm)
    return SensorBandDefinition(
        band_id=selection.band_id,
        segment=selection.segment,
        wavelength_nm=tuple(float(value) for value in wavelengths[mask]),
        rsr=tuple(float(value) for value in response[mask]),
        center_nm=_optional_float(band_row.get("center_wavelength_nm")),
        fwhm_nm=_optional_float(band_row.get("fwhm_nm")),
        support_min_nm=float(support_min_nm),
        support_max_nm=float(support_max_nm),
    )

def _load_rsrf_sensor_schema(sensor_id: str) -> SensorSRFSchema:
    rsrf_module = _load_rsrf_module()
    rsrf_root = _resolve_rsrf_root()
    if sensor_id in RSRF_SENSOR_BAND_SELECTIONS:
        bands = tuple(
            _sensor_band_definition_from_rsrf(rsrf_module, rsrf_root, sensor_id, selection)
            for selection in RSRF_SENSOR_BAND_SELECTIONS[sensor_id]
        )
        return SensorSRFSchema(sensor_id=sensor_id, bands=bands)

    _ensure_rsrf_sensor_definition_api(rsrf_module, "get_sensor_definition")
    try:
        sensor_definition = rsrf_module.get_sensor_definition(
            sensor_id,
            representation_variant=RSRF_REPRESENTATION_VARIANT,
            root=rsrf_root,
        )
    except Exception as exc:
        raise SensorSchemaError(
            "sensor_id could not be resolved through rsrf sensor definitions.",
            context={"sensor_id": sensor_id, "errors": list(getattr(exc, "errors", []))},
        ) from exc
    return _sensor_schema_from_rsrf_definition(sensor_definition)

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

def _simulate_source_retrieval_matrix_from_segments(
    hyperspectral_vnir: np.ndarray,
    hyperspectral_swir: np.ndarray,
    bands: Sequence[SensorBandDefinition],
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Simulate source retrieval inputs from prepared VNIR/SWIR arrays."""

    matrix = np.empty((hyperspectral_vnir.shape[0], len(bands)), dtype=dtype)
    for index, band in enumerate(bands):
        if band.segment == "vnir":
            source_rows = hyperspectral_vnir
        elif band.segment == "swir":
            source_rows = hyperspectral_swir
        else:
            raise SensorSchemaError("Unknown sensor segment.", context={"segment": band.segment, "band_id": band.band_id})
        response = _resample_band_response(band, segment_only=True)
        matrix[:, index] = _response_weighted_average(
            source_rows,
            response,
            error_message="Resampled SRF support must remain positive.",
            error_context={"band_id": band.band_id, "segment": band.segment},
        ).astype(dtype, copy=False)
    return matrix

def _load_prepared_sensor_schema_payload(
    path: Path,
    *,
    expected_schema_version: str,
) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
    payload = _read_json_document(
        path,
        error_factory=PreparedLibraryValidationError,
        document_name=path.name,
    )
    if not isinstance(payload, dict):
        raise PreparedLibraryValidationError(
            "sensor_schema.json must be a JSON object.",
            context={"path": str(path)},
        )

    schema_version = _optional_string(payload.get("schema_version"))
    if not schema_version:
        raise PreparedLibraryValidationError(
            "sensor_schema.json is missing schema_version.",
            context={"path": str(path)},
        )
    if schema_version != expected_schema_version:
        raise PreparedLibraryValidationError(
            "sensor_schema.json schema_version does not match manifest.json.",
            context={
                "path": str(path),
                "sensor_schema_version": schema_version,
                "manifest_schema_version": expected_schema_version,
            },
        )

    canonical_grid = payload.get("canonical_wavelength_grid")
    if not isinstance(canonical_grid, Mapping):
        raise PreparedLibraryValidationError(
            "sensor_schema.json is missing canonical_wavelength_grid.",
            context={"path": str(path)},
        )
    expected_grid = {
        "start_nm": CANONICAL_START_NM,
        "end_nm": CANONICAL_END_NM,
        "step_nm": 1,
    }
    actual_grid = {
        "start_nm": canonical_grid.get("start_nm"),
        "end_nm": canonical_grid.get("end_nm"),
        "step_nm": canonical_grid.get("step_nm"),
    }
    if actual_grid != expected_grid:
        raise PreparedLibraryValidationError(
            "sensor_schema.json canonical_wavelength_grid is incompatible with this package.",
            context={
                "path": str(path),
                "expected_canonical_wavelength_grid": expected_grid,
                "actual_canonical_wavelength_grid": actual_grid,
            },
        )

    sensors = payload.get("sensors")
    if not isinstance(sensors, list):
        raise PreparedLibraryValidationError(
            "sensor_schema.json sensors entry must be a list.",
            context={"path": str(path)},
        )
    if not all(isinstance(sensor_payload, dict) for sensor_payload in sensors):
        raise PreparedLibraryValidationError(
            "sensor_schema.json sensors entries must be JSON objects.",
            context={"path": str(path)},
        )
    sensor_payloads = [sensor_payload for sensor_payload in sensors if isinstance(sensor_payload, dict)]
    return payload, sensor_payloads

def _prepared_runtime_sensor_payload(schema: SensorSRFSchema) -> dict[str, object]:
    return schema.to_dict()

def load_sensor_schemas(
    srf_root: Path | None,
    *,
    required_sensor_ids: Sequence[str] | None = None,
) -> dict[str, SensorSRFSchema]:
    """Load sensor schemas from rsrf sensor-definition JSON files and built-ins."""

    schemas: dict[str, SensorSRFSchema] = {}
    resolved_srf_root = None if srf_root is None else Path(srf_root)
    json_paths: list[Path] = []
    rsrf_module = _load_rsrf_module()
    _ensure_rsrf_sensor_definition_api(rsrf_module, "load_sensor_definition")
    if resolved_srf_root is not None:
        if resolved_srf_root.exists():
            json_paths = sorted(path for path in resolved_srf_root.glob("*.json") if path.is_file())
        elif required_sensor_ids is None:
            raise SensorSchemaError("SRF root does not exist.", context={"srf_root": str(resolved_srf_root)})

    for path in json_paths:
        try:
            sensor_definition = rsrf_module.load_sensor_definition(path)
            schema = _sensor_schema_from_rsrf_definition(sensor_definition)
        except Exception as exc:
            raise SensorSchemaError(
                "Custom sensor SRF JSON must be a valid rsrf sensor definition.",
                context={"path": str(path), "errors": list(getattr(exc, "errors", []))},
            ) from exc
        if schema.sensor_id in schemas:
            raise SensorSchemaError(
                "Duplicate sensor_id encountered while loading SRF definitions.",
                context={"sensor_id": schema.sensor_id, "path": str(path)},
            )
        schemas[schema.sensor_id] = schema

    if required_sensor_ids is None:
        if not schemas:
            raise SensorSchemaError(
                "No sensor schema JSON files were found.",
                context={"srf_root": str(resolved_srf_root) if resolved_srf_root is not None else None},
            )
        return schemas

    for sensor_id in required_sensor_ids:
        if sensor_id in schemas:
            continue
        try:
            schemas[sensor_id] = _load_rsrf_sensor_schema(sensor_id)
        except SensorSchemaError:
            raise SensorSchemaError(
                "Requested source sensors could not be resolved.",
                context={
                    "missing_source_sensors": [sensor_id],
                    "srf_root": str(resolved_srf_root) if resolved_srf_root is not None else None,
                },
            ) from None

    if not schemas:
        raise SensorSchemaError("No sensor schemas could be resolved.")
    return schemas
