"""Public neutral input models for in-memory mapping runtime construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BandInput:
    """Neutral custom band input accepted by ``coerce_sensor_input``."""

    band_id: str | None = None
    center_wavelength_nm: float | None = None
    fwhm_nm: float | None = None
    response_definition: Mapping[str, Any] | None = None
    rsrf_sensor_id: str | None = None
    rsrf_band_id: str | None = None
    rsrf_representation_variant: str | None = None
    segment: str | None = None


@dataclass(frozen=True)
class SensorInput:
    """Neutral sensor input accepted by ``build_mapping_runtime``."""

    sensor_id: str | None = None
    bands: Sequence[BandInput | Mapping[str, Any]] = field(default_factory=tuple)
    band_id_policy: str = "preserve"
    segment_policy: str = "center_wavelength"


@dataclass(frozen=True)
class HyperspectralLibraryInput:
    """Neutral in-memory hyperspectral library input."""

    wavelengths_nm: Sequence[float]
    spectra: Sequence[Sequence[float]]
    sample_ids: Sequence[str]
    metadata_rows: Sequence[Mapping[str, object]] | None = None
    provenance_metadata: Sequence[Mapping[str, object]] | None = None
    source_id: str = "in_memory"
