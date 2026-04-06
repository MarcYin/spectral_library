"""Normalization and packaged-library workflows."""

from .coverage import filter_normalized_by_coverage
from .package import build_library_package
from .pipeline import SpectrumRecord, TARGET_WAVELENGTHS, normalize_sources
from .plots import EXPECTED_GRID_POINTS, generate_quality_plots

__all__ = [
    "EXPECTED_GRID_POINTS",
    "SpectrumRecord",
    "TARGET_WAVELENGTHS",
    "build_library_package",
    "filter_normalized_by_coverage",
    "generate_quality_plots",
    "normalize_sources",
]
