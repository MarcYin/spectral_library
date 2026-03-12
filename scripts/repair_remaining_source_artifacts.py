#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from whitsmooth_rust import robust_whittaker_irls_f64

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sample_processed_spectra_review import (
    CHUNK_SIZE,
    SECOND_ABSORPTION_END,
    SECOND_ABSORPTION_PAD,
    SECOND_ABSORPTION_START,
    SECOND_ABSORPTION_THRESHOLD,
    TAIL_DRIFT_THRESHOLD as REVIEW_TAIL_DRIFT_THRESHOLD,
    TAIL_EVAL_END,
    TAIL_EVAL_START,
    TAIL_FIT_END,
    TAIL_FIT_START,
    add_source_residual_thresholds,
    detect_flagged_bands,
    rowwise_abs_nanmax,
    rowwise_local_interp_deviation,
    rowwise_tail_extrapolation_drift,
)


WAVELENGTHS = np.arange(400, 2501, dtype=int)
REFLECTANCE_BOUNDS = (-0.05, 1.05)
CLIP_BOUNDS = (0.0, 1.0)

UNDERSTORY_LEFT_START = 2250
UNDERSTORY_LEFT_END = 2295
UNDERSTORY_REPLACE_START = 2300
UNDERSTORY_REPLACE_END = 2500
UNDERSTORY_BLEND_HALF_WINDOW = 20
UNDERSTORY_MAX_ABS_SLOPE = 5e-4
UNDERSTORY_1400_WINDOW = (1368, 1425)
UNDERSTORY_1400_LEFT = (1330, 1360)
UNDERSTORY_1400_RIGHT = (1435, 1465)
UNDERSTORY_1900_WINDOW = (1830, 2025)
UNDERSTORY_1900_LEFT = (1790, 1820)
UNDERSTORY_1900_RIGHT = (2030, 2060)
UNDERSTORY_WATER_BLEND_HALF_WINDOW = 15
UNDERSTORY_WATER_JUMP_THRESHOLD = 0.03

USGS_BLEND_HALF_WINDOW = 5
USGS_SHOULDER_POINTS = 16
USGS_VISIBLE_REPAIR_THRESHOLD = 0.05
USGS_VISIBLE_REPAIR_START = 400
USGS_VISIBLE_REPAIR_END = 900
USGS_VISIBLE_BLEND_HALF_WINDOW = 8
USGS_VISIBLE_SHOULDER_GAP = 4
USGS_VISIBLE_SHOULDER_WIDTH = 12

TAIL_BLEND_HALF_WINDOW = 20
TAIL_REPLACE_START = 2380
TAIL_REPLACE_END = 2500
TAIL_LEFT_START = 2300
TAIL_LEFT_END = 2370
TAIL_JUMP_THRESHOLD = 0.03
TAIL_DRIFT_THRESHOLD = 0.05

ROBUST_WEIGHT_THRESHOLD = 0.10
UNDERSTORY_ROBUST_LAM = 2_000.0
UNDERSTORY_ROBUST_ABS_THRESHOLD = 0.01
BRANCH_TAIL_LAM = 10_000.0
BSSL_TAIL_LAM = 12_000.0
SNOW_TAIL_LAM = 12_000.0
NGEE_TAIL_LAM = 8_000.0
HYSPIRI_TAIL_LAM = 8_000.0
TAIL_BLEND_EXTEND_START = 2280
TAIL_DROP_AFTER_2450 = 2450
SANTA_ABSORPTION2_WINDOW = (1850, 1950)
SANTA_ABSORPTION2_LEFT = (1800, 1840)
SANTA_ABSORPTION2_RIGHT = (1960, 2000)
SANTA_STRONG_SMOOTH_LAM = 50_000.0
SANTA_SMOOTH_CENTER_NM = 800
SANTA_SMOOTH_HALF_WINDOW = 50
SISPEC_ABSORPTION2_WINDOW = (1830, 1930)
SISPEC_ABSORPTION2_LEFT = (1780, 1820)
SISPEC_ABSORPTION2_RIGHT = (1940, 1980)

REPAIR_CANDIDATE_SOURCES = {
    "understory_estonia_czech",
    "bssl",
    "branch_tree_spectra_boreal_temperate",
    "hyspiri_ground_targets",
    "natural_snow_twigs",
    "ngee_arctic_2018",
    "santa_barbara_urban_reflectance",
    "sispec",
    "emit_adjusted_vegetation",
    "emit_l2a_surface",
}


def copy_static_inputs(base_root: Path, output_root: Path) -> None:
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("landcover_analysis/landcover_labels.csv"),
        Path("manifests/sources.csv"),
    ]:
        source_path = base_root / relative
        if not source_path.exists():
            continue
        target_path = output_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _compute_repair_metric_lookup(base_root: Path) -> dict[str, dict[str, float]]:
    spectra_path = base_root / "tabular" / "normalized_spectra.csv"
    header = pd.read_csv(spectra_path, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)
    usecols = ["source_id", "spectrum_id", "sample_name"] + spectral_columns
    metrics_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(spectra_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk[chunk["source_id"].isin(REPAIR_CANDIDATE_SOURCES)].copy()
        if chunk.empty:
            continue
        values = chunk[spectral_columns].to_numpy(dtype=float)
        flagged_all, smoothed = detect_flagged_bands(values, wavelengths)
        adjacent = np.abs(values[:, 1:] - values[:, :-1])
        visible_mask = (wavelengths[:-1] >= 400) & (wavelengths[:-1] < 700)
        swir_tail_mask = wavelengths[:-1] >= 2300
        residual = np.abs(values - smoothed)
        residual[~np.isfinite(values)] = np.nan
        metrics = chunk[["source_id", "spectrum_id", "sample_name"]].copy()
        metrics["flagged_band_count"] = flagged_all.sum(axis=1)
        metrics["max_abs_jump"] = rowwise_abs_nanmax(adjacent)
        metrics["max_abs_jump_visible"] = rowwise_abs_nanmax(adjacent[:, visible_mask])
        metrics["max_abs_jump_tail_2300"] = rowwise_abs_nanmax(adjacent[:, swir_tail_mask])
        metrics["max_abs_residual"] = rowwise_abs_nanmax(residual)
        metrics["max_interp_spike_absorption2"] = rowwise_local_interp_deviation(
            values,
            wavelengths,
            SECOND_ABSORPTION_START,
            SECOND_ABSORPTION_END,
            SECOND_ABSORPTION_PAD,
        )
        metrics["tail_end_drift_2400"] = rowwise_tail_extrapolation_drift(
            values,
            wavelengths,
            TAIL_FIT_START,
            TAIL_FIT_END,
            TAIL_EVAL_START,
            TAIL_EVAL_END,
        )
        metrics["mean_abs_residual"] = pd.DataFrame(residual).mean(axis=1, skipna=True).to_numpy(dtype=float)
        metrics["min_reflectance"] = pd.DataFrame(values).min(axis=1, skipna=True).to_numpy(dtype=float)
        metrics["max_reflectance"] = pd.DataFrame(values).max(axis=1, skipna=True).to_numpy(dtype=float)
        metrics["out_of_range_band_count"] = ((values < -0.05) | (values > 1.05)).sum(axis=1)
        metrics_frames.append(metrics)

    if not metrics_frames:
        return {}

    metrics = pd.concat(metrics_frames, ignore_index=True)
    metrics = add_source_residual_thresholds(metrics)
    emit_mask = metrics["source_id"].astype(str).str.startswith("emit_")
    santa_mask = metrics["source_id"].astype(str).eq("santa_barbara_urban_reflectance")
    metrics["is_suspicious"] = (
        (metrics["out_of_range_band_count"] > 0)
        | (metrics["max_abs_jump"] >= 0.15)
        | (metrics["max_abs_jump_visible"] >= 0.05)
        | (metrics["max_abs_jump_tail_2300"].fillna(0.0) >= 0.05)
        | (emit_mask & (metrics["max_interp_spike_absorption2"].fillna(0.0) >= SECOND_ABSORPTION_THRESHOLD))
        | (santa_mask & (metrics["tail_end_drift_2400"].fillna(0.0) >= REVIEW_TAIL_DRIFT_THRESHOLD))
        | (metrics["max_abs_residual"] >= metrics["source_residual_threshold"])
        | (
            (metrics["flagged_band_count"] >= metrics["source_flagged_band_threshold"])
            & (metrics["max_abs_residual"] >= 0.05)
        )
    )
    records = metrics.to_dict(orient="records")
    return {f"{row['source_id']}||{row['spectrum_id']}": row for row in records}


def _metric_ge(metrics: dict[str, float] | None, key: str, threshold: float) -> bool:
    if not metrics:
        return False
    value = metrics.get(key)
    if value is None or pd.isna(value):
        return False
    return float(value) >= threshold


def _metric_gt(metrics: dict[str, float] | None, key: str, threshold: float) -> bool:
    if not metrics:
        return False
    value = metrics.get(key)
    if value is None or pd.isna(value):
        return False
    return float(value) > threshold


def _should_attempt_repair(source_id: str, metrics: dict[str, float] | None) -> bool:
    if source_id == "santa_barbara_urban_reflectance":
        return True
    if metrics is None:
        return False
    if not bool(metrics.get("is_suspicious", False)):
        return False
    if source_id == "understory_estonia_czech":
        return _metric_ge(metrics, "flagged_band_count", float(metrics.get("source_flagged_band_threshold", 250.0))) and _metric_ge(
            metrics, "max_abs_residual", max(0.10, float(metrics.get("source_residual_threshold", 0.10)))
        )
    if source_id == "bssl":
        return (
            _metric_ge(metrics, "max_abs_jump_tail_2300", 0.05)
            or _metric_ge(metrics, "tail_end_drift_2400", 0.06)
            or _metric_gt(metrics, "out_of_range_band_count", 0.0)
        )
    if source_id == "branch_tree_spectra_boreal_temperate":
        return (
            _metric_ge(metrics, "max_abs_jump_tail_2300", 0.05)
            or _metric_ge(metrics, "tail_end_drift_2400", 0.07)
            or _metric_gt(metrics, "out_of_range_band_count", 0.0)
        )
    if source_id == "hyspiri_ground_targets":
        return (
            _metric_ge(metrics, "max_abs_jump_tail_2300", 0.05)
            or _metric_ge(metrics, "tail_end_drift_2400", 0.12)
            or _metric_gt(metrics, "out_of_range_band_count", 0.0)
        )
    if source_id == "natural_snow_twigs":
        return (
            _metric_ge(metrics, "max_abs_jump_tail_2300", 0.045)
            or _metric_ge(metrics, "tail_end_drift_2400", 0.05)
            or _metric_gt(metrics, "out_of_range_band_count", 0.0)
        )
    if source_id == "ngee_arctic_2018":
        return (
            _metric_ge(metrics, "max_abs_jump_tail_2300", 0.05)
            or _metric_ge(metrics, "tail_end_drift_2400", 0.05)
            or _metric_gt(metrics, "out_of_range_band_count", 0.0)
        )
    if source_id == "sispec":
        return _metric_ge(metrics, "max_interp_spike_absorption2", SECOND_ABSORPTION_THRESHOLD)
    if source_id in {"emit_adjusted_vegetation", "emit_l2a_surface"}:
        return _metric_ge(metrics, "max_abs_jump_visible", 0.05)
    return False


def _window_mask(start_nm: int, end_nm: int) -> np.ndarray:
    return (WAVELENGTHS >= start_nm) & (WAVELENGTHS <= end_nm)


def _parse_value(raw: str) -> float:
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def _fill_nan_linear(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return values
    x = np.arange(len(values), dtype=float)
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return values


def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1])) for group in groups if group.size > 0]


def _in_bounds(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= REFLECTANCE_BOUNDS[0]) & (values <= REFLECTANCE_BOUNDS[1])


def _robust_center(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmedian(finite))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    finite = np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    slope, intercept = np.polyfit(x[finite].astype(float), y[finite].astype(float), deg=1)
    return float(slope), float(intercept)


def _robust_smooth_segment(
    values: np.ndarray,
    *,
    start_nm: int,
    end_nm: int,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = _window_mask(start_nm, end_nm)
    segment = values[mask].astype(float, copy=True)
    filled = _fill_nan_linear(segment)
    x = WAVELENGTHS[mask].astype(np.float64)
    if np.count_nonzero(np.isfinite(filled)) < 4:
        return mask, segment, np.full_like(segment, np.nan), np.full_like(segment, np.nan)
    smoothed, weights = robust_whittaker_irls_f64(
        x,
        filled[np.newaxis, :],
        lam=lam,
        d=2,
        iterations=8,
        weighting="tukey",
        scale="mad",
        parallel=False,
        return_weights=True,
        merge_x_tol=0.0,
    )
    return mask, segment, np.asarray(smoothed[0], dtype=float), np.asarray(weights[0], dtype=float)


def _blend_weights(wavelengths: np.ndarray, start_nm: int, end_nm: int, half_window_nm: int) -> np.ndarray:
    weights = np.zeros(len(wavelengths), dtype=float)
    left_outer = start_nm - half_window_nm
    left_inner = start_nm + half_window_nm
    right_inner = end_nm - half_window_nm
    right_outer = end_nm + half_window_nm

    left_ramp = (wavelengths >= left_outer) & (wavelengths < left_inner)
    if left_inner > left_outer:
        weights[left_ramp] = (wavelengths[left_ramp] - left_outer) / float(left_inner - left_outer)

    middle = (wavelengths >= left_inner) & (wavelengths <= right_inner)
    weights[middle] = 1.0

    right_ramp = (wavelengths > right_inner) & (wavelengths <= right_outer)
    if right_outer > right_inner:
        weights[right_ramp] = (right_outer - wavelengths[right_ramp]) / float(right_outer - right_inner)

    return np.clip(weights, 0.0, 1.0)


def _transition_to_smoothed_weights(wavelengths: np.ndarray, center_nm: int, half_window_nm: int) -> np.ndarray:
    start_nm = center_nm - half_window_nm
    end_nm = center_nm + half_window_nm
    weights = np.zeros(len(wavelengths), dtype=float)
    ramp = (wavelengths > start_nm) & (wavelengths < end_nm)
    if end_nm > start_nm:
        weights[ramp] = (wavelengths[ramp] - start_nm) / float(end_nm - start_nm)
    weights[wavelengths >= end_nm] = 1.0
    return np.clip(weights, 0.0, 1.0)

def _repair_understory_tail_mid(values: np.ndarray) -> tuple[np.ndarray, dict[str, float] | None]:
    replace_mask = _window_mask(UNDERSTORY_REPLACE_START, UNDERSTORY_REPLACE_END)
    left_mask = _window_mask(UNDERSTORY_LEFT_START, UNDERSTORY_LEFT_END)
    blend_mask = _window_mask(
        UNDERSTORY_REPLACE_START - UNDERSTORY_BLEND_HALF_WINDOW,
        min(2500, UNDERSTORY_REPLACE_END + UNDERSTORY_BLEND_HALF_WINDOW),
    )

    left_values = values[left_mask]
    replace_values = values[replace_mask]
    left_valid = _in_bounds(left_values)
    if int(left_valid.sum()) < 8:
        return values, None

    early_mask = _window_mask(UNDERSTORY_LEFT_START, 2270)
    late_mask = _window_mask(2275, UNDERSTORY_LEFT_END)
    early_values = values[early_mask]
    late_values = values[late_mask]
    early_valid = _in_bounds(early_values)
    late_valid = _in_bounds(late_values)
    if int(early_valid.sum()) < 4 or int(late_valid.sum()) < 4:
        return values, None

    x0 = float(np.nanmean(WAVELENGTHS[early_mask][early_valid]))
    y0 = _robust_center(early_values[early_valid])
    x1 = float(np.nanmean(WAVELENGTHS[late_mask][late_valid]))
    y1 = _robust_center(late_values[late_valid])
    slope = (y1 - y0) / max(1.0, x1 - x0)
    slope = float(np.clip(slope, -UNDERSTORY_MAX_ABS_SLOPE, UNDERSTORY_MAX_ABS_SLOPE))
    anchor_x = x1
    anchor_y = y1
    predicted_center = anchor_y + slope * (WAVELENGTHS[replace_mask].astype(float) - anchor_x)
    deviation = np.abs(replace_values - predicted_center)
    adjacent_window = values[_window_mask(2330, 2500)]
    adjacent_jump = float(np.nanmax(np.abs(np.diff(adjacent_window)))) if adjacent_window.size > 1 else float("nan")
    out_of_range_count = int(np.count_nonzero(~_in_bounds(replace_values)))
    max_deviation = float(np.nanmax(deviation)) if np.isfinite(deviation).any() else float("nan")
    should = bool(out_of_range_count > 0 or adjacent_jump > 0.05 or max_deviation > 0.08)
    if not should:
        return values, None

    repaired = values.copy()
    predicted_blend = anchor_y + slope * (WAVELENGTHS[blend_mask].astype(float) - anchor_x)
    predicted_blend = np.clip(predicted_blend, 0.0, 1.0)
    weights = _blend_weights(
        WAVELENGTHS[blend_mask],
        UNDERSTORY_REPLACE_START,
        UNDERSTORY_REPLACE_END,
        UNDERSTORY_BLEND_HALF_WINDOW,
    )
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[replace_mask] = np.clip(predicted_center, 0.0, 1.0)
    diagnostics = {
        "repair_type": "understory_tail_2340_2500_constrained",
        "replace_start_nm": float(UNDERSTORY_REPLACE_START),
        "replace_end_nm": float(UNDERSTORY_REPLACE_END),
        "out_of_range_count": float(out_of_range_count),
        "max_abs_deviation": max_deviation,
        "max_abs_jump": adjacent_jump,
    }
    return repaired, diagnostics


def _iter_bad_segments(values: np.ndarray) -> list[tuple[int, int]]:
    bad = np.isfinite(values) & ((values < REFLECTANCE_BOUNDS[0]) | (values > REFLECTANCE_BOUNDS[1]))
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_bad in enumerate(bad):
        if is_bad and start is None:
            start = index
        elif (not is_bad) and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(values) - 1))
    return segments


def _repair_usgs_segments(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = values.copy()
    diagnostics_rows: list[dict[str, float]] = []
    for start_idx, end_idx in _iter_bad_segments(repaired):
        left_candidates = np.arange(max(0, start_idx - USGS_SHOULDER_POINTS), start_idx)
        right_candidates = np.arange(end_idx + 1, min(len(repaired), end_idx + 1 + USGS_SHOULDER_POINTS))
        left_good = left_candidates[_in_bounds(repaired[left_candidates])]
        right_good = right_candidates[_in_bounds(repaired[right_candidates])]

        repair_idx = np.arange(start_idx, end_idx + 1)
        wavelengths = WAVELENGTHS[repair_idx].astype(float)

        if left_good.size >= 2 and right_good.size >= 2:
            x0 = float(np.nanmean(WAVELENGTHS[left_good]))
            y0 = _robust_center(repaired[left_good])
            x1 = float(np.nanmean(WAVELENGTHS[right_good]))
            y1 = _robust_center(repaired[right_good])
            predicted = np.interp(wavelengths, [x0, x1], [y0, y1])
        elif left_good.size >= 2:
            fit = _fit_line(WAVELENGTHS[left_good].astype(float), repaired[left_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        elif right_good.size >= 2:
            fit = _fit_line(WAVELENGTHS[right_good].astype(float), repaired[right_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        else:
            continue

        predicted = np.clip(predicted, 0.0, 1.0)
        blend_start = max(0, start_idx - USGS_BLEND_HALF_WINDOW)
        blend_end = min(len(repaired) - 1, end_idx + USGS_BLEND_HALF_WINDOW)
        blend_idx = np.arange(blend_start, blend_end + 1)
        if left_good.size >= 1 and right_good.size >= 1:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(np.nanmean(WAVELENGTHS[left_good])), float(np.nanmean(WAVELENGTHS[right_good]))],
                [_robust_center(repaired[left_good]), _robust_center(repaired[right_good])],
            )
        else:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(WAVELENGTHS[repair_idx[0]]), float(WAVELENGTHS[repair_idx[-1]])],
                [float(predicted[0]), float(predicted[-1])],
            )
        predicted_blend = np.clip(predicted_blend, 0.0, 1.0)
        weights = _blend_weights(
            WAVELENGTHS[blend_idx],
            int(WAVELENGTHS[start_idx]),
            int(WAVELENGTHS[end_idx]),
            USGS_BLEND_HALF_WINDOW,
        )
        repaired[blend_idx] = (1.0 - weights) * repaired[blend_idx] + weights * predicted_blend
        repaired[repair_idx] = predicted
        diagnostics_rows.append(
            {
                "repair_type": "usgs_out_of_range_segment_interp",
                "replace_start_nm": float(WAVELENGTHS[start_idx]),
                "replace_end_nm": float(WAVELENGTHS[end_idx]),
                "out_of_range_count": float(end_idx - start_idx + 1),
                "segment_min": float(np.nanmin(values[repair_idx])),
                "segment_max": float(np.nanmax(values[repair_idx])),
            }
        )
    return repaired, diagnostics_rows


def _repair_out_of_range_segments(
    values: np.ndarray,
    *,
    repair_type: str,
    min_center_nm: int = 0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = values.copy()
    diagnostics_rows: list[dict[str, float]] = []
    for start_idx, end_idx in _iter_bad_segments(repaired):
        center_nm = int(WAVELENGTHS[(start_idx + end_idx) // 2])
        if center_nm < min_center_nm:
            continue
        left_candidates = np.arange(max(0, start_idx - USGS_SHOULDER_POINTS), start_idx)
        right_candidates = np.arange(end_idx + 1, min(len(repaired), end_idx + 1 + USGS_SHOULDER_POINTS))
        left_good = left_candidates[_in_bounds(repaired[left_candidates])]
        right_good = right_candidates[_in_bounds(repaired[right_candidates])]
        repair_idx = np.arange(start_idx, end_idx + 1)
        wavelengths = WAVELENGTHS[repair_idx].astype(float)
        if left_good.size >= 2 and right_good.size >= 2:
            x0 = float(np.nanmean(WAVELENGTHS[left_good]))
            y0 = _robust_center(repaired[left_good])
            x1 = float(np.nanmean(WAVELENGTHS[right_good]))
            y1 = _robust_center(repaired[right_good])
            predicted = np.interp(wavelengths, [x0, x1], [y0, y1])
        elif left_good.size >= 2:
            fit = _fit_line(WAVELENGTHS[left_good].astype(float), repaired[left_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        elif right_good.size >= 2:
            fit = _fit_line(WAVELENGTHS[right_good].astype(float), repaired[right_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        else:
            continue
        predicted = np.clip(predicted, *CLIP_BOUNDS)
        blend_start = max(0, start_idx - USGS_BLEND_HALF_WINDOW)
        blend_end = min(len(repaired) - 1, end_idx + USGS_BLEND_HALF_WINDOW)
        blend_idx = np.arange(blend_start, blend_end + 1)
        if left_good.size >= 1 and right_good.size >= 1:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(np.nanmean(WAVELENGTHS[left_good])), float(np.nanmean(WAVELENGTHS[right_good]))],
                [_robust_center(repaired[left_good]), _robust_center(repaired[right_good])],
            )
        else:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(WAVELENGTHS[repair_idx[0]]), float(WAVELENGTHS[repair_idx[-1]])],
                [float(predicted[0]), float(predicted[-1])],
            )
        predicted_blend = np.clip(predicted_blend, *CLIP_BOUNDS)
        weights = _blend_weights(
            WAVELENGTHS[blend_idx],
            int(WAVELENGTHS[start_idx]),
            int(WAVELENGTHS[end_idx]),
            USGS_BLEND_HALF_WINDOW,
        )
        repaired[blend_idx] = (1.0 - weights) * repaired[blend_idx] + weights * predicted_blend
        repaired[repair_idx] = predicted
        diagnostics_rows.append(
            {
                "repair_type": repair_type,
                "replace_start_nm": float(WAVELENGTHS[start_idx]),
                "replace_end_nm": float(WAVELENGTHS[end_idx]),
                "out_of_range_count": float(end_idx - start_idx + 1),
                "segment_min": float(np.nanmin(values[repair_idx])),
                "segment_max": float(np.nanmax(values[repair_idx])),
            }
        )
    return repaired, diagnostics_rows


def _window_max_jump(values: np.ndarray, start_nm: int, end_nm: int) -> float:
    mask = _window_mask(start_nm, end_nm)
    segment = values[mask]
    if segment.size < 2 or not np.isfinite(segment).any():
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(segment))))


def _repair_window_from_shoulders(
    values: np.ndarray,
    *,
    replace_start_nm: int,
    replace_end_nm: int,
    left_start_nm: int,
    left_end_nm: int,
    right_start_nm: int,
    right_end_nm: int,
    blend_half_window: int,
    repair_type: str,
    jump_threshold: float = 0.03,
    force_repair: bool = False,
) -> tuple[np.ndarray, dict[str, float] | None]:
    replace_mask = _window_mask(replace_start_nm, replace_end_nm)
    left_mask = _window_mask(left_start_nm, left_end_nm)
    right_mask = _window_mask(right_start_nm, right_end_nm)
    replace_values = values[replace_mask]
    left_values = values[left_mask]
    right_values = values[right_mask]
    left_valid = _in_bounds(left_values)
    right_valid = _in_bounds(right_values)
    if int(left_valid.sum()) < 4 or int(right_valid.sum()) < 4:
        return values, None

    out_of_range_count = int(np.count_nonzero(~_in_bounds(replace_values)))
    max_jump = _window_max_jump(values, replace_start_nm, replace_end_nm)
    should = force_repair or out_of_range_count > 0 or (np.isfinite(max_jump) and max_jump > jump_threshold)
    if not should:
        return values, None

    x0 = float(np.nanmean(WAVELENGTHS[left_mask][left_valid]))
    y0 = _robust_center(left_values[left_valid])
    x1 = float(np.nanmean(WAVELENGTHS[right_mask][right_valid]))
    y1 = _robust_center(right_values[right_valid])

    repaired = values.copy()
    blend_start_nm = max(WAVELENGTHS[0], replace_start_nm - blend_half_window)
    blend_end_nm = min(WAVELENGTHS[-1], replace_end_nm + blend_half_window)
    blend_mask = _window_mask(int(blend_start_nm), int(blend_end_nm))

    predicted_center = np.interp(WAVELENGTHS[replace_mask].astype(float), [x0, x1], [y0, y1])
    predicted_blend = np.interp(WAVELENGTHS[blend_mask].astype(float), [x0, x1], [y0, y1])
    predicted_center = np.clip(predicted_center, *CLIP_BOUNDS)
    predicted_blend = np.clip(predicted_blend, *CLIP_BOUNDS)
    weights = _blend_weights(WAVELENGTHS[blend_mask], replace_start_nm, replace_end_nm, blend_half_window)
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[replace_mask] = predicted_center
    diagnostics = {
        "repair_type": repair_type,
        "replace_start_nm": float(replace_start_nm),
        "replace_end_nm": float(replace_end_nm),
        "out_of_range_count": float(out_of_range_count),
        "max_abs_jump": float(max_jump) if np.isfinite(max_jump) else float("nan"),
        "left_center": float(y0),
        "right_center": float(y1),
    }
    return repaired, diagnostics


def _tail_drift_metric(values: np.ndarray, *, left_start_nm: int, left_end_nm: int, replace_end_nm: int) -> float:
    left_mask = _window_mask(left_start_nm, left_end_nm)
    left_values = values[left_mask]
    left_valid = _in_bounds(left_values)
    if int(left_valid.sum()) < 4:
        return float("nan")
    fit = _fit_line(WAVELENGTHS[left_mask][left_valid].astype(float), left_values[left_valid])
    if fit is None:
        return float("nan")
    slope, intercept = fit
    tail_mask = _window_mask(replace_end_nm - 10, replace_end_nm)
    tail_values = values[tail_mask]
    tail_valid = np.isfinite(tail_values)
    if int(tail_valid.sum()) < 3:
        return float("nan")
    predicted = slope * WAVELENGTHS[tail_mask][tail_valid].astype(float) + intercept
    return float(np.nanmax(np.abs(tail_values[tail_valid] - predicted)))


def _repair_tail_from_left_shoulder(
    values: np.ndarray,
    *,
    repair_type: str,
    replace_start_nm: int = TAIL_REPLACE_START,
    replace_end_nm: int = TAIL_REPLACE_END,
    left_start_nm: int = TAIL_LEFT_START,
    left_end_nm: int = TAIL_LEFT_END,
    blend_half_window: int = TAIL_BLEND_HALF_WINDOW,
    jump_threshold: float = TAIL_JUMP_THRESHOLD,
    drift_threshold: float = TAIL_DRIFT_THRESHOLD,
) -> tuple[np.ndarray, dict[str, float] | None]:
    replace_mask = _window_mask(replace_start_nm, replace_end_nm)
    left_mask = _window_mask(left_start_nm, left_end_nm)
    left_values = values[left_mask]
    left_valid = _in_bounds(left_values)
    if int(left_valid.sum()) < 8:
        return values, None
    fit = _fit_line(WAVELENGTHS[left_mask][left_valid].astype(float), left_values[left_valid])
    if fit is None:
        return values, None
    slope, intercept = fit
    replace_values = values[replace_mask]
    max_jump = _window_max_jump(values, replace_start_nm, replace_end_nm)
    drift = _tail_drift_metric(
        values,
        left_start_nm=left_start_nm,
        left_end_nm=left_end_nm,
        replace_end_nm=replace_end_nm,
    )
    out_of_range_count = int(np.count_nonzero(~_in_bounds(replace_values)))
    should = (
        out_of_range_count > 0
        or (np.isfinite(max_jump) and max_jump > jump_threshold)
        or (np.isfinite(drift) and drift > drift_threshold)
    )
    if not should:
        return values, None

    repaired = values.copy()
    blend_start_nm = max(WAVELENGTHS[0], replace_start_nm - blend_half_window)
    blend_mask = _window_mask(int(blend_start_nm), replace_end_nm)
    predicted_center = slope * WAVELENGTHS[replace_mask].astype(float) + intercept
    predicted_blend = slope * WAVELENGTHS[blend_mask].astype(float) + intercept
    predicted_center = np.clip(predicted_center, *CLIP_BOUNDS)
    predicted_blend = np.clip(predicted_blend, *CLIP_BOUNDS)
    weights = _blend_weights(WAVELENGTHS[blend_mask], replace_start_nm, replace_end_nm, blend_half_window)
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[replace_mask] = predicted_center
    diagnostics = {
        "repair_type": repair_type,
        "replace_start_nm": float(replace_start_nm),
        "replace_end_nm": float(replace_end_nm),
        "out_of_range_count": float(out_of_range_count),
        "max_abs_jump": float(max_jump) if np.isfinite(max_jump) else float("nan"),
        "tail_drift": float(drift) if np.isfinite(drift) else float("nan"),
    }
    return repaired, diagnostics


def _repair_usgs_visible_splice(values: np.ndarray, landcover_group: str) -> tuple[np.ndarray, dict[str, float] | None]:
    vis_mask = (WAVELENGTHS >= USGS_VISIBLE_REPAIR_START) & (WAVELENGTHS <= USGS_VISIBLE_REPAIR_END)
    vis_values = values[vis_mask]
    if vis_values.size < 2 or not np.isfinite(vis_values).any():
        return values, None
    diffs = np.abs(np.diff(vis_values))
    if not np.isfinite(diffs).any():
        return values, None
    jump_idx = int(np.nanargmax(diffs))
    jump_nm = int(USGS_VISIBLE_REPAIR_START + jump_idx)
    jump_value = float(diffs[jump_idx])
    if jump_value <= USGS_VISIBLE_REPAIR_THRESHOLD:
        return values, None
    if landcover_group == "vegetation" and 680 <= jump_nm <= 720:
        return values, None

    replace_start_nm = max(USGS_VISIBLE_REPAIR_START, jump_nm - USGS_VISIBLE_BLEND_HALF_WINDOW)
    replace_end_nm = min(USGS_VISIBLE_REPAIR_END, jump_nm + 1 + USGS_VISIBLE_BLEND_HALF_WINDOW)
    left_start_nm = max(USGS_VISIBLE_REPAIR_START, replace_start_nm - USGS_VISIBLE_SHOULDER_GAP - USGS_VISIBLE_SHOULDER_WIDTH)
    left_end_nm = max(USGS_VISIBLE_REPAIR_START, replace_start_nm - USGS_VISIBLE_SHOULDER_GAP)
    right_start_nm = min(USGS_VISIBLE_REPAIR_END, replace_end_nm + USGS_VISIBLE_SHOULDER_GAP)
    right_end_nm = min(USGS_VISIBLE_REPAIR_END, right_start_nm + USGS_VISIBLE_SHOULDER_WIDTH)
    return _repair_window_from_shoulders(
        values,
        replace_start_nm=replace_start_nm,
        replace_end_nm=replace_end_nm,
        left_start_nm=left_start_nm,
        left_end_nm=left_end_nm,
        right_start_nm=right_start_nm,
        right_end_nm=right_end_nm,
        blend_half_window=USGS_VISIBLE_BLEND_HALF_WINDOW,
        repair_type="usgs_visible_splice_interp",
        jump_threshold=USGS_VISIBLE_REPAIR_THRESHOLD,
        force_repair=True,
    )


def _repair_tail_with_robust_smoother(
    values: np.ndarray,
    *,
    smooth_start_nm: int,
    replace_start_nm: int,
    lam: float,
    repair_type: str,
    jump_threshold: float = TAIL_JUMP_THRESHOLD,
    drift_threshold: float = TAIL_DRIFT_THRESHOLD,
) -> tuple[np.ndarray, dict[str, float] | None]:
    max_jump = _window_max_jump(values, replace_start_nm, TAIL_REPLACE_END)
    drift = _tail_drift_metric(
        values,
        left_start_nm=max(400, smooth_start_nm),
        left_end_nm=max(smooth_start_nm + 5, replace_start_nm - 10),
        replace_end_nm=TAIL_REPLACE_END,
    )
    replace_mask = _window_mask(replace_start_nm, TAIL_REPLACE_END)
    replace_values = values[replace_mask]
    out_of_range_count = int(np.count_nonzero(~_in_bounds(replace_values)))
    should = (
        out_of_range_count > 0
        or (np.isfinite(max_jump) and max_jump > jump_threshold)
        or (np.isfinite(drift) and drift > drift_threshold)
    )
    if not should:
        return values, None

    mask, _, smoothed, _ = _robust_smooth_segment(values, start_nm=smooth_start_nm, end_nm=TAIL_REPLACE_END, lam=lam)
    if not np.isfinite(smoothed).any():
        return values, None
    repaired = values.copy()
    blend_mask = _window_mask(max(400, replace_start_nm - TAIL_BLEND_HALF_WINDOW), TAIL_REPLACE_END)
    smoothed_map = np.full_like(values, np.nan)
    smoothed_map[mask] = np.clip(smoothed, *CLIP_BOUNDS)
    predicted_blend = smoothed_map[blend_mask]
    valid = np.isfinite(predicted_blend)
    if not valid.any():
        return values, None
    predicted_blend = _fill_nan_linear(predicted_blend)
    weights = _blend_weights(WAVELENGTHS[blend_mask], replace_start_nm, TAIL_REPLACE_END, TAIL_BLEND_HALF_WINDOW)
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[replace_mask] = np.clip(smoothed_map[replace_mask], *CLIP_BOUNDS)
    return repaired, {
        "repair_type": repair_type,
        "replace_start_nm": float(replace_start_nm),
        "replace_end_nm": float(TAIL_REPLACE_END),
        "out_of_range_count": float(out_of_range_count),
        "max_abs_jump": float(max_jump) if np.isfinite(max_jump) else float("nan"),
        "tail_drift": float(drift) if np.isfinite(drift) else float("nan"),
    }


def _repair_tail_extrapolated_from_smoothed(
    values: np.ndarray,
    *,
    smooth_start_nm: int,
    drop_after_nm: int,
    lam: float,
    repair_type: str,
    use_last_value: bool,
    jump_threshold: float = TAIL_JUMP_THRESHOLD,
    drift_threshold: float = TAIL_DRIFT_THRESHOLD,
) -> tuple[np.ndarray, dict[str, float] | None]:
    max_jump = _window_max_jump(values, drop_after_nm, TAIL_REPLACE_END)
    drift = _tail_drift_metric(
        values,
        left_start_nm=max(400, smooth_start_nm),
        left_end_nm=max(smooth_start_nm + 5, drop_after_nm - 10),
        replace_end_nm=TAIL_REPLACE_END,
    )
    replace_mask = _window_mask(drop_after_nm, TAIL_REPLACE_END)
    replace_values = values[replace_mask]
    out_of_range_count = int(np.count_nonzero(~_in_bounds(replace_values)))
    should = (
        out_of_range_count > 0
        or (np.isfinite(max_jump) and max_jump > jump_threshold)
        or (np.isfinite(drift) and drift > drift_threshold)
    )
    if not should:
        return values, None

    mask, _, smoothed, _ = _robust_smooth_segment(values, start_nm=smooth_start_nm, end_nm=drop_after_nm, lam=lam)
    if not np.isfinite(smoothed).any():
        return values, None
    repaired = values.copy()
    full_smoothed = np.full_like(values, np.nan)
    full_smoothed[mask] = np.clip(smoothed, *CLIP_BOUNDS)

    drop_mask = _window_mask(drop_after_nm, TAIL_REPLACE_END)
    if use_last_value:
        last_val = float(full_smoothed[drop_after_nm - 400])
        predicted_tail = np.full(np.count_nonzero(drop_mask), last_val, dtype=float)
    else:
        fit_start_nm = max(smooth_start_nm, drop_after_nm - 20)
        fit_mask = _window_mask(fit_start_nm, drop_after_nm)
        fit_vals = full_smoothed[fit_mask]
        fit_good = np.isfinite(fit_vals)
        if int(fit_good.sum()) < 4:
            last_val = float(full_smoothed[drop_after_nm - 400])
            predicted_tail = np.full(np.count_nonzero(drop_mask), last_val, dtype=float)
        else:
            fit = _fit_line(WAVELENGTHS[fit_mask][fit_good].astype(float), fit_vals[fit_good])
            if fit is None:
                last_val = float(full_smoothed[drop_after_nm - 400])
                predicted_tail = np.full(np.count_nonzero(drop_mask), last_val, dtype=float)
            else:
                slope, intercept = fit
                predicted_tail = slope * WAVELENGTHS[drop_mask].astype(float) + intercept
    predicted_tail = np.clip(predicted_tail, *CLIP_BOUNDS)

    blend_mask = _window_mask(max(400, drop_after_nm - TAIL_BLEND_HALF_WINDOW), TAIL_REPLACE_END)
    predicted_full = repaired[blend_mask].copy()
    smoothed_prefix_mask = (WAVELENGTHS[blend_mask] < drop_after_nm) & np.isfinite(full_smoothed[blend_mask])
    predicted_full[smoothed_prefix_mask] = full_smoothed[blend_mask][smoothed_prefix_mask]
    predicted_full[WAVELENGTHS[blend_mask] >= drop_after_nm] = predicted_tail
    weights = _blend_weights(WAVELENGTHS[blend_mask], drop_after_nm, TAIL_REPLACE_END, TAIL_BLEND_HALF_WINDOW)
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_full
    repaired[drop_mask] = predicted_tail
    return repaired, {
        "repair_type": repair_type,
        "replace_start_nm": float(drop_after_nm),
        "replace_end_nm": float(TAIL_REPLACE_END),
        "out_of_range_count": float(out_of_range_count),
        "max_abs_jump": float(max_jump) if np.isfinite(max_jump) else float("nan"),
        "tail_drift": float(drift) if np.isfinite(drift) else float("nan"),
    }


def _apply_santa_strong_smooth_blend(values: np.ndarray) -> tuple[np.ndarray, dict[str, float] | None]:
    mask, _, smoothed, _ = _robust_smooth_segment(values, start_nm=400, end_nm=2500, lam=SANTA_STRONG_SMOOTH_LAM)
    if not mask.any() or not np.isfinite(smoothed).any():
        return values, None

    smoothed_full = values.copy()
    smoothed_full[mask] = np.clip(_fill_nan_linear(smoothed), *CLIP_BOUNDS)
    weights = _transition_to_smoothed_weights(WAVELENGTHS, SANTA_SMOOTH_CENTER_NM, SANTA_SMOOTH_HALF_WINDOW)

    repaired = values.copy()
    finite_original = np.isfinite(repaired)
    repaired[finite_original] = (
        (1.0 - weights[finite_original]) * repaired[finite_original]
        + weights[finite_original] * smoothed_full[finite_original]
    )
    repaired[~finite_original] = smoothed_full[~finite_original]

    delta = np.abs(repaired - values)
    return repaired, {
        "repair_type": "santa_post800_strong_smooth_blend",
        "replace_start_nm": float(SANTA_SMOOTH_CENTER_NM - SANTA_SMOOTH_HALF_WINDOW),
        "replace_end_nm": float(WAVELENGTHS[-1]),
        "center_nm": float(SANTA_SMOOTH_CENTER_NM),
        "blend_half_window_nm": float(SANTA_SMOOTH_HALF_WINDOW),
        "lambda": float(SANTA_STRONG_SMOOTH_LAM),
        "max_abs_delta": float(np.nanmax(delta)),
        "mean_abs_delta": float(np.nanmean(delta)),
    }


def _repair_understory_with_robust_smoother(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask, segment, smoothed, weights = _robust_smooth_segment(values, start_nm=400, end_nm=2500, lam=UNDERSTORY_ROBUST_LAM)
    if not np.isfinite(smoothed).any():
        return values, []
    finite = np.isfinite(segment)
    residual = np.abs(segment - smoothed)
    flagged = finite & ((weights < ROBUST_WEIGHT_THRESHOLD) | (~_in_bounds(segment))) & (residual > UNDERSTORY_ROBUST_ABS_THRESHOLD)
    if not flagged.any():
        return values, []
    repaired = values.copy()
    diagnostics: list[dict[str, float]] = []
    local_wavelengths = WAVELENGTHS[mask]
    for start_idx, end_idx in _contiguous_true_ranges(flagged):
        start_nm = int(local_wavelengths[start_idx])
        end_nm = int(local_wavelengths[end_idx])
        blend_mask = _window_mask(max(400, start_nm - 10), min(2500, end_nm + 10))
        predicted = np.full(np.count_nonzero(blend_mask), np.nan)
        local_blend = WAVELENGTHS[blend_mask]
        for i, nm in enumerate(local_blend):
            local_idx = int(nm - 400) - int(local_wavelengths[0] - 400)
            if 0 <= local_idx < len(smoothed):
                predicted[i] = smoothed[local_idx]
        predicted = _fill_nan_linear(predicted)
        weights_blend = _blend_weights(WAVELENGTHS[blend_mask], start_nm, end_nm, 10)
        repaired[blend_mask] = (1.0 - weights_blend) * repaired[blend_mask] + weights_blend * predicted
        replaced_mask = _window_mask(start_nm, end_nm)
        replaced_vals = np.full(np.count_nonzero(replaced_mask), np.nan)
        for i, nm in enumerate(WAVELENGTHS[replaced_mask]):
            local_idx = int(nm - local_wavelengths[0])
            if 0 <= local_idx < len(smoothed):
                replaced_vals[i] = smoothed[local_idx]
        repaired[replaced_mask] = np.clip(_fill_nan_linear(replaced_vals), *CLIP_BOUNDS)
        diagnostics.append(
            {
                "repair_type": "understory_full_robust_outlier_band_replace",
                "replace_start_nm": float(start_nm),
                "replace_end_nm": float(end_nm),
                "out_of_range_count": float(np.count_nonzero(~_in_bounds(segment[start_idx : end_idx + 1]))),
                "max_abs_jump": float(np.nanmax(np.abs(np.diff(segment[start_idx : end_idx + 1]))))
                if end_idx > start_idx
                else 0.0,
            }
        )
    return np.clip(repaired, *CLIP_BOUNDS), diagnostics
def _repair_understory_remaining_segments(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = values.copy()
    diagnostics_rows: list[dict[str, float]] = []
    for start_idx, end_idx in _iter_bad_segments(repaired):
        center_nm = int(WAVELENGTHS[(start_idx + end_idx) // 2])
        if center_nm < 1350:
            continue
        left_candidates = np.arange(max(0, start_idx - USGS_SHOULDER_POINTS), start_idx)
        right_candidates = np.arange(end_idx + 1, min(len(repaired), end_idx + 1 + USGS_SHOULDER_POINTS))
        left_good = left_candidates[_in_bounds(repaired[left_candidates])]
        right_good = right_candidates[_in_bounds(repaired[right_candidates])]
        if left_good.size < 2 and right_good.size < 2:
            continue
        repair_idx = np.arange(start_idx, end_idx + 1)
        wavelengths = WAVELENGTHS[repair_idx].astype(float)
        if left_good.size >= 2 and right_good.size >= 2:
            x0 = float(np.nanmean(WAVELENGTHS[left_good]))
            y0 = _robust_center(repaired[left_good])
            x1 = float(np.nanmean(WAVELENGTHS[right_good]))
            y1 = _robust_center(repaired[right_good])
            predicted = np.interp(wavelengths, [x0, x1], [y0, y1])
        elif left_good.size >= 2:
            fit = _fit_line(WAVELENGTHS[left_good].astype(float), repaired[left_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        else:
            fit = _fit_line(WAVELENGTHS[right_good].astype(float), repaired[right_good])
            if fit is None:
                continue
            slope, intercept = fit
            predicted = slope * wavelengths + intercept
        predicted = np.clip(predicted, 0.0, 1.0)
        blend_start = max(0, start_idx - USGS_BLEND_HALF_WINDOW)
        blend_end = min(len(repaired) - 1, end_idx + USGS_BLEND_HALF_WINDOW)
        blend_idx = np.arange(blend_start, blend_end + 1)
        if left_good.size >= 1 and right_good.size >= 1:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(np.nanmean(WAVELENGTHS[left_good])), float(np.nanmean(WAVELENGTHS[right_good]))],
                [_robust_center(repaired[left_good]), _robust_center(repaired[right_good])],
            )
        else:
            predicted_blend = np.interp(
                WAVELENGTHS[blend_idx].astype(float),
                [float(WAVELENGTHS[repair_idx[0]]), float(WAVELENGTHS[repair_idx[-1]])],
                [float(predicted[0]), float(predicted[-1])],
            )
        predicted_blend = np.clip(predicted_blend, 0.0, 1.0)
        weights = _blend_weights(
            WAVELENGTHS[blend_idx],
            int(WAVELENGTHS[start_idx]),
            int(WAVELENGTHS[end_idx]),
            USGS_BLEND_HALF_WINDOW,
        )
        repaired[blend_idx] = (1.0 - weights) * repaired[blend_idx] + weights * predicted_blend
        repaired[repair_idx] = predicted
        diagnostics_rows.append(
            {
                "repair_type": "understory_out_of_range_segment_interp",
                "replace_start_nm": float(WAVELENGTHS[start_idx]),
                "replace_end_nm": float(WAVELENGTHS[end_idx]),
                "out_of_range_count": float(end_idx - start_idx + 1),
                "segment_min": float(np.nanmin(values[repair_idx])),
                "segment_max": float(np.nanmax(values[repair_idx])),
            }
        )
    return repaired, diagnostics_rows


def _plot_examples(examples: list[dict[str, object]], output_path: Path, zoom_start: int, zoom_end: int) -> None:
    if not examples:
        return
    figure, axes = plt.subplots(len(examples), 2, figsize=(14, max(4, 3.0 * len(examples))), sharex=False)
    axes = np.atleast_2d(axes)
    for axis_row, example in zip(axes, examples):
        before = np.asarray(example["before"], dtype=float)
        after = np.asarray(example["after"], dtype=float)
        label = f"{example['source_id']} | {example['spectrum_id']}"
        axis_row[0].plot(WAVELENGTHS, before, color="#b2182b", linewidth=1.0, label="before")
        axis_row[0].plot(WAVELENGTHS, after, color="#2166ac", linewidth=1.0, label="after")
        axis_row[0].set_title(label)
        axis_row[0].grid(alpha=0.2)

        zoom_mask = (WAVELENGTHS >= zoom_start) & (WAVELENGTHS <= zoom_end)
        axis_row[1].plot(WAVELENGTHS[zoom_mask], before[zoom_mask], color="#b2182b", linewidth=1.0, label="before")
        axis_row[1].plot(WAVELENGTHS[zoom_mask], after[zoom_mask], color="#2166ac", linewidth=1.0, label="after")
        axis_row[1].set_title(f"{zoom_start}-{zoom_end} nm")
        axis_row[1].grid(alpha=0.2)

    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Reflectance")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair remaining source-specific artifacts after subset filtering.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "tabular").mkdir(parents=True, exist_ok=True)
    copy_static_inputs(base_root, output_root)
    repair_metric_lookup = _compute_repair_metric_lookup(base_root)

    input_path = base_root / "tabular" / "normalized_spectra.csv"
    output_path = output_root / "tabular" / "normalized_spectra.csv"
    diagnostics_rows: list[dict[str, object]] = []
    understory_examples: list[dict[str, object]] = []
    bssl_examples: list[dict[str, object]] = []
    branch_examples: list[dict[str, object]] = []
    hyspiri_examples: list[dict[str, object]] = []
    snow_examples: list[dict[str, object]] = []
    ngee_examples: list[dict[str, object]] = []
    santa_examples: list[dict[str, object]] = []
    sispec_examples: list[dict[str, object]] = []
    emit_examples: list[dict[str, object]] = []

    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as in_handle:
        reader = csv.DictReader(in_handle)
        assert reader.fieldnames is not None
        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                source_id = row["source_id"]
                if source_id not in {
                    "understory_estonia_czech",
                    "bssl",
                    "branch_tree_spectra_boreal_temperate",
                    "hyspiri_ground_targets",
                    "natural_snow_twigs",
                    "ngee_arctic_2018",
                    "santa_barbara_urban_reflectance",
                    "sispec",
                    "emit_adjusted_vegetation",
                    "emit_l2a_surface",
                }:
                    writer.writerow(row)
                    continue
                repair_metrics = repair_metric_lookup.get(f"{source_id}||{row['spectrum_id']}")
                if not _should_attempt_repair(source_id, repair_metrics):
                    writer.writerow(row)
                    continue

                before = np.asarray([_parse_value(row[f"nm_{wavelength}"]) for wavelength in WAVELENGTHS], dtype=float)
                after = before.copy()
                row_repairs: list[dict[str, object]] = []

                if source_id == "understory_estonia_czech":
                    for window, left, right, label in [
                        (UNDERSTORY_1400_WINDOW, UNDERSTORY_1400_LEFT, UNDERSTORY_1400_RIGHT, "understory_1400_window_interp"),
                        (UNDERSTORY_1900_WINDOW, UNDERSTORY_1900_LEFT, UNDERSTORY_1900_RIGHT, "understory_1900_window_interp"),
                    ]:
                        after, diag = _repair_window_from_shoulders(
                            after,
                            replace_start_nm=window[0],
                            replace_end_nm=window[1],
                            left_start_nm=left[0],
                            left_end_nm=left[1],
                            right_start_nm=right[0],
                            right_end_nm=right[1],
                            blend_half_window=UNDERSTORY_WATER_BLEND_HALF_WINDOW,
                            repair_type=label,
                            jump_threshold=UNDERSTORY_WATER_JUMP_THRESHOLD,
                        )
                        if diag is not None:
                            row_repairs.append(diag)
                    after, diag = _repair_understory_tail_mid(after)
                    if diag is not None:
                        row_repairs.append(diag)
                    after, segment_rows = _repair_out_of_range_segments(
                        after,
                        repair_type="understory_out_of_range_segment_interp",
                        min_center_nm=1350,
                    )
                    row_repairs.extend(segment_rows)
                    after, robust_rows = _repair_understory_with_robust_smoother(after)
                    row_repairs.extend(robust_rows)
                    if row_repairs and len(understory_examples) < 8:
                        understory_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "bssl":
                    after, diag = _repair_tail_with_robust_smoother(
                        after,
                        smooth_start_nm=2200,
                        replace_start_nm=2200,
                        lam=BSSL_TAIL_LAM,
                        repair_type="bssl_tail_robust_smooth",
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    if row_repairs and len(bssl_examples) < 8:
                        bssl_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "branch_tree_spectra_boreal_temperate":
                    after, diag = _repair_tail_with_robust_smoother(
                        after,
                        smooth_start_nm=2300,
                        replace_start_nm=2300,
                        lam=BRANCH_TAIL_LAM,
                        repair_type="branch_tree_tail_robust_smooth",
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    if row_repairs and len(branch_examples) < 8:
                        branch_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "hyspiri_ground_targets":
                    after, segment_rows = _repair_out_of_range_segments(
                        after,
                        repair_type="hyspiri_out_of_range_segment_interp",
                    )
                    row_repairs.extend(segment_rows)
                    after, diag = _repair_tail_extrapolated_from_smoothed(
                        after,
                        smooth_start_nm=2300,
                        drop_after_nm=TAIL_DROP_AFTER_2450,
                        lam=HYSPIRI_TAIL_LAM,
                        repair_type="hyspiri_tail_smoothed_extrapolated",
                        use_last_value=True,
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    if row_repairs:
                        after = np.clip(after, *CLIP_BOUNDS)
                    if row_repairs and len(hyspiri_examples) < 8:
                        hyspiri_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "natural_snow_twigs":
                    after, diag = _repair_tail_with_robust_smoother(
                        after,
                        smooth_start_nm=2350,
                        replace_start_nm=2350,
                        lam=SNOW_TAIL_LAM,
                        repair_type="natural_snow_tail_robust_smooth",
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    after, segment_rows = _repair_out_of_range_segments(
                        after,
                        repair_type="natural_snow_out_of_range_segment_interp",
                        min_center_nm=2300,
                    )
                    row_repairs.extend(segment_rows)
                    if row_repairs and len(snow_examples) < 8:
                        snow_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "ngee_arctic_2018":
                    after, diag = _repair_tail_extrapolated_from_smoothed(
                        after,
                        smooth_start_nm=2300,
                        drop_after_nm=TAIL_DROP_AFTER_2450,
                        lam=NGEE_TAIL_LAM,
                        repair_type="ngee_arctic_2018_tail_smoothed_extrapolated",
                        use_last_value=False,
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    after, segment_rows = _repair_out_of_range_segments(
                        after,
                        repair_type="ngee_arctic_2018_out_of_range_segment_interp",
                        min_center_nm=2300,
                    )
                    row_repairs.extend(segment_rows)
                    if row_repairs and len(ngee_examples) < 8:
                        ngee_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "santa_barbara_urban_reflectance":
                    after, diag = _apply_santa_strong_smooth_blend(after)
                    if diag is not None:
                        row_repairs.append(diag)
                    after, diag = _repair_window_from_shoulders(
                        after,
                        replace_start_nm=SANTA_ABSORPTION2_WINDOW[0],
                        replace_end_nm=SANTA_ABSORPTION2_WINDOW[1],
                        left_start_nm=SANTA_ABSORPTION2_LEFT[0],
                        left_end_nm=SANTA_ABSORPTION2_LEFT[1],
                        right_start_nm=SANTA_ABSORPTION2_RIGHT[0],
                        right_end_nm=SANTA_ABSORPTION2_RIGHT[1],
                        blend_half_window=10,
                        repair_type="santa_absorption2_neighbor_interp",
                        jump_threshold=0.03,
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    after, diag = _repair_tail_extrapolated_from_smoothed(
                        after,
                        smooth_start_nm=2300,
                        drop_after_nm=TAIL_DROP_AFTER_2450,
                        lam=NGEE_TAIL_LAM,
                        repair_type="santa_tail_smoothed_extrapolated",
                        use_last_value=True,
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    if row_repairs and len(santa_examples) < 8:
                        santa_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id == "sispec":
                    after, diag = _repair_window_from_shoulders(
                        after,
                        replace_start_nm=SISPEC_ABSORPTION2_WINDOW[0],
                        replace_end_nm=SISPEC_ABSORPTION2_WINDOW[1],
                        left_start_nm=SISPEC_ABSORPTION2_LEFT[0],
                        left_end_nm=SISPEC_ABSORPTION2_LEFT[1],
                        right_start_nm=SISPEC_ABSORPTION2_RIGHT[0],
                        right_end_nm=SISPEC_ABSORPTION2_RIGHT[1],
                        blend_half_window=10,
                        repair_type="sispec_absorption2_neighbor_interp",
                        jump_threshold=0.03,
                    )
                    if diag is not None:
                        row_repairs.append(diag)
                    if row_repairs and len(sispec_examples) < 8:
                        sispec_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )
                elif source_id in {"emit_adjusted_vegetation", "emit_l2a_surface"}:
                    emit_group = "vegetation" if source_id == "emit_adjusted_vegetation" else ""
                    after, diag = _repair_usgs_visible_splice(after, emit_group)
                    if diag is not None:
                        diag["repair_type"] = f"{source_id}_visible_splice_interp"
                        row_repairs.append(diag)
                    if row_repairs and len(emit_examples) < 8:
                        emit_examples.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "before": before,
                                "after": after.copy(),
                            }
                        )

                if row_repairs:
                    for wavelength in WAVELENGTHS:
                        row[f"nm_{wavelength}"] = f"{after[wavelength - 400]:.12g}"
                    for repair in row_repairs:
                        diagnostics_rows.append(
                            {
                                "source_id": source_id,
                                "spectrum_id": row["spectrum_id"],
                                "sample_name": row["sample_name"],
                                **repair,
                            }
                        )

                writer.writerow(row)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    diagnostics_csv = diagnostics_dir / "remaining_source_artifact_repairs.csv"
    diagnostics_frame.to_csv(diagnostics_csv, index=False)

    plots_dir = output_root / "plots"
    understory_plot = plots_dir / "understory_tail_remaining_repairs.png"
    bssl_plot = plots_dir / "bssl_tail_repairs.png"
    branch_plot = plots_dir / "branch_tree_tail_repairs.png"
    hyspiri_plot = plots_dir / "hyspiri_tail_repairs.png"
    snow_plot = plots_dir / "natural_snow_tail_repairs.png"
    ngee_plot = plots_dir / "ngee_arctic_2018_tail_repairs.png"
    santa_plot = plots_dir / "santa_absorption_tail_repairs.png"
    sispec_plot = plots_dir / "sispec_absorption_repairs.png"
    emit_plot = plots_dir / "emit_visible_repairs.png"
    _plot_examples(understory_examples, understory_plot, 2300, 2450)
    _plot_examples(bssl_examples, bssl_plot, 2280, 2500)
    _plot_examples(branch_examples, branch_plot, 2280, 2500)
    _plot_examples(hyspiri_examples, hyspiri_plot, 2280, 2500)
    _plot_examples(snow_examples, snow_plot, 2280, 2500)
    _plot_examples(ngee_examples, ngee_plot, 2280, 2500)
    _plot_examples(santa_examples, santa_plot, 700, 2500)
    _plot_examples(sispec_examples, sispec_plot, 1750, 2000)
    _plot_examples(emit_examples, emit_plot, 400, 1000)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "diagnostics_csv": str(diagnostics_csv),
        "understory_repaired_spectra": int(
            diagnostics_frame.loc[
                diagnostics_frame["source_id"].eq("understory_estonia_czech"), "spectrum_id"
            ].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "repair_rows": int(len(diagnostics_frame)),
        "understory_plot": str(understory_plot),
        "bssl_repaired_spectra": int(
            diagnostics_frame.loc[diagnostics_frame["source_id"].eq("bssl"), "spectrum_id"].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "branch_tree_repaired_spectra": int(
            diagnostics_frame.loc[
                diagnostics_frame["source_id"].eq("branch_tree_spectra_boreal_temperate"), "spectrum_id"
            ].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "hyspiri_repaired_spectra": int(
            diagnostics_frame.loc[diagnostics_frame["source_id"].eq("hyspiri_ground_targets"), "spectrum_id"].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "bssl_plot": str(bssl_plot),
        "branch_tree_plot": str(branch_plot),
        "hyspiri_plot": str(hyspiri_plot),
        "natural_snow_repaired_spectra": int(
            diagnostics_frame.loc[diagnostics_frame["source_id"].eq("natural_snow_twigs"), "spectrum_id"].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "ngee_arctic_2018_repaired_spectra": int(
            diagnostics_frame.loc[diagnostics_frame["source_id"].eq("ngee_arctic_2018"), "spectrum_id"].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "santa_repaired_spectra": int(
            diagnostics_frame.loc[
                diagnostics_frame["source_id"].eq("santa_barbara_urban_reflectance"), "spectrum_id"
            ].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "sispec_repaired_spectra": int(
            diagnostics_frame.loc[diagnostics_frame["source_id"].eq("sispec"), "spectrum_id"].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "emit_repaired_spectra": int(
            diagnostics_frame.loc[
                diagnostics_frame["source_id"].isin(["emit_adjusted_vegetation", "emit_l2a_surface"]), "spectrum_id"
            ].nunique()
        )
        if not diagnostics_frame.empty
        else 0,
        "natural_snow_plot": str(snow_plot),
        "ngee_plot": str(ngee_plot),
        "santa_plot": str(santa_plot),
        "sispec_plot": str(sispec_plot),
        "emit_plot": str(emit_plot),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
