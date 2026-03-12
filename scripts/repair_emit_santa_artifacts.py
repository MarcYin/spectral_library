#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sample_processed_spectra_review import (
    SECOND_ABSORPTION_END,
    SECOND_ABSORPTION_PAD,
    SECOND_ABSORPTION_START,
    SECOND_ABSORPTION_THRESHOLD,
    TAIL_DRIFT_THRESHOLD,
    TAIL_EVAL_END,
    TAIL_EVAL_START,
    TAIL_FIT_END,
    TAIL_FIT_START,
)


CHUNK_SIZE = 512
EMIT_PREFIX = "emit_"
SANTA_SOURCE_ID = "santa_barbara_urban_reflectance"
BLEND_HALF_WINDOW_NM = 50
EMIT_BROAD_LEFT_START = 1805
EMIT_BROAD_LEFT_END = 1815
EMIT_BROAD_EVAL_START = 1820
EMIT_BROAD_EVAL_END = 1930
EMIT_BROAD_RIGHT_START = 1935
EMIT_BROAD_RIGHT_END = 1945
EMIT_BROAD_DEV_THRESHOLD = 0.08


def copy_support_files(base_root: Path, output_root: Path) -> None:
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("landcover_analysis/landcover_labels.csv"),
        Path("landcover_analysis/landcover_group_summary.csv"),
    ]:
        source_path = base_root / relative
        if source_path.exists():
            target_path = output_root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _two_sided_blend_weights(
    wavelengths: np.ndarray,
    start_nm: int,
    end_nm: int,
    half_window_nm: int,
) -> np.ndarray:
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


def _left_boundary_blend_weights(
    wavelengths: np.ndarray,
    boundary_nm: int,
    half_window_nm: int,
) -> np.ndarray:
    weights = np.ones(len(wavelengths), dtype=float)
    outer = boundary_nm - half_window_nm
    inner = boundary_nm + half_window_nm
    weights[wavelengths < outer] = 0.0
    ramp = (wavelengths >= outer) & (wavelengths <= inner)
    if inner > outer:
        weights[ramp] = (wavelengths[ramp] - outer) / float(inner - outer)
    return np.clip(weights, 0.0, 1.0)


def emit_interp_fix(
    values: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = values.copy()
    deltas = np.zeros(values.shape, dtype=float)
    band_indices = np.where((wavelengths >= SECOND_ABSORPTION_START) & (wavelengths <= SECOND_ABSORPTION_END))[0]
    if band_indices.size == 0:
        return fixed, deltas

    max_iterations = int(len(band_indices) * 3)
    for row_index in range(values.shape[0]):
        row = fixed[row_index].copy()
        original = values[row_index]
        for _ in range(max_iterations):
            best_band_index = -1
            best_interp = np.nan
            best_delta = 0.0
            for band_index in band_indices:
                left_index = band_index - SECOND_ABSORPTION_PAD
                right_index = band_index + SECOND_ABSORPTION_PAD
                if left_index < 0 or right_index >= values.shape[1]:
                    continue
                y_left = row[left_index]
                y_center = row[band_index]
                y_right = row[right_index]
                if not (np.isfinite(y_left) and np.isfinite(y_center) and np.isfinite(y_right)):
                    continue
                x_left = float(wavelengths[left_index])
                x_right = float(wavelengths[right_index])
                x_center = float(wavelengths[band_index])
                interp = y_left + (y_right - y_left) * (x_center - x_left) / (x_right - x_left)
                delta = abs(y_center - interp)
                if delta > best_delta:
                    best_delta = delta
                    best_band_index = int(band_index)
                    best_interp = float(interp)
            if best_band_index < 0 or best_delta < SECOND_ABSORPTION_THRESHOLD:
                break
            row[best_band_index] = best_interp
        fixed[row_index] = row
        row_deltas = np.abs(original - row)
        changed = row_deltas > 0
        deltas[row_index, changed] = row_deltas[changed]
    return fixed, deltas


def emit_broad_shoulder_fix(
    values: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = values.copy()
    deltas = np.zeros(values.shape, dtype=float)
    left_mask = (wavelengths >= EMIT_BROAD_LEFT_START) & (wavelengths <= EMIT_BROAD_LEFT_END)
    right_mask = (wavelengths >= EMIT_BROAD_RIGHT_START) & (wavelengths <= EMIT_BROAD_RIGHT_END)
    eval_mask = (wavelengths >= EMIT_BROAD_EVAL_START) & (wavelengths <= EMIT_BROAD_EVAL_END)
    replace_mask = (wavelengths >= EMIT_BROAD_EVAL_START) & (wavelengths <= EMIT_BROAD_RIGHT_START)
    blend_start = EMIT_BROAD_EVAL_START - BLEND_HALF_WINDOW_NM
    blend_end = EMIT_BROAD_RIGHT_START + BLEND_HALF_WINDOW_NM
    blend_mask = (wavelengths >= blend_start) & (wavelengths <= blend_end)
    if not left_mask.any() or not right_mask.any() or not eval_mask.any() or not replace_mask.any() or not blend_mask.any():
        return fixed, deltas

    x0 = float(np.mean(wavelengths[left_mask]))
    x1 = float(np.mean(wavelengths[right_mask]))
    eval_x = wavelengths[eval_mask].astype(float)
    blend_x = wavelengths[blend_mask].astype(float)

    for row_index in range(values.shape[0]):
        row = values[row_index]
        left_y = row[left_mask]
        right_y = row[right_mask]
        if np.isfinite(left_y).sum() < 2 or np.isfinite(right_y).sum() < 2:
            continue
        y0 = float(np.nanmean(left_y))
        y1 = float(np.nanmean(right_y))
        if not (np.isfinite(y0) and np.isfinite(y1) and y0 > y1):
            continue

        eval_y = row[eval_mask]
        predicted_eval = y0 + (y1 - y0) * (eval_x - x0) / (x1 - x0)
        positive_deviation = eval_y - predicted_eval
        if not np.isfinite(positive_deviation).any():
            continue
        max_dev = float(np.nanmax(positive_deviation))
        if max_dev < EMIT_BROAD_DEV_THRESHOLD:
            continue

        predicted_blend = y0 + (y1 - y0) * (blend_x - x0) / (x1 - x0)
        predicted_blend = np.clip(predicted_blend, 0.0, 1.0)
        original_blend = row[blend_mask]
        valid = np.isfinite(original_blend)
        if not valid.any():
            continue
        weights = _two_sided_blend_weights(
            wavelengths[blend_mask],
            EMIT_BROAD_EVAL_START,
            EMIT_BROAD_RIGHT_START,
            BLEND_HALF_WINDOW_NM,
        )
        blended = (1.0 - weights) * original_blend + weights * predicted_blend
        fixed[row_index, blend_mask] = np.where(valid, blended, original_blend)
        row_delta = np.abs(original_blend[valid] - blended[valid])
        deltas[row_index, np.flatnonzero(blend_mask)[valid]] = row_delta
    return fixed, deltas


def santa_tail_fix(
    values: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = values.copy()
    deltas = np.zeros(values.shape, dtype=float)
    fit_mask = (wavelengths >= TAIL_FIT_START) & (wavelengths <= TAIL_FIT_END)
    eval_mask = (wavelengths >= TAIL_EVAL_START) & (wavelengths <= TAIL_EVAL_END)
    blend_start = TAIL_EVAL_START - BLEND_HALF_WINDOW_NM
    blend_mask = (wavelengths >= blend_start) & (wavelengths <= TAIL_EVAL_END)
    fit_x = wavelengths[fit_mask].astype(float)
    blend_x = wavelengths[blend_mask].astype(float)
    if fit_x.size < 2 or blend_x.size < 1:
        return fixed, deltas

    for row_index in range(values.shape[0]):
        fit_y = values[row_index, fit_mask]
        eval_y = values[row_index, eval_mask]
        fit_valid = np.isfinite(fit_y)
        eval_valid = np.isfinite(eval_y)
        if fit_valid.sum() < 2 or eval_valid.sum() < 1:
            continue
        slope, intercept = np.polyfit(fit_x[fit_valid], fit_y[fit_valid], 1)
        predicted = np.clip(slope * blend_x + intercept, 0.0, 1.0)
        eval_in_blend = eval_mask[blend_mask]
        predicted_eval = predicted[eval_in_blend]
        drift = np.abs(eval_y[eval_valid] - predicted_eval[eval_valid])
        if float(np.max(drift)) < TAIL_DRIFT_THRESHOLD:
            continue
        original_blend = fixed[row_index, blend_mask]
        blend_valid = np.isfinite(original_blend)
        weights = _left_boundary_blend_weights(
            wavelengths[blend_mask],
            TAIL_EVAL_START,
            BLEND_HALF_WINDOW_NM,
        )
        blended = (1.0 - weights) * original_blend + weights * predicted
        fixed[row_index, blend_mask] = np.where(blend_valid, blended, original_blend)
        deltas[row_index, np.flatnonzero(blend_mask)[blend_valid]] = np.abs(original_blend[blend_valid] - blended[blend_valid])
    return fixed, deltas


def apply_repairs(
    chunk: pd.DataFrame,
    spectral_columns: list[str],
    wavelengths: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]], Counter[str]]:
    values = chunk[spectral_columns].to_numpy(dtype=float)
    source_ids = chunk["source_id"].astype(str).to_numpy()
    spectrum_ids = chunk["spectrum_id"].astype(str).to_numpy()
    diagnostics: list[dict[str, object]] = []
    replaced_by_source: Counter[str] = Counter()

    emit_rows = np.flatnonzero(np.char.startswith(source_ids.astype(str), EMIT_PREFIX))
    if len(emit_rows) > 0:
        emit_fixed, emit_deltas = emit_interp_fix(values[emit_rows], wavelengths)
        emit_fixed, emit_broad_deltas = emit_broad_shoulder_fix(emit_fixed, wavelengths)
        values[emit_rows] = emit_fixed
        emit_counts = (emit_deltas > 0).sum(axis=1)
        emit_broad_counts = (emit_broad_deltas > 0).sum(axis=1)
        for local_index, replaced_count in enumerate(emit_counts):
            if int(replaced_count) == 0:
                continue
            row_deltas = emit_deltas[local_index][emit_deltas[local_index] > 0]
            source_id = source_ids[emit_rows[local_index]]
            replaced_by_source[source_id] += int(replaced_count)
            diagnostics.append(
                {
                    "source_id": source_id,
                    "spectrum_id": spectrum_ids[emit_rows[local_index]],
                    "repair_type": "emit_second_absorption_interp",
                    "replaced_band_count": int(replaced_count),
                    "mean_abs_delta": float(np.mean(row_deltas)),
                    "max_abs_delta": float(np.max(row_deltas)),
                }
            )
        for local_index, replaced_count in enumerate(emit_broad_counts):
            if int(replaced_count) == 0:
                continue
            row_deltas = emit_broad_deltas[local_index][emit_broad_deltas[local_index] > 0]
            source_id = source_ids[emit_rows[local_index]]
            replaced_by_source[source_id] += int(replaced_count)
            diagnostics.append(
                {
                    "source_id": source_id,
                    "spectrum_id": spectrum_ids[emit_rows[local_index]],
                    "repair_type": "emit_second_absorption_shoulder_interp",
                    "replaced_band_count": int(replaced_count),
                    "mean_abs_delta": float(np.mean(row_deltas)),
                    "max_abs_delta": float(np.max(row_deltas)),
                }
            )

    santa_rows = np.flatnonzero(source_ids == SANTA_SOURCE_ID)
    if len(santa_rows) > 0:
        santa_fixed, santa_deltas = santa_tail_fix(values[santa_rows], wavelengths)
        values[santa_rows] = santa_fixed
        santa_counts = (santa_deltas > 0).sum(axis=1)
        for local_index, replaced_count in enumerate(santa_counts):
            if int(replaced_count) == 0:
                continue
            row_deltas = santa_deltas[local_index][santa_deltas[local_index] > 0]
            replaced_by_source[SANTA_SOURCE_ID] += int(replaced_count)
            diagnostics.append(
                {
                    "source_id": SANTA_SOURCE_ID,
                    "spectrum_id": spectrum_ids[santa_rows[local_index]],
                    "repair_type": "santa_tail_linear_extrapolation",
                    "replaced_band_count": int(replaced_count),
                    "mean_abs_delta": float(np.mean(row_deltas)),
                    "max_abs_delta": float(np.max(row_deltas)),
                }
            )

    chunk.loc[:, spectral_columns] = values
    return chunk, diagnostics, replaced_by_source


def run_repair(base_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copy_support_files(base_root, output_root)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)

    header_written = False
    diagnostics_rows: list[dict[str, object]] = []
    replaced_by_source: Counter[str] = Counter()
    total_rows = 0

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            fixed_chunk, chunk_diags, chunk_counts = apply_repairs(chunk.copy(), spectral_columns, wavelengths)
            fixed_chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True
            diagnostics_rows.extend(chunk_diags)
            replaced_by_source.update(chunk_counts)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    if diagnostics_frame.empty:
        diagnostics_frame = pd.DataFrame(
            columns=[
                "source_id",
                "spectrum_id",
                "repair_type",
                "replaced_band_count",
                "mean_abs_delta",
                "max_abs_delta",
            ]
        )
    diagnostics_frame.to_csv(diagnostics_dir / "artifact_repairs.csv", index=False)

    summary_frame = (
        diagnostics_frame.groupby(["source_id", "repair_type"], dropna=False)
        .agg(
            repaired_spectra=("spectrum_id", "nunique"),
            replaced_bands=("replaced_band_count", "sum"),
            max_abs_delta=("max_abs_delta", "max"),
        )
        .reset_index()
        .sort_values(["repaired_spectra", "replaced_bands"], ascending=[False, False])
    )
    summary_frame.to_csv(diagnostics_dir / "artifact_repairs_summary.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "total_rows": int(total_rows),
        "repaired_spectra": int(diagnostics_frame["spectrum_id"].nunique()) if not diagnostics_frame.empty else 0,
        "repair_rows": int(len(diagnostics_frame)),
        "top_sources": summary_frame.head(10).to_dict(orient="records"),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair EMIT absorption spikes and Santa tail artifacts in normalized spectra.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    summary = run_repair(Path(args.base_root), Path(args.output_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
