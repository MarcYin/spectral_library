#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 512
SOURCE_ID = "ghisacasia_v001"
BLEND_HALF_WINDOW_NM = 50

DEEP_WINDOW_RULES = [
    {
        "repair_type": "ghisacasia_deep_absorption_1400_interp",
        "replace_start": 1330,
        "replace_end": 1455,
        "left_start": 1280,
        "left_end": 1320,
        "right_start": 1460,
        "right_end": 1500,
        "positive_dev_threshold": 0.015,
    },
    {
        "repair_type": "ghisacasia_deep_absorption_1900_interp",
        "replace_start": 1770,
        "replace_end": 1985,
        "left_start": 1720,
        "left_end": 1760,
        "right_start": 1990,
        "right_end": 2030,
        "positive_dev_threshold": 0.015,
    },
]

TAIL_RULE = {
    "repair_type": "ghisacasia_tail_linear_extrapolation",
    "fit_start": 2240,
    "fit_end": 2390,
    "replace_start": 2400,
    "replace_end": 2500,
    "drift_threshold": 0.05,
}


def copy_support_files(base_root: Path, output_root: Path) -> None:
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("landcover_analysis/landcover_labels.csv"),
        Path("landcover_analysis/landcover_group_summary.csv"),
        Path("landcover_analysis/landcover_analysis_summary.json"),
    ]:
        source_path = base_root / relative
        if source_path.exists():
            target_path = output_root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _anchored_line(
    wavelengths: np.ndarray,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
    replace_start: int,
    replace_end: int,
    row: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    left_mask = (wavelengths >= left_start) & (wavelengths <= left_end)
    right_mask = (wavelengths >= right_start) & (wavelengths <= right_end)
    replace_mask = (wavelengths >= replace_start) & (wavelengths <= replace_end)
    if not left_mask.any() or not right_mask.any() or not replace_mask.any():
        return None, float("nan")
    left_values = row[left_mask]
    right_values = row[right_mask]
    if np.isfinite(left_values).sum() < 2 or np.isfinite(right_values).sum() < 2:
        return None, float("nan")

    x0 = float(np.mean(wavelengths[left_mask]))
    x1 = float(np.mean(wavelengths[right_mask]))
    y0 = float(np.nanmean(left_values))
    y1 = float(np.nanmean(right_values))
    x = wavelengths[replace_mask].astype(float)
    predicted = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return np.clip(predicted, 0.0, 1.0), x0


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


def repair_deep_windows(values: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    fixed = values.copy()
    diagnostics: list[dict[str, object]] = []
    for row_index in range(values.shape[0]):
        row = fixed[row_index].copy()
        for rule in DEEP_WINDOW_RULES:
            predicted, _ = _anchored_line(
                wavelengths,
                rule["left_start"],
                rule["left_end"],
                rule["right_start"],
                rule["right_end"],
                rule["replace_start"],
                rule["replace_end"],
                row,
            )
            if predicted is None:
                continue
            replace_mask = (wavelengths >= rule["replace_start"]) & (wavelengths <= rule["replace_end"])
            current = row[replace_mask]
            positive_dev = current - predicted
            if not np.isfinite(positive_dev).any():
                continue
            max_dev = float(np.nanmax(positive_dev))
            if max_dev < float(rule["positive_dev_threshold"]):
                continue

            blend_start = rule["replace_start"] - BLEND_HALF_WINDOW_NM
            blend_end = rule["replace_end"] + BLEND_HALF_WINDOW_NM
            blend_mask = (wavelengths >= blend_start) & (wavelengths <= blend_end)
            predicted_blend, _ = _anchored_line(
                wavelengths,
                rule["left_start"],
                rule["left_end"],
                rule["right_start"],
                rule["right_end"],
                blend_start,
                blend_end,
                row,
            )
            if predicted_blend is None:
                continue
            current_blend = row[blend_mask]
            valid = np.isfinite(current_blend)
            weights = _two_sided_blend_weights(
                wavelengths[blend_mask],
                rule["replace_start"],
                rule["replace_end"],
                BLEND_HALF_WINDOW_NM,
            )
            blended = (1.0 - weights) * current_blend + weights * predicted_blend
            row[blend_mask] = np.where(valid, blended, current_blend)
            delta = np.abs(current_blend[valid] - blended[valid])
            diagnostics.append(
                {
                    "row_index": row_index,
                    "repair_type": str(rule["repair_type"]),
                    "replaced_band_count": int(valid.sum()),
                    "mean_abs_delta": float(np.mean(delta)),
                    "max_abs_delta": float(np.max(delta)),
                }
            )
        fixed[row_index] = row
    return fixed, diagnostics


def repair_tail(values: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    fixed = values.copy()
    diagnostics: list[dict[str, object]] = []
    fit_mask = (wavelengths >= TAIL_RULE["fit_start"]) & (wavelengths <= TAIL_RULE["fit_end"])
    replace_mask = (wavelengths >= TAIL_RULE["replace_start"]) & (wavelengths <= TAIL_RULE["replace_end"])
    fit_x = wavelengths[fit_mask].astype(float)
    blend_start = TAIL_RULE["replace_start"] - BLEND_HALF_WINDOW_NM
    blend_mask = (wavelengths >= blend_start) & (wavelengths <= TAIL_RULE["replace_end"])
    blend_x = wavelengths[blend_mask].astype(float)
    for row_index in range(values.shape[0]):
        fit_y = fixed[row_index, fit_mask]
        replace_y = fixed[row_index, replace_mask]
        fit_valid = np.isfinite(fit_y)
        replace_valid = np.isfinite(replace_y)
        if fit_valid.sum() < 2 or replace_valid.sum() < 1:
            continue
        slope, intercept = np.polyfit(fit_x[fit_valid], fit_y[fit_valid], 1)
        predicted = np.clip(slope * blend_x + intercept, 0.0, 1.0)
        replace_in_blend = replace_mask[blend_mask]
        predicted_replace = predicted[replace_in_blend]
        drift = np.abs(replace_y[replace_valid] - predicted_replace[replace_valid])
        max_drift = float(np.max(drift))
        if max_drift < float(TAIL_RULE["drift_threshold"]):
            continue
        current_blend = fixed[row_index, blend_mask]
        blend_valid = np.isfinite(current_blend)
        weights = _left_boundary_blend_weights(
            wavelengths[blend_mask],
            TAIL_RULE["replace_start"],
            BLEND_HALF_WINDOW_NM,
        )
        blended = (1.0 - weights) * current_blend + weights * predicted
        fixed[row_index, blend_mask] = np.where(blend_valid, blended, current_blend)
        blend_delta = np.abs(current_blend[blend_valid] - blended[blend_valid])
        diagnostics.append(
            {
                "row_index": row_index,
                "repair_type": str(TAIL_RULE["repair_type"]),
                "replaced_band_count": int(blend_valid.sum()),
                "mean_abs_delta": float(np.mean(blend_delta)),
                "max_abs_delta": float(np.max(blend_delta)),
            }
        )
    return fixed, diagnostics


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

    diagnostics_rows: list[dict[str, object]] = []
    total_rows = 0
    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            total_rows += len(chunk)
            keep = chunk.copy()
            mask = keep["source_id"].astype(str).eq(SOURCE_ID)
            if mask.any():
                source_rows = keep.loc[mask, spectral_columns].to_numpy(dtype=float)
                source_rows, deep_diags = repair_deep_windows(source_rows, wavelengths)
                source_rows, tail_diags = repair_tail(source_rows, wavelengths)
                keep.loc[mask, spectral_columns] = source_rows
                spectrum_ids = keep.loc[mask, "spectrum_id"].astype(str).tolist()
                for diag in deep_diags + tail_diags:
                    diagnostics_rows.append(
                        {
                            "source_id": SOURCE_ID,
                            "spectrum_id": spectrum_ids[int(diag["row_index"])],
                            "repair_type": diag["repair_type"],
                            "replaced_band_count": diag["replaced_band_count"],
                            "mean_abs_delta": diag["mean_abs_delta"],
                            "max_abs_delta": diag["max_abs_delta"],
                        }
                    )
            keep.to_csv(handle, index=False, header=not header_written)
            header_written = True

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
    diagnostics_frame.to_csv(diagnostics_dir / "ghisacasia_repairs.csv", index=False)
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
    summary_frame.to_csv(diagnostics_dir / "ghisacasia_repairs_summary.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "total_rows": int(total_rows),
        "repaired_spectra": int(diagnostics_frame["spectrum_id"].nunique()) if not diagnostics_frame.empty else 0,
        "repair_rows": int(len(diagnostics_frame)),
        "top_repairs": summary_frame.to_dict(orient="records"),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair ghisacasia deep-water-band and tail artifacts in normalized spectra.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_repair(Path(args.base_root), Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
