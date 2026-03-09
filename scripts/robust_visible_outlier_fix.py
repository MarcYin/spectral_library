#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from whitsmooth_rust import robust_whittaker_irls_f64


CHUNK_SIZE = 128
VISIBLE_START = 400
VISIBLE_END = 700
VISIBLE_LAM = 50.0
VISIBLE_ABS_THRESHOLD = 0.005


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    lam: float
    abs_threshold: float


DETECT_WINDOWS = [
    WindowSpec(400, 550, 100.0, 0.005),
    WindowSpec(550, 760, 50.0, 0.010),
    WindowSpec(760, 1350, 100.0, 0.010),
    WindowSpec(1350, 1450, 50.0, 0.010),
    WindowSpec(1450, 1800, 100.0, 0.010),
    WindowSpec(1800, 2000, 50.0, 0.010),
    WindowSpec(2000, 2450, 100.0, 0.010),
]
WEIGHT_THRESHOLD = 0.10


def copy_static_files(base_root: Path, output_root: Path) -> None:
    output_tabular = output_root / "tabular"
    output_landcover = output_root / "landcover_analysis"
    output_tabular.mkdir(parents=True, exist_ok=True)
    output_landcover.mkdir(parents=True, exist_ok=True)

    for name in ["spectra_metadata.csv", "source_summary.csv", "wavelength_grid.csv", "normalization_failures.csv"]:
        source_path = base_root / "tabular" / name
        if source_path.exists():
            shutil.copy2(source_path, output_tabular / name)

    for name in ["landcover_labels.csv"]:
        source_path = base_root / "landcover_analysis" / name
        if source_path.exists():
            shutil.copy2(source_path, output_landcover / name)


def detect_outliers(values: np.ndarray, source_ids: list[str], spectrum_ids: list[str]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    spectral_count = values.shape[1]
    flagged_all = np.zeros((values.shape[0], spectral_count), dtype=bool)
    outlier_rows: list[dict[str, object]] = []

    for window in DETECT_WINDOWS:
        start_idx = window.start - 400
        end_idx = window.end - 400 + 1
        window_values = values[:, start_idx:end_idx].astype(np.float64, copy=False)
        if window_values.size == 0:
            continue

        x = np.arange(window.start, window.end + 1, dtype=np.float64)
        smoothed, weights = robust_whittaker_irls_f64(
            x,
            window_values,
            lam=window.lam,
            d=2,
            iterations=8,
            weighting="tukey",
            scale="mad",
            parallel=True,
            return_weights=True,
            merge_x_tol=0.0,
        )
        residual = window_values - smoothed
        finite = np.isfinite(window_values)
        flagged = finite & (weights < WEIGHT_THRESHOLD) & (np.abs(residual) > window.abs_threshold)
        if not flagged.any():
            continue

        flagged_all[:, start_idx:end_idx] |= flagged

        for row_index in np.flatnonzero(flagged.any(axis=1)):
            band_mask = flagged[row_index]
            band_indices = np.flatnonzero(band_mask)
            outlier_rows.append(
                {
                    "source_id": source_ids[row_index],
                    "spectrum_id": spectrum_ids[row_index],
                    "window_start_nm": window.start,
                    "window_end_nm": window.end,
                    "flagged_band_count": int(len(band_indices)),
                    "first_flagged_nm": int(window.start + band_indices[0]),
                    "last_flagged_nm": int(window.start + band_indices[-1]),
                    "max_abs_residual": float(np.nanmax(np.abs(residual[row_index, band_mask]))),
                    "min_weight": float(np.nanmin(weights[row_index, band_mask])),
                }
            )

    band_counts = flagged_all.sum(axis=1)
    flag_frame = pd.DataFrame(
        {
            "source_id": source_ids,
            "spectrum_id": spectrum_ids,
            "detected_band_count": band_counts,
            "detected_visible_band_count": flagged_all[:, VISIBLE_START - 400 : VISIBLE_END - 400 + 1].sum(axis=1),
        }
    )
    flag_frame = flag_frame[(flag_frame["detected_band_count"] > 0) | (flag_frame["detected_visible_band_count"] > 0)].reset_index(drop=True)
    return flagged_all, flag_frame, pd.DataFrame(outlier_rows)


def replace_visible_bands(values: np.ndarray, flagged_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visible_slice = slice(VISIBLE_START - 400, VISIBLE_END - 400 + 1)
    visible_flags = flagged_all[:, visible_slice]
    if not visible_flags.any():
        return values, np.zeros(values.shape[0], dtype=int)

    visible_values = values[:, visible_slice].astype(np.float64, copy=True)
    x = np.arange(VISIBLE_START, VISIBLE_END + 1, dtype=np.float64)
    smoothed = robust_whittaker_irls_f64(
        x,
        visible_values,
        lam=VISIBLE_LAM,
        d=2,
        iterations=8,
        weighting="tukey",
        scale="mad",
        parallel=True,
        return_weights=False,
        merge_x_tol=0.0,
    )
    visible_values[visible_flags] = smoothed[visible_flags]
    values[:, visible_slice] = visible_values
    return values, visible_flags.sum(axis=1)


def process_chunk(chunk: pd.DataFrame, spectral_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Counter[str], Counter[str]]:
    values = chunk[spectral_columns].to_numpy(dtype=float)
    source_ids = chunk["source_id"].astype(str).tolist()
    spectrum_ids = chunk["spectrum_id"].astype(str).tolist()

    flagged_all, flag_frame, outlier_frame = detect_outliers(values, source_ids, spectrum_ids)
    values, replaced_visible_counts = replace_visible_bands(values, flagged_all)
    chunk.loc[:, spectral_columns] = values

    replace_frame = pd.DataFrame(
        {
            "source_id": source_ids,
            "spectrum_id": spectrum_ids,
            "replaced_visible_band_count": replaced_visible_counts,
        }
    )
    replace_frame = replace_frame[replace_frame["replaced_visible_band_count"] > 0].reset_index(drop=True)

    detected_by_source: Counter[str] = Counter()
    replaced_by_source: Counter[str] = Counter()
    for row in flag_frame.itertuples(index=False):
        detected_by_source[str(row.source_id)] += int(row.detected_band_count)
    for row in replace_frame.itertuples(index=False):
        replaced_by_source[str(row.source_id)] += int(row.replaced_visible_band_count)

    merged = flag_frame.merge(replace_frame, on=["source_id", "spectrum_id"], how="outer").fillna(0)
    merged["detected_band_count"] = merged["detected_band_count"].astype(int)
    merged["detected_visible_band_count"] = merged["detected_visible_band_count"].astype(int)
    merged["replaced_visible_band_count"] = merged["replaced_visible_band_count"].astype(int)
    return chunk, merged, outlier_frame, detected_by_source, replaced_by_source


def run_fix(base_root: Path, output_root: Path) -> dict[str, object]:
    copy_static_files(base_root, output_root)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    output_diag = output_root / "diagnostics"
    output_diag.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]

    detected_spectra = 0
    detected_bands = 0
    replaced_visible_spectra = 0
    replaced_visible_bands = 0
    detected_source_counter: Counter[str] = Counter()
    replaced_source_counter: Counter[str] = Counter()
    spectrum_flags: list[pd.DataFrame] = []
    outlier_windows: list[pd.DataFrame] = []

    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            fixed_chunk, flag_frame, outlier_frame, detected_by_source, replaced_by_source = process_chunk(chunk.copy(), spectral_columns)
            fixed_chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

            if not flag_frame.empty:
                detected_spectra += int((flag_frame["detected_band_count"] > 0).sum())
                detected_bands += int(flag_frame["detected_band_count"].sum())
                replaced_visible_spectra += int((flag_frame["replaced_visible_band_count"] > 0).sum())
                replaced_visible_bands += int(flag_frame["replaced_visible_band_count"].sum())
                spectrum_flags.append(flag_frame)
            detected_source_counter.update(detected_by_source)
            replaced_source_counter.update(replaced_by_source)

            if not outlier_frame.empty:
                outlier_windows.append(outlier_frame)

    spectrum_flags_frame = (
        pd.concat(spectrum_flags, ignore_index=True)
        if spectrum_flags
        else pd.DataFrame(
            columns=[
                "source_id",
                "spectrum_id",
                "detected_band_count",
                "detected_visible_band_count",
                "replaced_visible_band_count",
            ]
        )
    )
    spectrum_flags_frame.to_csv(output_diag / "spectrum_flag_counts.csv", index=False)

    outlier_windows_frame = (
        pd.concat(outlier_windows, ignore_index=True)
        if outlier_windows
        else pd.DataFrame(
            columns=[
                "source_id",
                "spectrum_id",
                "window_start_nm",
                "window_end_nm",
                "flagged_band_count",
                "first_flagged_nm",
                "last_flagged_nm",
                "max_abs_residual",
                "min_weight",
            ]
        )
    )
    outlier_windows_frame.to_csv(output_diag / "window_outlier_flags.csv", index=False)

    source_ids = sorted(set(detected_source_counter) | set(replaced_source_counter))
    source_summary = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "detected_band_count": int(detected_source_counter.get(source_id, 0)),
                "replaced_visible_band_count": int(replaced_source_counter.get(source_id, 0)),
            }
            for source_id in source_ids
        ]
    ).sort_values(["replaced_visible_band_count", "detected_band_count", "source_id"], ascending=[False, False, True])
    source_summary.to_csv(output_diag / "source_flag_summary.csv", index=False)

    return {
        "detected_spectra": detected_spectra,
        "detected_bands": detected_bands,
        "replaced_visible_spectra": replaced_visible_spectra,
        "replaced_visible_bands": replaced_visible_bands,
        "top_sources": source_summary.head(10).to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Use the robust all-spectra smoother only as a detector, then replace flagged visible bands for flagged spectra.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = run_fix(base_root, output_root)
    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "detector": {
            "smoother": "robust_whittaker_irls_f64",
            "weight_threshold": WEIGHT_THRESHOLD,
            "windows": [window.__dict__ for window in DETECT_WINDOWS],
        },
        "replacement": {
            "window_nm": [VISIBLE_START, VISIBLE_END],
            "lam": VISIBLE_LAM,
            "abs_threshold": VISIBLE_ABS_THRESHOLD,
            "mode": "replace_detected_visible_bands_only",
        },
        "stats": stats,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
