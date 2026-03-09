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


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    lam: float
    abs_threshold: float


WINDOWS = [
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


def process_chunk(
    chunk: pd.DataFrame,
    spectral_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Counter[str]]:
    values = chunk[spectral_columns].to_numpy(dtype=float)
    source_ids = chunk["source_id"].astype(str).tolist()
    spectrum_ids = chunk["spectrum_id"].astype(str).tolist()

    replaced_counts = np.zeros(values.shape[0], dtype=int)
    outlier_rows: list[dict[str, object]] = []
    by_source: Counter[str] = Counter()

    for window in WINDOWS:
        start_idx = window.start - 400
        end_idx = window.end - 400 + 1
        window_values = values[:, start_idx:end_idx].astype(np.float64, copy=True)
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

        window_values[flagged] = smoothed[flagged]
        values[:, start_idx:end_idx] = window_values

        for row_index in np.flatnonzero(flagged.any(axis=1)):
            band_mask = flagged[row_index]
            band_indices = np.flatnonzero(band_mask)
            band_count = int(len(band_indices))
            replaced_counts[row_index] += band_count
            by_source[source_ids[row_index]] += band_count
            outlier_rows.append(
                {
                    "source_id": source_ids[row_index],
                    "spectrum_id": spectrum_ids[row_index],
                    "window_start_nm": window.start,
                    "window_end_nm": window.end,
                    "flagged_band_count": band_count,
                    "first_flagged_nm": int(window.start + band_indices[0]),
                    "last_flagged_nm": int(window.start + band_indices[-1]),
                    "max_abs_residual": float(np.nanmax(np.abs(residual[row_index, band_mask]))),
                    "min_weight": float(np.nanmin(weights[row_index, band_mask])),
                }
            )

    chunk.loc[:, spectral_columns] = values
    flags = pd.DataFrame(
        {
            "source_id": source_ids,
            "spectrum_id": spectrum_ids,
            "flagged_band_count": replaced_counts,
        }
    )
    flags = flags[flags["flagged_band_count"] > 0].reset_index(drop=True)
    return chunk, flags, pd.DataFrame(outlier_rows), by_source


def run_fix(base_root: Path, output_root: Path) -> dict[str, object]:
    copy_static_files(base_root, output_root)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    output_diag = output_root / "diagnostics"
    output_diag.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]

    total_flagged_spectra = 0
    total_flagged_bands = 0
    source_counter: Counter[str] = Counter()
    spectrum_flags: list[pd.DataFrame] = []
    outlier_windows: list[pd.DataFrame] = []

    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            fixed_chunk, flag_frame, outlier_frame, by_source = process_chunk(chunk.copy(), spectral_columns)
            fixed_chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

            if not flag_frame.empty:
                total_flagged_spectra += int(len(flag_frame))
                total_flagged_bands += int(flag_frame["flagged_band_count"].sum())
                spectrum_flags.append(flag_frame)
            source_counter.update(by_source)

            if not outlier_frame.empty:
                outlier_windows.append(outlier_frame)

    spectrum_flags_frame = (
        pd.concat(spectrum_flags, ignore_index=True)
        if spectrum_flags
        else pd.DataFrame(columns=["source_id", "spectrum_id", "flagged_band_count"])
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

    source_summary = pd.DataFrame(
        [
            {"source_id": source_id, "flagged_band_count": int(count)}
            for source_id, count in sorted(source_counter.items(), key=lambda item: (-item[1], item[0]))
        ]
    )
    source_summary.to_csv(output_diag / "source_flag_summary.csv", index=False)

    return {
        "flagged_spectra": total_flagged_spectra,
        "flagged_bands": total_flagged_bands,
        "top_sources": source_summary.head(10).to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply robust Whittaker smoothing to all spectra, identify outlier bands, and replace obvious outliers.")
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
        "method": {
            "smoother": "robust_whittaker_irls_f64",
            "weight_threshold": WEIGHT_THRESHOLD,
            "windows": [window.__dict__ for window in WINDOWS],
        },
        "stats": stats,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
