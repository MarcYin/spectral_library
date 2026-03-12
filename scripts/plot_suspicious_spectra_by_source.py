#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHUNK_SIZE = 2048
FULL_WINDOW = (400, 2500)
VISIBLE_WINDOW = (400, 1000)
LANDCOVER_COLORS = {
    "soil": "#8C6D31",
    "vegetation": "#2E7D32",
    "urban": "#616161",
    "water": "#1976D2",
    "unlabeled": "#7B1FA2",
    "unclassified": "#7B1FA2",
}


def truncate_label(value: str, max_len: int = 22) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def nm_columns(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    header = pd.read_csv(csv_path, nrows=0)
    columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in columns], dtype=int)
    return wavelengths, columns


def load_flagged(flagged_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(flagged_csv, low_memory=False)
    frame["landcover_group"] = frame["landcover_group"].fillna("unlabeled").replace({"": "unlabeled"})
    frame["sample_key"] = frame["source_id"].astype(str) + "||" + frame["spectrum_id"].astype(str)
    return frame


def load_spectra_subset(spectra_csv: Path, sample_keys: set[str], spectral_columns: list[str]) -> pd.DataFrame:
    usecols = ["source_id", "spectrum_id", "sample_name"] + spectral_columns
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(spectra_csv, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["sample_key"] = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        subset = chunk[chunk["sample_key"].isin(sample_keys)].copy()
        if not subset.empty:
            parts.append(subset)
    if not parts:
        return pd.DataFrame(columns=usecols + ["sample_key"])
    return pd.concat(parts, ignore_index=True)


def plot_grid(
    frame: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: Path,
    title: str,
    wavelength_window: tuple[int, int],
) -> None:
    if frame.empty:
        return
    spectral_columns = [f"nm_{wavelength}" for wavelength in wavelengths]
    spectral_index = spectral_frame.set_index("sample_key")
    mask = (wavelengths >= wavelength_window[0]) & (wavelengths <= wavelength_window[1])
    n = len(frame)
    cols = 5
    rows = int(np.ceil(n / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(20, max(8, rows * 2.6)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, sample in zip(axes, frame.itertuples(index=False)):
        if sample.sample_key not in spectral_index.index:
            axis.set_visible(False)
            continue
        row = spectral_index.loc[sample.sample_key, spectral_columns]
        values = np.asarray(row, dtype=float)
        color = LANDCOVER_COLORS.get(str(sample.landcover_group), LANDCOVER_COLORS["unlabeled"])
        axis.plot(wavelengths[mask], values[mask], color=color, linewidth=1.0)
        axis.set_title(
            f"{truncate_label(sample.sample_name)}\nscore={float(sample.strange_score):.1f}",
            fontsize=7,
        )
        axis.grid(alpha=0.2)
    for axis in axes[n:]:
        axis.set_visible(False)
    for axis in axes[::cols]:
        if axis.get_visible():
            axis.set_ylabel("Reflectance")
    for axis in axes[-cols:]:
        if axis.get_visible():
            axis.set_xlabel("Wavelength (nm)")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_mean_overlay(
    frame: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if frame.empty:
        return
    spectral_columns = [f"nm_{wavelength}" for wavelength in wavelengths]
    merged = frame.merge(spectral_frame[["sample_key"] + spectral_columns], on="sample_key", how="inner")
    matrix = merged[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean_curve = np.nanmean(matrix, axis=0)
    p10 = np.nanpercentile(matrix, 10, axis=0)
    p90 = np.nanpercentile(matrix, 90, axis=0)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.fill_between(wavelengths, p10, p90, color="#9ecae1", alpha=0.35, label="p10-p90")
    axis.plot(wavelengths, mean_curve, color="#08519c", linewidth=1.8, label="mean suspicious")
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Reflectance")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot flagged suspicious spectra grouped by source.")
    parser.add_argument("--root", required=True, help="SIAC package root.")
    parser.add_argument(
        "--flagged-csv",
        default=None,
        help="Path to flagged suspicious spectra CSV. Defaults to <root>/full_review/flagged_suspicious_spectra.csv",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory. Defaults to <root>/full_review/source_suspicious_plots",
    )
    args = parser.parse_args()

    root = Path(args.root)
    flagged_csv = Path(args.flagged_csv) if args.flagged_csv else root / "full_review" / "flagged_suspicious_spectra.csv"
    output_root = Path(args.output_root) if args.output_root else root / "full_review" / "source_suspicious_plots"
    output_root.mkdir(parents=True, exist_ok=True)

    flagged = load_flagged(flagged_csv)
    spectra_csv = root / "tabular" / "siac_normalized_spectra.csv"
    wavelengths, spectral_columns = nm_columns(spectra_csv)
    spectra = load_spectra_subset(spectra_csv, set(flagged["sample_key"]), spectral_columns)

    summary_rows: list[dict[str, object]] = []
    for source_id, frame in flagged.groupby("source_id", sort=True):
        source_name = str(frame["source_name"].iloc[0])
        source_dir = output_root / source_id
        frame = frame.sort_values(["strange_score", "sample_name", "spectrum_id"], ascending=[False, True, True]).copy()
        plot_grid(
            frame,
            spectra,
            wavelengths,
            source_dir / "suspicious_full_spectrum.png",
            f"{source_id} suspicious spectra ({len(frame)})",
            FULL_WINDOW,
        )
        plot_grid(
            frame,
            spectra,
            wavelengths,
            source_dir / "suspicious_visible.png",
            f"{source_id} suspicious spectra, visible ({len(frame)})",
            VISIBLE_WINDOW,
        )
        plot_mean_overlay(
            frame,
            spectra,
            wavelengths,
            source_dir / "suspicious_mean_overlay.png",
            f"{source_id} suspicious mean and spread",
        )
        summary_rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "suspicious_spectra": int(len(frame)),
                "landcover_groups": "; ".join(
                    f"{group}:{count}" for group, count in Counter(frame["landcover_group"]).most_common()
                ),
                "full_plot": str(source_dir / "suspicious_full_spectrum.png"),
                "visible_plot": str(source_dir / "suspicious_visible.png"),
                "mean_overlay_plot": str(source_dir / "suspicious_mean_overlay.png"),
            }
        )

    summary_csv = output_root / "source_plot_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "source_name",
                "suspicious_spectra",
                "landcover_groups",
                "full_plot",
                "visible_plot",
                "mean_overlay_plot",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    run_summary = {
        "root": str(root),
        "flagged_csv": str(flagged_csv),
        "output_root": str(output_root),
        "total_suspicious_spectra": int(len(flagged)),
        "source_count": int(flagged["source_id"].nunique()),
        "plots_per_source": 3,
        "source_plot_summary": str(summary_csv),
    }
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
