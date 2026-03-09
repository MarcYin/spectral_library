#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_SOURCES = [
    "hyspiri_ground_targets",
    "ghisacasia_v001",
    "ngee_arctic_2018",
    "santa_barbara_urban_reflectance",
]
WINDOWS_BY_SOURCE = {
    "hyspiri_ground_targets": [(1330, 1455), (1770, 1985), (2300, 2500)],
    "ghisacasia_v001": [(1330, 1455), (1770, 1985), (2300, 2500)],
    "ngee_arctic_2018": [(1330, 1455), (1770, 1985), (2300, 2500)],
    "santa_barbara_urban_reflectance": [(2300, 2500)],
}
CHUNK_SIZE = 512


def mean_curves(dataset_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    csv_path = dataset_root / "tabular" / "normalized_spectra.csv"
    header = pd.read_csv(csv_path, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)
    usecols = ["source_id"] + spectral_columns

    sums = {source_id: np.zeros(len(wavelengths), dtype=float) for source_id in TARGET_SOURCES}
    counts = {source_id: np.zeros(len(wavelengths), dtype=float) for source_id in TARGET_SOURCES}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk[chunk["source_id"].isin(TARGET_SOURCES)]
        if chunk.empty:
            continue
        for source_id, frame in chunk.groupby("source_id"):
            values = frame[spectral_columns].to_numpy(dtype=float)
            valid = np.isfinite(values)
            sums[str(source_id)] += np.where(valid, values, 0.0).sum(axis=0)
            counts[str(source_id)] += valid.sum(axis=0)

    means: dict[str, np.ndarray] = {}
    for source_id in TARGET_SOURCES:
        with np.errstate(divide="ignore", invalid="ignore"):
            means[source_id] = sums[source_id] / counts[source_id]
    return wavelengths, means


def plot_overview(
    wavelengths: np.ndarray,
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for axis, source_id in zip(axes.ravel(), TARGET_SOURCES):
        axis.plot(wavelengths, before[source_id], color="#4C78A8", linewidth=1.4, label="Before")
        axis.plot(wavelengths, after[source_id], color="#F58518", linewidth=1.4, label="After")
        for start, end in WINDOWS_BY_SOURCE[source_id]:
            axis.axvspan(start, end, color="#9e9e9e", alpha=0.12)
        axis.set_title(source_id)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Mean reflectance")
        axis.grid(alpha=0.2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Source-specific absorption smoothing overview")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_deltas(
    wavelengths: np.ndarray,
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_id in TARGET_SOURCES:
        for start, end in WINDOWS_BY_SOURCE[source_id]:
            mask = (wavelengths >= start) & (wavelengths <= end)
            delta = np.abs(after[source_id][mask] - before[source_id][mask])
            rows.append(
                {
                    "source_id": source_id,
                    "start_nm": start,
                    "end_nm": end,
                    "mean_abs_delta": float(np.nanmean(delta)),
                    "max_abs_delta": float(np.nanmax(delta)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot before/after source-specific absorption smoothing means.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    fixed_root = Path(args.fixed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    wavelengths, before = mean_curves(base_root)
    _, after = mean_curves(fixed_root)

    overview_path = output_root / "absorption_smoothing_overview.png"
    plot_overview(wavelengths, before, after, overview_path)
    delta_path = output_root / "absorption_smoothing_window_deltas.csv"
    deltas = write_deltas(wavelengths, before, after, delta_path)

    summary = {
        "base_root": str(base_root),
        "fixed_root": str(fixed_root),
        "output_root": str(output_root),
        "overview_plot": str(overview_path),
        "delta_table": str(delta_path),
        "window_count": int(len(deltas)),
    }
    (output_root / "absorption_smoothing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
