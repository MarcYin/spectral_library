#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANDCOVER_ORDER = ["soil", "vegetation", "urban", "water"]
CHUNK_SIZE = 2048


def build_group_map(labels_path: Path) -> dict[str, str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    keys = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    return dict(zip(keys, labels["landcover_group"]))


def aggregate_dataset(
    normalized_csv: Path,
    group_map: dict[str, str],
    wavelengths: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int]]:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    pooled_sums = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}
    pooled_counts = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}
    source_sums: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(len(wavelengths), dtype=float))
    source_counts: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(len(wavelengths), dtype=float))
    source_spectra: dict[tuple[str, str], int] = defaultdict(int)

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        groups = keys.map(group_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].isin(LANDCOVER_ORDER)]
        if chunk.empty:
            continue

        for group, group_frame in chunk.groupby("landcover_group"):
            values = group_frame[columns[2:]].to_numpy(dtype=float)
            valid = np.isfinite(values)
            pooled_sums[group] += np.where(valid, values, 0.0).sum(axis=0)
            pooled_counts[group] += valid.sum(axis=0)

            for source_id, source_frame in group_frame.groupby("source_id"):
                key = (group, str(source_id))
                source_values = source_frame[columns[2:]].to_numpy(dtype=float)
                source_valid = np.isfinite(source_values)
                source_sums[key] += np.where(source_valid, source_values, 0.0).sum(axis=0)
                source_counts[key] += source_valid.sum(axis=0)
                source_spectra[key] += int(len(source_frame))

    return pooled_sums, pooled_counts, source_sums, source_counts, source_spectra


def build_curves(
    pooled_sums: dict[str, np.ndarray],
    pooled_counts: dict[str, np.ndarray],
    source_sums: dict[tuple[str, str], np.ndarray],
    source_counts: dict[tuple[str, str], np.ndarray],
    wavelengths: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], dict[str, int]]:
    pooled_means: dict[str, np.ndarray] = {}
    source_balanced_means: dict[str, np.ndarray] = {}
    common_support_means: dict[str, np.ndarray] = {}
    source_support_counts: dict[str, np.ndarray] = {}
    source_totals: dict[str, int] = {}
    common_source_totals: dict[str, int] = {}

    for group in LANDCOVER_ORDER:
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled_means[group] = pooled_sums[group] / pooled_counts[group]

        group_sources = [source_id for current_group, source_id in source_sums if current_group == group]
        source_totals[group] = len(group_sources)
        if not group_sources:
            source_balanced_means[group] = np.full(len(wavelengths), np.nan)
            common_support_means[group] = np.full(len(wavelengths), np.nan)
            source_support_counts[group] = np.zeros(len(wavelengths), dtype=float)
            common_source_totals[group] = 0
            continue

        per_source_means = []
        per_source_supported = []
        common_support_sources = []
        for source_id in group_sources:
            key = (group, source_id)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_values = source_sums[key] / source_counts[key]
            per_source_means.append(mean_values)
            supported = source_counts[key] > 0
            per_source_supported.append(supported)
            if bool(np.all(supported)):
                common_support_sources.append(mean_values)

        matrix = np.vstack(per_source_means)
        support_matrix = np.vstack(per_source_supported)
        support_counts = support_matrix.sum(axis=0)
        source_support_counts[group] = support_counts
        with np.errstate(divide="ignore", invalid="ignore"):
            source_balanced_means[group] = np.nansum(matrix, axis=0) / support_counts

        common_source_totals[group] = len(common_support_sources)
        if common_support_sources:
            common_matrix = np.vstack(common_support_sources)
            common_support_means[group] = np.nanmean(common_matrix, axis=0)
        else:
            common_support_means[group] = np.full(len(wavelengths), np.nan)

    return (
        pooled_means,
        source_balanced_means,
        common_support_means,
        source_support_counts,
        source_totals,
        common_source_totals,
    )


def curves_to_frame(
    dataset_label: str,
    wavelengths: np.ndarray,
    pooled_means: dict[str, np.ndarray],
    source_balanced_means: dict[str, np.ndarray],
    common_support_means: dict[str, np.ndarray],
    source_support_counts: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in LANDCOVER_ORDER:
        for wavelength, pooled, source_balanced, common_support, source_support in zip(
            wavelengths,
            pooled_means[group],
            source_balanced_means[group],
            common_support_means[group],
            source_support_counts[group],
        ):
            rows.append(
                {
                    "dataset": dataset_label,
                    "landcover_group": group,
                    "wavelength_nm": int(wavelength),
                    "pooled_mean": float(pooled) if np.isfinite(pooled) else np.nan,
                    "source_balanced_mean": float(source_balanced) if np.isfinite(source_balanced) else np.nan,
                    "common_support_source_mean": float(common_support) if np.isfinite(common_support) else np.nan,
                    "source_support_count": int(source_support),
                }
            )
    return pd.DataFrame(rows)


def summary_to_frame(
    dataset_label: str,
    wavelengths: np.ndarray,
    pooled_means: dict[str, np.ndarray],
    source_balanced_means: dict[str, np.ndarray],
    common_support_means: dict[str, np.ndarray],
    pooled_counts: dict[str, np.ndarray],
    source_totals: dict[str, int],
    common_source_totals: dict[str, int],
) -> pd.DataFrame:
    key_wavelengths = [400, 450, 460, 500]
    rows: list[dict[str, object]] = []
    for group in LANDCOVER_ORDER:
        row = {
            "dataset": dataset_label,
            "landcover_group": group,
            "source_total": source_totals[group],
            "common_support_source_total": common_source_totals[group],
        }
        for wavelength in key_wavelengths:
            idx = int(wavelength - wavelengths[0])
            row[f"pooled_nm_{wavelength}"] = float(pooled_means[group][idx]) if np.isfinite(pooled_means[group][idx]) else np.nan
            row[f"source_balanced_nm_{wavelength}"] = float(source_balanced_means[group][idx]) if np.isfinite(source_balanced_means[group][idx]) else np.nan
            row[f"common_support_nm_{wavelength}"] = float(common_support_means[group][idx]) if np.isfinite(common_support_means[group][idx]) else np.nan
            row[f"pooled_support_nm_{wavelength}"] = int(pooled_counts[group][idx])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dataset(
    output_path: Path,
    title: str,
    wavelengths: np.ndarray,
    pooled_means: dict[str, np.ndarray],
    source_balanced_means: dict[str, np.ndarray],
    common_support_means: dict[str, np.ndarray],
    source_support_counts: dict[str, np.ndarray],
    source_totals: dict[str, int],
    common_source_totals: dict[str, int],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        axis.plot(wavelengths, pooled_means[group], color="#868e96", linewidth=1.7, label="Raw pooled mean")
        axis.plot(wavelengths, source_balanced_means[group], color="#d9480f", linewidth=1.8, label="Equal-weight source mean")
        if np.isfinite(common_support_means[group]).any():
            axis.plot(wavelengths, common_support_means[group], color="#0b7285", linewidth=1.8, label="Common-support source mean")
        axis.axvline(500, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{group} (sources={source_totals[group]}, common={common_source_totals[group]})"
        )

        support_axis = axis.twinx()
        support_axis.plot(wavelengths, source_support_counts[group], color="#f08c00", linewidth=1.0, alpha=0.35)
        support_axis.set_ylabel("Source support", color="#f08c00")
        support_axis.tick_params(axis="y", colors="#f08c00")

    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean reflectance")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def process_dataset(
    dataset_label: str,
    normalized_csv: Path,
    group_map: dict[str, str],
    wavelengths: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_sums, pooled_counts, source_sums, source_counts, _ = aggregate_dataset(normalized_csv, group_map, wavelengths)
    (
        pooled_means,
        source_balanced_means,
        common_support_means,
        source_support_counts,
        source_totals,
        common_source_totals,
    ) = build_curves(pooled_sums, pooled_counts, source_sums, source_counts, wavelengths)

    curves = curves_to_frame(
        dataset_label,
        wavelengths,
        pooled_means,
        source_balanced_means,
        common_support_means,
        source_support_counts,
    )
    summary = summary_to_frame(
        dataset_label,
        wavelengths,
        pooled_means,
        source_balanced_means,
        common_support_means,
        pooled_counts,
        source_totals,
        common_source_totals,
    )

    plot_dataset(
        output_dir / f"source_balanced_{dataset_label}.png",
        f"{dataset_label.replace('_', ' ').title()} blue-region composites",
        wavelengths,
        pooled_means,
        source_balanced_means,
        common_support_means,
        source_support_counts,
        source_totals,
        common_source_totals,
    )
    return curves, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pooled means against source-balanced landcover composites.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--repaired-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wavelength-start", type=int, default=400)
    parser.add_argument("--wavelength-end", type=int, default=550)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    repaired_root = Path(args.repaired_root) if args.repaired_root else None
    output_dir = Path(args.output_dir) if args.output_dir else base_root / "landcover_analysis" / "source_balanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.arange(args.wavelength_start, args.wavelength_end + 1, dtype=int)
    group_map = build_group_map(base_root / "landcover_analysis" / "landcover_labels.csv")

    base_curves, base_summary = process_dataset(
        "original_normalized",
        base_root / "tabular" / "normalized_spectra.csv",
        group_map,
        wavelengths,
        output_dir,
    )

    frames = [base_curves]
    summaries = [base_summary]
    generated_plots = [str(output_dir / "source_balanced_original_normalized.png")]

    if repaired_root is not None:
        repaired_curves, repaired_summary = process_dataset(
            "repaired_dataset",
            repaired_root / "tabular" / "normalized_spectra.csv",
            group_map,
            wavelengths,
            output_dir,
        )
        frames.append(repaired_curves)
        summaries.append(repaired_summary)
        generated_plots.append(str(output_dir / "source_balanced_repaired_dataset.png"))

    pd.concat(frames, ignore_index=True).to_csv(output_dir / "source_balanced_curves.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(output_dir / "source_balanced_summary.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "repaired_root": str(repaired_root) if repaired_root is not None else "",
        "output_dir": str(output_dir),
        "plots": generated_plots,
        "wavelength_start": int(wavelengths[0]),
        "wavelength_end": int(wavelengths[-1]),
    }
    (output_dir / "source_balanced_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
