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


LANDCOVER_ORDER = ["soil", "vegetation", "urban", "water"]
CHUNK_SIZE = 2048


def load_wavelengths(csv_path: Path) -> np.ndarray:
    header = pd.read_csv(csv_path, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    return np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)


def build_group_map(labels_path: Path) -> dict[str, str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    keys = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    return dict(zip(keys, labels["landcover_group"].astype(str)))


def aggregate_group_means(normalized_csv: Path, group_map: dict[str, str], wavelengths: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    sums = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}
    counts = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk.assign(landcover_group=keys.map(group_map))
        chunk = chunk[chunk["landcover_group"].isin(LANDCOVER_ORDER)]
        if chunk.empty:
            continue

        spectral = chunk[columns[2:]].to_numpy(dtype=float)
        valid = np.isfinite(spectral)
        for group, group_frame in chunk.groupby("landcover_group"):
            group_values = group_frame[columns[2:]].to_numpy(dtype=float)
            group_valid = np.isfinite(group_values)
            sums[str(group)] += np.where(group_valid, group_values, 0.0).sum(axis=0)
            counts[str(group)] += group_valid.sum(axis=0)

    means: dict[str, np.ndarray] = {}
    for group in LANDCOVER_ORDER:
        with np.errstate(divide="ignore", invalid="ignore"):
            means[group] = sums[group] / counts[group]
    return means, counts


def sample_changed_spectra(
    flag_counts_path: Path,
    labels_path: Path,
    per_group: int,
    seed: int,
) -> pd.DataFrame:
    flags = pd.read_csv(flag_counts_path, low_memory=False)
    flags = flags[flags["replaced_visible_band_count"] > 0].copy()
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    merged = flags.merge(labels, on=["source_id", "spectrum_id"], how="left")
    merged = merged[merged["landcover_group"].isin(LANDCOVER_ORDER)].copy()

    sampled: list[pd.DataFrame] = []
    for group in LANDCOVER_ORDER:
        group_frame = merged[merged["landcover_group"] == group].copy()
        if group_frame.empty:
            continue
        take = min(per_group, len(group_frame))
        sampled.append(group_frame.sample(n=take, random_state=seed + LANDCOVER_ORDER.index(group)))

    if not sampled:
        return pd.DataFrame(columns=list(merged.columns))

    sampled_frame = pd.concat(sampled, ignore_index=True)
    sampled_frame = sampled_frame.sort_values(
        by=["landcover_group", "replaced_visible_band_count", "source_id", "spectrum_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return sampled_frame


def extract_spectra_rows(normalized_csv: Path, wavelengths: np.ndarray, sample_keys: set[str]) -> pd.DataFrame:
    columns = ["source_id", "spectrum_id", "sample_name"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        subset = chunk[keys.isin(sample_keys)].copy()
        if not subset.empty:
            rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(rows, ignore_index=True)
    combined["sample_key"] = combined["source_id"].astype(str) + "||" + combined["spectrum_id"].astype(str)
    return combined


def plot_group_means(
    wavelengths: np.ndarray,
    base_means: dict[str, np.ndarray],
    fixed_means: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        axis.plot(wavelengths, base_means[group], label="Before", color="#4C78A8", linewidth=1.8)
        axis.plot(wavelengths, fixed_means[group], label="After", color="#F58518", linewidth=1.8)
        axis.set_title(group.title())
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Mean reflectance")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Full-Spectrum Mean Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_group_deltas(
    wavelengths: np.ndarray,
    base_means: dict[str, np.ndarray],
    fixed_means: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        delta = fixed_means[group] - base_means[group]
        axis.plot(wavelengths, delta, color="#54A24B", linewidth=1.6)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"{group.title()} (After - Before)")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Reflectance delta")
        axis.grid(alpha=0.2)
    fig.suptitle("Full-Spectrum Mean Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_random_samples(
    wavelengths: np.ndarray,
    sampled: pd.DataFrame,
    base_rows: pd.DataFrame,
    fixed_rows: pd.DataFrame,
    output_full: Path,
    output_visible: Path,
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame()

    base_index = base_rows.set_index("sample_key")
    fixed_index = fixed_rows.set_index("sample_key")
    spectral_columns = [f"nm_{wavelength}" for wavelength in wavelengths]
    visible_mask = (wavelengths >= 400) & (wavelengths <= 700)

    n = len(sampled)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig_full, axes_full = plt.subplots(rows, cols, figsize=(16, rows * 3.8), sharex=True)
    fig_vis, axes_vis = plt.subplots(rows, cols, figsize=(16, rows * 3.8), sharex=True)
    axes_full = np.atleast_1d(axes_full).ravel()
    axes_vis = np.atleast_1d(axes_vis).ravel()

    summary_rows: list[dict[str, object]] = []
    for axis_full, axis_vis, sample in zip(axes_full, axes_vis, sampled.itertuples(index=False)):
        sample_key = f"{sample.source_id}||{sample.spectrum_id}"
        if sample_key not in base_index.index or sample_key not in fixed_index.index:
            continue
        before = base_index.loc[sample_key, spectral_columns].to_numpy(dtype=float)
        after = fixed_index.loc[sample_key, spectral_columns].to_numpy(dtype=float)
        changed = np.isfinite(before) & np.isfinite(after) & (np.abs(after - before) > 1e-12)

        axis_full.plot(wavelengths, before, color="#4C78A8", linewidth=1.2, label="Before")
        axis_full.plot(wavelengths, after, color="#F58518", linewidth=1.2, label="After")
        axis_full.scatter(wavelengths[changed], after[changed], color="#E45756", s=6, alpha=0.7, label="Changed bands")
        axis_full.set_title(
            f"{sample.landcover_group} | {sample.source_id}\n{sample.spectrum_id} ({int(sample.replaced_visible_band_count)} visible bands)"
        )
        axis_full.set_xlabel("Wavelength (nm)")
        axis_full.set_ylabel("Reflectance")
        axis_full.grid(alpha=0.2)

        axis_vis.plot(wavelengths[visible_mask], before[visible_mask], color="#4C78A8", linewidth=1.4, label="Before")
        axis_vis.plot(wavelengths[visible_mask], after[visible_mask], color="#F58518", linewidth=1.4, label="After")
        axis_vis.scatter(
            wavelengths[visible_mask & changed],
            after[visible_mask & changed],
            color="#E45756",
            s=10,
            alpha=0.75,
            label="Changed bands",
        )
        axis_vis.set_title(
            f"{sample.landcover_group} | {sample.source_id}\n{sample.spectrum_id}"
        )
        axis_vis.set_xlabel("Wavelength (nm)")
        axis_vis.set_ylabel("Reflectance")
        axis_vis.grid(alpha=0.2)

        summary_rows.append(
            {
                "landcover_group": sample.landcover_group,
                "source_id": sample.source_id,
                "spectrum_id": sample.spectrum_id,
                "sample_name": base_index.loc[sample_key, "sample_name"],
                "detected_band_count": int(sample.detected_band_count),
                "detected_visible_band_count": int(sample.detected_visible_band_count),
                "replaced_visible_band_count": int(sample.replaced_visible_band_count),
                "full_changed_band_count": int(changed.sum()),
                "max_abs_delta": float(np.nanmax(np.abs(after - before))),
                "mean_abs_delta_visible": float(np.nanmean(np.abs(after[visible_mask] - before[visible_mask]))),
            }
        )

    for figure_axes in (axes_full, axes_vis):
        for axis in figure_axes[n:]:
            axis.set_visible(False)

    handles, labels = axes_full[0].get_legend_handles_labels()
    fig_full.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig_full.suptitle("Random Changed Spectra: Full Spectrum")
    fig_full.tight_layout(rect=(0, 0, 1, 0.96))
    fig_full.savefig(output_full, dpi=180)
    plt.close(fig_full)

    handles, labels = axes_vis[0].get_legend_handles_labels()
    fig_vis.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig_vis.suptitle("Random Changed Spectra: Visible Window")
    fig_vis.tight_layout(rect=(0, 0, 1, 0.96))
    fig_vis.savefig(output_visible, dpi=180)
    plt.close(fig_vis)

    return pd.DataFrame(summary_rows)


def write_summary(
    sampled: pd.DataFrame,
    group_means_before: dict[str, np.ndarray],
    group_means_after: dict[str, np.ndarray],
    wavelengths: np.ndarray,
    base_label: str,
    fixed_label: str,
    output_path: Path,
) -> None:
    summary = {
        "base_dataset": base_label,
        "fixed_dataset": fixed_label,
        "sample_count": int(len(sampled)),
        "groups_sampled": sampled["landcover_group"].value_counts().to_dict() if not sampled.empty else {},
        "mean_delta_visible_max_abs": {},
    }
    visible_mask = (wavelengths >= 400) & (wavelengths <= 700)
    for group in LANDCOVER_ORDER:
        delta = group_means_after[group][visible_mask] - group_means_before[group][visible_mask]
        summary["mean_delta_visible_max_abs"][group] = float(np.nanmax(np.abs(delta)))
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full-spectrum and random before/after comparisons.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260309)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    wavelengths = load_wavelengths(args.fixed_root / "tabular" / "normalized_spectra.csv")
    base_group_map = build_group_map(args.base_root / "landcover_analysis" / "landcover_labels.csv")
    fixed_group_map = build_group_map(args.fixed_root / "landcover_analysis" / "landcover_labels.csv")

    base_means, _ = aggregate_group_means(args.base_root / "tabular" / "normalized_spectra.csv", base_group_map, wavelengths)
    fixed_means, _ = aggregate_group_means(args.fixed_root / "tabular" / "normalized_spectra.csv", fixed_group_map, wavelengths)
    plot_group_means(wavelengths, base_means, fixed_means, args.output_root / "full_mean_curve_comparison.png")
    plot_group_deltas(wavelengths, base_means, fixed_means, args.output_root / "full_mean_delta.png")

    sampled = sample_changed_spectra(
        args.fixed_root / "diagnostics" / "spectrum_flag_counts.csv",
        args.fixed_root / "landcover_analysis" / "landcover_labels.csv",
        per_group=args.samples_per_group,
        seed=args.seed,
    )
    sampled.to_csv(args.output_root / "random_sample_spectra.csv", index=False)

    if not sampled.empty:
        sample_keys = set(sampled["source_id"].astype(str) + "||" + sampled["spectrum_id"].astype(str))
        base_rows = extract_spectra_rows(args.base_root / "tabular" / "normalized_spectra.csv", wavelengths, sample_keys)
        fixed_rows = extract_spectra_rows(args.fixed_root / "tabular" / "normalized_spectra.csv", wavelengths, sample_keys)
        sample_summary = plot_random_samples(
            wavelengths,
            sampled,
            base_rows,
            fixed_rows,
            args.output_root / "random_spectra_full_comparison.png",
            args.output_root / "random_spectra_visible_comparison.png",
        )
        sample_summary.to_csv(args.output_root / "random_sample_summary.csv", index=False)

    write_summary(
        sampled,
        base_means,
        fixed_means,
        wavelengths,
        args.base_root.name,
        args.fixed_root.name,
        args.output_root / "comparison_summary.json",
    )


if __name__ == "__main__":
    main()
