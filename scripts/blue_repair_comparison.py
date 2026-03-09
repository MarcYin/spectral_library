#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANDCOVER_ORDER = ["soil", "vegetation", "urban", "water"]
WAVELENGTHS = np.arange(400, 551, dtype=int)
WAVELENGTH_COLUMNS = [f"nm_{wavelength}" for wavelength in WAVELENGTHS]
BOUNDARY_COLUMNS = ["nm_499", "nm_500", "nm_501"]
CHUNK_SIZE = 2048


@dataclass
class CurveAggregate:
    spectra_count: dict[str, int]
    sums: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    boundary_jump_sum: dict[str, float]
    boundary_curvature_sum: dict[str, float]
    boundary_count: dict[str, int]


def aggregate_spectra(csv_path: Path, group_map: dict[str, str]) -> CurveAggregate:
    spectra_count = {group: 0 for group in LANDCOVER_ORDER}
    sums = {group: np.zeros(len(WAVELENGTHS), dtype=float) for group in LANDCOVER_ORDER}
    counts = {group: np.zeros(len(WAVELENGTHS), dtype=float) for group in LANDCOVER_ORDER}
    boundary_jump_sum = {group: 0.0 for group in LANDCOVER_ORDER}
    boundary_curvature_sum = {group: 0.0 for group in LANDCOVER_ORDER}
    boundary_count = {group: 0 for group in LANDCOVER_ORDER}

    usecols = ["source_id", "spectrum_id"] + WAVELENGTH_COLUMNS
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        groups = keys.map(group_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].isin(LANDCOVER_ORDER)]
        if chunk.empty:
            continue

        for group, group_frame in chunk.groupby("landcover_group"):
            spectra_count[group] += int(len(group_frame))
            values = group_frame[WAVELENGTH_COLUMNS].to_numpy(dtype=float)
            valid = np.isfinite(values)
            sums[group] += np.where(valid, values, 0.0).sum(axis=0)
            counts[group] += valid.sum(axis=0)

            boundary = group_frame[BOUNDARY_COLUMNS].to_numpy(dtype=float)
            jump = np.abs(boundary[:, 2] - boundary[:, 1])
            curvature = np.abs(boundary[:, 0] - 2.0 * boundary[:, 1] + boundary[:, 2])
            finite = np.isfinite(jump) & np.isfinite(curvature)
            boundary_jump_sum[group] += float(jump[finite].sum())
            boundary_curvature_sum[group] += float(curvature[finite].sum())
            boundary_count[group] += int(finite.sum())

    return CurveAggregate(
        spectra_count=spectra_count,
        sums=sums,
        counts=counts,
        boundary_jump_sum=boundary_jump_sum,
        boundary_curvature_sum=boundary_curvature_sum,
        boundary_count=boundary_count,
    )


def means_from_aggregate(aggregate: CurveAggregate) -> dict[str, np.ndarray]:
    means: dict[str, np.ndarray] = {}
    for group in LANDCOVER_ORDER:
        with np.errstate(divide="ignore", invalid="ignore"):
            means[group] = aggregate.sums[group] / aggregate.counts[group]
    return means


def load_full_group_map(labels_path: Path) -> dict[str, str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    keys = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    return dict(zip(keys, labels["landcover_group"]))


def load_blue_subset_map(flags_path: Path) -> dict[str, str]:
    flags = pd.read_csv(
        flags_path,
        usecols=["source_id", "spectrum_id", "landcover_group", "filled_noisy_blue_edge_points"],
        low_memory=False,
    )
    flags = flags[
        (flags["landcover_group"].isin(LANDCOVER_ORDER))
        & (flags["filled_noisy_blue_edge_points"] > 0)
    ].copy()
    keys = flags["source_id"].astype(str) + "||" + flags["spectrum_id"].astype(str)
    return dict(zip(keys, flags["landcover_group"]))


def curves_to_frame(curves: dict[str, dict[str, np.ndarray]], counts: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison_set, group_curves in curves.items():
        group_counts = counts[comparison_set]
        for group in LANDCOVER_ORDER:
            if group not in group_curves:
                continue
            mean_values = group_curves[group]
            support_counts = group_counts[group]
            if not np.isfinite(mean_values).any():
                continue
            for wavelength, mean_value, support in zip(WAVELENGTHS, mean_values, support_counts):
                rows.append(
                    {
                        "comparison_set": comparison_set,
                        "landcover_group": group,
                        "wavelength_nm": int(wavelength),
                        "mean_reflectance": float(mean_value) if np.isfinite(mean_value) else np.nan,
                        "support_count": int(support),
                    }
                )
    return pd.DataFrame(rows)


def boundary_metrics_frame(
    full_before: CurveAggregate,
    old_before: CurveAggregate,
    old_after: CurveAggregate,
    new_before: CurveAggregate,
    new_after: CurveAggregate,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in LANDCOVER_ORDER:
        rows.append(
            {
                "landcover_group": group,
                "full_group_spectra": full_before.spectra_count[group],
                "old_subset_spectra": old_before.spectra_count[group],
                "new_subset_spectra": new_before.spectra_count[group],
                "old_before_mean_abs_jump_500_501": old_before.boundary_jump_sum[group] / old_before.boundary_count[group]
                if old_before.boundary_count[group]
                else np.nan,
                "old_after_mean_abs_jump_500_501": old_after.boundary_jump_sum[group] / old_after.boundary_count[group]
                if old_after.boundary_count[group]
                else np.nan,
                "new_before_mean_abs_jump_500_501": new_before.boundary_jump_sum[group] / new_before.boundary_count[group]
                if new_before.boundary_count[group]
                else np.nan,
                "new_after_mean_abs_jump_500_501": new_after.boundary_jump_sum[group] / new_after.boundary_count[group]
                if new_after.boundary_count[group]
                else np.nan,
                "old_before_mean_abs_curvature_499_500_501": old_before.boundary_curvature_sum[group] / old_before.boundary_count[group]
                if old_before.boundary_count[group]
                else np.nan,
                "old_after_mean_abs_curvature_499_500_501": old_after.boundary_curvature_sum[group] / old_after.boundary_count[group]
                if old_after.boundary_count[group]
                else np.nan,
                "new_before_mean_abs_curvature_499_500_501": new_before.boundary_curvature_sum[group] / new_before.boundary_count[group]
                if new_before.boundary_count[group]
                else np.nan,
                "new_after_mean_abs_curvature_499_500_501": new_after.boundary_curvature_sum[group] / new_after.boundary_count[group]
                if new_after.boundary_count[group]
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    output_path: Path,
    title_prefix: str,
    before_means: dict[str, np.ndarray],
    after_means: dict[str, np.ndarray] | None = None,
    full_means: dict[str, np.ndarray] | None = None,
    before_counts: dict[str, int] | None = None,
    after_counts: dict[str, int] | None = None,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        before_curve = before_means.get(group)
        if before_curve is None or not np.isfinite(before_curve).any():
            axis.set_visible(False)
            continue

        if full_means is not None:
            full_curve = full_means.get(group)
            if full_curve is not None and np.isfinite(full_curve).any():
                axis.plot(WAVELENGTHS, full_curve, color="#adb5bd", linewidth=1.3, label="Full-group original")

        axis.plot(WAVELENGTHS, before_curve, color="#495057", linewidth=1.7, label="Subset original")
        if after_means is not None:
            after_curve = after_means.get(group)
            if after_curve is not None and np.isfinite(after_curve).any():
                axis.plot(WAVELENGTHS, after_curve, color="#0b7285", linewidth=1.9, label="Subset repaired")

        spectrum_note = ""
        if before_counts is not None:
            spectrum_note = f"n={before_counts.get(group, 0)}"
        if after_counts is not None and before_counts is not None and after_counts.get(group, 0) != before_counts.get(group, 0):
            spectrum_note = f"n={before_counts.get(group, 0)}/{after_counts.get(group, 0)}"
        title = group if not spectrum_note else f"{group} ({spectrum_note})"
        axis.set_title(title)
        axis.axvline(500, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.grid(alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean reflectance")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(title_prefix)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate clean before/after comparisons for blue-region PCA repairs.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    output_dir = Path(args.output_dir) if args.output_dir else new_root / "plots" / "clean_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_table = base_root / "tabular" / "normalized_spectra.csv"
    old_table = old_root / "tabular" / "normalized_spectra.csv"
    new_table = new_root / "tabular" / "normalized_spectra.csv"

    full_group_map = load_full_group_map(base_root / "landcover_analysis" / "landcover_labels.csv")
    old_subset_map = load_blue_subset_map(old_root / "tabular" / "postprocess_flags.csv")
    new_subset_map = load_blue_subset_map(new_root / "tabular" / "postprocess_flags.csv")

    full_before = aggregate_spectra(base_table, full_group_map)
    old_before = aggregate_spectra(base_table, old_subset_map)
    old_after = aggregate_spectra(old_table, old_subset_map)
    new_before = aggregate_spectra(base_table, new_subset_map)
    new_after = aggregate_spectra(new_table, new_subset_map)

    full_means = means_from_aggregate(full_before)
    old_before_means = means_from_aggregate(old_before)
    old_after_means = means_from_aggregate(old_after)
    new_before_means = means_from_aggregate(new_before)
    new_after_means = means_from_aggregate(new_after)

    boundary_metrics = boundary_metrics_frame(full_before, old_before, old_after, new_before, new_after)
    boundary_metrics.to_csv(output_dir / "blue_subset_boundary_metrics.csv", index=False)

    counts_frame = pd.DataFrame(
        {
            "landcover_group": LANDCOVER_ORDER,
            "full_group_spectra": [full_before.spectra_count[group] for group in LANDCOVER_ORDER],
            "old_subset_spectra": [old_before.spectra_count[group] for group in LANDCOVER_ORDER],
            "new_subset_spectra": [new_before.spectra_count[group] for group in LANDCOVER_ORDER],
        }
    )
    counts_frame.to_csv(output_dir / "blue_subset_counts.csv", index=False)

    curves = {
        "full_group_original": full_means,
        "old_subset_original": old_before_means,
        "old_subset_repaired": old_after_means,
        "new_subset_original": new_before_means,
        "new_subset_repaired": new_after_means,
    }
    curve_counts = {
        "full_group_original": full_before.counts,
        "old_subset_original": old_before.counts,
        "old_subset_repaired": old_after.counts,
        "new_subset_original": new_before.counts,
        "new_subset_repaired": new_after.counts,
    }
    curves_to_frame(curves, curve_counts).to_csv(output_dir / "blue_curve_means.csv", index=False)

    plot_comparison(
        output_dir / "blue_full_group_baseline.png",
        "Original full-group mean spectra in the blue region",
        before_means=full_means,
        before_counts=full_before.spectra_count,
    )
    plot_comparison(
        output_dir / "blue_old_subset_before_after.png",
        "Old full-spectrum blue repair subset",
        before_means=old_before_means,
        after_means=old_after_means,
        full_means=full_means,
        before_counts=old_before.spectra_count,
        after_counts=old_after.spectra_count,
    )
    plot_comparison(
        output_dir / "blue_new_subset_before_after.png",
        "New 501-900 to 400-500 blue repair subset",
        before_means=new_before_means,
        after_means=new_after_means,
        full_means=full_means,
        before_counts=new_before.spectra_count,
        after_counts=new_after.spectra_count,
    )

    summary = {
        "base_root": str(base_root),
        "old_root": str(old_root),
        "new_root": str(new_root),
        "output_dir": str(output_dir),
        "full_group_spectra": counts_frame.set_index("landcover_group")["full_group_spectra"].to_dict(),
        "old_subset_spectra": counts_frame.set_index("landcover_group")["old_subset_spectra"].to_dict(),
        "new_subset_spectra": counts_frame.set_index("landcover_group")["new_subset_spectra"].to_dict(),
    }
    (output_dir / "blue_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
