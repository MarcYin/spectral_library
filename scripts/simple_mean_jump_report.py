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
TOP_SOURCE_CONTRIBUTORS = 8
TOP_SPECTRUM_OUTLIERS = 10
MAD_EPS = 1e-9


def build_group_map(labels_path: Path) -> dict[str, str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    keys = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    return dict(zip(keys, labels["landcover_group"].astype(str)))


def aggregate_curves(
    normalized_csv: Path,
    group_map: dict[str, str],
    wavelengths: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    pooled_sums = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}
    pooled_counts = {group: np.zeros(len(wavelengths), dtype=float) for group in LANDCOVER_ORDER}
    source_sums: dict[tuple[str, str], np.ndarray] = {}
    source_counts: dict[tuple[str, str], np.ndarray] = {}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk.assign(landcover_group=keys.map(group_map))
        chunk = chunk[chunk["landcover_group"].isin(LANDCOVER_ORDER)]
        if chunk.empty:
            continue

        for group, group_frame in chunk.groupby("landcover_group"):
            group_values = group_frame[columns[2:]].to_numpy(dtype=float)
            group_valid = np.isfinite(group_values)
            pooled_sums[str(group)] += np.where(group_valid, group_values, 0.0).sum(axis=0)
            pooled_counts[str(group)] += group_valid.sum(axis=0)

        for (group, source_id), source_frame in chunk.groupby(["landcover_group", "source_id"]):
            key = (str(group), str(source_id))
            source_values = source_frame[columns[2:]].to_numpy(dtype=float)
            source_valid = np.isfinite(source_values)
            if key not in source_sums:
                source_sums[key] = np.zeros(len(wavelengths), dtype=float)
                source_counts[key] = np.zeros(len(wavelengths), dtype=float)
            source_sums[key] += np.where(source_valid, source_values, 0.0).sum(axis=0)
            source_counts[key] += source_valid.sum(axis=0)

    return pooled_sums, pooled_counts, source_sums, source_counts


def compute_pooled_means(
    pooled_sums: dict[str, np.ndarray],
    pooled_counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pooled_means: dict[str, np.ndarray] = {}
    for group in LANDCOVER_ORDER:
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled_means[group] = pooled_sums[group] / pooled_counts[group]
    return pooled_means


def summarize_top_jumps(
    dataset_label: str,
    wavelengths: np.ndarray,
    pooled_means: dict[str, np.ndarray],
    pooled_counts: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in LANDCOVER_ORDER:
        mean_curve = pooled_means[group]
        if len(mean_curve) < 2:
            continue
        jumps = mean_curve[1:] - mean_curve[:-1]
        finite = np.isfinite(jumps)
        if not finite.any():
            continue
        jump_indices = np.flatnonzero(finite)
        best_idx = int(jump_indices[np.argmax(np.abs(jumps[finite]))])
        rows.append(
            {
                "dataset": dataset_label,
                "landcover_group": group,
                "left_nm": int(wavelengths[best_idx]),
                "right_nm": int(wavelengths[best_idx + 1]),
                "left_mean": float(mean_curve[best_idx]),
                "right_mean": float(mean_curve[best_idx + 1]),
                "jump_value": float(jumps[best_idx]),
                "left_support": int(pooled_counts[group][best_idx]),
                "right_support": int(pooled_counts[group][best_idx + 1]),
                "support_delta": int(pooled_counts[group][best_idx + 1] - pooled_counts[group][best_idx]),
            }
        )
    return pd.DataFrame(rows)


def compute_contributors(
    dataset_label: str,
    jump_summary: pd.DataFrame,
    source_sums: dict[tuple[str, str], np.ndarray],
    source_counts: dict[tuple[str, str], np.ndarray],
    wavelengths: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for jump_row in jump_summary.itertuples(index=False):
        group = str(jump_row.landcover_group)
        source_ids = [source_id for current_group, source_id in source_sums if current_group == group]
        if not source_ids:
            continue

        left_idx = int(jump_row.left_nm - wavelengths[0])
        right_idx = int(jump_row.right_nm - wavelengths[0])

        counts_matrix = np.vstack([source_counts[(group, source_id)] for source_id in source_ids])
        with np.errstate(divide="ignore", invalid="ignore"):
            means_matrix = np.vstack([source_sums[(group, source_id)] / source_counts[(group, source_id)] for source_id in source_ids])

        left_total = float(counts_matrix[:, left_idx].sum())
        right_total = float(counts_matrix[:, right_idx].sum())
        left_weights = counts_matrix[:, left_idx] / left_total if left_total > 0 else np.zeros(len(source_ids))
        right_weights = counts_matrix[:, right_idx] / right_total if right_total > 0 else np.zeros(len(source_ids))
        left_means = means_matrix[:, left_idx]
        right_means = means_matrix[:, right_idx]

        contribution = np.nan_to_num(right_weights * right_means, nan=0.0) - np.nan_to_num(
            left_weights * left_means,
            nan=0.0,
        )
        order = np.argsort(np.abs(np.nan_to_num(contribution, nan=0.0)))[::-1][:TOP_SOURCE_CONTRIBUTORS]
        for rank, source_idx in enumerate(order, start=1):
            rows.append(
                {
                    "dataset": dataset_label,
                    "landcover_group": group,
                    "left_nm": int(jump_row.left_nm),
                    "right_nm": int(jump_row.right_nm),
                    "source_id": source_ids[source_idx],
                    "contributor_rank": rank,
                    "left_support": int(counts_matrix[source_idx, left_idx]),
                    "right_support": int(counts_matrix[source_idx, right_idx]),
                    "left_mean": float(left_means[source_idx]) if np.isfinite(left_means[source_idx]) else np.nan,
                    "right_mean": float(right_means[source_idx]) if np.isfinite(right_means[source_idx]) else np.nan,
                    "net_contribution": float(contribution[source_idx]) if np.isfinite(contribution[source_idx]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def find_spectrum_outliers(
    dataset_label: str,
    normalized_csv: Path,
    group_map: dict[str, str],
    jump_summary: pd.DataFrame,
    wavelengths: np.ndarray,
) -> pd.DataFrame:
    jump_targets = {
        str(row.landcover_group): (int(row.left_nm), int(row.right_nm))
        for row in jump_summary.itertuples(index=False)
    }
    if not jump_targets:
        return pd.DataFrame()

    columns = ["source_id", "spectrum_id", "sample_name"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    robust_stats: dict[str, tuple[float, float]] = {}

    jump_values_by_group: dict[str, list[float]] = {group: [] for group in jump_targets}
    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk.assign(landcover_group=keys.map(group_map))
        chunk = chunk[chunk["landcover_group"].isin(jump_targets)]
        if chunk.empty:
            continue

        for group, frame in chunk.groupby("landcover_group"):
            left_nm, right_nm = jump_targets[str(group)]
            left_values = frame[f"nm_{left_nm}"].to_numpy(dtype=float)
            right_values = frame[f"nm_{right_nm}"].to_numpy(dtype=float)
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            if finite.any():
                jump_values_by_group[str(group)].extend((right_values[finite] - left_values[finite]).tolist())

    for group, values in jump_values_by_group.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)) * 1.4826)
        robust_stats[group] = (median, max(mad, MAD_EPS))

    rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk.assign(landcover_group=keys.map(group_map))
        chunk = chunk[chunk["landcover_group"].isin(jump_targets)]
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            group = str(row.landcover_group)
            if group not in robust_stats:
                continue
            left_nm, right_nm = jump_targets[group]
            left_value = getattr(row, f"nm_{left_nm}")
            right_value = getattr(row, f"nm_{right_nm}")
            if not np.isfinite(left_value) or not np.isfinite(right_value):
                continue
            jump_value = float(right_value - left_value)
            median, mad = robust_stats[group]
            rows.append(
                {
                    "dataset": dataset_label,
                    "landcover_group": group,
                    "left_nm": left_nm,
                    "right_nm": right_nm,
                    "source_id": str(row.source_id),
                    "spectrum_id": str(row.spectrum_id),
                    "sample_name": str(row.sample_name),
                    "jump_value": jump_value,
                    "group_median_jump": median,
                    "robust_zscore": abs(jump_value - median) / mad,
                }
            )

    if not rows:
        return pd.DataFrame()

    outliers = pd.DataFrame(rows)
    return (
        outliers.sort_values(
            ["dataset", "landcover_group", "robust_zscore"],
            ascending=[True, True, False],
        )
        .groupby(["dataset", "landcover_group"], as_index=False, group_keys=False)
        .head(TOP_SPECTRUM_OUTLIERS)
        .reset_index(drop=True)
    )


def plot_mean_curves(
    output_path: Path,
    wavelengths: np.ndarray,
    original_means: dict[str, np.ndarray],
    fixed_means: dict[str, np.ndarray],
    original_jumps: pd.DataFrame,
    fixed_jumps: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        axis.plot(wavelengths, original_means[group], color="#6c757d", linewidth=1.8, label="Original mean")
        axis.plot(wavelengths, fixed_means[group], color="#0b7285", linewidth=1.8, label="Fixed mean")
        original_jump = original_jumps[original_jumps["landcover_group"] == group].iloc[0]
        fixed_jump = fixed_jumps[fixed_jumps["landcover_group"] == group].iloc[0]
        axis.axvline(float(original_jump["left_nm"]), color="#6c757d", linestyle="--", linewidth=0.9, alpha=0.55)
        axis.axvline(float(fixed_jump["left_nm"]), color="#0b7285", linestyle="--", linewidth=0.9, alpha=0.55)
        axis.set_title(group)
        axis.grid(alpha=0.25)
        axis.text(
            0.02,
            0.98,
            f"orig max jump: {int(original_jump['left_nm'])}->{int(original_jump['right_nm'])} ({float(original_jump['jump_value']):+.4f})\n"
            f"fix max jump: {int(fixed_jump['left_nm'])}->{int(fixed_jump['right_nm'])} ({float(fixed_jump['jump_value']):+.4f})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean reflectance")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Simple landcover mean curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_jump_profiles(
    output_path: Path,
    wavelengths: np.ndarray,
    original_means: dict[str, np.ndarray],
    fixed_means: dict[str, np.ndarray],
) -> None:
    jump_wavelengths = wavelengths[1:]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        original_jump = original_means[group][1:] - original_means[group][:-1]
        fixed_jump = fixed_means[group][1:] - fixed_means[group][:-1]
        axis.plot(jump_wavelengths, original_jump, color="#6c757d", linewidth=1.5, label="Original step")
        axis.plot(jump_wavelengths, fixed_jump, color="#0b7285", linewidth=1.5, label="Fixed step")
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(group)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Right wavelength of adjacent step (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean step size")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Adjacent wavelength jumps in landcover mean curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_single_dataset_means(
    output_path: Path,
    title: str,
    wavelengths: np.ndarray,
    means: dict[str, np.ndarray],
    jumps: pd.DataFrame,
    color: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, group in zip(axes.ravel(), LANDCOVER_ORDER):
        axis.plot(wavelengths, means[group], color=color, linewidth=1.8)
        jump_row = jumps[jumps["landcover_group"] == group].iloc[0]
        axis.axvline(float(jump_row["left_nm"]), color=color, linestyle="--", linewidth=0.9, alpha=0.6)
        axis.set_title(group)
        axis.grid(alpha=0.25)
        axis.text(
            0.02,
            0.98,
            f"max jump: {int(jump_row['left_nm'])}->{int(jump_row['right_nm'])} ({float(jump_row['jump_value']):+.4f})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean reflectance")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_findings(
    output_path: Path,
    jump_summary: pd.DataFrame,
    contributors: pd.DataFrame,
) -> None:
    lines = ["# Simple Jump Findings", ""]
    for group in LANDCOVER_ORDER:
        lines.append(f"## {group}")
        for dataset in ("original", "fixed"):
            row = jump_summary[
                (jump_summary["dataset"] == dataset) & (jump_summary["landcover_group"] == group)
            ].iloc[0]
            lines.append(
                f"- {dataset}: max jump at {int(row['left_nm'])}->{int(row['right_nm'])} nm, "
                f"jump={float(row['jump_value']):+.6f}, support {int(row['left_support'])}->{int(row['right_support'])}."
            )
            top_sources = contributors[
                (contributors["dataset"] == dataset)
                & (contributors["landcover_group"] == group)
            ].head(3)
            if not top_sources.empty:
                source_text = ", ".join(
                    f"{source_row.source_id} ({float(source_row.net_contribution):+.4f})"
                    for source_row in top_sources.itertuples(index=False)
                )
                lines.append(f"  Top source contributors: {source_text}.")
        lines.append("")

    fixed_rows = {
        str(row.landcover_group): row
        for row in jump_summary[jump_summary["dataset"] == "fixed"].itertuples(index=False)
    }
    lines.extend(["## Proposed Fixes"])
    soil_jump = abs(float(fixed_rows["soil"].jump_value))
    vegetation_jump = abs(float(fixed_rows["vegetation"].jump_value))
    urban_jump = abs(float(fixed_rows["urban"].jump_value))
    water_jump = abs(float(fixed_rows["water"].jump_value))

    if soil_jump <= 0.005:
        lines.append("- `soil`: leave unchanged. The current mean is already smooth enough for this check.")
    else:
        lines.append("- `soil`: still has a meaningful jump; inspect the top spectrum outliers before changing the mean again.")

    if vegetation_jump <= 0.005:
        lines.append("- `vegetation`: fix is sufficient. The dominant visible-window jump is now small enough for this check.")
    else:
        lines.append("- `vegetation`: still has a meaningful residual jump; inspect the top contributing source and the flagged spectra around that band.")

    if urban_jump <= 0.005:
        lines.append("- `urban`: fix is sufficient. The dominant visible-window jump is now small enough for this check.")
    else:
        lines.append("- `urban`: still has a meaningful support-mix jump; use source-balanced curves or extend source-specific blue-edge repairs.")

    if water_jump <= 0.005:
        lines.append("- `water`: leave unchanged. No further action is needed from this jump check.")
    else:
        lines.append("- `water`: inspect the top contributing source before changing the mean.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simple mean-curve jump diagnostics for original vs fixed normalized spectra.")
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--fixed-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wavelength-start", type=int, default=400)
    parser.add_argument("--wavelength-end", type=int, default=550)
    args = parser.parse_args()

    original_root = Path(args.original_root)
    fixed_root = Path(args.fixed_root)
    output_dir = Path(args.output_dir) if args.output_dir else fixed_root / "landcover_analysis" / "simple_mean_jump_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.arange(args.wavelength_start, args.wavelength_end + 1, dtype=int)

    original_group_map = build_group_map(original_root / "landcover_analysis" / "landcover_labels.csv")
    fixed_group_map = build_group_map(fixed_root / "landcover_analysis" / "landcover_labels.csv")

    original_sums, original_counts, original_source_sums, original_source_counts = aggregate_curves(
        original_root / "tabular" / "normalized_spectra.csv",
        original_group_map,
        wavelengths,
    )
    fixed_sums, fixed_counts, fixed_source_sums, fixed_source_counts = aggregate_curves(
        fixed_root / "tabular" / "normalized_spectra.csv",
        fixed_group_map,
        wavelengths,
    )

    original_means = compute_pooled_means(original_sums, original_counts)
    fixed_means = compute_pooled_means(fixed_sums, fixed_counts)

    original_jumps = summarize_top_jumps("original", wavelengths, original_means, original_counts)
    fixed_jumps = summarize_top_jumps("fixed", wavelengths, fixed_means, fixed_counts)
    jump_summary = pd.concat([original_jumps, fixed_jumps], ignore_index=True)
    jump_summary.to_csv(output_dir / "group_jump_summary.csv", index=False)

    contributors = pd.concat(
        [
            compute_contributors("original", original_jumps, original_source_sums, original_source_counts, wavelengths),
            compute_contributors("fixed", fixed_jumps, fixed_source_sums, fixed_source_counts, wavelengths),
        ],
        ignore_index=True,
    )
    contributors.to_csv(output_dir / "top_source_contributors.csv", index=False)

    outliers = pd.concat(
        [
            find_spectrum_outliers("original", original_root / "tabular" / "normalized_spectra.csv", original_group_map, original_jumps, wavelengths),
            find_spectrum_outliers("fixed", fixed_root / "tabular" / "normalized_spectra.csv", fixed_group_map, fixed_jumps, wavelengths),
        ],
        ignore_index=True,
    )
    outliers.to_csv(output_dir / "top_spectrum_outliers.csv", index=False)

    plot_mean_curves(
        output_dir / "mean_curve_comparison.png",
        wavelengths,
        original_means,
        fixed_means,
        original_jumps,
        fixed_jumps,
    )
    plot_single_dataset_means(
        output_dir / "mean_curve_original.png",
        "Original mean curves by landcover",
        wavelengths,
        original_means,
        original_jumps,
        "#6c757d",
    )
    plot_single_dataset_means(
        output_dir / "mean_curve_fixed.png",
        "Fixed mean curves by landcover",
        wavelengths,
        fixed_means,
        fixed_jumps,
        "#0b7285",
    )
    plot_jump_profiles(
        output_dir / "mean_step_comparison.png",
        wavelengths,
        original_means,
        fixed_means,
    )
    write_findings(output_dir / "findings.md", jump_summary, contributors)

    summary = {
        "original_root": str(original_root),
        "fixed_root": str(fixed_root),
        "output_dir": str(output_dir),
        "plots": [
            str(output_dir / "mean_curve_comparison.png"),
            str(output_dir / "mean_curve_original.png"),
            str(output_dir / "mean_curve_fixed.png"),
            str(output_dir / "mean_step_comparison.png"),
        ],
        "tables": [
            str(output_dir / "group_jump_summary.csv"),
            str(output_dir / "top_source_contributors.csv"),
            str(output_dir / "top_spectrum_outliers.csv"),
            str(output_dir / "findings.md"),
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
