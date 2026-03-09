#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LANDCOVER_ORDER = ["soil", "vegetation", "urban", "water"]
CHUNK_SIZE = 2048
TOP_JUMPS_PER_GROUP = 5
TOP_SOURCE_CONTRIBUTORS = 8
TOP_SPECTRUM_OUTLIERS = 25
MAD_EPS = 1e-9


def build_label_maps(labels_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"].isin(LANDCOVER_ORDER)].copy()
    key_series = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    group_map = dict(zip(key_series, labels["landcover_group"]))
    source_map = dict(zip(key_series, labels["source_id"].astype(str)))
    return group_map, source_map


def aggregate_source_curves(
    normalized_csv: Path,
    group_map: dict[str, str],
    wavelengths: np.ndarray,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    source_sums: dict[tuple[str, str], np.ndarray] = {}
    source_counts: dict[tuple[str, str], np.ndarray] = {}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        groups = keys.map(group_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].isin(LANDCOVER_ORDER)]
        if chunk.empty:
            continue

        for (group, source_id), frame in chunk.groupby(["landcover_group", "source_id"]):
            values = frame[columns[2:]].to_numpy(dtype=float)
            valid = np.isfinite(values)
            key = (str(group), str(source_id))
            if key not in source_sums:
                source_sums[key] = np.zeros(len(wavelengths), dtype=float)
                source_counts[key] = np.zeros(len(wavelengths), dtype=float)
            source_sums[key] += np.where(valid, values, 0.0).sum(axis=0)
            source_counts[key] += valid.sum(axis=0)

    return source_sums, source_counts


def compute_group_jump_tables(
    source_sums: dict[tuple[str, str], np.ndarray],
    source_counts: dict[tuple[str, str], np.ndarray],
    wavelengths: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jump_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []

    for group in LANDCOVER_ORDER:
        source_ids = [source_id for current_group, source_id in source_sums if current_group == group]
        if not source_ids:
            continue

        counts_matrix = np.vstack([source_counts[(group, source_id)] for source_id in source_ids])
        with np.errstate(divide="ignore", invalid="ignore"):
            means_matrix = np.vstack([source_sums[(group, source_id)] / source_counts[(group, source_id)] for source_id in source_ids])

        pooled_support = counts_matrix.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled_mean = np.nansum(means_matrix * counts_matrix, axis=0) / pooled_support

        adjacent_jump = pooled_mean[1:] - pooled_mean[:-1]
        jump_order = np.argsort(np.abs(adjacent_jump))[::-1][:TOP_JUMPS_PER_GROUP]
        for jump_rank, idx in enumerate(jump_order, start=1):
            left_nm = int(wavelengths[idx])
            right_nm = int(wavelengths[idx + 1])
            left_support = float(pooled_support[idx])
            right_support = float(pooled_support[idx + 1])
            jump_value = float(adjacent_jump[idx])

            left_total = left_support if left_support > 0 else np.nan
            right_total = right_support if right_support > 0 else np.nan
            left_weights = counts_matrix[:, idx] / left_total if np.isfinite(left_total) else np.zeros(len(source_ids))
            right_weights = counts_matrix[:, idx + 1] / right_total if np.isfinite(right_total) else np.zeros(len(source_ids))
            left_means = means_matrix[:, idx]
            right_means = means_matrix[:, idx + 1]

            contribution = right_weights * right_means - left_weights * left_means
            composition = (right_weights - left_weights) * right_means
            within_source = left_weights * (right_means - left_means)

            jump_rows.append(
                {
                    "landcover_group": group,
                    "jump_rank": jump_rank,
                    "left_nm": left_nm,
                    "right_nm": right_nm,
                    "pooled_left_mean": float(pooled_mean[idx]),
                    "pooled_right_mean": float(pooled_mean[idx + 1]),
                    "pooled_jump": jump_value,
                    "left_support": int(left_support),
                    "right_support": int(right_support),
                    "support_delta": int(right_support - left_support),
                }
            )

            contributor_order = np.argsort(np.abs(contribution))[::-1][:TOP_SOURCE_CONTRIBUTORS]
            for contributor_rank, source_idx in enumerate(contributor_order, start=1):
                contribution_rows.append(
                    {
                        "landcover_group": group,
                        "jump_rank": jump_rank,
                        "left_nm": left_nm,
                        "right_nm": right_nm,
                        "source_id": source_ids[source_idx],
                        "contributor_rank": contributor_rank,
                        "left_support": int(counts_matrix[source_idx, idx]),
                        "right_support": int(counts_matrix[source_idx, idx + 1]),
                        "left_weight": float(left_weights[source_idx]),
                        "right_weight": float(right_weights[source_idx]),
                        "left_mean": float(left_means[source_idx]) if np.isfinite(left_means[source_idx]) else np.nan,
                        "right_mean": float(right_means[source_idx]) if np.isfinite(right_means[source_idx]) else np.nan,
                        "net_contribution": float(contribution[source_idx]),
                        "composition_contribution": float(composition[source_idx]),
                        "within_source_contribution": float(within_source[source_idx]),
                    }
                )

    return pd.DataFrame(jump_rows), pd.DataFrame(contribution_rows)


def find_spectrum_outliers(
    normalized_csv: Path,
    group_map: dict[str, str],
    jump_table: pd.DataFrame,
    wavelengths: np.ndarray,
) -> pd.DataFrame:
    target_pairs: dict[str, set[tuple[int, int]]] = {}
    for group, frame in jump_table.groupby("landcover_group"):
        target_pairs[group] = {(int(row.left_nm), int(row.right_nm)) for row in frame.itertuples(index=False)}

    columns = ["source_id", "spectrum_id", "sample_name"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    jump_values_by_group_pair: dict[tuple[str, int, int], list[float]] = {}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        groups = keys.map(group_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].isin(target_pairs)]
        if chunk.empty:
            continue

        for group, frame in chunk.groupby("landcover_group"):
            values = frame[[f"nm_{wavelength}" for wavelength in wavelengths]].to_numpy(dtype=float)
            for left_nm, right_nm in target_pairs[group]:
                left_idx = int(left_nm - wavelengths[0])
                right_idx = int(right_nm - wavelengths[0])
                jump = values[:, right_idx] - values[:, left_idx]
                finite = np.isfinite(jump)
                if finite.any():
                    jump_values_by_group_pair.setdefault((group, left_nm, right_nm), []).extend(jump[finite].tolist())

    robust_stats: dict[tuple[str, int, int], tuple[float, float]] = {}
    for key, values in jump_values_by_group_pair.items():
        array = np.asarray(values, dtype=float)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)) * 1.4826)
        robust_stats[key] = (median, max(mad, MAD_EPS))

    outlier_rows: list[dict[str, object]] = []
    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        groups = keys.map(group_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].isin(target_pairs)]
        if chunk.empty:
            continue

        for row in chunk.itertuples(index=False):
            group = str(row.landcover_group)
            if group not in target_pairs:
                continue
            for left_nm, right_nm in target_pairs[group]:
                stat_key = (group, left_nm, right_nm)
                if stat_key not in robust_stats:
                    continue
                left_value = getattr(row, f"nm_{left_nm}")
                right_value = getattr(row, f"nm_{right_nm}")
                if not np.isfinite(left_value) or not np.isfinite(right_value):
                    continue
                jump = float(right_value - left_value)
                median, mad = robust_stats[stat_key]
                robust_z = abs(jump - median) / mad
                outlier_rows.append(
                    {
                        "landcover_group": group,
                        "left_nm": left_nm,
                        "right_nm": right_nm,
                        "source_id": str(row.source_id),
                        "spectrum_id": str(row.spectrum_id),
                        "sample_name": str(row.sample_name),
                        "jump_value": jump,
                        "group_median_jump": median,
                        "group_mad_jump": mad,
                        "robust_zscore": robust_z,
                    }
                )

    if not outlier_rows:
        return pd.DataFrame()

    outliers = pd.DataFrame(outlier_rows)
    outliers = (
        outliers.sort_values(
            ["landcover_group", "left_nm", "right_nm", "robust_zscore"],
            ascending=[True, True, True, False],
        )
        .groupby(["landcover_group", "left_nm", "right_nm"], as_index=False, group_keys=False)
        .head(TOP_SPECTRUM_OUTLIERS)
        .reset_index(drop=True)
    )
    return outliers


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose blue-region jumps by source and individual spectrum.")
    parser.add_argument("--normalized-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wavelength-start", type=int, default=400)
    parser.add_argument("--wavelength-end", type=int, default=550)
    args = parser.parse_args()

    normalized_root = Path(args.normalized_root)
    output_dir = Path(args.output_dir) if args.output_dir else normalized_root / "landcover_analysis" / "blue_jump_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.arange(args.wavelength_start, args.wavelength_end + 1, dtype=int)
    group_map, _ = build_label_maps(normalized_root / "landcover_analysis" / "landcover_labels.csv")

    source_sums, source_counts = aggregate_source_curves(
        normalized_root / "tabular" / "normalized_spectra.csv",
        group_map,
        wavelengths,
    )
    jump_table, contribution_table = compute_group_jump_tables(source_sums, source_counts, wavelengths)
    jump_table.to_csv(output_dir / "group_top_jumps.csv", index=False)
    contribution_table.to_csv(output_dir / "source_jump_contributions.csv", index=False)

    outliers = find_spectrum_outliers(
        normalized_root / "tabular" / "normalized_spectra.csv",
        group_map,
        jump_table,
        wavelengths,
    )
    outliers.to_csv(output_dir / "blue_jump_spectrum_outliers.csv", index=False)

    summary = {
        "normalized_root": str(normalized_root),
        "output_dir": str(output_dir),
        "groups_analyzed": sorted(jump_table["landcover_group"].unique().tolist()) if not jump_table.empty else [],
        "top_jumps_rows": int(len(jump_table)),
        "source_contribution_rows": int(len(contribution_table)),
        "spectrum_outlier_rows": int(len(outliers)),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
