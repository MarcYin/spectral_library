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


CHUNK_SIZE = 2048
URBAN_MEAN_MIN_END_NM = 2450.0
LANDCOVER_COLORS = {
    "soil": "#8C6D31",
    "vegetation": "#2E7D32",
    "urban": "#616161",
    "water": "#1976D2",
    "unclassified": "#7B1FA2",
}


def truncate_label(value: str, max_len: int = 28) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def load_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tabular = root / "tabular"
    metadata = pd.read_csv(tabular / "siac_spectra_metadata.csv", low_memory=False)
    source_summary = pd.read_csv(tabular / "siac_source_summary.csv", low_memory=False)
    landcover_summary = pd.read_csv(tabular / "siac_landcover_summary.csv", low_memory=False)
    excluded_path = tabular / "siac_excluded_sources.csv"
    if excluded_path.exists():
        excluded = pd.read_csv(excluded_path, low_memory=False)
    else:
        excluded = pd.DataFrame(columns=["source_id", "source_name", "available_spectra_count", "exclusion_reason"])

    metadata["landcover_group"] = metadata["landcover_group"].fillna("unclassified")
    metadata["classification_rule"] = metadata["classification_rule"].fillna("unclassified")
    metadata["sample_key"] = metadata["source_id"].astype(str) + "||" + metadata["spectrum_id"].astype(str)
    return metadata, source_summary, landcover_summary, excluded


def nm_columns(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    header = pd.read_csv(csv_path, nrows=0)
    columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in columns], dtype=int)
    return wavelengths, columns


def load_sampled_spectra(csv_path: Path, sample_keys: set[str], spectral_columns: list[str]) -> pd.DataFrame:
    usecols = ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"] + spectral_columns
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        subset = chunk[keys.isin(sample_keys)].copy()
        if not subset.empty:
            subset["sample_key"] = subset["source_id"].astype(str) + "||" + subset["spectrum_id"].astype(str)
            parts.append(subset)
    if not parts:
        return pd.DataFrame(columns=usecols + ["sample_key"])
    return pd.concat(parts, ignore_index=True)


def sample_group(frame: pd.DataFrame, group_col: str, limit: int, seed: int) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for index, (_, group_frame) in enumerate(frame.groupby(group_col, sort=True)):
        take = min(limit, len(group_frame))
        samples.append(group_frame.sample(n=take, random_state=seed + index * 997))
    if not samples:
        return frame.head(0).copy()
    return pd.concat(samples, ignore_index=True).sort_values([group_col, "sample_name", "spectrum_id"]).reset_index(drop=True)


def plot_bar(frame: pd.DataFrame, category_col: str, value_col: str, title: str, output_path: Path, top_n: int = 20) -> None:
    subset = frame.sort_values(value_col, ascending=False).head(top_n).copy()
    figure, axis = plt.subplots(figsize=(12, max(4, 0.45 * len(subset))))
    axis.barh(subset[category_col].astype(str), subset[value_col].astype(float), color="#4C78A8")
    axis.invert_yaxis()
    axis.set_xlabel(value_col.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def compute_group_means(
    spectra_csv: Path,
    group_lookup: pd.DataFrame,
    group_col: str,
    spectral_columns: list[str],
) -> pd.DataFrame:
    lookup = group_lookup[["sample_key", group_col]].drop_duplicates().rename(columns={group_col: "__group"})
    totals: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}

    usecols = ["source_id", "spectrum_id"] + spectral_columns
    for chunk in pd.read_csv(spectra_csv, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["sample_key"] = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        merged = chunk.merge(lookup, on="sample_key", how="inner")
        if merged.empty:
            continue
        for group_name, frame in merged.groupby("__group", sort=False):
            matrix = frame[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(matrix)
            value_sum = np.where(finite, matrix, 0.0).sum(axis=0)
            value_count = finite.sum(axis=0).astype(float)
            if group_name not in totals:
                totals[group_name] = value_sum
                counts[group_name] = value_count
            else:
                totals[group_name] += value_sum
                counts[group_name] += value_count

    rows: list[dict[str, object]] = []
    for group_name in sorted(totals):
        mean = np.divide(
            totals[group_name],
            counts[group_name],
            out=np.full_like(totals[group_name], np.nan, dtype=float),
            where=counts[group_name] > 0,
        )
        row: dict[str, object] = {group_col: group_name}
        row.update({column: float(value) for column, value in zip(spectral_columns, mean)})
        rows.append(row)
    return pd.DataFrame(rows)


def filter_mean_curve_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    filtered = metadata.copy()
    native_max = pd.to_numeric(filtered["native_max_nm"], errors="coerce")
    urban_mask = filtered["landcover_group"].astype(str) == "urban"
    return filtered[~urban_mask | (native_max >= URBAN_MEAN_MIN_END_NM)].copy()


def plot_mean_curves(
    means: pd.DataFrame,
    group_col: str,
    wavelengths: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    spectral_columns = [f"nm_{wavelength}" for wavelength in wavelengths]
    figure, axis = plt.subplots(figsize=(12, 6))
    for _, row in means.iterrows():
        label = str(row[group_col])
        color = LANDCOVER_COLORS.get(label, None)
        axis.plot(wavelengths, row[spectral_columns].to_numpy(dtype=float), label=label, linewidth=1.6, color=color)
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("Reflectance")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_random_grid(
    sample_frame: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: Path,
    title: str,
    wavelength_window: tuple[int, int] | None = None,
) -> None:
    sample_count = len(sample_frame)
    cols = 10
    rows = int(np.ceil(sample_count / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(24, max(10, rows * 2.2)), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    spectral_columns = [f"nm_{wavelength}" for wavelength in wavelengths]
    if wavelength_window is None:
        band_mask = np.ones(len(wavelengths), dtype=bool)
    else:
        band_mask = (wavelengths >= wavelength_window[0]) & (wavelengths <= wavelength_window[1])

    spectral_index = spectral_frame.set_index("sample_key")
    for axis, sample in zip(axes, sample_frame.itertuples(index=False)):
        if sample.sample_key not in spectral_index.index:
            axis.set_visible(False)
            continue
        row = spectral_index.loc[sample.sample_key, spectral_columns]
        values = np.asarray(row, dtype=float)
        color = LANDCOVER_COLORS.get(str(sample.landcover_group), LANDCOVER_COLORS["unclassified"])
        axis.plot(wavelengths[band_mask], values[band_mask], color=color, linewidth=1.0)
        axis.set_title(f"{truncate_label(sample.sample_name, 18)}\n{sample.landcover_group}", fontsize=7)
        axis.grid(alpha=0.2)

    for axis in axes[sample_count:]:
        axis.set_visible(False)

    for axis in axes[::cols]:
        axis.set_ylabel("Reflectance")
    for axis in axes[-cols:]:
        if axis.get_visible():
            axis.set_xlabel("Wavelength (nm)")

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visualizations for a built SIAC spectral library package.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-per-source", type=int, default=100)
    parser.add_argument("--max-per-landcover", type=int, default=100)
    parser.add_argument("--top-n-sources", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata, source_summary, landcover_summary, excluded = load_frames(root)
    spectra_csv = root / "tabular" / "siac_normalized_spectra.csv"
    wavelengths, spectral_columns = nm_columns(spectra_csv)

    plot_bar(source_summary, "source_id", "spectra_count", "SIAC Exported Spectra by Source", output_root / "source_counts.png", top_n=25)
    plot_bar(
        source_summary.sort_values("mean_coverage_fraction", ascending=False),
        "source_id",
        "mean_coverage_fraction",
        "Mean Coverage Fraction by Source",
        output_root / "source_mean_coverage.png",
        top_n=25,
    )
    plot_bar(landcover_summary, "landcover_group", "spectra_count", "SIAC Spectra by Landcover", output_root / "landcover_counts.png", top_n=10)
    if not excluded.empty:
        plot_bar(
            excluded.sort_values("available_spectra_count", ascending=False),
            "source_id",
            "available_spectra_count",
            "Excluded Sources Retained as Metadata",
            output_root / "excluded_sources.png",
            top_n=10,
        )

    mean_curve_metadata = filter_mean_curve_metadata(metadata)
    landcover_means = compute_group_means(spectra_csv, mean_curve_metadata, "landcover_group", spectral_columns)
    plot_mean_curves(
        landcover_means,
        "landcover_group",
        wavelengths,
        output_root / "landcover_mean_curves.png",
        "Mean Spectra by Landcover Group",
    )

    top_sources = source_summary[source_summary["included_in_spectra"] == True].sort_values("spectra_count", ascending=False).head(args.top_n_sources)
    top_meta = metadata[metadata["source_id"].isin(top_sources["source_id"].tolist())].copy()
    top_source_means = compute_group_means(spectra_csv, top_meta, "source_id", spectral_columns)
    plot_mean_curves(
        top_source_means,
        "source_id",
        wavelengths,
        output_root / "top_source_mean_curves.png",
        f"Mean Spectra for Top {len(top_source_means)} Sources",
    )

    source_samples = sample_group(metadata, "source_id", args.max_per_source, args.seed)
    landcover_samples = sample_group(metadata, "landcover_group", args.max_per_landcover, args.seed + 500_000)
    sample_keys = set(pd.concat([source_samples["sample_key"], landcover_samples["sample_key"]], ignore_index=True).tolist())
    sampled_spectra = load_sampled_spectra(spectra_csv, sample_keys, spectral_columns)

    source_plot_rows: list[dict[str, object]] = []
    for source_id, frame in source_samples.groupby("source_id", sort=True):
        source_dir = output_root / "random_by_source" / source_id
        plot_random_grid(
            frame.reset_index(drop=True),
            sampled_spectra[sampled_spectra["source_id"] == source_id].copy(),
            wavelengths,
            source_dir / "random_100_full_spectrum.png",
            f"{source_id} | random spectra",
        )
        plot_random_grid(
            frame.reset_index(drop=True),
            sampled_spectra[sampled_spectra["source_id"] == source_id].copy(),
            wavelengths,
            source_dir / "random_100_visible.png",
            f"{source_id} | random spectra (400-700 nm)",
            wavelength_window=(400, 700),
        )
        source_plot_rows.append(
            {
                "source_id": source_id,
                "sample_count": int(len(frame)),
                "full_plot": str(source_dir / "random_100_full_spectrum.png"),
                "visible_plot": str(source_dir / "random_100_visible.png"),
            }
        )

    landcover_plot_rows: list[dict[str, object]] = []
    for group, frame in landcover_samples.groupby("landcover_group", sort=True):
        plot_random_grid(
            frame.reset_index(drop=True),
            sampled_spectra[sampled_spectra["landcover_group"] == group].copy(),
            wavelengths,
            output_root / f"{group}_random_100_full_spectrum.png",
            f"{group} | random spectra",
        )
        plot_random_grid(
            frame.reset_index(drop=True),
            sampled_spectra[sampled_spectra["landcover_group"] == group].copy(),
            wavelengths,
            output_root / f"{group}_random_100_visible.png",
            f"{group} | random spectra (400-700 nm)",
            wavelength_window=(400, 700),
        )
        landcover_plot_rows.append(
            {
                "landcover_group": group,
                "sample_count": int(len(frame)),
                "full_plot": str(output_root / f"{group}_random_100_full_spectrum.png"),
                "visible_plot": str(output_root / f"{group}_random_100_visible.png"),
            }
        )

    pd.DataFrame(source_plot_rows).to_csv(output_root / "source_plot_summary.csv", index=False)
    pd.DataFrame(landcover_plot_rows).to_csv(output_root / "landcover_plot_summary.csv", index=False)
    source_samples.to_csv(output_root / "source_sample_index.csv", index=False)
    landcover_samples.to_csv(output_root / "landcover_sample_index.csv", index=False)
    landcover_means.to_csv(output_root / "landcover_mean_curves.csv", index=False)
    top_source_means.to_csv(output_root / "top_source_mean_curves.csv", index=False)

    summary = {
        "root": str(root),
        "total_exported_spectra": int(len(metadata)),
        "exported_source_count": int(metadata["source_id"].nunique()),
        "excluded_source_count": int(len(excluded)),
        "landcover_mean_curve_spectra": int(len(mean_curve_metadata)),
        "urban_mean_curve_min_end_nm": URBAN_MEAN_MIN_END_NM,
        "landcover_groups": sorted(metadata["landcover_group"].dropna().astype(str).unique().tolist()),
        "source_plot_count": int(len(source_plot_rows)),
        "landcover_plot_count": int(len(landcover_plot_rows)),
        "output_root": str(output_root),
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
