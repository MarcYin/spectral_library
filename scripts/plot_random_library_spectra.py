#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHUNK_SIZE = 2048
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


def load_metadata(root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        root / "tabular" / "spectra_metadata.csv",
        usecols=["source_id", "source_name", "spectrum_id", "sample_name"],
        low_memory=False,
    )
    labels = pd.read_csv(
        root / "landcover_analysis" / "landcover_labels.csv",
        usecols=["source_id", "spectrum_id", "landcover_group", "classification_rule"],
        low_memory=False,
    )
    merged = metadata.merge(labels, on=["source_id", "spectrum_id"], how="left")
    merged["landcover_group"] = merged["landcover_group"].fillna("unclassified")
    merged["classification_rule"] = merged["classification_rule"].fillna("unclassified")
    merged["sample_key"] = merged["source_id"].astype(str) + "||" + merged["spectrum_id"].astype(str)
    return merged


def sample_source_frame(frame: pd.DataFrame, max_per_source: int, seed: int) -> pd.DataFrame:
    target = min(max_per_source, len(frame))
    groups = sorted(frame["landcover_group"].astype(str).unique().tolist())
    sampled_parts: list[pd.DataFrame] = []
    remaining = target

    for offset, group in enumerate(groups):
        group_frame = frame[frame["landcover_group"] == group]
        if group_frame.empty:
            continue
        take = min(len(group_frame), max(1, target // len(groups)))
        sampled = group_frame.sample(n=take, random_state=seed + offset)
        sampled_parts.append(sampled)
        remaining -= len(sampled)

    combined = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["sample_key"]) if sampled_parts else frame.head(0)
    if remaining > 0:
        leftover = frame[~frame["sample_key"].isin(combined["sample_key"])].copy()
        if not leftover.empty:
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=seed + 97)
            combined = pd.concat([combined, extra], ignore_index=True)

    return combined.sort_values(["landcover_group", "sample_name", "spectrum_id"]).reset_index(drop=True)


def select_samples(metadata: pd.DataFrame, max_per_source: int, seed: int) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for idx, (source_id, frame) in enumerate(metadata.groupby("source_id", sort=True)):
        sampled = sample_source_frame(frame.copy(), max_per_source=max_per_source, seed=seed + idx * 1000)
        sampled_frames.append(sampled)
    return pd.concat(sampled_frames, ignore_index=True)


def sanitize_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "unclassified"


def sample_subcategory_frame(frame: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    take = min(max_per_group, len(frame))
    if take <= 0:
        return frame.head(0).copy()
    return frame.sample(n=take, random_state=seed).sort_values(["sample_name", "spectrum_id"]).reset_index(drop=True)


def select_subcategory_samples(metadata: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for idx, ((source_id, classification_rule), frame) in enumerate(
        metadata.groupby(["source_id", "classification_rule"], sort=True)
    ):
        sampled = sample_subcategory_frame(frame.copy(), max_per_group=max_per_group, seed=seed + idx * 997)
        sampled["subcategory_slug"] = sanitize_slug(classification_rule)
        sampled_frames.append(sampled)
    return pd.concat(sampled_frames, ignore_index=True) if sampled_frames else metadata.head(0).copy()


def load_sampled_spectra(root: Path, sample_keys: set[str]) -> tuple[np.ndarray, pd.DataFrame]:
    csv_path = root / "tabular" / "normalized_spectra.csv"
    header = pd.read_csv(csv_path, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)

    rows: list[pd.DataFrame] = []
    usecols = ["source_id", "spectrum_id"] + spectral_columns
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        subset = chunk[keys.isin(sample_keys)].copy()
        if not subset.empty:
            subset["sample_key"] = subset["source_id"].astype(str) + "||" + subset["spectrum_id"].astype(str)
            rows.append(subset)

    spectra = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=usecols + ["sample_key"])
    return wavelengths, spectra


def split_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def plot_source_grid(
    source_frame: pd.DataFrame,
    spectral_frame: pd.DataFrame,
    wavelengths: np.ndarray,
    output_path: Path,
    title: str,
    wavelength_window: tuple[int, int] | None = None,
) -> None:
    sample_count = len(source_frame)
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
    for axis, sample in zip(axes, source_frame.itertuples(index=False)):
        if sample.sample_key not in spectral_index.index:
            axis.set_visible(False)
            continue

        row = spectral_index.loc[sample.sample_key, spectral_columns]
        values = np.asarray(row, dtype=float)
        color = LANDCOVER_COLORS.get(str(sample.landcover_group), LANDCOVER_COLORS["unclassified"])
        axis.plot(wavelengths[band_mask], values[band_mask], color=color, linewidth=1.0)
        axis.set_title(
            f"{truncate_label(sample.sample_name, 22)}\n{sample.landcover_group}",
            fontsize=7,
        )
        axis.grid(alpha=0.2)

    for axis in axes[sample_count:]:
        axis.set_visible(False)

    for axis in axes[::cols]:
        axis.set_ylabel("Reflectance")
    for axis in axes[-cols:]:
        if axis.get_visible():
            axis.set_xlabel("Wavelength (nm)")

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=group)
        for group, color in LANDCOVER_COLORS.items()
        if group in source_frame["landcover_group"].unique()
    ]
    if legend_handles:
        figure.legend(handles=legend_handles, loc="upper center", ncol=min(5, len(legend_handles)), frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate up to 100 random spectra plots for each spectral library.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-per-source", type=int, default=100)
    parser.add_argument("--max-per-subcategory", type=int, default=100)
    parser.add_argument("--source-ids", default="")
    parser.add_argument("--seed", type=int, default=20260309)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(root)
    selected_source_ids = set(split_csv_arg(args.source_ids))
    if selected_source_ids:
        metadata = metadata[metadata["source_id"].astype(str).isin(selected_source_ids)].copy()
    sampled = select_samples(metadata, max_per_source=args.max_per_source, seed=args.seed)
    subcategory_sampled = select_subcategory_samples(
        metadata,
        max_per_group=args.max_per_subcategory,
        seed=args.seed + 500_000,
    )
    combined = (
        pd.concat([sampled, subcategory_sampled], ignore_index=True)
        .drop_duplicates(subset=["sample_key"])
        .reset_index(drop=True)
    )
    sampled.to_csv(output_root / "sample_index.csv", index=False)
    subcategory_sampled.to_csv(output_root / "subcategory_sample_index.csv", index=False)

    wavelengths, spectral_frame = load_sampled_spectra(root, set(combined["sample_key"].tolist()))

    summary_rows: list[dict[str, object]] = []
    for source_id, frame in sampled.groupby("source_id", sort=True):
        source_dir = output_root / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.copy().reset_index(drop=True)
        frame.to_csv(source_dir / "random_sample_manifest.csv", index=False)

        source_spectra = spectral_frame[spectral_frame["source_id"] == source_id].copy()
        source_name = str(frame["source_name"].iloc[0])

        plot_source_grid(
            frame,
            source_spectra,
            wavelengths,
            source_dir / "random_100_full_spectrum.png",
            title=f"{source_id} | {source_name} | random spectra",
        )
        plot_source_grid(
            frame,
            source_spectra,
            wavelengths,
            source_dir / "random_100_visible.png",
            title=f"{source_id} | {source_name} | random spectra (400-700 nm)",
            wavelength_window=(400, 700),
        )

        group_counts = frame["landcover_group"].value_counts().to_dict()
        summary_rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "sample_count": int(len(frame)),
                "landcover_groups": json.dumps(group_counts, sort_keys=True),
                "full_plot": str(source_dir / "random_100_full_spectrum.png"),
                "visible_plot": str(source_dir / "random_100_visible.png"),
            }
        )

    subcategory_rows: list[dict[str, object]] = []
    for (source_id, classification_rule), frame in subcategory_sampled.groupby(
        ["source_id", "classification_rule"],
        sort=True,
    ):
        source_dir = output_root / source_id / "subcategories"
        source_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.copy().reset_index(drop=True)
        source_spectra = spectral_frame[spectral_frame["source_id"] == source_id].copy()
        source_name = str(frame["source_name"].iloc[0])
        subcategory_slug = str(frame["subcategory_slug"].iloc[0])
        sub_dir = source_dir / subcategory_slug
        sub_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(sub_dir / "random_sample_manifest.csv", index=False)

        title_base = f"{source_id} | {classification_rule}"
        plot_source_grid(
            frame,
            source_spectra,
            wavelengths,
            sub_dir / "random_100_full_spectrum.png",
            title=f"{title_base} | {source_name} | random spectra",
        )
        plot_source_grid(
            frame,
            source_spectra,
            wavelengths,
            sub_dir / "random_100_visible.png",
            title=f"{title_base} | {source_name} | random spectra (400-700 nm)",
            wavelength_window=(400, 700),
        )

        landcover_counts = frame["landcover_group"].value_counts().to_dict()
        subcategory_rows.append(
            {
                "source_id": source_id,
                "source_name": source_name,
                "classification_rule": classification_rule,
                "subcategory_slug": subcategory_slug,
                "sample_count": int(len(frame)),
                "landcover_groups": json.dumps(landcover_counts, sort_keys=True),
                "full_plot": str(sub_dir / "random_100_full_spectrum.png"),
                "visible_plot": str(sub_dir / "random_100_visible.png"),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("source_id").reset_index(drop=True)
    summary.to_csv(output_root / "plot_summary.csv", index=False)
    subcategory_summary = (
        pd.DataFrame(subcategory_rows)
        .sort_values(["source_id", "classification_rule"])
        .reset_index(drop=True)
    )
    subcategory_summary.to_csv(output_root / "subcategory_plot_summary.csv", index=False)
    (output_root / "run_summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "output_root": str(output_root),
                "source_count": int(summary["source_id"].nunique()),
                "subcategory_count": int(subcategory_summary.shape[0]),
                "total_sampled_spectra": int(len(sampled)),
                "total_subcategory_sampled_spectra": int(len(subcategory_sampled)),
                "max_per_source": int(args.max_per_source),
                "max_per_subcategory": int(args.max_per_subcategory),
                "seed": int(args.seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
