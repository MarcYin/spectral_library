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


TARGET_GROUP = "vegetation"
CHUNK_SIZE = 2048


def build_group_keys(labels_path: Path) -> set[str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"] == TARGET_GROUP].copy()
    return set(labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str))


def aggregate_source_means(
    normalized_csv: Path,
    allowed_keys: set[str],
    wavelengths: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    source_sums: dict[str, np.ndarray] = {}
    source_counts: dict[str, np.ndarray] = {}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk[keys.isin(allowed_keys)]
        if chunk.empty:
            continue
        for source_id, frame in chunk.groupby("source_id"):
            values = frame[columns[2:]].to_numpy(dtype=float)
            valid = np.isfinite(values)
            source_id = str(source_id)
            if source_id not in source_sums:
                source_sums[source_id] = np.zeros(len(wavelengths), dtype=float)
                source_counts[source_id] = np.zeros(len(wavelengths), dtype=float)
            source_sums[source_id] += np.where(valid, values, 0.0).sum(axis=0)
            source_counts[source_id] += valid.sum(axis=0)

    sources = sorted(source_sums)
    means = np.vstack(
        [
            np.divide(
                source_sums[source_id],
                source_counts[source_id],
                out=np.full_like(source_sums[source_id], np.nan),
                where=source_counts[source_id] > 0,
            )
            for source_id in sources
        ]
    )
    counts = np.vstack([source_counts[source_id] for source_id in sources])
    return sources, means, counts


def plot_diagnostic(
    output_path: Path,
    wavelengths: np.ndarray,
    pooled_mean: np.ndarray,
    top_sources: list[str],
    means: np.ndarray,
    weights_at_peak: dict[str, float],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(wavelengths, pooled_mean, color="#0b7285", linewidth=2.2, label="Vegetation mean")
    for source_id, source_curve in zip(top_sources, means):
        label = f"{source_id} (weight@437={weights_at_peak.get(source_id, 0.0):.3f})"
        axes[0].plot(wavelengths, source_curve, linewidth=1.2, alpha=0.9, label=label)
    axes[0].axvline(437, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("Mean reflectance")
    axes[0].set_title("Vegetation 430-450 nm mean and dominant source curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    step = pooled_mean[1:] - pooled_mean[:-1]
    axes[1].plot(wavelengths[1:], step, color="#d9480f", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].axvline(437, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Adjacent step")
    axes[1].set_title("Vegetation mean adjacent steps")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the vegetation blue-region peak by source.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wavelength-start", type=int, default=430)
    parser.add_argument("--wavelength-end", type=int, default=450)
    parser.add_argument("--top-sources", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "landcover_analysis" / "vegetation_blue_peak"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.arange(args.wavelength_start, args.wavelength_end + 1, dtype=int)
    allowed_keys = build_group_keys(root / "landcover_analysis" / "landcover_labels.csv")
    sources, means, counts = aggregate_source_means(root / "tabular" / "normalized_spectra.csv", allowed_keys, wavelengths)

    pooled_support = counts.sum(axis=0)
    pooled_mean = np.nansum(np.nan_to_num(means) * counts, axis=0) / pooled_support
    peak_idx = int(np.argmax(pooled_mean))
    peak_nm = int(wavelengths[peak_idx])
    peak_weights = counts[:, peak_idx] / pooled_support[peak_idx]
    peak_contrib = peak_weights * np.nan_to_num(means[:, peak_idx])

    order = np.argsort(np.abs(peak_contrib))[::-1]
    top_indices = order[: args.top_sources]
    top_source_ids = [sources[idx] for idx in top_indices]
    top_frame = pd.DataFrame(
        {
            "source_id": [sources[idx] for idx in top_indices],
            "weight_at_peak": [float(peak_weights[idx]) for idx in top_indices],
            "mean_at_peak": [float(means[idx, peak_idx]) for idx in top_indices],
            "contribution_at_peak": [float(peak_contrib[idx]) for idx in top_indices],
        }
    )
    top_frame.to_csv(output_dir / "top_source_contributions_at_peak.csv", index=False)

    curve_rows: list[dict[str, object]] = []
    for idx in top_indices:
        for wavelength, value, count in zip(wavelengths, means[idx], counts[idx]):
            curve_rows.append(
                {
                    "source_id": sources[idx],
                    "wavelength_nm": int(wavelength),
                    "mean_reflectance": float(value) if np.isfinite(value) else np.nan,
                    "support": int(count),
                }
            )
    pd.DataFrame(curve_rows).to_csv(output_dir / "top_source_curves.csv", index=False)

    pooled_frame = pd.DataFrame(
        {
            "wavelength_nm": wavelengths,
            "pooled_mean": pooled_mean,
            "pooled_step": np.concatenate([[np.nan], pooled_mean[1:] - pooled_mean[:-1]]),
            "pooled_support": pooled_support.astype(int),
        }
    )
    pooled_frame.to_csv(output_dir / "vegetation_pooled_curve.csv", index=False)

    plot_diagnostic(
        output_dir / "vegetation_blue_peak.png",
        wavelengths,
        pooled_mean,
        top_source_ids,
        means[top_indices],
        {sources[idx]: float(peak_weights[idx]) for idx in top_indices},
    )

    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "peak_nm": peak_nm,
        "peak_value": float(pooled_mean[peak_idx]),
        "top_sources": top_source_ids,
        "plot": str(output_dir / "vegetation_blue_peak.png"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
