#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from whitsmooth_rust import robust_whittaker_irls_f64

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_ID = "santa_barbara_urban_reflectance"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 20260311
DEFAULT_LAMBDA = 25000.0


def resolve_inputs(root: Path) -> tuple[Path, Path]:
    metadata_csv = root / "tabular" / "siac_spectra_metadata.csv"
    spectra_csv = root / "tabular" / "siac_normalized_spectra.csv"
    if not metadata_csv.exists() or not spectra_csv.exists():
        raise FileNotFoundError(f"Missing SIAC tables under {root}")
    return metadata_csv, spectra_csv


def fill_for_smoothing(values: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for row_index in range(values.shape[0]):
        row = values[row_index]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        filled[row_index] = np.interp(wavelengths, wavelengths[finite], row[finite])
    return filled


def sample_santa_spectra(
    metadata_csv: Path,
    spectra_csv: Path,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    metadata = pd.read_csv(
        metadata_csv,
        usecols=["source_id", "source_name", "spectrum_id", "sample_name", "landcover_group"],
        low_memory=False,
    )
    metadata["source_id"] = metadata["source_id"].astype(str)
    metadata["spectrum_id"] = metadata["spectrum_id"].astype(str)
    santa = metadata[metadata["source_id"].eq(SOURCE_ID)].copy()
    santa = santa.sort_values("spectrum_id").reset_index(drop=True)
    if santa.empty:
        raise ValueError(f"No {SOURCE_ID} spectra found in {metadata_csv}")

    rng = random.Random(seed)
    sample_count = min(sample_size, len(santa))
    selected_ids = rng.sample(santa["spectrum_id"].tolist(), sample_count)
    selected = santa[santa["spectrum_id"].isin(selected_ids)].copy()
    selected["order_key"] = selected["spectrum_id"].map({spectrum_id: index for index, spectrum_id in enumerate(selected_ids)})
    selected = selected.sort_values("order_key").drop(columns="order_key").reset_index(drop=True)

    header = pd.read_csv(spectra_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    spectra = pd.read_csv(
        spectra_csv,
        usecols=["source_id", "spectrum_id", "sample_name"] + spectral_columns,
        low_memory=False,
    )
    spectra["source_id"] = spectra["source_id"].astype(str)
    spectra["spectrum_id"] = spectra["spectrum_id"].astype(str)
    subset = spectra[spectra["source_id"].eq(SOURCE_ID) & spectra["spectrum_id"].isin(selected_ids)].copy()
    subset["order_key"] = subset["spectrum_id"].map({spectrum_id: index for index, spectrum_id in enumerate(selected_ids)})
    subset = subset.sort_values("order_key").drop(columns="order_key").reset_index(drop=True)
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=np.float64)
    return selected, subset, spectral_columns, wavelengths


def strong_smooth(values: np.ndarray, wavelengths: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    filled = fill_for_smoothing(values, wavelengths)
    smoothed, weights = robust_whittaker_irls_f64(
        wavelengths.astype(np.float64),
        filled.astype(np.float64),
        lam=float(lam),
        d=2,
        iterations=10,
        weighting="tukey",
        scale="mad",
        parallel=True,
        return_weights=True,
        merge_x_tol=0.0,
    )
    smoothed = np.clip(smoothed, 0.0, 1.0)
    return smoothed, weights


def rowwise_abs_nanmax(values: np.ndarray) -> np.ndarray:
    masked = np.where(np.isfinite(values), np.abs(values), -np.inf)
    maximum = masked.max(axis=1)
    maximum[maximum == -np.inf] = np.nan
    return maximum


def compute_metrics(values: np.ndarray, smoothed: np.ndarray, wavelengths: np.ndarray) -> pd.DataFrame:
    adjacent_before = rowwise_abs_nanmax(values[:, 1:] - values[:, :-1])
    adjacent_after = rowwise_abs_nanmax(smoothed[:, 1:] - smoothed[:, :-1])
    tail_mask = wavelengths[:-1] >= 2300
    adjacent_tail_before = rowwise_abs_nanmax((values[:, 1:] - values[:, :-1])[:, tail_mask])
    adjacent_tail_after = rowwise_abs_nanmax((smoothed[:, 1:] - smoothed[:, :-1])[:, tail_mask])
    residual = np.abs(values - smoothed)
    roughness_before = np.nanmean(np.abs(np.diff(values, n=2, axis=1)), axis=1)
    roughness_after = np.nanmean(np.abs(np.diff(smoothed, n=2, axis=1)), axis=1)
    return pd.DataFrame(
        {
            "max_abs_jump_before": adjacent_before,
            "max_abs_jump_after": adjacent_after,
            "max_abs_jump_tail_2300_before": adjacent_tail_before,
            "max_abs_jump_tail_2300_after": adjacent_tail_after,
            "mean_abs_delta": np.nanmean(residual, axis=1),
            "max_abs_delta": np.nanmax(residual, axis=1),
            "roughness_before": roughness_before,
            "roughness_after": roughness_after,
        }
    )


def plot_grid(
    path: Path,
    spectra: pd.DataFrame,
    smoothed: np.ndarray,
    spectral_columns: list[str],
    wavelengths: np.ndarray,
    min_nm: int,
    max_nm: int,
    title: str,
) -> None:
    mask = (wavelengths >= min_nm) & (wavelengths <= max_nm)
    xs = wavelengths[mask]
    fig, axes = plt.subplots(10, 10, figsize=(22, 22), constrained_layout=True)
    axes = axes.ravel()
    for axis, (_, row), smoothed_row in zip(axes, spectra.iterrows(), smoothed):
        original = row[spectral_columns].to_numpy(dtype=float)[mask]
        axis.plot(xs, original, color="#7f8c8d", linewidth=0.8, alpha=0.9)
        axis.plot(xs, smoothed_row[mask], color="#c0392b", linewidth=0.9)
        axis.set_title(str(row["spectrum_id"]).replace(f"{SOURCE_ID}_", ""), fontsize=6)
        axis.grid(alpha=0.15)
        axis.tick_params(labelsize=5)
    for axis in axes[len(spectra) :]:
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean(path: Path, values: np.ndarray, smoothed: np.ndarray, wavelengths: np.ndarray) -> None:
    mean_before = np.nanmean(values, axis=0)
    mean_after = np.nanmean(smoothed, axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].plot(wavelengths, mean_before, color="#7f8c8d", label="before")
    axes[0].plot(wavelengths, mean_after, color="#c0392b", label="after")
    axes[0].set_title("Mean Santa Barbara spectrum: before vs strong robust smoothing")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(wavelengths, mean_after - mean_before, color="#2c3e50")
    axes[1].set_title("Mean delta (after - before)")
    axes[1].grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show before/after strong robust smoothing on random Santa Barbara urban spectra.")
    parser.add_argument("--root", required=True, help="SIAC package root.")
    parser.add_argument("--output-root", default="", help="Output folder. Defaults to <root>/santa_strong_smoothing_demo.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else root / "santa_strong_smoothing_demo"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_csv, spectra_csv = resolve_inputs(root)
    selected, spectra, spectral_columns, wavelengths = sample_santa_spectra(
        metadata_csv,
        spectra_csv,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    values = spectra[spectral_columns].to_numpy(dtype=float)
    smoothed, weights = strong_smooth(values, wavelengths, lam=args.lam)

    metrics = pd.concat([selected.reset_index(drop=True), compute_metrics(values, smoothed, wavelengths)], axis=1)
    metrics.to_csv(output_root / "sample_metrics.csv", index=False)
    selected.to_csv(output_root / "sample_index.csv", index=False)

    plot_grid(
        output_root / "random_100_before_after_full.png",
        spectra,
        smoothed,
        spectral_columns,
        wavelengths,
        400,
        2500,
        f"Santa Barbara urban spectra: before (gray) vs strong robust smoothing (red), lambda={args.lam:g}",
    )
    plot_grid(
        output_root / "random_100_before_after_visible.png",
        spectra,
        smoothed,
        spectral_columns,
        wavelengths,
        400,
        700,
        f"Santa Barbara urban spectra visible window: before vs after, lambda={args.lam:g}",
    )
    plot_grid(
        output_root / "random_100_before_after_absorption2.png",
        spectra,
        smoothed,
        spectral_columns,
        wavelengths,
        1800,
        2000,
        f"Santa Barbara urban spectra 1800-2000 nm: before vs after, lambda={args.lam:g}",
    )
    plot_grid(
        output_root / "random_100_before_after_tail.png",
        spectra,
        smoothed,
        spectral_columns,
        wavelengths,
        2300,
        2500,
        f"Santa Barbara urban spectra 2300-2500 nm: before vs after, lambda={args.lam:g}",
    )
    plot_mean(output_root / "mean_before_after.png", values, smoothed, wavelengths)

    summary = {
        "root": str(root),
        "source_id": SOURCE_ID,
        "sample_size": int(len(selected)),
        "seed": int(args.seed),
        "lambda": float(args.lam),
        "output_root": str(output_root),
        "mean_max_abs_jump_before": float(metrics["max_abs_jump_before"].mean()),
        "mean_max_abs_jump_after": float(metrics["max_abs_jump_after"].mean()),
        "mean_tail_jump_before": float(metrics["max_abs_jump_tail_2300_before"].mean()),
        "mean_tail_jump_after": float(metrics["max_abs_jump_tail_2300_after"].mean()),
        "mean_roughness_before": float(metrics["roughness_before"].mean()),
        "mean_roughness_after": float(metrics["roughness_after"].mean()),
        "mean_abs_delta": float(metrics["mean_abs_delta"].mean()),
        "mean_weight": float(np.nanmean(weights)),
    }
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
