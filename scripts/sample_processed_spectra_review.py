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
from whitsmooth_rust import robust_whittaker_irls_f64


SAMPLE_SEED = 20260310
DEFAULT_SAMPLE_SIZE = 1000
CHUNK_SIZE = 2048
WEIGHT_THRESHOLD = 0.10
MAX_ROBUST_Z = 50.0

SECOND_ABSORPTION_START = 1850
SECOND_ABSORPTION_END = 1950
SECOND_ABSORPTION_PAD = 3
SECOND_ABSORPTION_THRESHOLD = 0.05

TAIL_FIT_START = 2350
TAIL_FIT_END = 2400
TAIL_EVAL_START = 2401
TAIL_EVAL_END = 2500
TAIL_DRIFT_THRESHOLD = 0.20
SOURCE_RESIDUAL_QUANTILE_SMALL = 0.995
SOURCE_RESIDUAL_QUANTILE_LARGE = 0.999
SOURCE_RESIDUAL_LARGE_N = 1000


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    lam: float
    abs_threshold: float


WINDOWS = [
    WindowSpec(400, 550, 100.0, 0.005),
    WindowSpec(550, 760, 50.0, 0.010),
    WindowSpec(760, 1350, 100.0, 0.010),
    WindowSpec(1350, 1450, 50.0, 0.010),
    WindowSpec(1450, 1800, 100.0, 0.010),
    WindowSpec(1800, 2000, 50.0, 0.010),
    WindowSpec(2000, 2450, 100.0, 0.010),
]


def resolve_inputs(base_root: Path) -> tuple[Path, Path]:
    siac_meta = base_root / "tabular" / "siac_spectra_metadata.csv"
    siac_spectra = base_root / "tabular" / "siac_normalized_spectra.csv"
    base_meta = base_root / "tabular" / "spectra_metadata.csv"
    base_spectra = base_root / "tabular" / "normalized_spectra.csv"
    if siac_meta.exists() and siac_spectra.exists():
        return siac_meta, siac_spectra
    return base_meta, base_spectra


def read_metadata(metadata_csv: Path) -> pd.DataFrame:
    columns = ["source_id", "source_name", "spectrum_id", "sample_name"]
    header = pd.read_csv(metadata_csv, nrows=0)
    if "landcover_group" in header.columns:
        columns.append("landcover_group")
    metadata = pd.read_csv(metadata_csv, usecols=columns, low_memory=False)
    if "landcover_group" not in metadata.columns:
        metadata["landcover_group"] = ""
    metadata["landcover_group"] = metadata["landcover_group"].fillna("")
    return metadata


def sample_metadata(metadata: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(metadata) <= sample_size:
        return metadata.copy().reset_index(drop=True)
    return metadata.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def load_sample_spectra(spectra_csv: Path, sampled_meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    sample_keys = set(
        sampled_meta["source_id"].astype(str) + "||" + sampled_meta["spectrum_id"].astype(str)
    )
    header = pd.read_csv(spectra_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)

    rows: list[pd.DataFrame] = []
    usecols = ["source_id", "spectrum_id", "sample_name"] + spectral_columns
    if "landcover_group" in header.columns:
        usecols.append("landcover_group")
    for chunk in pd.read_csv(spectra_csv, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        subset = chunk[keys.isin(sample_keys)].copy()
        if not subset.empty:
            rows.append(subset)

    if not rows:
        raise ValueError(f"No spectra loaded from {spectra_csv}")

    spectra = pd.concat(rows, ignore_index=True)
    spectra["sample_key"] = spectra["source_id"].astype(str) + "||" + spectra["spectrum_id"].astype(str)
    sampled_meta = sampled_meta.copy()
    sampled_meta["sample_key"] = sampled_meta["source_id"].astype(str) + "||" + sampled_meta["spectrum_id"].astype(str)
    drop_columns = ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"]
    merged = sampled_meta.merge(
        spectra.drop(columns=drop_columns, errors="ignore"),
        on="sample_key",
        how="inner",
    )
    return merged, wavelengths, spectral_columns


def _fill_for_smoothing(values: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for row_index in range(values.shape[0]):
        row = values[row_index]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        finite_x = wavelengths[finite]
        finite_y = row[finite]
        filled[row_index] = np.interp(wavelengths, finite_x, finite_y)
    return filled


def detect_flagged_bands(values: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flagged_all = np.zeros(values.shape, dtype=bool)
    smoothed_reference = np.full(values.shape, np.nan, dtype=float)
    filled = _fill_for_smoothing(values, wavelengths)

    for window in WINDOWS:
        start_idx = int(window.start - wavelengths[0])
        end_idx = int(window.end - wavelengths[0] + 1)
        window_values = filled[:, start_idx:end_idx].astype(np.float64, copy=False)
        x = np.arange(window.start, window.end + 1, dtype=np.float64)
        smoothed, weights = robust_whittaker_irls_f64(
            x,
            window_values,
            lam=window.lam,
            d=2,
            iterations=8,
            weighting="tukey",
            scale="mad",
            parallel=True,
            return_weights=True,
            merge_x_tol=0.0,
        )
        smoothed_reference[:, start_idx:end_idx] = smoothed
        residual = window_values - smoothed
        original = values[:, start_idx:end_idx]
        finite = np.isfinite(original)
        flagged = finite & (weights < WEIGHT_THRESHOLD) & (np.abs(residual) > window.abs_threshold)
        flagged_all[:, start_idx:end_idx] |= flagged

    return flagged_all, smoothed_reference


def robust_z(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros(values.shape[0], dtype=float)
    if not finite.any():
        return result
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values[finite] - median)))
    scale = max(1e-9, 1.4826 * mad)
    result[finite] = np.minimum(MAX_ROBUST_Z, np.maximum(0.0, (values[finite] - median) / scale))
    return result


def add_source_residual_thresholds(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    residual_thresholds: dict[str, float] = {}
    flagged_band_thresholds: dict[str, float] = {}
    for source_id, group in metrics.groupby("source_id", dropna=False):
        values = group["max_abs_residual"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        band_values = group["flagged_band_count"].to_numpy(dtype=float)
        finite_band_values = band_values[np.isfinite(band_values)]
        if finite.size == 0:
            residual_threshold = 0.10
        else:
            quantile = (
                SOURCE_RESIDUAL_QUANTILE_LARGE
                if finite.size >= SOURCE_RESIDUAL_LARGE_N
                else SOURCE_RESIDUAL_QUANTILE_SMALL
            )
            residual_threshold = max(0.10, float(np.nanquantile(finite, quantile)))
        if finite_band_values.size == 0:
            flagged_band_threshold = 250.0
        else:
            quantile = (
                SOURCE_RESIDUAL_QUANTILE_LARGE
                if finite_band_values.size >= SOURCE_RESIDUAL_LARGE_N
                else SOURCE_RESIDUAL_QUANTILE_SMALL
            )
            flagged_band_threshold = max(250.0, float(np.nanquantile(finite_band_values, quantile)))
        residual_thresholds[str(source_id)] = residual_threshold
        flagged_band_thresholds[str(source_id)] = flagged_band_threshold
    metrics["source_residual_threshold"] = metrics["source_id"].astype(str).map(residual_thresholds).astype(float)
    metrics["source_flagged_band_threshold"] = metrics["source_id"].astype(str).map(flagged_band_thresholds).astype(float)
    return metrics


def rowwise_abs_nanmax(values: np.ndarray) -> np.ndarray:
    masked = np.where(np.isfinite(values), np.abs(values), -np.inf)
    maximum = masked.max(axis=1)
    maximum[maximum == -np.inf] = np.nan
    return maximum


def rowwise_local_interp_deviation(
    values: np.ndarray,
    wavelengths: np.ndarray,
    start_nm: int,
    end_nm: int,
    pad: int,
) -> np.ndarray:
    row_count = values.shape[0]
    result = np.full(row_count, np.nan, dtype=float)
    band_indices = np.where((wavelengths >= start_nm) & (wavelengths <= end_nm))[0]
    for band_index in band_indices:
        left_index = band_index - pad
        right_index = band_index + pad
        if left_index < 0 or right_index >= values.shape[1]:
            continue
        y_left = values[:, left_index]
        y_center = values[:, band_index]
        y_right = values[:, right_index]
        valid = np.isfinite(y_left) & np.isfinite(y_center) & np.isfinite(y_right)
        if not valid.any():
            continue
        x_left = float(wavelengths[left_index])
        x_right = float(wavelengths[right_index])
        x_center = float(wavelengths[band_index])
        interp = y_left + (y_right - y_left) * (x_center - x_left) / (x_right - x_left)
        delta = np.abs(y_center - interp)
        replace = valid & (np.isnan(result) | (delta > result))
        result[replace] = delta[replace]
    return result


def rowwise_tail_extrapolation_drift(
    values: np.ndarray,
    wavelengths: np.ndarray,
    fit_start_nm: int,
    fit_end_nm: int,
    eval_start_nm: int,
    eval_end_nm: int,
) -> np.ndarray:
    row_count = values.shape[0]
    result = np.full(row_count, np.nan, dtype=float)
    fit_mask = (wavelengths >= fit_start_nm) & (wavelengths <= fit_end_nm)
    eval_mask = (wavelengths >= eval_start_nm) & (wavelengths <= eval_end_nm)
    fit_x = wavelengths[fit_mask].astype(float)
    eval_x = wavelengths[eval_mask].astype(float)
    if fit_x.size < 2 or eval_x.size < 1:
        return result

    for row_index in range(row_count):
        fit_y = values[row_index, fit_mask]
        eval_y = values[row_index, eval_mask]
        fit_valid = np.isfinite(fit_y)
        eval_valid = np.isfinite(eval_y)
        if fit_valid.sum() < 2 or eval_valid.sum() < 1:
            continue
        slope, intercept = np.polyfit(fit_x[fit_valid], fit_y[fit_valid], 1)
        predicted = slope * eval_x[eval_valid] + intercept
        result[row_index] = float(np.max(np.abs(eval_y[eval_valid] - predicted)))
    return result


def score_sample(sampled: pd.DataFrame, wavelengths: np.ndarray, spectral_columns: list[str]) -> pd.DataFrame:
    values = sampled[spectral_columns].to_numpy(dtype=float)
    flagged_all, smoothed = detect_flagged_bands(values, wavelengths)

    adjacent = np.abs(np.diff(values, axis=1))
    visible_mask = (wavelengths[:-1] >= 400) & (wavelengths[:-1] < 700)
    swir_tail_mask = wavelengths[:-1] >= 2300
    residual = np.abs(values - smoothed)
    residual[~np.isfinite(values)] = np.nan

    metrics = sampled[
        ["source_id", "source_name", "spectrum_id", "sample_name", "landcover_group"]
    ].copy()
    metrics["flagged_band_count"] = flagged_all.sum(axis=1)
    metrics["max_abs_jump"] = rowwise_abs_nanmax(adjacent)
    metrics["max_abs_jump_visible"] = rowwise_abs_nanmax(adjacent[:, visible_mask])
    metrics["max_abs_jump_tail_2300"] = rowwise_abs_nanmax(adjacent[:, swir_tail_mask])
    metrics["max_abs_residual"] = rowwise_abs_nanmax(residual)
    metrics["max_interp_spike_absorption2"] = rowwise_local_interp_deviation(
        values,
        wavelengths,
        SECOND_ABSORPTION_START,
        SECOND_ABSORPTION_END,
        SECOND_ABSORPTION_PAD,
    )
    metrics["tail_end_drift_2400"] = rowwise_tail_extrapolation_drift(
        values,
        wavelengths,
        TAIL_FIT_START,
        TAIL_FIT_END,
        TAIL_EVAL_START,
        TAIL_EVAL_END,
    )
    metrics["mean_abs_residual"] = np.nanmean(residual, axis=1)
    metrics["min_reflectance"] = np.nanmin(values, axis=1)
    metrics["max_reflectance"] = np.nanmax(values, axis=1)
    metrics["out_of_range_band_count"] = ((values < -0.05) | (values > 1.05)).sum(axis=1)
    metrics["valid_band_count"] = np.isfinite(values).sum(axis=1)
    metrics = add_source_residual_thresholds(metrics)

    score = (
        3.0 * robust_z(metrics["flagged_band_count"].to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_jump"].to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_jump_visible"].to_numpy(dtype=float))
        + 1.5 * robust_z(metrics["max_abs_jump_tail_2300"].to_numpy(dtype=float))
        + 2.5 * robust_z(metrics["max_interp_spike_absorption2"].fillna(0.0).to_numpy(dtype=float))
        + 2.5 * robust_z(metrics["tail_end_drift_2400"].fillna(0.0).to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_residual"].to_numpy(dtype=float))
        + 1.5 * robust_z(metrics["mean_abs_residual"].to_numpy(dtype=float))
        + 4.0 * robust_z(metrics["out_of_range_band_count"].to_numpy(dtype=float))
    )
    metrics["strange_score"] = score

    emit_mask = metrics["source_id"].astype(str).str.startswith("emit_")
    santa_mask = metrics["source_id"].astype(str).eq("santa_barbara_urban_reflectance")
    thresholds = (
        (metrics["flagged_band_count"] >= 15)
        | (metrics["max_abs_jump"] >= 0.08)
        | (metrics["max_abs_jump_visible"] >= 0.05)
        | (metrics["max_abs_jump_tail_2300"] >= 0.05)
        | (emit_mask & (metrics["max_interp_spike_absorption2"].fillna(0.0) >= SECOND_ABSORPTION_THRESHOLD))
        | (santa_mask & (metrics["tail_end_drift_2400"].fillna(0.0) >= TAIL_DRIFT_THRESHOLD))
        | (metrics["max_abs_residual"] >= metrics["source_residual_threshold"])
        | (
            (metrics["flagged_band_count"] >= metrics["source_flagged_band_threshold"])
            & (metrics["max_abs_residual"] >= 0.05)
        )
        | (metrics["out_of_range_band_count"] > 0)
    )
    metrics["is_strange"] = thresholds
    return metrics.sort_values(["is_strange", "strange_score"], ascending=[False, False]).reset_index(drop=True)


def plot_suspicious_spectra(
    sampled: pd.DataFrame,
    ranked_metrics: pd.DataFrame,
    wavelengths: np.ndarray,
    spectral_columns: list[str],
    output_dir: Path,
    top_n: int,
) -> None:
    suspicious = ranked_metrics.head(top_n).copy()
    if suspicious.empty:
        return
    joined = suspicious.merge(
        sampled[["source_id", "spectrum_id"] + spectral_columns],
        on=["source_id", "spectrum_id"],
        how="left",
    )
    rows = int(np.ceil(len(joined) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 3.0), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    visible_mask = (wavelengths >= 400) & (wavelengths <= 700)

    for axis, row in zip(axes, joined.itertuples(index=False)):
        values = np.asarray([getattr(row, column) for column in spectral_columns], dtype=float)
        axis.plot(wavelengths, values, color="#0b7285", linewidth=1.1)
        axis.set_title(
            f"{row.source_id} | {row.spectrum_id}\nscore={row.strange_score:.2f}, flags={int(row.flagged_band_count)}"
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Reflectance")
        axis.grid(alpha=0.2)
    for axis in axes[len(joined) :]:
        axis.set_visible(False)
    fig.suptitle("Top suspicious sampled spectra")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "top_suspicious_full.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 3.0), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, row in zip(axes, joined.itertuples(index=False)):
        values = np.asarray([getattr(row, column) for column in spectral_columns], dtype=float)
        axis.plot(wavelengths[visible_mask], values[visible_mask], color="#e67700", linewidth=1.2)
        axis.set_title(f"{row.source_id} | {row.spectrum_id}")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Reflectance")
        axis.grid(alpha=0.2)
    for axis in axes[len(joined) :]:
        axis.set_visible(False)
    fig.suptitle("Top suspicious sampled spectra (visible window)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "top_suspicious_visible.png", dpi=180)
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, sample_size: int, seed: int, output_path: Path) -> dict[str, object]:
    strange = metrics[metrics["is_strange"]].copy()
    by_source = (
        strange.groupby("source_id")
        .agg(
            strange_spectra=("spectrum_id", "count"),
            max_score=("strange_score", "max"),
            mean_score=("strange_score", "mean"),
        )
        .sort_values(["strange_spectra", "max_score"], ascending=[False, False])
        .reset_index()
    )
    by_landcover = (
        strange.assign(landcover_group=strange["landcover_group"].replace("", "unlabeled"))
        .groupby("landcover_group")
        .agg(strange_spectra=("spectrum_id", "count"))
        .sort_values("strange_spectra", ascending=False)
        .reset_index()
    )
    summary = {
        "sample_size": int(sample_size),
        "seed": int(seed),
        "strange_spectra": int(len(strange)),
        "strange_fraction": float(len(strange) / max(1, len(metrics))),
        "top_sources": by_source.head(10).to_dict(orient="records"),
        "top_landcover_groups": by_landcover.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomly sample processed spectra and flag suspicious curves.")
    parser.add_argument("--base-root", default="build/siac_spectral_library_v1")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root) if args.output_root else base_root / "sample_review_1000"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_csv, spectra_csv = resolve_inputs(base_root)
    metadata = read_metadata(metadata_csv)
    sampled_meta = sample_metadata(metadata, args.sample_size, args.seed)
    sampled_meta.to_csv(output_root / "sample_index.csv", index=False)

    sampled, wavelengths, spectral_columns = load_sample_spectra(spectra_csv, sampled_meta)
    metrics = score_sample(sampled, wavelengths, spectral_columns)
    metrics.to_csv(output_root / "sample_metrics.csv", index=False)
    metrics[metrics["is_strange"]].to_csv(output_root / "flagged_strange_spectra.csv", index=False)

    plot_suspicious_spectra(sampled, metrics, wavelengths, spectral_columns, output_root, top_n=args.top_n)
    summary = write_summary(metrics, len(sampled), args.seed, output_root / "review_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
