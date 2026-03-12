#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sample_processed_spectra_review import (
    CHUNK_SIZE,
    SECOND_ABSORPTION_END,
    SECOND_ABSORPTION_PAD,
    SECOND_ABSORPTION_START,
    SECOND_ABSORPTION_THRESHOLD,
    TAIL_DRIFT_THRESHOLD,
    TAIL_EVAL_END,
    TAIL_EVAL_START,
    TAIL_FIT_END,
    TAIL_FIT_START,
    add_source_residual_thresholds,
    detect_flagged_bands,
    plot_suspicious_spectra,
    read_metadata,
    resolve_inputs,
    robust_z,
    rowwise_local_interp_deviation,
    rowwise_abs_nanmax,
    rowwise_tail_extrapolation_drift,
)


def compute_chunk_metrics(chunk: pd.DataFrame, spectral_columns: list[str], wavelengths) -> pd.DataFrame:
    values = chunk[spectral_columns].to_numpy(dtype=float)
    flagged_all, smoothed = detect_flagged_bands(values, wavelengths)
    adjacent = abs(values[:, 1:] - values[:, :-1])
    visible_mask = (wavelengths[:-1] >= 400) & (wavelengths[:-1] < 700)
    swir_tail_mask = wavelengths[:-1] >= 2300
    residual = abs(values - smoothed)
    residual[~pd.notna(values)] = float("nan")

    metrics = chunk[["source_id", "source_name", "spectrum_id", "sample_name", "landcover_group"]].copy()
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
    metrics["mean_abs_residual"] = pd.DataFrame(residual).mean(axis=1, skipna=True).to_numpy(dtype=float)
    metrics["min_reflectance"] = pd.DataFrame(values).min(axis=1, skipna=True).to_numpy(dtype=float)
    metrics["max_reflectance"] = pd.DataFrame(values).max(axis=1, skipna=True).to_numpy(dtype=float)
    metrics["out_of_range_band_count"] = ((values < -0.05) | (values > 1.05)).sum(axis=1)
    metrics["valid_band_count"] = pd.notna(values).sum(axis=1)
    return metrics


def finalize_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = add_source_residual_thresholds(metrics)
    score = (
        3.0 * robust_z(metrics["flagged_band_count"].to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_jump"].to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_jump_visible"].to_numpy(dtype=float))
        + 1.5 * robust_z(metrics["max_abs_jump_tail_2300"].fillna(0.0).to_numpy(dtype=float))
        + 2.5 * robust_z(metrics["max_interp_spike_absorption2"].fillna(0.0).to_numpy(dtype=float))
        + 2.5 * robust_z(metrics["tail_end_drift_2400"].fillna(0.0).to_numpy(dtype=float))
        + 2.0 * robust_z(metrics["max_abs_residual"].to_numpy(dtype=float))
        + 1.5 * robust_z(metrics["mean_abs_residual"].to_numpy(dtype=float))
        + 4.0 * robust_z(metrics["out_of_range_band_count"].to_numpy(dtype=float))
    )
    metrics = metrics.copy()
    metrics["strange_score"] = score
    emit_mask = metrics["source_id"].astype(str).str.startswith("emit_")
    santa_mask = metrics["source_id"].astype(str).eq("santa_barbara_urban_reflectance")
    metrics["is_suspicious"] = (
        (metrics["out_of_range_band_count"] > 0)
        | (metrics["max_abs_jump"] >= 0.15)
        | (metrics["max_abs_jump_visible"] >= 0.05)
        | (metrics["max_abs_jump_tail_2300"].fillna(0.0) >= 0.05)
        | (emit_mask & (metrics["max_interp_spike_absorption2"].fillna(0.0) >= SECOND_ABSORPTION_THRESHOLD))
        | (santa_mask & (metrics["tail_end_drift_2400"].fillna(0.0) >= TAIL_DRIFT_THRESHOLD))
        | (metrics["max_abs_residual"] >= metrics["source_residual_threshold"])
        | (
            (metrics["flagged_band_count"] >= metrics["source_flagged_band_threshold"])
            & (metrics["max_abs_residual"] >= 0.05)
        )
    )
    return metrics.sort_values(["is_suspicious", "strange_score"], ascending=[False, False]).reset_index(drop=True)


def iter_joined_chunks(metadata: pd.DataFrame, spectra_csv: Path):
    header = pd.read_csv(spectra_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    usecols = ["source_id", "spectrum_id", "sample_name"] + spectral_columns
    if "landcover_group" in header.columns:
        usecols.append("landcover_group")

    meta = metadata.copy()
    meta["source_id"] = meta["source_id"].astype(str)
    meta["spectrum_id"] = meta["spectrum_id"].astype(str)
    meta["sample_key"] = meta["source_id"] + "||" + meta["spectrum_id"]
    for chunk in pd.read_csv(spectra_csv, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["source_id"] = chunk["source_id"].astype(str)
        chunk["spectrum_id"] = chunk["spectrum_id"].astype(str)
        chunk["sample_key"] = chunk["source_id"] + "||" + chunk["spectrum_id"]
        drop_columns = ["landcover_group"]
        joined = chunk.merge(
            meta.drop(columns=[col for col in drop_columns if col not in meta.columns], errors="ignore"),
            on="sample_key",
            how="left",
            suffixes=("", "_meta"),
        )
        for column in ["source_id_meta", "spectrum_id_meta", "sample_name_meta"]:
            if column in joined.columns:
                joined = joined.drop(columns=[column])
        if "landcover_group_x" in joined.columns or "landcover_group_y" in joined.columns:
            joined["landcover_group"] = joined.get("landcover_group_x", "").fillna("").astype(str)
            if "landcover_group_y" in joined.columns:
                joined["landcover_group"] = joined["landcover_group"].replace("", pd.NA).fillna(joined["landcover_group_y"]).fillna("")
            joined = joined.drop(columns=[col for col in ["landcover_group_x", "landcover_group_y"] if col in joined.columns])
        joined["landcover_group"] = joined.get("landcover_group", "").fillna("")
        joined["source_name"] = joined["source_name"].fillna(joined["source_id"])
        yield joined, spectral_columns


def extract_top_spectra(spectra_csv: Path, top_metrics: pd.DataFrame, spectral_columns: list[str]) -> pd.DataFrame:
    wanted = set(top_metrics["source_id"].astype(str) + "||" + top_metrics["spectrum_id"].astype(str))
    rows: list[pd.DataFrame] = []
    usecols = ["source_id", "spectrum_id", "sample_name"] + spectral_columns
    header = pd.read_csv(spectra_csv, nrows=0)
    if "landcover_group" in header.columns:
        usecols.append("landcover_group")
    for chunk in pd.read_csv(spectra_csv, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["source_id"] = chunk["source_id"].astype(str)
        chunk["spectrum_id"] = chunk["spectrum_id"].astype(str)
        keys = chunk["source_id"] + "||" + chunk["spectrum_id"]
        subset = chunk[keys.isin(wanted)].copy()
        if not subset.empty:
            subset["source_name"] = subset["source_id"]
            rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=usecols + ["source_name"])
    combined = pd.concat(rows, ignore_index=True)
    combined["source_name"] = combined["source_name"].fillna(combined["source_id"])
    return combined


def build_summary(metrics: pd.DataFrame, output_root: Path) -> dict[str, object]:
    suspicious = metrics[metrics["is_suspicious"]].copy()
    suspicious["landcover_group"] = suspicious["landcover_group"].replace("", "unlabeled")
    by_source = (
        suspicious.groupby("source_id")
        .agg(
            suspicious_spectra=("spectrum_id", "count"),
            max_score=("strange_score", "max"),
            mean_score=("strange_score", "mean"),
            max_abs_residual=("max_abs_residual", "max"),
            max_abs_jump=("max_abs_jump", "max"),
        )
        .sort_values(["suspicious_spectra", "max_score"], ascending=[False, False])
        .reset_index()
    )
    by_landcover = (
        suspicious.groupby("landcover_group")
        .agg(suspicious_spectra=("spectrum_id", "count"))
        .sort_values("suspicious_spectra", ascending=False)
        .reset_index()
    )
    by_source.to_csv(output_root / "suspicious_by_source.csv", index=False)
    by_landcover.to_csv(output_root / "suspicious_by_landcover.csv", index=False)
    summary = {
        "total_spectra": int(len(metrics)),
        "suspicious_spectra": int(len(suspicious)),
        "suspicious_fraction": float(len(suspicious) / max(1, len(metrics))),
        "top_sources": by_source.head(15).to_dict(orient="records"),
        "top_landcover_groups": by_landcover.to_dict(orient="records"),
    }
    (output_root / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run suspicious-spectrum review over the full processed dataset.")
    parser.add_argument("--base-root", default="build/siac_spectral_library_v1")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--top-n", type=int, default=24)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root) if args.output_root else base_root / "full_review"
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_csv, spectra_csv = resolve_inputs(base_root)
    metadata = read_metadata(metadata_csv)

    header = pd.read_csv(spectra_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = pd.Index([int(column.split("_", 1)[1]) for column in spectral_columns]).to_numpy(dtype=int)

    metric_frames: list[pd.DataFrame] = []
    for joined, spectral_columns in iter_joined_chunks(metadata, spectra_csv):
        metric_frames.append(compute_chunk_metrics(joined, spectral_columns, wavelengths))

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics = finalize_scores(metrics)
    metrics.to_csv(output_root / "all_metrics.csv", index=False)
    metrics[metrics["is_suspicious"]].to_csv(output_root / "flagged_suspicious_spectra.csv", index=False)

    top_metrics = metrics.head(args.top_n).copy()
    top_spectra = extract_top_spectra(spectra_csv, top_metrics, spectral_columns)
    top_spectra = top_spectra.merge(
        top_metrics[["source_id", "spectrum_id", "source_name", "landcover_group"]],
        on=["source_id", "spectrum_id"],
        how="left",
        suffixes=("", "_meta"),
    )
    plot_suspicious_spectra(
        top_spectra,
        top_metrics,
        wavelengths,
        spectral_columns,
        output_root,
        top_n=args.top_n,
    )

    summary = build_summary(metrics, output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
