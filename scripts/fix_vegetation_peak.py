#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 2048
SOURCE_ID = "ghisaconus_v001"
TARGET_GROUP = "vegetation"
ANCHOR_LEFT = 420
ANCHOR_RIGHT = 500
REPLACE_START = 430
REPLACE_END = 500


def copy_static_files(base_root: Path, output_root: Path) -> None:
    output_tabular = output_root / "tabular"
    output_landcover = output_root / "landcover_analysis"
    output_tabular.mkdir(parents=True, exist_ok=True)
    output_landcover.mkdir(parents=True, exist_ok=True)

    for name in ["spectra_metadata.csv", "source_summary.csv", "wavelength_grid.csv", "normalization_failures.csv"]:
        source_path = base_root / "tabular" / name
        if source_path.exists():
            shutil.copy2(source_path, output_tabular / name)

    for name in ["landcover_labels.csv"]:
        source_path = base_root / "landcover_analysis" / name
        if source_path.exists():
            shutil.copy2(source_path, output_landcover / name)


def load_vegetation_keys(labels_path: Path) -> tuple[set[str], set[str]]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    labels = labels[labels["landcover_group"] == TARGET_GROUP].copy()
    labels["key"] = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    vegetation_keys = set(labels["key"])
    target_keys = set(labels.loc[labels["source_id"] == SOURCE_ID, "key"])
    return vegetation_keys, target_keys


def build_template(normalized_csv: Path, vegetation_keys: set[str], wavelengths: np.ndarray) -> np.ndarray:
    columns = ["source_id", "spectrum_id"] + [f"nm_{wavelength}" for wavelength in wavelengths]
    source_sums: dict[str, np.ndarray] = {}
    source_counts: dict[str, np.ndarray] = {}

    for chunk in pd.read_csv(normalized_csv, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
        chunk = chunk[keys.isin(vegetation_keys)]
        if chunk.empty:
            continue

        for source_id, frame in chunk.groupby("source_id"):
            source_id = str(source_id)
            if source_id == SOURCE_ID:
                continue
            values = frame[columns[2:]].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if source_id not in source_sums:
                source_sums[source_id] = np.zeros(len(wavelengths), dtype=float)
                source_counts[source_id] = np.zeros(len(wavelengths), dtype=float)
            source_sums[source_id] += np.where(valid, values, 0.0).sum(axis=0)
            source_counts[source_id] += valid.sum(axis=0)

    if not source_sums:
        raise RuntimeError("No vegetation sources available to build the GHISACONUS replacement template.")

    per_source_means = []
    for source_id in sorted(source_sums):
        per_source_means.append(
            np.divide(
                source_sums[source_id],
                source_counts[source_id],
                out=np.full_like(source_sums[source_id], np.nan),
                where=source_counts[source_id] > 0,
            )
        )
    return np.nanmean(np.vstack(per_source_means), axis=0)


def replace_ghisaconus_segment(frame: pd.DataFrame, template: np.ndarray, wavelengths: np.ndarray) -> tuple[int, int]:
    template_left = float(template[ANCHOR_LEFT - wavelengths[0]])
    template_right = float(template[ANCHOR_RIGHT - wavelengths[0]])
    template_span = template_right - template_left
    if not np.isfinite(template_span) or abs(template_span) < 1e-12:
        return 0, 0

    replace_columns = [f"nm_{wavelength}" for wavelength in range(REPLACE_START, REPLACE_END + 1)]
    left_values = frame[f"nm_{ANCHOR_LEFT}"].to_numpy(dtype=float)
    right_values = frame[f"nm_{ANCHOR_RIGHT}"].to_numpy(dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if not valid.any():
        return 0, 0

    template_segment = template[REPLACE_START - wavelengths[0] : REPLACE_END - wavelengths[0] + 1]
    normalized_segment = (template_segment - template_left) / template_span
    replacement = left_values[valid].reshape(-1, 1) + (right_values[valid] - left_values[valid]).reshape(-1, 1) * normalized_segment.reshape(1, -1)
    frame.loc[valid, replace_columns] = replacement
    return int(valid.sum()), int(replacement.size)


def write_fixed_dataset(base_root: Path, output_root: Path) -> dict[str, object]:
    copy_static_files(base_root, output_root)

    wavelengths = np.arange(ANCHOR_LEFT, ANCHOR_RIGHT + 1, dtype=int)
    vegetation_keys, target_keys = load_vegetation_keys(base_root / "landcover_analysis" / "landcover_labels.csv")
    template = build_template(base_root / "tabular" / "normalized_spectra.csv", vegetation_keys, wavelengths)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"

    replaced_rows = 0
    replaced_values = 0
    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
            target_mask = keys.isin(target_keys)
            if target_mask.any():
                target_chunk = chunk.loc[target_mask].copy()
                row_count, value_count = replace_ghisaconus_segment(target_chunk, template, wavelengths)
                if row_count:
                    replaced_rows += row_count
                    replaced_values += value_count
                    chunk.loc[target_mask, target_chunk.columns] = target_chunk.values
            chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

    return {
        "source_id": SOURCE_ID,
        "anchor_left_nm": ANCHOR_LEFT,
        "anchor_right_nm": ANCHOR_RIGHT,
        "replace_start_nm": REPLACE_START,
        "replace_end_nm": REPLACE_END,
        "rows_replaced": replaced_rows,
        "values_replaced": replaced_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix the vegetation blue-region peak by replacing the GHISACONUS 430-500 nm segment with a template-scaled curve.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = write_fixed_dataset(base_root, output_root)
    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "rule": {
            "target_source": SOURCE_ID,
            "landcover_group": TARGET_GROUP,
            "replacement_method": "source_balanced_other_vegetation_template_scaled_to_nm_420_and_nm_500",
            "replace_range_nm": [REPLACE_START, REPLACE_END],
        },
        "stats": stats,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
