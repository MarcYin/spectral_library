#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 2048
VEGETATION_SOURCE_ID = "ghisaconus_v001"
VEGETATION_BRIDGE_START = 430
VEGETATION_BRIDGE_END = 436
VEGETATION_ANCHOR_RIGHT = 437
URBAN_RANGE_START = 400
URBAN_RANGE_END = 550
URBAN_FIT_POINTS = 10


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


def build_group_map(labels_path: Path) -> dict[str, str]:
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    keys = labels["source_id"].astype(str) + "||" + labels["spectrum_id"].astype(str)
    return dict(zip(keys, labels["landcover_group"].astype(str)))


def bridge_ghisaconus_blue_edge(frame: pd.DataFrame) -> int:
    bridge_columns = [f"nm_{wavelength}" for wavelength in range(VEGETATION_BRIDGE_START, VEGETATION_BRIDGE_END + 1)]
    right_column = f"nm_{VEGETATION_ANCHOR_RIGHT}"

    values = frame[bridge_columns].to_numpy(dtype=float)
    left_anchor = values[:, 0]
    right_anchor = frame[right_column].to_numpy(dtype=float)
    valid = np.isfinite(left_anchor) & np.isfinite(right_anchor)
    if not valid.any():
        return 0

    x = np.arange(VEGETATION_BRIDGE_START, VEGETATION_BRIDGE_END + 1, dtype=float)
    denominator = float(VEGETATION_ANCHOR_RIGHT - VEGETATION_BRIDGE_START)
    fractions = ((x - VEGETATION_BRIDGE_START) / denominator).reshape(1, -1)
    interpolated = left_anchor[valid].reshape(-1, 1) + (right_anchor[valid] - left_anchor[valid]).reshape(-1, 1) * fractions
    frame.loc[valid, bridge_columns] = interpolated
    return int(valid.sum())


def backfill_urban_blue_edge(frame: pd.DataFrame) -> tuple[int, Counter[str]]:
    columns = [f"nm_{wavelength}" for wavelength in range(URBAN_RANGE_START, URBAN_RANGE_END + 1)]
    wavelengths = np.arange(URBAN_RANGE_START, URBAN_RANGE_END + 1, dtype=float)
    values = frame[columns].to_numpy(dtype=float)

    repaired_rows = 0
    by_source: Counter[str] = Counter()

    for row_index in range(values.shape[0]):
        row = values[row_index]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        first_idx = int(np.flatnonzero(finite)[0])
        if first_idx == 0:
            continue

        available_idx = np.flatnonzero(finite)
        fit_idx = available_idx[: min(URBAN_FIT_POINTS, len(available_idx))]
        fit_x = wavelengths[fit_idx]
        fit_y = row[fit_idx]
        if len(fit_idx) >= 2:
            slope, intercept = np.polyfit(fit_x, fit_y, 1)
            predicted = slope * wavelengths[:first_idx] + intercept
        else:
            predicted = np.full(first_idx, fit_y[0], dtype=float)

        row[:first_idx] = np.clip(predicted, 0.0, 1.0)
        repaired_rows += 1
        by_source[str(frame.iloc[row_index]["source_id"])] += 1

    frame.loc[:, columns] = values
    return repaired_rows, by_source


def write_fixed_dataset(base_root: Path, output_root: Path) -> dict[str, object]:
    copy_static_files(base_root, output_root)
    group_map = build_group_map(base_root / "landcover_analysis" / "landcover_labels.csv")

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"

    vegetation_rows = 0
    urban_rows = 0
    urban_by_source: Counter[str] = Counter()

    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            keys = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
            groups = keys.map(group_map)
            chunk = chunk.assign(landcover_group=groups)

            vegetation_mask = (chunk["source_id"] == VEGETATION_SOURCE_ID) & (chunk["landcover_group"] == "vegetation")
            if vegetation_mask.any():
                vegetation_rows += bridge_ghisaconus_blue_edge(chunk.loc[vegetation_mask].copy())
                bridged = chunk.loc[vegetation_mask].copy()
                bridge_ghisaconus_blue_edge(bridged)
                chunk.loc[vegetation_mask, bridged.columns] = bridged.values

            urban_mask = chunk["landcover_group"] == "urban"
            if urban_mask.any():
                urban_chunk = chunk.loc[urban_mask].copy()
                repaired_count, repaired_sources = backfill_urban_blue_edge(urban_chunk)
                if repaired_count:
                    urban_rows += repaired_count
                    urban_by_source.update(repaired_sources)
                    chunk.loc[urban_mask, urban_chunk.columns] = urban_chunk.values

            chunk = chunk.drop(columns=["landcover_group"])
            chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

    return {
        "vegetation_rows_bridged": int(vegetation_rows),
        "urban_rows_backfilled": int(urban_rows),
        "urban_rows_backfilled_by_source": dict(sorted(urban_by_source.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply vegetation and urban blue-edge fixes while leaving soil and water unchanged.")
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
        "rules": {
            "ghisaconus_v001_bridge_nm_430_436_to_nm_437": True,
            "urban_leading_blue_gap_backfill_from_first_measured_bands": True,
            "soil_unchanged": True,
            "water_unchanged": True,
        },
        "stats": stats,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
