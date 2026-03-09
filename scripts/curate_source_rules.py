#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BLUE_FILL_COLUMNS = [f"nm_{wavelength}" for wavelength in range(400, 500)]
CHUNK_SIZE = 2048
SANTA_BARBARA_SOURCE_ID = "santa_barbara_urban_reflectance"
SANTA_BARBARA_SCALE_FIX = 10000.0


def resolve_tabular_root(dataset_root: Path) -> Path:
    candidate = dataset_root / "tabular"
    return candidate if candidate.exists() else dataset_root


def load_frames(base_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tabular_root = resolve_tabular_root(base_root)
    metadata = pd.read_csv(
        tabular_root / "spectra_metadata.csv",
        low_memory=False,
    )
    labels = pd.read_csv(
        base_root / "landcover_analysis" / "landcover_labels.csv",
        low_memory=False,
    )
    source_summary = pd.read_csv(
        tabular_root / "source_summary.csv",
        low_memory=False,
    )
    return metadata, labels, source_summary


def build_rule_frames(metadata: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = metadata.merge(
        labels[["source_id", "spectrum_id", "landcover_group"]],
        on=["source_id", "spectrum_id"],
        how="left",
    )
    frame["key"] = frame["source_id"].astype(str) + "||" + frame["spectrum_id"].astype(str)

    drop_existing = frame["source_id"] == "existing_urban_reference_data"
    drop_ossl = (
        (frame["source_id"] == "ossl")
        & (frame["landcover_group"] == "soil")
        & (frame["native_min_nm"] >= 452.0)
    )

    keep_mask = ~(drop_existing | drop_ossl)
    kept_frame = frame[keep_mask].copy()

    stats = {
        "base_spectra": int(len(frame)),
        "kept_spectra": int(len(kept_frame)),
        "dropped_existing_urban_reference_data": int(drop_existing.sum()),
        "dropped_ossl_partial_blue_soil": int(drop_ossl.sum()),
        "kept_ossl_full_blue_soil": int(
            (
                (kept_frame["source_id"] == "ossl")
                & (kept_frame["landcover_group"] == "soil")
            ).sum()
        ),
        "ghisaconus_gapfill_targets": int(
            (
                (kept_frame["source_id"] == "ghisaconus_v001")
                & (kept_frame["landcover_group"] == "vegetation")
            ).sum()
        ),
        "santa_barbara_scale_fix_targets": int(
            (
                (kept_frame["source_id"] == SANTA_BARBARA_SOURCE_ID)
                & (kept_frame["value_scale_applied"] == 1.0)
            ).sum()
        ),
    }
    return kept_frame, stats


def load_ghisaconus_replacements(repaired_root: Path) -> pd.DataFrame:
    tabular_root = resolve_tabular_root(repaired_root)
    usecols = ["source_id", "spectrum_id"] + BLUE_FILL_COLUMNS
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        tabular_root / "normalized_spectra.csv",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = chunk[chunk["source_id"] == "ghisaconus_v001"].copy()
        if not chunk.empty:
            rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["key"] + BLUE_FILL_COLUMNS)
    frame = pd.concat(rows, ignore_index=True)
    frame["key"] = frame["source_id"].astype(str) + "||" + frame["spectrum_id"].astype(str)
    return frame.drop(columns=["source_id", "spectrum_id"]).set_index("key")


def rebuild_source_summary(filtered_metadata: pd.DataFrame, original_source_summary: pd.DataFrame) -> pd.DataFrame:
    failure_lookup = dict(zip(original_source_summary["source_id"], original_source_summary["failure_count"]))
    rows: list[dict[str, object]] = []
    for source_id, frame in filtered_metadata.groupby("source_id", sort=True):
        parser_counts = frame["parser"].value_counts().sort_index()
        parser_string = ",".join(f"{parser}:{count}" for parser, count in parser_counts.items())
        rows.append(
            {
                "source_id": source_id,
                "source_name": frame["source_name"].iloc[0],
                "ingest_role": frame["ingest_role"].iloc[0],
                "failure_count": int(failure_lookup.get(source_id, 0)),
                "normalized_spectra": int(len(frame)),
                "parsers": parser_string,
            }
        )
    return pd.DataFrame(rows)


def copy_static_files(base_root: Path, output_root: Path) -> None:
    base_tabular = resolve_tabular_root(base_root)
    output_tabular = output_root / "tabular"
    output_landcover = output_root / "landcover_analysis"
    output_tabular.mkdir(parents=True, exist_ok=True)
    output_landcover.mkdir(parents=True, exist_ok=True)

    for name in ["wavelength_grid.csv", "normalization_failures.csv"]:
        source_path = base_tabular / name
        if source_path.exists():
            shutil.copy2(source_path, output_tabular / name)


def write_curated_dataset(
    base_root: Path,
    repaired_root: Path,
    output_root: Path,
) -> dict[str, int]:
    metadata, labels, source_summary = load_frames(base_root)
    kept_frame, stats = build_rule_frames(metadata, labels)
    keep_keys = set(kept_frame["key"])
    ghisa_keys = set(
        kept_frame.loc[
            (kept_frame["source_id"] == "ghisaconus_v001")
            & (kept_frame["landcover_group"] == "vegetation"),
            "key",
        ]
    )
    santa_barbara_keys = set(
        kept_frame.loc[
            (kept_frame["source_id"] == SANTA_BARBARA_SOURCE_ID)
            & (kept_frame["value_scale_applied"] == 1.0),
            "key",
        ]
    )
    replacements = load_ghisaconus_replacements(repaired_root)

    copy_static_files(base_root, output_root)
    output_tabular = output_root / "tabular"
    output_landcover = output_root / "landcover_analysis"

    filtered_metadata = kept_frame.drop(columns=["key", "landcover_group"])
    if santa_barbara_keys:
        santa_mask = (
            (filtered_metadata["source_id"] == SANTA_BARBARA_SOURCE_ID)
            & (filtered_metadata["spectrum_id"].astype(str).radd(f"{SANTA_BARBARA_SOURCE_ID}||").isin(santa_barbara_keys))
        )
        filtered_metadata.loc[santa_mask, "value_scale_applied"] = SANTA_BARBARA_SCALE_FIX
    filtered_metadata.to_csv(output_tabular / "spectra_metadata.csv", index=False)

    filtered_labels = labels.copy()
    filtered_labels["key"] = filtered_labels["source_id"].astype(str) + "||" + filtered_labels["spectrum_id"].astype(str)
    filtered_labels = filtered_labels[filtered_labels["key"].isin(keep_keys)].drop(columns=["key"])
    filtered_labels.to_csv(output_landcover / "landcover_labels.csv", index=False)

    rebuilt_source_summary = rebuild_source_summary(filtered_metadata, source_summary)
    rebuilt_source_summary.to_csv(output_tabular / "source_summary.csv", index=False)

    base_tabular = resolve_tabular_root(base_root)
    base_csv = base_tabular / "normalized_spectra.csv"
    output_csv = output_tabular / "normalized_spectra.csv"

    ghisa_filled_rows = 0
    santa_barbara_fixed_rows = 0
    header_written = False
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(base_csv, chunksize=CHUNK_SIZE, low_memory=False):
            chunk["key"] = chunk["source_id"].astype(str) + "||" + chunk["spectrum_id"].astype(str)
            chunk = chunk[chunk["key"].isin(keep_keys)].copy()
            if chunk.empty:
                continue

            ghisa_mask = chunk["key"].isin(ghisa_keys)
            if ghisa_mask.any():
                ghisa_chunk = chunk.loc[ghisa_mask, ["key"] + BLUE_FILL_COLUMNS].copy()
                donor = replacements.reindex(ghisa_chunk["key"])
                valid_rows = donor.notna().all(axis=1)
                if valid_rows.any():
                    valid_keys = ghisa_chunk.loc[valid_rows.to_numpy(), "key"].tolist()
                    donor_values = donor.loc[valid_rows].to_numpy(dtype=float)
                    row_mask = chunk["key"].isin(valid_keys)
                    chunk.loc[row_mask, BLUE_FILL_COLUMNS] = donor_values
                    ghisa_filled_rows += int(len(valid_keys))

            santa_mask = chunk["key"].isin(santa_barbara_keys)
            if santa_mask.any():
                spectral_columns = [column for column in chunk.columns if column.startswith("nm_")]
                chunk.loc[santa_mask, spectral_columns] = (
                    chunk.loc[santa_mask, spectral_columns].astype(float) / SANTA_BARBARA_SCALE_FIX
                )
                santa_barbara_fixed_rows += int(santa_mask.sum())

            chunk = chunk.drop(columns=["key"])
            chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

    stats["ghisaconus_gapfilled_rows"] = int(ghisa_filled_rows)
    stats["santa_barbara_scale_fixed_rows"] = int(santa_barbara_fixed_rows)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply source-specific curation rules to the normalized spectral dataset.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    repaired_root = Path(args.repaired_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = write_curated_dataset(base_root, repaired_root, output_root)
    summary = {
        "base_root": str(base_root),
        "repaired_root": str(repaired_root),
        "output_root": str(output_root),
        "rules": {
            "ossl_soil_keep_only_native_min_nm_below_452": True,
            "ghisaconus_v001_fill_nm_400_499_from_repaired_dataset": True,
            "remove_existing_urban_reference_data": True,
            "santa_barbara_urban_reflectance_rescale_rows_with_value_scale_1_to_10000": True,
        },
        "stats": stats,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
