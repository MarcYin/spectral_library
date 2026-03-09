#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 2048


def copy_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def filter_large_csv(source_path: Path, target_path: Path, excluded: set[str]) -> dict[str, int]:
    kept_rows = 0
    dropped_rows = 0
    header_written = False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(source_path, chunksize=CHUNK_SIZE, low_memory=False):
            source_ids = chunk["source_id"].astype(str)
            mask = ~source_ids.isin(excluded)
            kept = chunk[mask]
            dropped_rows += int((~mask).sum())
            kept_rows += int(mask.sum())
            if not kept.empty:
                kept.to_csv(handle, index=False, header=not header_written)
                header_written = True
    if not header_written:
        pd.read_csv(source_path, nrows=0).to_csv(target_path, index=False)
    return {"kept_rows": kept_rows, "dropped_rows": dropped_rows}


def filter_small_csv(source_path: Path, target_path: Path, excluded: set[str]) -> dict[str, int]:
    frame = pd.read_csv(source_path, low_memory=False)
    before = len(frame)
    if "source_id" in frame.columns:
        frame = frame[~frame["source_id"].astype(str).isin(excluded)].copy()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target_path, index=False)
    return {"kept_rows": int(len(frame)), "dropped_rows": int(before - len(frame))}


def append_manifest_note(manifest_path: Path, excluded: set[str]) -> None:
    frame = pd.read_csv(manifest_path, low_memory=False)
    notes = frame["notes"].fillna("").astype(str)
    excluded_note = "Excluded from curated merged database due to unstable visible-band behavior."
    mask = frame["source_id"].astype(str).isin(excluded)
    frame.loc[mask, "notes"] = notes[mask].apply(
        lambda current: excluded_note if not current else f"{current} {excluded_note}"
    )
    frame.to_csv(manifest_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exclude one or more sources from a normalized dataset tree.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exclude-source", action="append", required=True)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    excluded = {value.strip() for value in args.exclude_source if value.strip()}

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy files that do not need filtering.
    copy_if_exists(base_root / "tabular" / "wavelength_grid.csv", output_root / "tabular" / "wavelength_grid.csv")
    copy_if_exists(base_root / "tabular" / "normalization_failures.csv", output_root / "tabular" / "normalization_failures.csv")

    summaries: dict[str, dict[str, int]] = {}
    for relative_path in [
        Path("tabular/normalized_spectra.csv"),
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("landcover_analysis/landcover_labels.csv"),
        Path("diagnostics/source_flag_summary.csv"),
        Path("diagnostics/spectrum_flag_counts.csv"),
        Path("diagnostics/window_outlier_flags.csv"),
    ]:
        source_path = base_root / relative_path
        target_path = output_root / relative_path
        if not source_path.exists():
            continue
        if relative_path.name in {"normalized_spectra.csv", "spectra_metadata.csv"}:
            summaries[str(relative_path)] = filter_large_csv(source_path, target_path, excluded)
        else:
            summaries[str(relative_path)] = filter_small_csv(source_path, target_path, excluded)

    if args.manifest:
        manifest_path = output_root / "manifests" / Path(args.manifest).name
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, manifest_path)
        append_manifest_note(manifest_path, excluded)

    total_dropped = sum(item["dropped_rows"] for item in summaries.values())
    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "excluded_sources": sorted(excluded),
        "tables": summaries,
        "total_dropped_rows": int(total_dropped),
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
