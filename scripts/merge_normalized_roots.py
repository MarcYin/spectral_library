#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def resolve_tabular_root(root: Path) -> Path:
    candidate = root / "tabular"
    if candidate.exists():
        return candidate
    return root


def copy_wavelength_grid(input_roots: list[Path], output_root: Path) -> str:
    first_path = resolve_tabular_root(input_roots[0]) / "wavelength_grid.csv"
    if not first_path.exists():
        raise FileNotFoundError(f"Missing wavelength grid: {first_path}")
    reference = first_path.read_text(encoding="utf-8")
    for root in input_roots[1:]:
        candidate = resolve_tabular_root(root) / "wavelength_grid.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing wavelength grid: {candidate}")
        text = candidate.read_text(encoding="utf-8")
        if text != reference:
            raise ValueError(f"Wavelength grid mismatch: {candidate} does not match {first_path}")
    target = output_root / "tabular" / "wavelength_grid.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first_path, target)
    return str(target)


def stream_concat_csv(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    key_columns: list[str] | None = None,
) -> tuple[int, int]:
    written_rows = 0
    duplicate_rows = 0
    header: list[str] | None = None
    seen_keys: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.DictWriter[str] | None = None
        for path in input_paths:
            if not path.exists():
                continue
            with path.open("r", newline="", encoding="utf-8", errors="replace") as in_handle:
                reader = csv.DictReader(in_handle)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = list(reader.fieldnames)
                    writer = csv.DictWriter(out_handle, fieldnames=header)
                    writer.writeheader()
                elif list(reader.fieldnames) != header:
                    raise ValueError(f"CSV header mismatch in {path}")
                assert writer is not None
                for row in reader:
                    if key_columns:
                        key = "\t".join(str(row[column]) for column in key_columns)
                        if key in seen_keys:
                            duplicate_rows += 1
                            continue
                        seen_keys.add(key)
                    writer.writerow(row)
                    written_rows += 1
    return written_rows, duplicate_rows


def build_parser_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_id", "normalized_spectra", "parsers"])

    rows: list[dict[str, object]] = []
    for source_id, group in frame.groupby("source_id", sort=False):
        parser_counts = Counter(group["parser"].fillna("unknown"))
        rows.append(
            {
                "source_id": source_id,
                "normalized_spectra": int(len(group.index)),
                "parsers": ",".join(f"{name}:{count}" for name, count in sorted(parser_counts.items())),
            }
        )
    return pd.DataFrame(rows)


def rebuild_source_summary(output_root: Path) -> dict[str, int]:
    tabular_root = output_root / "tabular"
    metadata = pd.read_csv(tabular_root / "spectra_metadata.csv", low_memory=False)
    failures_path = tabular_root / "normalization_failures.csv"
    failures = pd.read_csv(failures_path, low_memory=False) if failures_path.exists() else pd.DataFrame(columns=["source_id"])

    source_info = (
        metadata.groupby("source_id", as_index=False)
        .agg(
            source_name=("source_name", "first"),
            ingest_role=("ingest_role", "first"),
        )
    )
    parser_summary = build_parser_summary(metadata)
    failure_summary = (
        failures.groupby("source_id", as_index=False).size().rename(columns={"size": "failure_count"})
        if not failures.empty and "source_id" in failures.columns
        else pd.DataFrame(columns=["source_id", "failure_count"])
    )
    source_summary = (
        source_info.merge(parser_summary, on="source_id", how="left")
        .merge(failure_summary, on="source_id", how="left")
        .fillna({"normalized_spectra": 0, "parsers": "", "failure_count": 0})
    )
    source_summary["normalized_spectra"] = source_summary["normalized_spectra"].astype(int)
    source_summary["failure_count"] = source_summary["failure_count"].astype(int)
    source_summary = source_summary[
        ["source_id", "source_name", "ingest_role", "failure_count", "normalized_spectra", "parsers"]
    ].sort_values("source_id")
    source_summary.to_csv(tabular_root / "source_summary.csv", index=False)
    return {
        "source_count": int(source_summary["source_id"].nunique()),
        "normalized_spectra": int(source_summary["normalized_spectra"].sum()),
        "failure_count": int(source_summary["failure_count"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple normalized roots into one normalized dataset.")
    parser.add_argument("--input-roots", required=True, help="Comma-separated normalized dataset roots.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default="manifests/sources.csv")
    args = parser.parse_args()

    input_roots = [Path(value.strip()) for value in args.input_roots.split(",") if value.strip()]
    if not input_roots:
        raise SystemExit("No input roots provided.")

    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "tabular").mkdir(parents=True, exist_ok=True)

    wavelength_grid_path = copy_wavelength_grid(input_roots, output_root)

    metadata_rows, metadata_duplicates = stream_concat_csv(
        [resolve_tabular_root(root) / "spectra_metadata.csv" for root in input_roots],
        output_root / "tabular" / "spectra_metadata.csv",
        key_columns=["source_id", "spectrum_id"],
    )
    spectra_rows, spectra_duplicates = stream_concat_csv(
        [resolve_tabular_root(root) / "normalized_spectra.csv" for root in input_roots],
        output_root / "tabular" / "normalized_spectra.csv",
        key_columns=["source_id", "spectrum_id"],
    )
    failure_rows, _ = stream_concat_csv(
        [resolve_tabular_root(root) / "normalization_failures.csv" for root in input_roots],
        output_root / "tabular" / "normalization_failures.csv",
        key_columns=None,
    )

    manifest_target = output_root / "manifests" / "sources.csv"
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.manifest), manifest_target)

    source_summary = rebuild_source_summary(output_root)

    summary = {
        "input_roots": [str(root) for root in input_roots],
        "output_root": str(output_root),
        "manifest": str(Path(args.manifest)),
        "wavelength_grid": wavelength_grid_path,
        "spectra_metadata_rows": metadata_rows,
        "normalized_spectra_rows": spectra_rows,
        "normalization_failure_rows": failure_rows,
        "spectra_metadata_duplicates_skipped": metadata_duplicates,
        "normalized_spectra_duplicates_skipped": spectra_duplicates,
        "source_summary": source_summary,
    }
    (output_root / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
