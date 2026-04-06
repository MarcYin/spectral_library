from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .plots import EXPECTED_GRID_POINTS


def _resolve_tabular_root(normalized_root: Path) -> Path:
    candidate = normalized_root / "tabular"
    if candidate.exists():
        return candidate
    return normalized_root


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required normalized table is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _stream_filter_normalized_spectra(input_path: Path, output_path: Path, keep_keys: set[str]) -> int:
    kept_rows = 0
    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
        reader = csv.DictReader(input_handle)
        with output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = f"{row['source_id']}\t{row['spectrum_id']}"
                if key not in keep_keys:
                    continue
                writer.writerow(row)
                kept_rows += 1
    return kept_rows


def _build_parser_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_id", "normalized_spectra", "parsers"])

    rows: list[dict[str, Any]] = []
    for source_id, group in frame.groupby("source_id", sort=False):
        parser_counts = Counter(group["parser"].fillna("unknown"))
        parser_text = ",".join(f"{name}:{count}" for name, count in sorted(parser_counts.items()))
        rows.append(
            {
                "source_id": source_id,
                "normalized_spectra": int(len(group.index)),
                "parsers": parser_text,
            }
        )
    return pd.DataFrame(rows)


def filter_normalized_by_coverage(
    normalized_root: Path,
    output_root: Path,
    *,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    if min_coverage < 0 or min_coverage > 1:
        raise ValueError("min_coverage must be between 0 and 1")

    tabular_root = _resolve_tabular_root(normalized_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_tabular = output_root / "tabular"
    output_tabular.mkdir(parents=True, exist_ok=True)

    source_summary_path = tabular_root / "source_summary.csv"
    spectra_metadata_path = tabular_root / "spectra_metadata.csv"
    normalized_spectra_path = tabular_root / "normalized_spectra.csv"

    source_summary = _read_required_csv(source_summary_path)
    metadata = _read_required_csv(spectra_metadata_path)
    metadata["normalized_points"] = pd.to_numeric(metadata["normalized_points"], errors="coerce")
    metadata["coverage_fraction"] = metadata["normalized_points"] / EXPECTED_GRID_POINTS

    retained_metadata = metadata[metadata["coverage_fraction"] >= min_coverage].copy()
    retained_metadata = retained_metadata.drop(columns=["coverage_fraction"])
    retained_metadata.to_csv(output_tabular / "spectra_metadata.csv", index=False)

    parser_summary = _build_parser_summary(retained_metadata)
    summary = source_summary.copy()
    summary["failure_count"] = pd.to_numeric(summary.get("failure_count"), errors="coerce").fillna(0).astype(int)
    summary = summary.drop(columns=["normalized_spectra", "parsers"], errors="ignore")
    summary = summary.merge(parser_summary, on="source_id", how="left")
    summary["normalized_spectra"] = pd.to_numeric(summary["normalized_spectra"], errors="coerce").fillna(0).astype(int)
    summary["parsers"] = summary["parsers"].fillna("")
    summary.to_csv(output_tabular / "source_summary.csv", index=False)

    keep_keys = {
        f"{source_id}\t{spectrum_id}"
        for source_id, spectrum_id in retained_metadata[["source_id", "spectrum_id"]].itertuples(index=False, name=None)
    }
    kept_rows = _stream_filter_normalized_spectra(normalized_spectra_path, output_tabular / "normalized_spectra.csv", keep_keys)

    copied_files: list[str] = []
    for name in ["wavelength_grid.csv", "normalization_failures.csv"]:
        source_path = tabular_root / name
        if source_path.exists():
            shutil.copy2(source_path, output_tabular / name)
            copied_files.append(name)

    summary_payload = {
        "input_tabular_root": str(tabular_root),
        "output_tabular_root": str(output_tabular),
        "min_coverage": min_coverage,
        "input_spectra": int(len(metadata.index)),
        "retained_spectra": int(len(retained_metadata.index)),
        "dropped_spectra": int(len(metadata.index) - len(retained_metadata.index)),
        "written_normalized_rows": kept_rows,
        "retained_sources": int(retained_metadata["source_id"].nunique()) if "source_id" in retained_metadata.columns else 0,
        "copied_files": copied_files,
    }
    summary_path = output_root / "coverage_filter_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_payload["summary_path"] = str(summary_path)
    return summary_payload
