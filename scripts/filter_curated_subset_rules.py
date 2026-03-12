#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


CURATED_RULES = [
    {
        "name": "exclude_bssl_longwave_only_subset",
        "source_id": "bssl",
        "native_min_nm_min": 600.0,
        "native_spacing_nm_min": 5.0,
        "reason": "bssl 600-3000 nm 5 nm subset creates a false onset at 600 nm in pooled soil means",
    },
]


def stream_filter_normalized_spectra(input_path: Path, output_path: Path, keep_keys: set[str]) -> int:
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


def build_parser_summary(frame: pd.DataFrame) -> pd.DataFrame:
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


def filter_small_table(source_path: Path, target_path: Path, keep_keys: set[str]) -> dict[str, int]:
    frame = pd.read_csv(source_path, low_memory=False)
    before = len(frame.index)
    if {"source_id", "spectrum_id"}.issubset(frame.columns):
        keys = frame["source_id"].astype(str) + "\t" + frame["spectrum_id"].astype(str)
        frame = frame[keys.isin(keep_keys)].copy()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target_path, index=False)
    return {"kept_rows": int(len(frame.index)), "dropped_rows": int(before - len(frame.index))}


def apply_rules(metadata: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    exclude_mask = pd.Series(False, index=metadata.index)
    diagnostics_rows: list[dict[str, Any]] = []
    for rule in CURATED_RULES:
        rule_mask = metadata["source_id"].astype(str) == str(rule["source_id"])
        if "native_min_nm_min" in rule:
            rule_mask &= pd.to_numeric(metadata["native_min_nm"], errors="coerce") >= float(rule["native_min_nm_min"])
        if "native_spacing_nm_min" in rule:
            rule_mask &= pd.to_numeric(metadata["native_spacing_nm"], errors="coerce") >= float(rule["native_spacing_nm_min"])
        matched = metadata[rule_mask].copy()
        if matched.empty:
            continue
        matched["rule_name"] = str(rule["name"])
        matched["exclusion_reason"] = str(rule["reason"])
        diagnostics_rows.extend(matched.to_dict(orient="records"))
        exclude_mask |= rule_mask
    diagnostics = pd.DataFrame(diagnostics_rows)
    return exclude_mask, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply curated spectrum-subset exclusion rules to a normalized dataset root.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_tabular = base_root / "tabular"
    output_tabular = output_root / "tabular"
    output_tabular.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(input_tabular / "spectra_metadata.csv", low_memory=False)
    exclude_mask, diagnostics = apply_rules(metadata)
    retained_metadata = metadata[~exclude_mask].copy()
    retained_metadata.to_csv(output_tabular / "spectra_metadata.csv", index=False)

    keep_keys = {
        f"{source_id}\t{spectrum_id}"
        for source_id, spectrum_id in retained_metadata[["source_id", "spectrum_id"]].itertuples(index=False, name=None)
    }
    kept_rows = stream_filter_normalized_spectra(
        input_tabular / "normalized_spectra.csv",
        output_tabular / "normalized_spectra.csv",
        keep_keys,
    )

    source_summary = pd.read_csv(input_tabular / "source_summary.csv", low_memory=False)
    parser_summary = build_parser_summary(retained_metadata)
    rebuilt_source_summary = source_summary.drop(columns=["normalized_spectra", "parsers"], errors="ignore").merge(
        parser_summary,
        on="source_id",
        how="left",
    )
    rebuilt_source_summary["normalized_spectra"] = (
        pd.to_numeric(rebuilt_source_summary["normalized_spectra"], errors="coerce").fillna(0).astype(int)
    )
    rebuilt_source_summary["parsers"] = rebuilt_source_summary["parsers"].fillna("")
    rebuilt_source_summary = rebuilt_source_summary[rebuilt_source_summary["normalized_spectra"] > 0].reset_index(drop=True)
    rebuilt_source_summary.to_csv(output_tabular / "source_summary.csv", index=False)

    copied_files: list[str] = []
    for relative in [
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("manifests/sources.csv"),
    ]:
        source_path = base_root / relative
        if source_path.exists():
            target_path = output_root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_files.append(str(relative))

    filtered_tables: dict[str, dict[str, int]] = {}
    for relative in [Path("landcover_analysis/landcover_labels.csv")]:
        source_path = base_root / relative
        if source_path.exists():
            filtered_tables[str(relative)] = filter_small_table(source_path, output_root / relative, keep_keys)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_dir / "excluded_spectrum_subsets.csv", index=False)

    by_source = (
        metadata.assign(excluded=exclude_mask)
        .groupby("source_id", dropna=False)["excluded"]
        .agg(total="size", excluded="sum")
        .reset_index()
    )
    by_source["retained"] = by_source["total"] - by_source["excluded"]
    by_source["excluded_fraction"] = by_source["excluded"] / by_source["total"]
    by_source.to_csv(diagnostics_dir / "subset_filter_by_source.csv", index=False)

    rule_summary = (
        diagnostics.groupby(["rule_name", "source_id"], dropna=False)
        .agg(excluded_spectra=("spectrum_id", "nunique"))
        .reset_index()
        if not diagnostics.empty
        else pd.DataFrame(columns=["rule_name", "source_id", "excluded_spectra"])
    )
    rule_summary.to_csv(diagnostics_dir / "subset_filter_rule_summary.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "input_spectra": int(len(metadata.index)),
        "retained_spectra": int(len(retained_metadata.index)),
        "excluded_spectra": int(exclude_mask.sum()),
        "retained_sources": int(retained_metadata["source_id"].nunique()),
        "written_normalized_rows": kept_rows,
        "copied_files": copied_files,
        "filtered_tables": filtered_tables,
        "rules": CURATED_RULES,
    }
    output_summary = output_root / "curated_subset_filter_summary.json"
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
