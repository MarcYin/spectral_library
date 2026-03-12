#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CHUNK_SIZE = 256
BLEND_HALF_WINDOW_NM = 50


@dataclass(frozen=True)
class WindowRule:
    name: str
    replace_start: int
    replace_end: int
    fit_start: int
    fit_end: int


RULES_BY_SOURCE: dict[str, list[WindowRule]] = {
    "hyspiri_ground_targets": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500),
    ],
    "ghisacasia_v001": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500),
    ],
    "ngee_arctic_2018": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500),
    ],
    "santa_barbara_urban_reflectance": [
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500),
    ],
}


def copy_support_files(source_root: Path, output_root: Path) -> None:
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("manifests/sources.csv"),
    ]:
        source_path = source_root / relative
        if source_path.exists():
            target_path = output_root / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def max_adjacent_jump(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    filtered = values[finite]
    return float(np.max(np.abs(np.diff(filtered))))


def mean_abs_second_diff(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    filtered = values[finite]
    if filtered.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(filtered, n=2))))


def boundary_jump(values: np.ndarray, wavelengths: np.ndarray, start_nm: int, end_nm: int) -> float:
    replace_idx = np.flatnonzero((wavelengths >= start_nm) & (wavelengths <= end_nm))
    if replace_idx.size == 0:
        return 0.0
    first = int(replace_idx[0])
    last = int(replace_idx[-1])
    jumps: list[float] = []
    if first > 0 and np.isfinite(values[first - 1]) and np.isfinite(values[first]):
        jumps.append(float(abs(values[first] - values[first - 1])))
    if last < len(values) - 1 and np.isfinite(values[last]) and np.isfinite(values[last + 1]):
        jumps.append(float(abs(values[last + 1] - values[last])))
    return max(jumps) if jumps else 0.0


def score_window(values: np.ndarray, wavelengths: np.ndarray, rule: WindowRule) -> tuple[float, float, float]:
    return (
        boundary_jump(values, wavelengths, rule.replace_start, rule.replace_end),
        max_adjacent_jump(values),
        mean_abs_second_diff(values),
    )


def assert_same_keys(*chunks: pd.DataFrame) -> None:
    first = chunks[0][["source_id", "spectrum_id"]].astype(str)
    for chunk in chunks[1:]:
        current = chunk[["source_id", "spectrum_id"]].astype(str)
        if not first.equals(current):
            raise RuntimeError("Chunk ordering mismatch between variant roots.")


def run_selection(base_root: Path, old_root: Path, guarded_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copy_support_files(guarded_root, output_root)

    base_csv = base_root / "tabular" / "normalized_spectra.csv"
    old_csv = old_root / "tabular" / "normalized_spectra.csv"
    guarded_csv = guarded_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(base_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)

    diagnostics: list[dict[str, object]] = []
    selection_counts = {"base": 0, "old": 0, "guarded": 0}
    header_written = False

    base_iter = pd.read_csv(base_csv, chunksize=CHUNK_SIZE, low_memory=False)
    old_iter = pd.read_csv(old_csv, chunksize=CHUNK_SIZE, low_memory=False)
    guarded_iter = pd.read_csv(guarded_csv, chunksize=CHUNK_SIZE, low_memory=False)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for base_chunk, old_chunk, guarded_chunk in zip(base_iter, old_iter, guarded_iter):
            assert_same_keys(base_chunk, old_chunk, guarded_chunk)

            output_chunk = guarded_chunk.copy()
            base_values = base_chunk[spectral_columns].to_numpy(dtype=float)
            old_values = old_chunk[spectral_columns].to_numpy(dtype=float)
            guarded_values = guarded_chunk[spectral_columns].to_numpy(dtype=float)
            output_values = guarded_values.copy()
            source_ids = base_chunk["source_id"].astype(str).to_numpy()
            spectrum_ids = base_chunk["spectrum_id"].astype(str).to_numpy()

            for row_index, source_id in enumerate(source_ids):
                rules = RULES_BY_SOURCE.get(source_id)
                if not rules:
                    continue
                for rule in rules:
                    blend_start = max(rule.fit_start, rule.replace_start - BLEND_HALF_WINDOW_NM)
                    blend_end = min(rule.fit_end, rule.replace_end + BLEND_HALF_WINDOW_NM)
                    blend_mask = (wavelengths >= blend_start) & (wavelengths <= blend_end)
                    if not blend_mask.any():
                        continue
                    local_wavelengths = wavelengths[blend_mask]
                    base_window = base_values[row_index, blend_mask]
                    old_window = old_values[row_index, blend_mask]
                    guarded_window = guarded_values[row_index, blend_mask]
                    scores = {
                        "base": score_window(base_window, local_wavelengths, rule),
                        "old": score_window(old_window, local_wavelengths, rule),
                        "guarded": score_window(guarded_window, local_wavelengths, rule),
                    }
                    selected = min(scores, key=scores.get)
                    if selected == "base":
                        output_values[row_index, blend_mask] = base_window
                    elif selected == "old":
                        output_values[row_index, blend_mask] = old_window
                    else:
                        output_values[row_index, blend_mask] = guarded_window
                    selection_counts[selected] += 1
                    diagnostics.append(
                        {
                            "source_id": source_id,
                            "spectrum_id": spectrum_ids[row_index],
                            "window_name": rule.name,
                            "selected_variant": selected,
                            "base_score": scores["base"][0] + scores["base"][1] + scores["base"][2],
                            "old_score": scores["old"][0] + scores["old"][1] + scores["old"][2],
                            "guarded_score": scores["guarded"][0] + scores["guarded"][1] + scores["guarded"][2],
                            "base_boundary_jump": scores["base"][0],
                            "old_boundary_jump": scores["old"][0],
                            "guarded_boundary_jump": scores["guarded"][0],
                        }
                    )

            output_chunk.loc[:, spectral_columns] = output_values
            output_chunk.to_csv(handle, index=False, header=not header_written)
            header_written = True

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame.to_csv(diagnostics_dir / "variant_selection.csv", index=False)
    summary_frame = (
        diagnostics_frame.groupby(["source_id", "window_name", "selected_variant"], dropna=False)
        .size()
        .rename("selection_count")
        .reset_index()
        .sort_values(["source_id", "window_name", "selection_count"], ascending=[True, True, False])
    )
    summary_frame.to_csv(diagnostics_dir / "variant_selection_summary.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "old_root": str(old_root),
        "guarded_root": str(guarded_root),
        "output_root": str(output_root),
        "selection_counts": selection_counts,
        "diagnostic_rows": int(len(diagnostics_frame)),
    }
    (output_root / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the best source-specific absorption repair variant per spectrum/window.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--guarded-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_selection(
        Path(args.base_root),
        Path(args.old_root),
        Path(args.guarded_root),
        Path(args.output_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
