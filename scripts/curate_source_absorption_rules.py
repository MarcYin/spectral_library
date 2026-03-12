#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from plot_absorption_smoothing_overview import mean_curves, plot_overview, write_deltas
from whitsmooth_rust import robust_whittaker_irls_f64


CHUNK_SIZE = 512
EXCLUDED_SOURCES = {"probefield_aligned", "probefield_preprocessed"}
BLEND_HALF_WINDOW_NM = 50
JUMP_GUARD_ABS_TOL = 5e-4


@dataclass(frozen=True)
class WindowRule:
    name: str
    replace_start: int
    replace_end: int
    fit_start: int
    fit_end: int
    lam: float


RULES_BY_SOURCE: dict[str, list[WindowRule]] = {
    "hyspiri_ground_targets": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500, 120.0),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030, 120.0),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500, 80.0),
    ],
    "ghisacasia_v001": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500, 120.0),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030, 120.0),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500, 80.0),
    ],
    "ngee_arctic_2018": [
        WindowRule("deep_absorption_1400", 1330, 1455, 1280, 1500, 120.0),
        WindowRule("deep_absorption_1900", 1770, 1985, 1720, 2030, 120.0),
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500, 80.0),
    ],
    "santa_barbara_urban_reflectance": [
        WindowRule("water_absorption_2300_plus", 2300, 2500, 2240, 2500, 80.0),
    ],
}


def copy_if_exists(source_path: Path, target_path: Path) -> None:
    if source_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def filter_table(source_path: Path, target_path: Path, excluded: set[str]) -> dict[str, int]:
    kept_rows = 0
    dropped_rows = 0
    header_written = False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(source_path, chunksize=CHUNK_SIZE, low_memory=False):
            mask = ~chunk["source_id"].astype(str).isin(excluded)
            kept = chunk[mask]
            kept_rows += int(mask.sum())
            dropped_rows += int((~mask).sum())
            if not kept.empty:
                kept.to_csv(handle, index=False, header=not header_written)
                header_written = True
    if not header_written:
        pd.read_csv(source_path, nrows=0).to_csv(target_path, index=False)
    return {"kept_rows": kept_rows, "dropped_rows": dropped_rows}


def append_manifest_notes(manifest_path: Path) -> None:
    frame = pd.read_csv(manifest_path, low_memory=False)
    notes = frame["notes"].fillna("").astype(str)
    excluded_note = "Excluded from curated merged database by source curation rule."
    smooth_note = "Source-specific smoothing applied in deep water absorption bands and/or >2300 nm."

    excluded_mask = frame["source_id"].astype(str).isin(EXCLUDED_SOURCES)
    frame.loc[excluded_mask, "notes"] = notes[excluded_mask].apply(
        lambda current: excluded_note if not current else f"{current} {excluded_note}"
    )

    smooth_mask = frame["source_id"].astype(str).isin(RULES_BY_SOURCE)
    notes = frame["notes"].fillna("").astype(str)
    frame.loc[smooth_mask, "notes"] = notes[smooth_mask].apply(
        lambda current: smooth_note if not current else f"{current} {smooth_note}"
    )
    frame.to_csv(manifest_path, index=False)


def blend_weights(
    wavelengths: np.ndarray,
    start_nm: int,
    end_nm: int,
    half_window_nm: int,
) -> np.ndarray:
    weights = np.zeros(len(wavelengths), dtype=float)
    left_outer = start_nm - half_window_nm
    left_inner = start_nm + half_window_nm
    right_inner = end_nm - half_window_nm
    right_outer = end_nm + half_window_nm

    left_ramp = (wavelengths >= left_outer) & (wavelengths < left_inner)
    if left_inner > left_outer:
        weights[left_ramp] = (wavelengths[left_ramp] - left_outer) / float(left_inner - left_outer)

    middle = (wavelengths >= left_inner) & (wavelengths <= right_inner)
    weights[middle] = 1.0

    right_ramp = (wavelengths > right_inner) & (wavelengths <= right_outer)
    if right_outer > right_inner:
        weights[right_ramp] = (right_outer - wavelengths[right_ramp]) / float(right_outer - right_inner)

    return np.clip(weights, 0.0, 1.0)


def max_adjacent_jump(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    filtered = values[finite]
    return float(np.max(np.abs(np.diff(filtered))))


def boundary_jump_max(
    values: np.ndarray,
    wavelengths: np.ndarray,
    start_nm: int,
    end_nm: int,
) -> float:
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


def accept_candidate(
    original_values: np.ndarray,
    candidate_values: np.ndarray,
    wavelengths: np.ndarray,
    start_nm: int,
    end_nm: int,
) -> bool:
    original_jump = max_adjacent_jump(original_values)
    candidate_jump = max_adjacent_jump(candidate_values)
    original_boundary = boundary_jump_max(original_values, wavelengths, start_nm, end_nm)
    candidate_boundary = boundary_jump_max(candidate_values, wavelengths, start_nm, end_nm)
    return (
        candidate_jump <= original_jump + JUMP_GUARD_ABS_TOL
        and candidate_boundary <= original_boundary + JUMP_GUARD_ABS_TOL
    )


def apply_rules(
    values: np.ndarray,
    source_ids: np.ndarray,
    spectrum_ids: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], Counter[str], Counter[str]]:
    diagnostics: list[dict[str, object]] = []
    replaced_by_source: Counter[str] = Counter()
    spectra_by_source: Counter[str] = Counter()

    for source_id, rules in RULES_BY_SOURCE.items():
        row_indices = np.flatnonzero(source_ids == source_id)
        if len(row_indices) == 0:
            continue

        source_values = values[row_indices].astype(np.float64, copy=True)
        source_replaced_any = np.zeros(len(row_indices), dtype=bool)
        for rule in rules:
            fit_mask = (wavelengths >= rule.fit_start) & (wavelengths <= rule.fit_end)
            replace_mask = (wavelengths >= rule.replace_start) & (wavelengths <= rule.replace_end)
            blend_start = max(rule.fit_start, rule.replace_start - BLEND_HALF_WINDOW_NM)
            blend_end = min(rule.fit_end, rule.replace_end + BLEND_HALF_WINDOW_NM)
            blend_mask = (wavelengths >= blend_start) & (wavelengths <= blend_end)
            if not fit_mask.any() or not replace_mask.any() or not blend_mask.any():
                continue

            fit_values = source_values[:, fit_mask]
            x = wavelengths[fit_mask].astype(np.float64)
            smoothed = robust_whittaker_irls_f64(
                x,
                fit_values,
                lam=rule.lam,
                d=2,
                iterations=8,
                weighting="tukey",
                scale="mad",
                parallel=True,
                return_weights=False,
                merge_x_tol=0.0,
            )

            blend_in_fit = blend_mask[fit_mask]
            before = fit_values[:, blend_in_fit]
            after = smoothed[:, blend_in_fit]
            replaceable = np.isfinite(after)
            if not replaceable.any():
                continue

            original_values = source_values[:, blend_mask].copy()
            blend_wavelengths = wavelengths[blend_mask]
            weights = blend_weights(
                blend_wavelengths,
                rule.replace_start,
                rule.replace_end,
                BLEND_HALF_WINDOW_NM,
            )[None, :]
            blended = (1.0 - weights) * original_values + weights * after
            replaced_counts = np.zeros(len(row_indices), dtype=int)

            for local_idx in range(len(row_indices)):
                row_original = original_values[local_idx]
                row_replaceable = replaceable[local_idx] & np.isfinite(row_original)
                replaced_count = int(row_replaceable.sum())
                if int(replaced_count) == 0:
                    continue
                row_candidate = row_original.copy()
                row_candidate[row_replaceable] = blended[local_idx, row_replaceable]
                if not accept_candidate(
                    row_original,
                    row_candidate,
                    blend_wavelengths,
                    rule.replace_start,
                    rule.replace_end,
                ):
                    continue
                source_values[local_idx, blend_mask] = row_candidate
                delta = np.abs(row_candidate[row_replaceable] - row_original[row_replaceable])
                replaced_counts[local_idx] = replaced_count
                source_replaced_any[local_idx] = True
                replaced_by_source[source_id] += replaced_count
                diagnostics.append(
                    {
                        "source_id": source_id,
                        "spectrum_id": spectrum_ids[row_indices[local_idx]],
                        "window_name": rule.name,
                        "replace_start_nm": int(rule.replace_start),
                        "replace_end_nm": int(rule.replace_end),
                        "replaced_band_count": int(replaced_count),
                        "mean_abs_delta": float(np.nanmean(delta)),
                        "max_abs_delta": float(np.nanmax(delta)),
                    }
                )

        values[row_indices] = source_values
        spectra_by_source[source_id] += int(source_replaced_any.sum())

    return values, diagnostics, replaced_by_source, spectra_by_source


def run_curation(base_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copy_if_exists(base_root / "tabular" / "wavelength_grid.csv", output_root / "tabular" / "wavelength_grid.csv")
    copy_if_exists(base_root / "tabular" / "normalization_failures.csv", output_root / "tabular" / "normalization_failures.csv")

    table_summaries: dict[str, dict[str, int]] = {}
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("landcover_analysis/landcover_labels.csv"),
    ]:
        source_path = base_root / relative
        if source_path.exists():
            table_summaries[str(relative)] = filter_table(source_path, output_root / relative, EXCLUDED_SOURCES)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(input_csv, nrows=0)
    spectral_columns = [column for column in header.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)

    kept_rows = 0
    dropped_rows = 0
    header_written = False
    diagnostics_rows: list[dict[str, object]] = []
    replaced_by_source: Counter[str] = Counter()
    spectra_by_source: Counter[str] = Counter()

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False):
            mask = ~chunk["source_id"].astype(str).isin(EXCLUDED_SOURCES)
            kept = chunk[mask].copy()
            kept_rows += int(mask.sum())
            dropped_rows += int((~mask).sum())
            if kept.empty:
                continue

            values = kept[spectral_columns].to_numpy(dtype=float)
            source_ids = kept["source_id"].astype(str).to_numpy()
            spectrum_ids = kept["spectrum_id"].astype(str).to_numpy()
            values, chunk_diags, chunk_replaced, chunk_spectra = apply_rules(values, source_ids, spectrum_ids, wavelengths)
            kept.loc[:, spectral_columns] = values
            kept.to_csv(handle, index=False, header=not header_written)
            header_written = True

            diagnostics_rows.extend(chunk_diags)
            replaced_by_source.update(chunk_replaced)
            spectra_by_source.update(chunk_spectra)

    table_summaries["tabular/normalized_spectra.csv"] = {"kept_rows": kept_rows, "dropped_rows": dropped_rows}

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    if diagnostics_frame.empty:
        diagnostics_frame = pd.DataFrame(
            columns=[
                "source_id",
                "spectrum_id",
                "window_name",
                "replace_start_nm",
                "replace_end_nm",
                "replaced_band_count",
                "mean_abs_delta",
                "max_abs_delta",
            ]
        )
    diagnostics_frame.to_csv(diagnostics_dir / "source_specific_smoothing.csv", index=False)

    summary_frame = pd.DataFrame(
        [
            {
                "source_id": source_id,
                "spectra_smoothed": int(spectra_by_source.get(source_id, 0)),
                "bands_replaced": int(replaced_by_source.get(source_id, 0)),
            }
            for source_id in sorted(set(replaced_by_source) | set(spectra_by_source))
        ]
    )
    summary_frame.to_csv(diagnostics_dir / "source_specific_smoothing_summary.csv", index=False)

    manifest_source = base_root / "manifests" / "sources.csv"
    if manifest_source.exists():
        manifest_target = output_root / "manifests" / "sources.csv"
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_source, manifest_target)
        append_manifest_notes(manifest_target)

    plots_dir = output_root / "plots"
    wavelengths, before = mean_curves(base_root)
    _, after = mean_curves(output_root)
    overview_path = plots_dir / "absorption_smoothing_overview.png"
    plot_overview(wavelengths, before, after, overview_path)
    delta_path = plots_dir / "absorption_smoothing_window_deltas.csv"
    deltas = write_deltas(wavelengths, before, after, delta_path)
    summary_path = plots_dir / "absorption_smoothing_summary.json"
    plot_summary = {
        "base_root": str(base_root),
        "fixed_root": str(output_root),
        "output_root": str(plots_dir),
        "overview_plot": str(overview_path),
        "delta_table": str(delta_path),
        "window_count": int(len(deltas)),
    }
    summary_path.write_text(json.dumps(plot_summary, indent=2), encoding="utf-8")

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "excluded_sources": sorted(EXCLUDED_SOURCES),
        "smoothed_sources": sorted(RULES_BY_SOURCE),
        "tables": table_summaries,
        "smoothing": {
            "source_counts": {
                source_id: {
                    "spectra_smoothed": int(spectra_by_source.get(source_id, 0)),
                    "bands_replaced": int(replaced_by_source.get(source_id, 0)),
                }
                for source_id in sorted(set(replaced_by_source) | set(spectra_by_source))
            },
            "diagnostic_rows": int(len(diagnostics_frame)),
        },
        "plots": plot_summary,
    }
    (output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply source exclusions and targeted absorption smoothing rules.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_curation(Path(args.base_root), Path(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
