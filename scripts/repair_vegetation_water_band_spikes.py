#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOW_RULES = [
    {
        "name": "water_absorption_1400",
        "left_start": 1320,
        "left_end": 1360,
        "replace_start": 1380,
        "replace_end": 1425,
        "right_start": 1455,
        "right_end": 1495,
        "dev_threshold": 0.12,
        "jump_threshold": 0.04,
        "blend_half_window_nm": 50,
    },
    {
        "name": "water_absorption_1900",
        "left_start": 1760,
        "left_end": 1800,
        "replace_start": 1875,
        "replace_end": 1935,
        "right_start": 1995,
        "right_end": 2035,
        "dev_threshold": 0.15,
        "jump_threshold": 0.05,
        "blend_half_window_nm": 50,
    },
]

POINT_SPIKE_SOURCE_IDS = {"understory_estonia_czech"}
POINT_SPIKE_RULES = [
    {
        "name": "point_spike_1400",
        "replace_start": 1368,
        "replace_end": 1425,
        "pad_nm": 8,
        "deviation_threshold": 0.08,
        "jump_threshold": 0.05,
        "max_iterations": 120,
    },
    {
        "name": "point_spike_1900",
        "replace_start": 1800,
        "replace_end": 1935,
        "pad_nm": 8,
        "deviation_threshold": 0.08,
        "jump_threshold": 0.05,
        "max_iterations": 160,
    },
]


def copy_support_files(base_root: Path, output_root: Path) -> None:
    for relative in [
        Path("tabular/spectra_metadata.csv"),
        Path("tabular/source_summary.csv"),
        Path("tabular/wavelength_grid.csv"),
        Path("tabular/normalization_failures.csv"),
        Path("manifests/sources.csv"),
    ]:
        source_path = base_root / relative
        if not source_path.exists():
            continue
        target_path = output_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def to_float(value: str) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text)


def load_vegetation_keys(labels_path: Path) -> set[str]:
    labels = pd.read_csv(labels_path, low_memory=False)
    vegetation = labels[labels["landcover_group"] == "vegetation"].copy()
    return set((vegetation["source_id"].astype(str) + "\t" + vegetation["spectrum_id"].astype(str)).tolist())


def blend_weights(wavelengths: np.ndarray, start_nm: int, end_nm: int, half_window_nm: int) -> np.ndarray:
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


def detect_rule(values: np.ndarray, wavelengths: np.ndarray, rule: dict[str, object]) -> tuple[bool, float, float, np.ndarray]:
    left_mask = (wavelengths >= int(rule["left_start"])) & (wavelengths <= int(rule["left_end"]))
    center_mask = (wavelengths >= int(rule["replace_start"])) & (wavelengths <= int(rule["replace_end"]))
    right_mask = (wavelengths >= int(rule["right_start"])) & (wavelengths <= int(rule["right_end"]))

    left_values = values[left_mask]
    center_values = values[center_mask]
    right_values = values[right_mask]
    if np.isfinite(left_values).sum() < 2 or np.isfinite(center_values).sum() < 2 or np.isfinite(right_values).sum() < 2:
        return False, float("nan"), float("nan"), np.full(center_values.shape, np.nan, dtype=float)

    y0 = float(np.nanmean(left_values))
    y1 = float(np.nanmean(right_values))
    predicted = np.interp(
        wavelengths[center_mask].astype(float),
        [float(np.mean(wavelengths[left_mask])), float(np.mean(wavelengths[right_mask]))],
        [y0, y1],
    )
    deviation = center_values - predicted
    max_dev = float(np.nanmax(deviation)) if np.isfinite(deviation).any() else float("nan")
    max_jump = float(np.nanmax(np.abs(np.diff(center_values)))) if np.isfinite(center_values).sum() >= 2 else float("nan")
    flagged = bool(
        np.isfinite(max_dev)
        and np.isfinite(max_jump)
        and max_dev > float(rule["dev_threshold"])
        and max_jump > float(rule["jump_threshold"])
    )
    return flagged, max_dev, max_jump, predicted


def repair_rule(values: np.ndarray, wavelengths: np.ndarray, rule: dict[str, object], predicted_center: np.ndarray) -> np.ndarray:
    repaired = values.copy()
    replace_start = int(rule["replace_start"])
    replace_end = int(rule["replace_end"])
    half_window = int(rule["blend_half_window_nm"])
    blend_mask = (wavelengths >= replace_start - half_window) & (wavelengths <= replace_end + half_window)
    center_mask = (wavelengths >= replace_start) & (wavelengths <= replace_end)
    predicted_all = np.interp(
        wavelengths[blend_mask].astype(float),
        [float(np.mean(wavelengths[(wavelengths >= int(rule["left_start"])) & (wavelengths <= int(rule["left_end"]))])),
         float(np.mean(wavelengths[(wavelengths >= int(rule["right_start"])) & (wavelengths <= int(rule["right_end"]))]))],
        [float(np.nanmean(values[(wavelengths >= int(rule["left_start"])) & (wavelengths <= int(rule["left_end"]))])),
         float(np.nanmean(values[(wavelengths >= int(rule["right_start"])) & (wavelengths <= int(rule["right_end"]))]))],
    )
    weights = blend_weights(wavelengths[blend_mask], replace_start, replace_end, half_window)
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_all
    repaired[center_mask] = predicted_center
    return repaired


def repair_point_spikes(
    values: np.ndarray,
    wavelengths: np.ndarray,
    rule: dict[str, object],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = values.copy()
    diagnostics: list[dict[str, float]] = []
    replace_indices = np.flatnonzero(
        (wavelengths >= int(rule["replace_start"])) & (wavelengths <= int(rule["replace_end"]))
    )
    pad_nm = int(rule["pad_nm"])
    max_iterations = int(rule["max_iterations"])
    for _ in range(max_iterations):
        best_index = -1
        best_interp = float("nan")
        best_deviation = 0.0
        best_jump = 0.0
        for band_index in replace_indices:
            left_index = band_index - pad_nm
            right_index = band_index + pad_nm
            if left_index < 0 or right_index >= len(repaired):
                continue
            y_left = repaired[left_index]
            y_center = repaired[band_index]
            y_right = repaired[right_index]
            if not (np.isfinite(y_left) and np.isfinite(y_center) and np.isfinite(y_right)):
                continue
            x_left = float(wavelengths[left_index])
            x_right = float(wavelengths[right_index])
            x_center = float(wavelengths[band_index])
            interp = y_left + (y_right - y_left) * (x_center - x_left) / (x_right - x_left)
            deviation = abs(y_center - interp)
            left_jump = abs(y_center - repaired[band_index - 1]) if band_index > 0 and np.isfinite(repaired[band_index - 1]) else 0.0
            right_jump = abs(repaired[band_index + 1] - y_center) if band_index + 1 < len(repaired) and np.isfinite(repaired[band_index + 1]) else 0.0
            jump = max(left_jump, right_jump)
            if deviation > float(rule["deviation_threshold"]) and jump > float(rule["jump_threshold"]) and deviation > best_deviation:
                best_index = int(band_index)
                best_interp = float(interp)
                best_deviation = float(deviation)
                best_jump = float(jump)
        if best_index < 0:
            break
        diagnostics.append(
            {
                "band_nm": float(wavelengths[best_index]),
                "deviation": best_deviation,
                "jump": best_jump,
            }
        )
        repaired[best_index] = best_interp
    return repaired, diagnostics


def plot_examples(example_rows: list[dict[str, object]], output_path: Path) -> None:
    if not example_rows:
        return
    figure, axes = plt.subplots(len(example_rows), 2, figsize=(12, max(4, 3.1 * len(example_rows))), sharex=False)
    axes = np.atleast_2d(axes)
    wavelengths = np.arange(400, 2501, dtype=int)
    for axis_row, example in zip(axes, example_rows):
        before = np.asarray(example["before"], dtype=float)
        after = np.asarray(example["after"], dtype=float)
        source_id = str(example["source_id"])
        spectrum_id = str(example["spectrum_id"])
        axis_row[0].plot(wavelengths, before, color="#b2182b", linewidth=1.0, label="before")
        axis_row[0].plot(wavelengths, after, color="#2166ac", linewidth=1.0, label="after")
        axis_row[0].set_title(f"{source_id} | {spectrum_id}")
        axis_row[0].grid(alpha=0.2)

        zoom_mask = ((wavelengths >= 1280) & (wavelengths <= 1510)) | ((wavelengths >= 1740) & (wavelengths <= 2045))
        axis_row[1].plot(wavelengths[zoom_mask], before[zoom_mask], color="#b2182b", linewidth=1.0, label="before")
        axis_row[1].plot(wavelengths[zoom_mask], after[zoom_mask], color="#2166ac", linewidth=1.0, label="after")
        axis_row[1].set_title("deep water windows")
        axis_row[1].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Reflectance")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair vegetation water-band spikes using outside interpolation detection.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--labels-path", default="")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    labels_path = Path(args.labels_path) if args.labels_path else base_root / "landcover_analysis" / "landcover_labels.csv"
    vegetation_keys = load_vegetation_keys(labels_path)

    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "tabular").mkdir(parents=True, exist_ok=True)
    copy_support_files(base_root, output_root)
    if labels_path.exists():
        target_labels = output_root / "landcover_analysis" / "landcover_labels.csv"
        target_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_path, target_labels)

    input_csv = base_root / "tabular" / "normalized_spectra.csv"
    output_csv = output_root / "tabular" / "normalized_spectra.csv"
    wavelengths = np.arange(400, 2501, dtype=int)
    example_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    with input_csv.open("r", newline="", encoding="utf-8", errors="replace") as input_handle:
        reader = csv.DictReader(input_handle)
        assert reader.fieldnames is not None
        with output_csv.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = f"{row['source_id']}\t{row['spectrum_id']}"
                values = np.asarray([to_float(row[f"nm_{wavelength}"]) for wavelength in wavelengths], dtype=float)
                if key in vegetation_keys:
                    before = values.copy()
                    changed = False
                    for rule in WINDOW_RULES:
                        flagged, max_dev, max_jump, predicted_center = detect_rule(values, wavelengths, rule)
                        if not flagged:
                            continue
                        repaired = repair_rule(values, wavelengths, rule, predicted_center)
                        diagnostics_rows.append(
                            {
                                "source_id": row["source_id"],
                                "spectrum_id": row["spectrum_id"],
                                "sample_name": row["sample_name"],
                                "window_name": str(rule["name"]),
                                "max_positive_deviation": max_dev,
                                "max_adjacent_jump": max_jump,
                            }
                        )
                        values = repaired
                        changed = True
                    if row["source_id"] in POINT_SPIKE_SOURCE_IDS:
                        for rule in POINT_SPIKE_RULES:
                            repaired, point_diags = repair_point_spikes(values, wavelengths, rule)
                            if not point_diags:
                                continue
                            for point_diag in point_diags:
                                diagnostics_rows.append(
                                    {
                                        "source_id": row["source_id"],
                                        "spectrum_id": row["spectrum_id"],
                                        "sample_name": row["sample_name"],
                                        "window_name": str(rule["name"]),
                                        "max_positive_deviation": float(point_diag["deviation"]),
                                        "max_adjacent_jump": float(point_diag["jump"]),
                                        "band_nm": int(point_diag["band_nm"]),
                                    }
                                )
                            values = repaired
                            changed = True
                    if changed:
                        for wavelength in wavelengths:
                            row[f"nm_{wavelength}"] = f"{values[wavelength - 400]:.12g}"
                        if len(example_rows) < 8:
                            example_rows.append(
                                {
                                    "source_id": row["source_id"],
                                    "spectrum_id": row["spectrum_id"],
                                    "before": before,
                                    "after": values.copy(),
                                }
                            )
                writer.writerow(row)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    diagnostics_frame.to_csv(diagnostics_dir / "vegetation_water_band_repairs.csv", index=False)
    plot_examples(example_rows, output_root / "plots" / "vegetation_water_band_repairs.png")

    by_source = (
        diagnostics_frame.groupby("source_id", as_index=False)
        .agg(
            repaired_windows=("window_name", "size"),
            repaired_spectra=("spectrum_id", pd.Series.nunique),
        )
        .sort_values(["repaired_spectra", "repaired_windows", "source_id"], ascending=[False, False, True])
        if not diagnostics_frame.empty
        else pd.DataFrame(columns=["source_id", "repaired_windows", "repaired_spectra"])
    )
    by_source.to_csv(diagnostics_dir / "vegetation_water_band_repairs_by_source.csv", index=False)

    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "labels_path": str(labels_path),
        "repaired_window_count": int(len(diagnostics_rows)),
        "repaired_spectra_count": int(diagnostics_frame["spectrum_id"].nunique()) if not diagnostics_frame.empty else 0,
        "diagnostics_csv": str(diagnostics_dir / "vegetation_water_band_repairs.csv"),
        "diagnostics_by_source_csv": str(diagnostics_dir / "vegetation_water_band_repairs_by_source.csv"),
        "example_plot": str(output_root / "plots" / "vegetation_water_band_repairs.png"),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
