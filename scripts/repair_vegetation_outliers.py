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


WAVELENGTHS = np.arange(400, 2501, dtype=int)
UNDERSTORY_REFLECTANCE_BOUNDS = (-0.05, 1.2)
UNDERSTORY_WINDOW_RULES = [
    {
        "name": "understory_absorption_1400_interp",
        "left_start": 1320,
        "left_end": 1350,
        "replace_start": 1355,
        "replace_end": 1435,
        "right_start": 1455,
        "right_end": 1495,
        "blend_half_window_nm": 40,
        "deviation_threshold": 0.08,
        "jump_threshold": 0.05,
    },
    {
        "name": "understory_absorption_1900_interp",
        "left_start": 1760,
        "left_end": 1800,
        "replace_start": 1820,
        "replace_end": 2025,
        "right_start": 2060,
        "right_end": 2100,
        "blend_half_window_nm": 65,
        "deviation_threshold": 0.10,
        "jump_threshold": 0.08,
    },
]
UNDERSTORY_TAIL_RULE = {
    "name": "understory_tail_linear_extrapolation",
    "fit_start": 2300,
    "fit_end": 2375,
    "blend_start": 2350,
    "blend_end": 2400,
    "replace_start": 2401,
    "replace_end": 2500,
    "jump_threshold": 0.08,
}


def copy_static_inputs(base_root: Path, output_root: Path) -> None:
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


def _window_mask(start_nm: int, end_nm: int) -> np.ndarray:
    return (WAVELENGTHS >= start_nm) & (WAVELENGTHS <= end_nm)


def _blend_weights(wavelengths: np.ndarray, start_nm: int, end_nm: int, half_window_nm: int) -> np.ndarray:
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


def _robust_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmedian(finite))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return None
    coef = np.polyfit(x[finite].astype(float), y[finite].astype(float), deg=1)
    return float(coef[0]), float(coef[1])


def _detect_understory_window(values: np.ndarray, rule: dict[str, object]) -> tuple[bool, dict[str, float], np.ndarray]:
    left_mask = _window_mask(int(rule["left_start"]), int(rule["left_end"]))
    center_mask = _window_mask(int(rule["replace_start"]), int(rule["replace_end"]))
    right_mask = _window_mask(int(rule["right_start"]), int(rule["right_end"]))
    left_values = values[left_mask]
    center_values = values[center_mask]
    right_values = values[right_mask]
    center_wavelengths = WAVELENGTHS[center_mask].astype(float)

    diagnostics = {
        "out_of_range_count": float(
            np.count_nonzero(
                (np.isfinite(center_values))
                & (
                    (center_values < UNDERSTORY_REFLECTANCE_BOUNDS[0])
                    | (center_values > UNDERSTORY_REFLECTANCE_BOUNDS[1])
                )
            )
        ),
        "max_abs_deviation": float("nan"),
        "max_abs_jump": float("nan"),
    }
    if np.isfinite(left_values).sum() < 2 or np.isfinite(center_values).sum() < 2 or np.isfinite(right_values).sum() < 2:
        return False, diagnostics, np.full(center_values.shape, np.nan, dtype=float)

    y0 = _robust_mean(left_values)
    y1 = _robust_mean(right_values)
    predicted = np.interp(
        center_wavelengths,
        [float(np.mean(WAVELENGTHS[left_mask])), float(np.mean(WAVELENGTHS[right_mask]))],
        [y0, y1],
    )
    deviation = center_values - predicted
    diagnostics["max_abs_deviation"] = float(np.nanmax(np.abs(deviation)))
    diagnostics["max_abs_jump"] = float(np.nanmax(np.abs(np.diff(center_values))))
    should = bool(
        diagnostics["out_of_range_count"] > 0
        or diagnostics["max_abs_deviation"] > float(rule["deviation_threshold"])
        or diagnostics["max_abs_jump"] > float(rule["jump_threshold"])
    )
    return should, diagnostics, predicted


def _repair_understory_window(values: np.ndarray, rule: dict[str, object], predicted_center: np.ndarray) -> np.ndarray:
    repaired = values.copy()
    blend_mask = _window_mask(
        int(rule["replace_start"]) - int(rule["blend_half_window_nm"]),
        int(rule["replace_end"]) + int(rule["blend_half_window_nm"]),
    )
    center_mask = _window_mask(int(rule["replace_start"]), int(rule["replace_end"]))
    left_mask = _window_mask(int(rule["left_start"]), int(rule["left_end"]))
    right_mask = _window_mask(int(rule["right_start"]), int(rule["right_end"]))
    y0 = _robust_mean(values[left_mask])
    y1 = _robust_mean(values[right_mask])
    predicted_blend = np.interp(
        WAVELENGTHS[blend_mask].astype(float),
        [float(np.mean(WAVELENGTHS[left_mask])), float(np.mean(WAVELENGTHS[right_mask]))],
        [y0, y1],
    )
    predicted_blend = np.clip(predicted_blend, UNDERSTORY_REFLECTANCE_BOUNDS[0], UNDERSTORY_REFLECTANCE_BOUNDS[1])
    weights = _blend_weights(
        WAVELENGTHS[blend_mask],
        int(rule["replace_start"]),
        int(rule["replace_end"]),
        int(rule["blend_half_window_nm"]),
    )
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[center_mask] = np.clip(predicted_center, UNDERSTORY_REFLECTANCE_BOUNDS[0], UNDERSTORY_REFLECTANCE_BOUNDS[1])
    return repaired


def _detect_understory_tail(values: np.ndarray) -> tuple[bool, dict[str, float], np.ndarray]:
    fit_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["fit_start"]), int(UNDERSTORY_TAIL_RULE["fit_end"]))
    replace_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["replace_start"]), int(UNDERSTORY_TAIL_RULE["replace_end"]))
    blend_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["blend_start"]), int(UNDERSTORY_TAIL_RULE["blend_end"]))
    fit = _fit_line(WAVELENGTHS[fit_mask].astype(float), values[fit_mask])
    diagnostics = {
        "tail_min": float(np.nanmin(values[replace_mask])),
        "tail_max": float(np.nanmax(values[replace_mask])),
        "tail_max_adjacent_jump": float(np.nanmax(np.abs(np.diff(values[_window_mask(2400, 2500)])))),
        "tail_line_deviation": float("nan"),
    }
    if fit is None:
        return False, diagnostics, np.full(replace_mask.sum(), np.nan, dtype=float)
    slope, intercept = fit
    predicted = slope * WAVELENGTHS[replace_mask].astype(float) + intercept
    predicted = np.clip(predicted, UNDERSTORY_REFLECTANCE_BOUNDS[0], UNDERSTORY_REFLECTANCE_BOUNDS[1])
    actual = values[replace_mask]
    if np.isfinite(actual).sum():
        diagnostics["tail_line_deviation"] = float(np.nanmax(np.abs(actual - predicted)))
    should = bool(
        diagnostics["tail_min"] < UNDERSTORY_REFLECTANCE_BOUNDS[0]
        or diagnostics["tail_max"] > UNDERSTORY_REFLECTANCE_BOUNDS[1]
        or diagnostics["tail_max_adjacent_jump"] > float(UNDERSTORY_TAIL_RULE["jump_threshold"])
        or diagnostics["tail_line_deviation"] > 0.08
    )
    return should, diagnostics, predicted


def _repair_understory_tail(values: np.ndarray, predicted_tail: np.ndarray) -> np.ndarray:
    repaired = values.copy()
    fit_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["fit_start"]), int(UNDERSTORY_TAIL_RULE["fit_end"]))
    blend_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["blend_start"]), int(UNDERSTORY_TAIL_RULE["blend_end"]))
    replace_mask = _window_mask(int(UNDERSTORY_TAIL_RULE["replace_start"]), int(UNDERSTORY_TAIL_RULE["replace_end"]))
    fit = _fit_line(WAVELENGTHS[fit_mask].astype(float), values[fit_mask])
    assert fit is not None
    slope, intercept = fit
    predicted_blend = slope * WAVELENGTHS[blend_mask].astype(float) + intercept
    predicted_blend = np.clip(predicted_blend, UNDERSTORY_REFLECTANCE_BOUNDS[0], UNDERSTORY_REFLECTANCE_BOUNDS[1])
    weights = np.linspace(0.0, 1.0, blend_mask.sum())
    repaired[blend_mask] = (1.0 - weights) * repaired[blend_mask] + weights * predicted_blend
    repaired[replace_mask] = predicted_tail
    return repaired


def plot_examples(example_rows: list[dict[str, object]], output_path: Path) -> None:
    if not example_rows:
        return
    figure, axes = plt.subplots(len(example_rows), 3, figsize=(15, max(4, 3.2 * len(example_rows))), sharex=False)
    axes = np.atleast_2d(axes)
    for axis_row, example in zip(axes, example_rows):
        before = np.asarray(example["before"], dtype=float)
        after = np.asarray(example["after"], dtype=float)
        label = str(example["spectrum_id"])
        axis_row[0].plot(WAVELENGTHS, before, color="#b2182b", linewidth=1.0, label="before")
        axis_row[0].plot(WAVELENGTHS, after, color="#2166ac", linewidth=1.0, label="after")
        axis_row[0].set_title(f"{label} | full spectrum")
        axis_row[0].grid(alpha=0.2)

        deep_mask = ((WAVELENGTHS >= 1300) & (WAVELENGTHS <= 2050))
        axis_row[1].plot(WAVELENGTHS[deep_mask], before[deep_mask], color="#b2182b", linewidth=1.0, label="before")
        axis_row[1].plot(WAVELENGTHS[deep_mask], after[deep_mask], color="#2166ac", linewidth=1.0, label="after")
        axis_row[1].set_title("1400/1900 windows")
        axis_row[1].grid(alpha=0.2)

        tail_mask = WAVELENGTHS >= 2300
        axis_row[2].plot(WAVELENGTHS[tail_mask], before[tail_mask], color="#b2182b", linewidth=1.0, label="before")
        axis_row[2].plot(WAVELENGTHS[tail_mask], after[tail_mask], color="#2166ac", linewidth=1.0, label="after")
        axis_row[2].set_title("2300-2500 nm tail")
        axis_row[2].grid(alpha=0.2)

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
    parser = argparse.ArgumentParser(description="Repair source-specific vegetation outlier artifacts in a normalized dataset.")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "tabular").mkdir(parents=True, exist_ok=True)
    copy_static_inputs(base_root, output_root)

    input_path = base_root / "tabular" / "normalized_spectra.csv"
    output_path = output_root / "tabular" / "normalized_spectra.csv"
    repaired_examples: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []

    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as in_handle:
        reader = csv.DictReader(in_handle)
        assert reader.fieldnames is not None
        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["source_id"] != "understory_estonia_czech":
                    writer.writerow(row)
                    continue

                before = np.asarray([float(row[f"nm_{wavelength}"]) for wavelength in WAVELENGTHS], dtype=float)
                after = before.copy()
                spectrum_repairs: list[dict[str, float | str]] = []

                for rule in UNDERSTORY_WINDOW_RULES:
                    should_repair, diagnostics, predicted_center = _detect_understory_window(after, rule)
                    if not should_repair:
                        continue
                    after = _repair_understory_window(after, rule, predicted_center)
                    spectrum_repairs.append(
                        {
                            "repair_type": str(rule["name"]),
                            **diagnostics,
                        }
                    )

                should_repair_tail, tail_diagnostics, predicted_tail = _detect_understory_tail(after)
                if should_repair_tail:
                    after = _repair_understory_tail(after, predicted_tail)
                    spectrum_repairs.append(
                        {
                            "repair_type": str(UNDERSTORY_TAIL_RULE["name"]),
                            **tail_diagnostics,
                        }
                    )

                if spectrum_repairs:
                    for wavelength in WAVELENGTHS:
                        row[f"nm_{wavelength}"] = f"{after[wavelength - 400]:.12g}"
                    if len(repaired_examples) < 8:
                        repaired_examples.append(
                            {
                                "source_id": row["source_id"],
                                "spectrum_id": row["spectrum_id"],
                                "sample_name": row["sample_name"],
                                "before": before,
                                "after": after,
                            }
                        )
                    for repair in spectrum_repairs:
                        repair_rows.append(
                            {
                                "source_id": row["source_id"],
                                "spectrum_id": row["spectrum_id"],
                                "sample_name": row["sample_name"],
                                **repair,
                            }
                        )

                writer.writerow(row)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    repairs_frame = pd.DataFrame(repair_rows)
    repairs_frame.to_csv(diagnostics_dir / "vegetation_outlier_repairs.csv", index=False)
    example_plot = output_root / "plots" / "understory_estonia_repairs.png"
    plot_examples(repaired_examples, example_plot)

    repaired_spectra = (
        int(repairs_frame["spectrum_id"].nunique())
        if not repairs_frame.empty and "spectrum_id" in repairs_frame
        else 0
    )
    summary = {
        "base_root": str(base_root),
        "output_root": str(output_root),
        "repaired_spectra": repaired_spectra,
        "repair_rows": int(len(repair_rows)),
        "repaired_source_ids": sorted(repairs_frame["source_id"].unique().tolist()) if not repairs_frame.empty else [],
        "diagnostics_csv": str(diagnostics_dir / "vegetation_outlier_repairs.csv"),
        "example_plot": str(example_plot),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
