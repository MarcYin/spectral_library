#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_spectra(root: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    spectra_csv = root / "tabular" / "normalized_spectra.csv"
    frame = pd.read_csv(spectra_csv, low_memory=False)
    spectral_columns = [column for column in frame.columns if column.startswith("nm_")]
    wavelengths = np.asarray([int(column.split("_", 1)[1]) for column in spectral_columns], dtype=int)
    return frame, wavelengths, spectral_columns


def select_examples(repairs_csv: Path, repair_type: str, top_n: int, per_source: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(repairs_csv)
    frame = frame[frame["repair_type"].eq(repair_type)].copy()
    if per_source > 0:
        parts = []
        for _, source_frame in frame.sort_values("max_abs_delta", ascending=False).groupby("source_id", sort=False):
            parts.append(source_frame.head(per_source))
        frame = pd.concat(parts, ignore_index=True) if parts else frame.head(0)
        return frame.sort_values("max_abs_delta", ascending=False).head(top_n)
    return frame.sort_values("max_abs_delta", ascending=False).head(top_n)


def plot_examples(
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
    examples: pd.DataFrame,
    wavelengths: np.ndarray,
    spectral_columns: list[str],
    output_path: Path,
    xlim: tuple[int, int],
) -> None:
    rows = len(examples)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.8 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    before_index = before_frame.set_index(["source_id", "spectrum_id"])
    after_index = after_frame.set_index(["source_id", "spectrum_id"])

    for axis, (_, example) in zip(axes, examples.iterrows()):
        key = (example["source_id"], example["spectrum_id"])
        before = before_index.loc[key, spectral_columns].to_numpy(dtype=float)
        after = after_index.loc[key, spectral_columns].to_numpy(dtype=float)
        changed = np.abs(after - before) > 0

        axis.plot(wavelengths, before, color="#9c6b30", linewidth=1.1, label="Before")
        axis.plot(wavelengths, after, color="#006d77", linewidth=1.3, label="After")
        if changed.any():
            change_x = wavelengths[changed]
            axis.axvspan(int(change_x.min()), int(change_x.max()), color="#d62828", alpha=0.08)
        axis.set_xlim(*xlim)
        axis.grid(alpha=0.2, linewidth=0.4)
        axis.set_ylabel("Reflectance")
        axis.set_title(
            f'{example["source_id"]} | {example["spectrum_id"]} | {example["repair_type"]} | max delta {example["max_abs_delta"]:.3f}',
            fontsize=9,
        )

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot before/after examples for repaired spectra.")
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    parser.add_argument("--repairs-csv", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    before_root = Path(args.before_root)
    after_root = Path(args.after_root)
    repairs_csv = Path(args.repairs_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    before_frame, wavelengths, spectral_columns = load_spectra(before_root)
    after_frame, _, _ = load_spectra(after_root)

    emit_examples = select_examples(
        repairs_csv,
        "emit_second_absorption_shoulder_interp",
        top_n=6,
        per_source=3,
    )
    santa_examples = select_examples(
        repairs_csv,
        "santa_tail_linear_extrapolation",
        top_n=6,
        per_source=6,
    )

    if not emit_examples.empty:
        emit_examples.to_csv(output_root / "emit_shoulder_examples.csv", index=False)
        plot_examples(
            before_frame,
            after_frame,
            emit_examples,
            wavelengths,
            spectral_columns,
            output_root / "emit_shoulder_examples_zoom.png",
            (1760, 1960),
        )
        plot_examples(
            before_frame,
            after_frame,
            emit_examples,
            wavelengths,
            spectral_columns,
            output_root / "emit_shoulder_examples_full.png",
            (400, 2500),
        )

    if not santa_examples.empty:
        santa_examples.to_csv(output_root / "santa_tail_examples.csv", index=False)
        plot_examples(
            before_frame,
            after_frame,
            santa_examples,
            wavelengths,
            spectral_columns,
            output_root / "santa_tail_examples_zoom.png",
            (2325, 2500),
        )
        plot_examples(
            before_frame,
            after_frame,
            santa_examples,
            wavelengths,
            spectral_columns,
            output_root / "santa_tail_examples_full.png",
            (400, 2500),
        )

    summary = {
        "before_root": str(before_root),
        "after_root": str(after_root),
        "repairs_csv": str(repairs_csv),
        "emit_example_count": int(len(emit_examples)),
        "santa_example_count": int(len(santa_examples)),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
