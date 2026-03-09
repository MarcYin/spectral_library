from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_GRID_START_NM = 400
EXPECTED_GRID_END_NM = 2500
EXPECTED_GRID_POINTS = (EXPECTED_GRID_END_NM - EXPECTED_GRID_START_NM) + 1
MAX_SCATTER_POINTS = 20000


def _resolve_tabular_root(normalized_root: Path) -> Path:
    candidate = normalized_root / "tabular"
    if candidate.exists():
        return candidate
    return normalized_root


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required quality input is missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _prepare_plot_backend() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate quality plots") from exc

    plt.style.use("tableau-colorblind10")
    return plt


def _save_figure(plt: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_empty(plt: Any, output_path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=15, fontweight="bold")
    axis.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    _save_figure(plt, output_path)


def _plot_source_counts(plt: Any, source_summary: pd.DataFrame, output_path: Path, top_n_sources: int) -> None:
    counts = source_summary.sort_values("normalized_spectra", ascending=False).head(top_n_sources)
    if counts.empty:
        _plot_empty(plt, output_path, "Normalized Spectra by Source", "No normalized spectra are available.")
        return

    counts = counts.sort_values("normalized_spectra")
    figure, axis = plt.subplots(figsize=(10, max(5.5, 0.35 * len(counts))))
    axis.barh(counts["source_id"], counts["normalized_spectra"], color="#2c7fb8")
    axis.set_title(f"Top {len(counts)} Sources by Normalized Spectra")
    axis.set_xlabel("Normalized spectra")
    axis.set_ylabel("Source")
    _save_figure(plt, output_path)


def _plot_failure_rates(plt: Any, source_summary: pd.DataFrame, output_path: Path, top_n_sources: int) -> None:
    summary = source_summary.copy()
    summary["attempted"] = summary["normalized_spectra"].fillna(0) + summary["failure_count"].fillna(0)
    summary = summary[summary["attempted"] > 0]
    if summary.empty or summary["failure_count"].fillna(0).sum() == 0:
        _plot_empty(plt, output_path, "Normalization Failure Rate by Source", "No normalization failures were recorded.")
        return

    summary["failure_rate"] = summary["failure_count"] / summary["attempted"]
    summary = summary.sort_values(["failure_rate", "failure_count"], ascending=False).head(top_n_sources)
    summary = summary.sort_values("failure_rate")
    figure, axis = plt.subplots(figsize=(10, max(5.5, 0.35 * len(summary))))
    axis.barh(summary["source_id"], summary["failure_rate"], color="#d95f0e")
    axis.set_title(f"Top {len(summary)} Sources by Failure Rate")
    axis.set_xlabel("Failure rate")
    axis.set_ylabel("Source")
    axis.set_xlim(0, 1)
    _save_figure(plt, output_path)


def _plot_native_spacing_histogram(plt: Any, metadata: pd.DataFrame, output_path: Path) -> None:
    spacing = metadata["native_spacing_nm"].dropna()
    spacing = spacing[spacing > 0]
    if spacing.empty:
        _plot_empty(plt, output_path, "Native Spectral Spacing", "No native spacing metadata are available.")
        return

    upper_bound = float(spacing.quantile(0.98))
    clipped = spacing.clip(upper=max(upper_bound, spacing.max()))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(clipped, bins=50, color="#2ca25f", edgecolor="white")
    axis.set_title("Native Spectral Spacing Distribution")
    axis.set_xlabel("Native spacing (nm)")
    axis.set_ylabel("Spectra")
    _save_figure(plt, output_path)


def _plot_native_range_scatter(plt: Any, metadata: pd.DataFrame, output_path: Path) -> None:
    scatter = metadata.dropna(subset=["native_min_nm", "native_max_nm"])
    if scatter.empty:
        _plot_empty(plt, output_path, "Native Spectral Range Coverage", "No native wavelength ranges are available.")
        return

    if len(scatter) > MAX_SCATTER_POINTS:
        scatter = scatter.sample(n=MAX_SCATTER_POINTS, random_state=0)

    figure, axis = plt.subplots(figsize=(8.5, 6))
    hexbin = axis.hexbin(
        scatter["native_min_nm"],
        scatter["native_max_nm"],
        gridsize=40,
        cmap="viridis",
        mincnt=1,
    )
    axis.axvline(EXPECTED_GRID_START_NM, color="black", linestyle="--", linewidth=1)
    axis.axhline(EXPECTED_GRID_END_NM, color="black", linestyle="--", linewidth=1)
    axis.set_title("Native Spectral Range Coverage")
    axis.set_xlabel("Native minimum wavelength (nm)")
    axis.set_ylabel("Native maximum wavelength (nm)")
    colorbar = figure.colorbar(hexbin, ax=axis)
    colorbar.set_label("Spectra per bin")
    _save_figure(plt, output_path)


def _plot_normalized_coverage_histogram(plt: Any, metadata: pd.DataFrame, output_path: Path) -> None:
    coverage = metadata["normalized_points"].dropna() / EXPECTED_GRID_POINTS
    coverage = coverage[(coverage >= 0) & (coverage <= 1.05)]
    if coverage.empty:
        _plot_empty(plt, output_path, "Normalized Grid Coverage", "No normalized coverage metadata are available.")
        return

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(coverage, bins=40, color="#756bb1", edgecolor="white")
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("Normalized Grid Coverage Fraction")
    axis.set_xlabel("Coverage fraction of 400-2500 nm grid")
    axis.set_ylabel("Spectra")
    _save_figure(plt, output_path)


def _plot_failure_reasons(plt: Any, failures: pd.DataFrame, output_path: Path) -> None:
    if failures.empty or "reason" not in failures.columns:
        _plot_empty(plt, output_path, "Failure Reasons", "No failure records are available.")
        return

    counts = failures["reason"].fillna("unspecified").value_counts().head(15)
    counts = counts.sort_values()
    figure, axis = plt.subplots(figsize=(10, max(5.5, 0.34 * len(counts))))
    axis.barh(counts.index, counts.values, color="#cb181d")
    axis.set_title("Top Normalization Failure Reasons")
    axis.set_xlabel("Failure rows")
    axis.set_ylabel("Reason")
    _save_figure(plt, output_path)


def _plot_parser_counts(plt: Any, metadata: pd.DataFrame, output_path: Path) -> None:
    if metadata.empty or "parser" not in metadata.columns:
        _plot_empty(plt, output_path, "Parser Mix", "No parser metadata are available.")
        return

    counts = metadata["parser"].fillna("unknown").value_counts().head(15)
    counts = counts.sort_values()
    figure, axis = plt.subplots(figsize=(10, max(5.5, 0.34 * len(counts))))
    axis.barh(counts.index, counts.values, color="#636363")
    axis.set_title("Top Parsers by Normalized Spectra")
    axis.set_xlabel("Spectra")
    axis.set_ylabel("Parser")
    _save_figure(plt, output_path)


def generate_quality_plots(
    normalized_root: Path,
    output_root: Path | None = None,
    *,
    top_n_sources: int = 20,
) -> dict[str, Any]:
    plt = _prepare_plot_backend()
    tabular_root = _resolve_tabular_root(normalized_root)
    output_dir = output_root or (normalized_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = _read_csv(tabular_root / "source_summary.csv")
    metadata = _read_csv(tabular_root / "spectra_metadata.csv")
    failures = _read_csv(tabular_root / "normalization_failures.csv", required=False)

    source_summary = _coerce_numeric(source_summary, ["normalized_spectra", "failure_count"])
    metadata = _coerce_numeric(
        metadata,
        ["native_wavelength_count", "native_min_nm", "native_max_nm", "native_spacing_nm", "normalized_points"],
    )

    source_summary = source_summary.fillna({"normalized_spectra": 0, "failure_count": 0})
    metadata["coverage_fraction"] = metadata["normalized_points"] / EXPECTED_GRID_POINTS

    plots = {
        "source_counts": output_dir / "source_counts.png",
        "source_failure_rates": output_dir / "source_failure_rates.png",
        "native_spacing_hist": output_dir / "native_spacing_hist.png",
        "native_range_scatter": output_dir / "native_range_scatter.png",
        "normalized_coverage_hist": output_dir / "normalized_coverage_hist.png",
        "failure_reasons": output_dir / "failure_reasons.png",
        "parser_counts": output_dir / "parser_counts.png",
    }

    _plot_source_counts(plt, source_summary, plots["source_counts"], top_n_sources)
    _plot_failure_rates(plt, source_summary, plots["source_failure_rates"], top_n_sources)
    _plot_native_spacing_histogram(plt, metadata, plots["native_spacing_hist"])
    _plot_native_range_scatter(plt, metadata, plots["native_range_scatter"])
    _plot_normalized_coverage_histogram(plt, metadata, plots["normalized_coverage_hist"])
    _plot_failure_reasons(plt, failures, plots["failure_reasons"])
    _plot_parser_counts(plt, metadata, plots["parser_counts"])

    failure_count = int(source_summary["failure_count"].fillna(0).sum())
    coverage_fraction = metadata["coverage_fraction"].dropna()
    metrics = {
        "tabular_root": str(tabular_root),
        "output_root": str(output_dir),
        "source_count": int(metadata["source_id"].nunique()) if "source_id" in metadata.columns else 0,
        "normalized_spectra": int(source_summary["normalized_spectra"].fillna(0).sum()),
        "failure_rows": int(len(failures.index)),
        "failure_count_from_summary": failure_count,
        "coverage_fraction_mean": float(coverage_fraction.mean()) if not coverage_fraction.empty else None,
        "coverage_fraction_p05": float(coverage_fraction.quantile(0.05)) if not coverage_fraction.empty else None,
        "coverage_fraction_p95": float(coverage_fraction.quantile(0.95)) if not coverage_fraction.empty else None,
        "plots": {name: str(path) for name, path in plots.items()},
    }
    metrics_path = output_dir / "quality_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics
