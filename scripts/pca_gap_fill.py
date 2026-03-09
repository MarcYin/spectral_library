#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LANDCOVER_GROUPS = ["soil", "vegetation", "water", "urban"]
WAVELENGTHS = np.arange(400, 2501, dtype=int)
BAND_COLUMNS = [f"nm_{wavelength}" for wavelength in WAVELENGTHS]
DEEP_REPAIR_BANDS = {
    "water_abs_1400": (1330, 1455),
    "water_abs_1900": (1770, 1985),
}
BLUE_REPAIR_NAME = "blue_edge_400_500"
BLUE_MODEL_RANGE = (400, 900)
BLUE_TARGET_RANGE = (400, 500)
BLUE_PREDICTOR_RANGE = (501, 900)
TRAIN_NORMALIZED_POINTS_MIN = 2090
TRAIN_NATIVE_MIN_MAX = (400.0, 2490.0)
TRAIN_BLUE_NORMALIZED_POINTS_MIN = 480
TRAIN_BLUE_NATIVE_MIN_MAX = (400.0, 900.0)
TRAIN_SAMPLE_MAX = 5000
MAX_COMPONENTS = 12
MIN_COMPONENTS = 3
VARIANCE_TARGET = 0.995
RIDGE_LAMBDA = 1e-6
RANDOM_SEED = 42
CHUNK_SIZE = 512


@dataclass
class PcaModel:
    group: str
    wavelengths: np.ndarray
    global_idx: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    training_min: np.ndarray
    training_max: np.ndarray
    noise_thresholds: dict[str, float]
    training_rows: int
    candidate_rows: int
    excluded_noisy_rows: int
    predictor_idx: np.ndarray | None = None
    target_idx: np.ndarray | None = None


def resolve_tabular_root(normalized_root: Path) -> Path:
    candidate = normalized_root / "tabular"
    if candidate.exists():
        return candidate
    return normalized_root


def range_indices(range_nm: tuple[int, int], wavelengths: np.ndarray) -> np.ndarray:
    start_nm, end_nm = range_nm
    mask = (wavelengths >= start_nm) & (wavelengths <= end_nm)
    return np.flatnonzero(mask)


def band_indices_for_wavelengths(
    wavelengths: np.ndarray,
    repair_bands: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    for name, repair_range in repair_bands.items():
        bands[name] = range_indices(repair_range, wavelengths)
    return bands


def load_group_metadata(tabular_root: Path, labels_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        tabular_root / "spectra_metadata.csv",
        usecols=[
            "source_id",
            "spectrum_id",
            "sample_name",
            "native_min_nm",
            "native_max_nm",
            "normalized_points",
        ],
        low_memory=False,
    )
    labels = pd.read_csv(
        labels_path,
        usecols=["source_id", "spectrum_id", "landcover_group"],
        low_memory=False,
    )
    frame = metadata.merge(labels, on=["source_id", "spectrum_id"], how="left")
    frame = frame[frame["landcover_group"].isin(LANDCOVER_GROUPS)].copy()
    return frame


def select_training_keys(
    group_metadata: pd.DataFrame,
    normalized_points_min: int,
    native_min_max: tuple[float, float],
) -> tuple[dict[str, str], dict[str, int]]:
    candidates = group_metadata[
        (group_metadata["normalized_points"] >= normalized_points_min)
        & (group_metadata["native_min_nm"] <= native_min_max[0])
        & (group_metadata["native_max_nm"] >= native_min_max[1])
    ].copy()

    rng = np.random.default_rng(RANDOM_SEED)
    selected_map: dict[str, str] = {}
    counts: dict[str, int] = {}
    for group in LANDCOVER_GROUPS:
        group_rows = candidates[candidates["landcover_group"] == group]
        counts[group] = int(len(group_rows))
        if group_rows.empty:
            continue
        if len(group_rows) > TRAIN_SAMPLE_MAX:
            choice = rng.choice(group_rows.index.to_numpy(), size=TRAIN_SAMPLE_MAX, replace=False)
            group_rows = group_rows.loc[np.sort(choice)]
        for source_id, spectrum_id in group_rows[["source_id", "spectrum_id"]].itertuples(index=False, name=None):
            selected_map[f"{source_id}\t{spectrum_id}"] = group
    return selected_map, counts


def compute_band_noise_metric(values: np.ndarray, band_idx: np.ndarray) -> float:
    band_values = values[band_idx]
    if not np.isfinite(band_values).all():
        return float("inf")
    if len(band_values) < 5:
        return 0.0
    second_diff = np.diff(band_values, n=2)
    if second_diff.size == 0:
        return 0.0
    return float(np.median(np.abs(second_diff)))


def load_training_spectra(
    normalized_spectra_path: Path,
    selected_map: dict[str, str],
    spectral_columns: list[str],
) -> dict[str, np.ndarray]:
    training_rows: dict[str, list[np.ndarray]] = {group: [] for group in LANDCOVER_GROUPS}
    usecols = ["source_id", "spectrum_id"] + spectral_columns
    for chunk in pd.read_csv(normalized_spectra_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        keys = chunk["source_id"] + "\t" + chunk["spectrum_id"]
        groups = keys.map(selected_map)
        chunk = chunk.assign(landcover_group=groups)
        chunk = chunk[chunk["landcover_group"].notna()]
        if chunk.empty:
            continue
        for group, group_frame in chunk.groupby("landcover_group"):
            matrix = group_frame[spectral_columns].to_numpy(dtype=float)
            training_rows[group].append(matrix)
    result: dict[str, np.ndarray] = {}
    for group in LANDCOVER_GROUPS:
        if training_rows[group]:
            result[group] = np.vstack(training_rows[group])
        else:
            result[group] = np.empty((0, len(spectral_columns)), dtype=float)
    return result


def fit_models(
    training_spectra: dict[str, np.ndarray],
    candidate_counts: dict[str, int],
    wavelengths: np.ndarray,
    global_idx: np.ndarray,
    repair_bands: dict[str, tuple[int, int]],
    predictor_range: tuple[int, int] | None = None,
    target_range: tuple[int, int] | None = None,
) -> dict[str, PcaModel]:
    band_idx_map = band_indices_for_wavelengths(wavelengths, repair_bands)
    predictor_idx = range_indices(predictor_range, wavelengths) if predictor_range is not None else None
    target_idx = range_indices(target_range, wavelengths) if target_range is not None else None
    models: dict[str, PcaModel] = {}
    for group, matrix in training_spectra.items():
        if matrix.size == 0:
            continue

        finite_mask = np.isfinite(matrix).all(axis=1)
        matrix = matrix[finite_mask]
        if len(matrix) == 0:
            continue

        band_metrics: dict[str, np.ndarray] = {}
        for band_name, band_idx in band_idx_map.items():
            band_metrics[band_name] = np.array([compute_band_noise_metric(row, band_idx) for row in matrix], dtype=float)

        thresholds: dict[str, float] = {}
        keep_mask = np.ones(len(matrix), dtype=bool)
        for band_name, metrics in band_metrics.items():
            finite_metrics = metrics[np.isfinite(metrics)]
            if len(finite_metrics) == 0:
                thresholds[band_name] = float("inf")
                keep_mask &= np.isfinite(metrics)
                continue
            median = float(np.median(finite_metrics))
            mad = float(np.median(np.abs(finite_metrics - median)) * 1.4826)
            percentile = float(np.quantile(finite_metrics, 0.99))
            threshold = max(percentile, median + 6.0 * mad, 1e-6)
            thresholds[band_name] = threshold
            keep_mask &= metrics <= threshold

        clean_matrix = matrix[keep_mask]
        if len(clean_matrix) < 8:
            clean_matrix = matrix

        centered = clean_matrix - clean_matrix.mean(axis=0)
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        explained = singular_values**2
        explained_ratio = explained / explained.sum() if explained.sum() > 0 else np.zeros_like(explained)
        cumulative = np.cumsum(explained_ratio)
        component_count = int(np.searchsorted(cumulative, VARIANCE_TARGET) + 1) if cumulative.size else 1
        component_count = max(MIN_COMPONENTS, min(MAX_COMPONENTS, component_count, len(clean_matrix) - 1, len(BAND_COLUMNS)))
        component_count = max(1, component_count)
        components = vt[:component_count].T

        models[group] = PcaModel(
            group=group,
            wavelengths=wavelengths,
            global_idx=global_idx,
            mean=clean_matrix.mean(axis=0),
            components=components,
            explained_variance_ratio=explained_ratio[:component_count],
            training_min=np.nanmin(clean_matrix, axis=0),
            training_max=np.nanmax(clean_matrix, axis=0),
            noise_thresholds=thresholds,
            training_rows=int(len(clean_matrix)),
            candidate_rows=int(candidate_counts.get(group, 0)),
            excluded_noisy_rows=int(len(matrix) - len(clean_matrix)),
            predictor_idx=predictor_idx,
            target_idx=target_idx,
        )
    return models


def save_models(models: dict[str, PcaModel], models_dir: Path, summary_name: str, file_prefix: str = "") -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, model in models.items():
        np.savez_compressed(
            models_dir / f"{file_prefix}{group}_pca_model.npz",
            wavelengths=model.wavelengths,
            global_idx=model.global_idx,
            mean=model.mean,
            components=model.components,
            explained_variance_ratio=model.explained_variance_ratio,
            training_min=model.training_min,
            training_max=model.training_max,
        )
        row = {
            "landcover_group": group,
            "candidate_rows": model.candidate_rows,
            "training_rows": model.training_rows,
            "excluded_noisy_rows": model.excluded_noisy_rows,
            "components": model.components.shape[1],
            "explained_variance_ratio_sum": float(model.explained_variance_ratio.sum()),
        }
        for band_name, threshold in model.noise_thresholds.items():
            row[f"{band_name}_threshold"] = threshold
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(models_dir / summary_name, index=False)
    return summary


def reconstruct_with_pca(
    values: np.ndarray,
    model: PcaModel,
    replace_mask: np.ndarray,
    observed_local_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    local_values = values[model.global_idx]
    local_replace_mask = replace_mask[model.global_idx]
    if observed_local_idx is None:
        observed_mask = np.isfinite(local_values) & ~local_replace_mask
    else:
        observed_mask = np.zeros(len(local_values), dtype=bool)
        finite_predictors = np.isfinite(local_values[observed_local_idx]) & ~local_replace_mask[observed_local_idx]
        observed_mask[observed_local_idx] = finite_predictors
    if observed_mask.sum() < max(8, model.components.shape[1]):
        return values, False

    design = model.components[observed_mask, :]
    target = local_values[observed_mask] - model.mean[observed_mask]
    lhs = design.T @ design + np.eye(model.components.shape[1]) * RIDGE_LAMBDA
    rhs = design.T @ target
    coefficients = np.linalg.solve(lhs, rhs)
    predicted = model.mean + model.components @ coefficients
    clipped = np.clip(predicted, model.training_min - 0.05, model.training_max + 0.05)

    repaired = values.copy()
    local_repaired = local_values.copy()
    local_repaired[local_replace_mask] = clipped[local_replace_mask]
    repaired[model.global_idx] = local_repaired
    return repaired, True


def copy_static_tables(tabular_root: Path, output_tabular: Path) -> None:
    output_tabular.mkdir(parents=True, exist_ok=True)
    for name in ["wavelength_grid.csv", "source_summary.csv", "spectra_metadata.csv", "normalization_failures.csv"]:
        source_path = tabular_root / name
        if source_path.exists():
            shutil.copy2(source_path, output_tabular / name)


def postprocess_spectra(
    normalized_spectra_path: Path,
    output_path: Path,
    labels_map: dict[str, str],
    full_models: dict[str, PcaModel],
    blue_models: dict[str, PcaModel],
) -> tuple[pd.DataFrame, dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_rows: list[dict[str, object]] = []
    group_totals = {
        "processed_spectra": 0,
        "filled_points": 0,
        "filled_gap_points": 0,
        "filled_noisy_blue_edge_points": 0,
        "filled_noisy_deep_band_points": 0,
        "skipped_unclassified": 0,
        "skipped_no_model": 0,
    }

    deep_band_idx_map = band_indices_for_wavelengths(WAVELENGTHS, DEEP_REPAIR_BANDS)
    blue_target_global_idx = range_indices(BLUE_TARGET_RANGE, WAVELENGTHS)
    header_written = False
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        for chunk in pd.read_csv(normalized_spectra_path, chunksize=CHUNK_SIZE, low_memory=False):
            spectral_values = chunk[BAND_COLUMNS].to_numpy(dtype=float)
            processed_rows = []
            for row_offset, row in enumerate(chunk.itertuples(index=False)):
                source_id = row.source_id
                spectrum_id = row.spectrum_id
                sample_name = row.sample_name
                key = f"{source_id}\t{spectrum_id}"
                group = labels_map.get(key, "")

                values = spectral_values[row_offset].copy()
                if group not in LANDCOVER_GROUPS:
                    group_totals["skipped_unclassified"] += 1
                    repaired = values
                    filled_gap_points = 0
                    filled_noisy_blue_points = 0
                    filled_noisy_deep_points = 0
                    noisy_bands: list[str] = []
                    model_used = ""
                    spectrum_processed = False
                elif group not in full_models and group not in blue_models:
                    group_totals["skipped_no_model"] += 1
                    repaired = values
                    filled_gap_points = 0
                    filled_noisy_blue_points = 0
                    filled_noisy_deep_points = 0
                    noisy_bands = []
                    model_used = ""
                    spectrum_processed = False
                else:
                    repaired = values.copy()
                    gap_mask = ~np.isfinite(values)
                    filled_gap_points = 0
                    filled_noisy_blue_points = 0
                    filled_noisy_deep_points = 0
                    noisy_bands = []
                    model_tags: list[str] = []
                    spectrum_processed = False

                    full_replace_mask = gap_mask.copy()
                    full_replace_mask[blue_target_global_idx] = False
                    deep_noisy_bands: list[str] = []
                    for band_name, band_idx in deep_band_idx_map.items():
                        band_values = values[band_idx]
                        if not np.isfinite(band_values).all():
                            full_replace_mask[band_idx] = True
                            continue
                        metric = compute_band_noise_metric(values, band_idx)
                        model = full_models.get(group)
                        if model and metric > model.noise_thresholds.get(band_name, float("inf")):
                            full_replace_mask[band_idx] = True
                            deep_noisy_bands.append(band_name)

                    full_model = full_models.get(group)
                    if full_model and full_replace_mask.any():
                        repaired_full, repaired_ok = reconstruct_with_pca(repaired, full_model, full_replace_mask)
                        if repaired_ok:
                            repaired = repaired_full
                            full_gap_points = int((gap_mask & full_replace_mask).sum())
                            full_deep_points = int(full_replace_mask.sum()) - full_gap_points
                            filled_gap_points += full_gap_points
                            filled_noisy_deep_points += max(0, full_deep_points)
                            noisy_bands.extend(deep_noisy_bands)
                            model_tags.append(f"{group}_full")
                            spectrum_processed = True

                    blue_model = blue_models.get(group)
                    blue_replace_mask = np.zeros(len(values), dtype=bool)
                    blue_replace_mask[blue_target_global_idx] = gap_mask[blue_target_global_idx]
                    blue_metric = compute_band_noise_metric(values, blue_target_global_idx)
                    blue_is_noisy = (
                        blue_model is not None
                        and np.isfinite(blue_metric)
                        and blue_metric > blue_model.noise_thresholds.get(BLUE_REPAIR_NAME, float("inf"))
                    )
                    if blue_is_noisy:
                        blue_replace_mask[blue_target_global_idx] = True

                    if blue_model and blue_replace_mask.any():
                        repaired_blue, repaired_ok = reconstruct_with_pca(
                            repaired,
                            blue_model,
                            blue_replace_mask,
                            observed_local_idx=blue_model.predictor_idx,
                        )
                        if repaired_ok:
                            repaired = repaired_blue
                            blue_gap_points = int((gap_mask & blue_replace_mask).sum())
                            blue_noisy_points = int(blue_replace_mask.sum()) - blue_gap_points
                            filled_gap_points += blue_gap_points
                            filled_noisy_blue_points += max(0, blue_noisy_points)
                            if blue_is_noisy:
                                noisy_bands.append(BLUE_REPAIR_NAME)
                            model_tags.append(f"{group}_blue")
                            spectrum_processed = True

                    if spectrum_processed:
                        group_totals["processed_spectra"] += 1
                        group_totals["filled_points"] += filled_gap_points + filled_noisy_blue_points + filled_noisy_deep_points
                        group_totals["filled_gap_points"] += filled_gap_points
                        group_totals["filled_noisy_blue_edge_points"] += filled_noisy_blue_points
                        group_totals["filled_noisy_deep_band_points"] += filled_noisy_deep_points
                    model_used = "+".join(model_tags)

                processed_row = {
                    "source_id": source_id,
                    "spectrum_id": spectrum_id,
                    "sample_name": sample_name,
                }
                for wavelength, value in zip(WAVELENGTHS, repaired):
                    processed_row[f"nm_{wavelength}"] = value
                processed_rows.append(processed_row)

                stats_rows.append(
                    {
                        "source_id": source_id,
                        "spectrum_id": spectrum_id,
                        "sample_name": sample_name,
                        "landcover_group": group,
                        "model_used": model_used,
                        "filled_gap_points": filled_gap_points,
                        "filled_noisy_blue_edge_points": filled_noisy_blue_points,
                        "filled_noisy_deep_band_points": filled_noisy_deep_points,
                        "filled_points_total": filled_gap_points + filled_noisy_blue_points + filled_noisy_deep_points,
                        "flagged_noisy_bands": ",".join(sorted(set(noisy_bands))),
                    }
                )

            frame = pd.DataFrame(processed_rows)
            frame.to_csv(output_handle, index=False, header=not header_written)
            header_written = True

    stats_frame = pd.DataFrame(stats_rows)
    return stats_frame, group_totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Gap fill normalized spectra with landcover-specific PCA models.")
    parser.add_argument("--normalized-root", required=True)
    parser.add_argument("--labels-path", default="")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    normalized_root = Path(args.normalized_root)
    tabular_root = resolve_tabular_root(normalized_root)
    labels_path = Path(args.labels_path) if args.labels_path else normalized_root / "landcover_analysis" / "landcover_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Landcover labels file is missing: {labels_path}")

    output_root = Path(args.output_root) if args.output_root else normalized_root.with_name(normalized_root.name + "_pca_gapfilled")
    output_tabular = output_root / "tabular"
    output_models = output_root / "models"

    group_metadata = load_group_metadata(tabular_root, labels_path)
    labels_map = {
        f"{source_id}\t{spectrum_id}": group
        for source_id, spectrum_id, group in group_metadata[["source_id", "spectrum_id", "landcover_group"]].itertuples(index=False, name=None)
    }

    full_selected_map, full_candidate_counts = select_training_keys(
        group_metadata,
        TRAIN_NORMALIZED_POINTS_MIN,
        TRAIN_NATIVE_MIN_MAX,
    )
    blue_selected_map, blue_candidate_counts = select_training_keys(
        group_metadata,
        TRAIN_BLUE_NORMALIZED_POINTS_MIN,
        TRAIN_BLUE_NATIVE_MIN_MAX,
    )

    blue_window_idx = range_indices(BLUE_MODEL_RANGE, WAVELENGTHS)
    blue_window_wavelengths = WAVELENGTHS[blue_window_idx]
    blue_window_columns = [BAND_COLUMNS[idx] for idx in blue_window_idx]

    full_training_spectra = load_training_spectra(tabular_root / "normalized_spectra.csv", full_selected_map, BAND_COLUMNS)
    blue_training_spectra = load_training_spectra(tabular_root / "normalized_spectra.csv", blue_selected_map, blue_window_columns)

    full_models = fit_models(
        full_training_spectra,
        full_candidate_counts,
        WAVELENGTHS,
        np.arange(len(WAVELENGTHS), dtype=int),
        DEEP_REPAIR_BANDS,
    )
    blue_models = fit_models(
        blue_training_spectra,
        blue_candidate_counts,
        blue_window_wavelengths,
        blue_window_idx,
        {BLUE_REPAIR_NAME: BLUE_TARGET_RANGE},
        predictor_range=BLUE_PREDICTOR_RANGE,
        target_range=BLUE_TARGET_RANGE,
    )

    full_model_summary = save_models(full_models, output_models, "pca_model_summary.csv")
    blue_model_summary = save_models(blue_models, output_models, "blue_pca_model_summary.csv", file_prefix="blue_")

    copy_static_tables(tabular_root, output_tabular)
    stats_frame, totals = postprocess_spectra(
        tabular_root / "normalized_spectra.csv",
        output_tabular / "normalized_spectra.csv",
        labels_map,
        full_models,
        blue_models,
    )
    stats_frame.to_csv(output_tabular / "postprocess_flags.csv", index=False)

    by_group = (
        stats_frame[stats_frame["landcover_group"].isin(LANDCOVER_GROUPS)]
        .groupby("landcover_group", as_index=False)
        .agg(
            spectra=("spectrum_id", "count"),
            modeled=("model_used", lambda values: int((pd.Series(values) != "").sum())),
            filled_gap_points=("filled_gap_points", "sum"),
            filled_noisy_blue_edge_points=("filled_noisy_blue_edge_points", "sum"),
            filled_noisy_deep_band_points=("filled_noisy_deep_band_points", "sum"),
            filled_points_total=("filled_points_total", "sum"),
        )
    )
    by_group.to_csv(output_root / "postprocess_group_summary.csv", index=False)

    summary = {
        "input_root": str(normalized_root),
        "output_root": str(output_root),
        "labels_path": str(labels_path),
        "models_trained": full_model_summary["landcover_group"].tolist() if not full_model_summary.empty else [],
        "blue_models_trained": blue_model_summary["landcover_group"].tolist() if not blue_model_summary.empty else [],
        "candidate_counts": full_candidate_counts,
        "blue_candidate_counts": blue_candidate_counts,
        "totals": totals,
    }
    summary_path = output_root / "postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
