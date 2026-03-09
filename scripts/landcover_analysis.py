#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LANDCOVER_GROUPS = ["soil", "vegetation", "water", "urban"]
GROUP_COLORS = {
    "soil": "#8c510a",
    "vegetation": "#238b45",
    "water": "#2b8cbe",
    "urban": "#636363",
    "unclassified": "#bdbdbd",
}
PURE_SOURCE_GROUPS = {
    "ossl": "soil",
    "bssl": "soil",
    "hsdos": "soil",
    "probefield_preprocessed": "soil",
    "probefield_aligned": "soil",
    "neospectra_soil_nir": "soil",
    "mediterranean_woodlands": "vegetation",
    "montesinho_plants": "vegetation",
    "ghisaconus_v001": "vegetation",
    "ghisacasia_v001": "vegetation",
    "emit_adjusted_vegetation": "vegetation",
    "german_urban_image_spectra": "urban",
    "brussels_image_urban_materials": "urban",
    "brussels_apex_2015_ground_truth": "urban",
    "hymap_2007_urban_ground_truth": "urban",
    "santa_barbara_urban_reflectance": "urban",
    "existing_urban_reference_data": "urban",
}
USGS_CHAPTER_GROUPS = {
    "ChapterA_ArtificialMaterials": "urban",
    "ChapterC_Coatings": "urban",
    "ChapterS_SoilsAndMixtures": "soil",
    "ChapterV_Vegetation": "vegetation",
    "ChapterL_Liquids": "water",
}
ECOSTRESS_PREFIX_GROUPS = {
    "manmade": "urban",
    "soil": "soil",
    "vegetation": "vegetation",
    "nonphotosyntheticvegetation": "vegetation",
    "water": "water",
}
EMIT_FILE_GROUPS = {
    "filtered_veg": "vegetation",
    "filtered_ocean": "water",
    "surface_Liquids": "water",
}
GROUP_KEYWORDS = {
    "soil": ["soil", "loam", "clay", "sand", "silt", "mud", "peat", "sediment", "dust"],
    "vegetation": [
        "veget",
        "leaf",
        "grass",
        "tree",
        "crop",
        "forest",
        "plant",
        "canopy",
        "pine",
        "oak",
        "needle",
        "shrub",
        "lichen",
        "moss",
        "woodland",
    ],
    "water": ["water", "seawater", "ocean", "river", "lake", "pond", "snow", "ice", "liquid", "brine"],
    "urban": [
        "urban",
        "asphalt",
        "concrete",
        "brick",
        "roof",
        "road",
        "building",
        "metal",
        "paint",
        "glass",
        "cement",
        "fabric",
        "plaster",
        "paving",
        "shingle",
        "tile",
        "tar",
    ],
}
FEATURE_WAVELENGTHS = list(range(400, 2501, 50))
PLOT_WAVELENGTHS = list(range(400, 2501, 10))
FIT_SAMPLE_MAX = 5000
PLOT_SAMPLE_MAX = 1200
SCATTER_SAMPLE_MAX = 2500
RANDOM_SEED = 42


def resolve_tabular_root(normalized_root: Path) -> Path:
    candidate = normalized_root / "tabular"
    if candidate.exists():
        return candidate
    return normalized_root


def load_manifest_map(manifest_path: Path) -> dict[str, dict[str, str]]:
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    return manifest.set_index("source_id").to_dict(orient="index")


def classify_landcover(row: pd.Series, manifest_info: dict[str, str]) -> tuple[str | None, str]:
    source_id = row["source_id"]
    input_path = row["input_path"]
    sample_name = str(row["sample_name"])

    if source_id in PURE_SOURCE_GROUPS:
        return PURE_SOURCE_GROUPS[source_id], f"source:{source_id}"

    if source_id == "usgs_v7":
        suffix = input_path.split("ASCIIdata_splib07b_cvASD/")[-1]
        chapter = suffix.split("/")[0]
        if chapter in USGS_CHAPTER_GROUPS:
            return USGS_CHAPTER_GROUPS[chapter], f"usgs:{chapter}"

    if source_id == "ecostress_v1":
        filename = Path(input_path).name
        prefix = filename.split(".")[0]
        if prefix in ECOSTRESS_PREFIX_GROUPS:
            return ECOSTRESS_PREFIX_GROUPS[prefix], f"ecostress:{prefix}"

    if source_id == "emit_l2a_surface":
        stem = Path(input_path).stem
        if stem in EMIT_FILE_GROUPS:
            return EMIT_FILE_GROUPS[stem], f"emit:{stem}"

    subsection = manifest_info.get("subsection", "").lower()
    source_name = manifest_info.get("name", "").lower()
    spectral_type = manifest_info.get("spectral_type", "").lower()
    if subsection == "soil":
        return "soil", f"subsection:{subsection}"
    if subsection.startswith("vegetation"):
        return "vegetation", f"subsection:{subsection}"
    if subsection == "urban" or "urban" in source_name:
        return "urban", f"subsection_or_name:{subsection or source_name}"

    text = " ".join([source_name, spectral_type, sample_name.lower(), input_path.lower()])
    hits: list[str] = []
    for group, keywords in GROUP_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            hits.append(group)
    if len(hits) == 1:
        return hits[0], "keyword"
    return None, "unclassified"


def build_labels(metadata_path: Path, manifest_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    labels = []
    for row in metadata.itertuples(index=False):
        record = row._asdict()
        group, rule = classify_landcover(pd.Series(record), manifest_map.get(record["source_id"], {}))
        labels.append(
            {
                "source_id": record["source_id"],
                "spectrum_id": record["spectrum_id"],
                "sample_name": record["sample_name"],
                "landcover_group": group if group in LANDCOVER_GROUPS else "",
                "classification_rule": rule,
            }
        )
    return pd.DataFrame(labels)


def read_spectra_columns(normalized_spectra_path: Path, columns: list[str]) -> pd.DataFrame:
    usecols = ["source_id", "spectrum_id", "sample_name"] + columns
    return pd.read_csv(normalized_spectra_path, usecols=usecols, low_memory=False)


def robust_scale_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0) * 1.4826
    std = np.nanstd(values, axis=0)
    scale = np.where(mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0))
    return median, scale


def project_pca(values: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = values.mean(axis=0)
    centered = values - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    projected = centered @ components
    return projected, components, center


def fit_kmeans(values: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_clusters = max(1, min(n_clusters, len(values)))
    centroids = values[rng.choice(len(values), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(30):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            mask = new_labels == cluster_id
            if mask.any():
                new_centroids[cluster_id] = values[mask].mean(axis=0)
            else:
                new_centroids[cluster_id] = values[rng.integers(0, len(values))]
        if np.array_equal(new_labels, labels):
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    return centroids, labels


def assign_cluster_metrics(projected_all: np.ndarray, projected_fit: np.ndarray, fit_labels: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    distances_all = np.linalg.norm(projected_all[:, None, :] - centroids[None, :, :], axis=2)
    labels_all = distances_all.argmin(axis=1)
    chosen_all = distances_all[np.arange(len(distances_all)), labels_all]

    distances_fit = np.linalg.norm(projected_fit[:, None, :] - centroids[None, :, :], axis=2)
    chosen_fit = distances_fit[np.arange(len(distances_fit)), fit_labels]
    thresholds: dict[int, float] = {}
    for cluster_id in range(len(centroids)):
        cluster_distances = chosen_fit[fit_labels == cluster_id]
        if len(cluster_distances) == 0:
            thresholds[cluster_id] = float(np.quantile(chosen_fit, 0.99))
            continue
        median = float(np.median(cluster_distances))
        mad = float(np.median(np.abs(cluster_distances - median)) * 1.4826)
        percentile = float(np.quantile(cluster_distances, 0.99))
        thresholds[cluster_id] = max(percentile, median + 4.0 * mad, 1e-6)
    return labels_all, chosen_all, thresholds


def analyze_group_features(features: pd.DataFrame, group: str) -> pd.DataFrame:
    band_columns = [column for column in features.columns if column.startswith("nm_")]
    values = features[band_columns].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    matrix = values.to_numpy(dtype=float)

    rng = np.random.default_rng(RANDOM_SEED + LANDCOVER_GROUPS.index(group))
    fit_size = min(FIT_SAMPLE_MAX, len(features))
    fit_indices = np.sort(rng.choice(len(features), size=fit_size, replace=False))
    fit_matrix = matrix[fit_indices]

    median, scale = robust_scale_fit(fit_matrix)
    scaled_all = (matrix - median) / scale
    scaled_fit = scaled_all[fit_indices]

    projected_fit, components, center = project_pca(scaled_fit, n_components=min(3, scaled_fit.shape[0], scaled_fit.shape[1]))
    if projected_fit.shape[1] < 3:
        projected_fit = np.pad(projected_fit, ((0, 0), (0, 3 - projected_fit.shape[1])))
        components = np.pad(components, ((0, 0), (0, 3 - components.shape[1])))
    projected_all = (scaled_all - center) @ components

    if group in {"soil", "urban"} and len(features) >= 1000:
        n_clusters = 4
    elif group == "vegetation" and len(features) >= 300:
        n_clusters = 3
    else:
        n_clusters = 2 if len(features) >= 20 else 1

    centroids, fit_labels = fit_kmeans(projected_fit[:, :3], n_clusters, RANDOM_SEED + len(features))
    labels_all, distances_all, thresholds = assign_cluster_metrics(projected_all[:, :3], projected_fit[:, :3], fit_labels, centroids)

    result = features[["source_id", "spectrum_id", "sample_name", "landcover_group"]].copy()
    result["pc1"] = projected_all[:, 0]
    result["pc2"] = projected_all[:, 1]
    result["pc3"] = projected_all[:, 2]
    result["cluster_id"] = labels_all
    result["cluster_distance"] = distances_all
    result["cluster_threshold"] = result["cluster_id"].map(thresholds)
    result["outlier_score"] = result["cluster_distance"] / result["cluster_threshold"]
    result["is_outlier"] = result["cluster_distance"] > result["cluster_threshold"]
    return result


def load_plot_sample(normalized_spectra_path: Path, analyzed: pd.DataFrame) -> pd.DataFrame:
    sample_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for group in LANDCOVER_GROUPS:
        group_rows = analyzed[analyzed["landcover_group"] == group]
        if group_rows.empty:
            continue
        take = min(PLOT_SAMPLE_MAX, len(group_rows))
        if take == len(group_rows):
            sample_rows.append(group_rows)
        else:
            indices = np.sort(rng.choice(group_rows.index.to_numpy(), size=take, replace=False))
            sample_rows.append(group_rows.loc[indices])
    sample_keys = pd.concat(sample_rows, ignore_index=True)
    key_map = sample_keys.set_index(["source_id", "spectrum_id"])[["landcover_group", "cluster_id", "is_outlier", "outlier_score"]]

    usecols = ["source_id", "spectrum_id", "sample_name"] + [f"nm_{wavelength}" for wavelength in PLOT_WAVELENGTHS]
    frames = []
    for chunk in pd.read_csv(normalized_spectra_path, usecols=usecols, chunksize=20000, low_memory=False):
        merged = chunk.merge(key_map, on=["source_id", "spectrum_id"], how="inner")
        if not merged.empty:
            frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def plot_classification_counts(labels: pd.DataFrame, output_path: Path) -> None:
    counts = labels["landcover_group"].replace("", "unclassified").value_counts()
    ordered = counts.reindex(LANDCOVER_GROUPS + ["unclassified"], fill_value=0)
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.bar(ordered.index, ordered.values, color=[GROUP_COLORS[group] for group in ordered.index])
    axis.set_title("Landcover Classification Counts")
    axis.set_ylabel("Spectra")
    _finalize_plot(output_path)


def plot_typical_signatures(plot_sample: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for axis, group in zip(axes.flatten(), LANDCOVER_GROUPS):
        group_frame = plot_sample[plot_sample["landcover_group"] == group]
        if group_frame.empty:
            axis.axis("off")
            axis.set_title(group.title())
            continue
        curve_columns = [f"nm_{wavelength}" for wavelength in PLOT_WAVELENGTHS]
        matrix = group_frame[curve_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        median = np.nanmedian(matrix, axis=0)
        lower = np.nanquantile(matrix, 0.10, axis=0)
        upper = np.nanquantile(matrix, 0.90, axis=0)
        axis.fill_between(PLOT_WAVELENGTHS, lower, upper, color=GROUP_COLORS[group], alpha=0.2)
        axis.plot(PLOT_WAVELENGTHS, median, color=GROUP_COLORS[group], linewidth=2.0)
        axis.set_title(f"{group.title()} median and 10-90% range")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Reflectance")
    _finalize_plot(output_path)


def plot_cluster_scatter(analyzed: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    rng = np.random.default_rng(RANDOM_SEED)
    for axis, group in zip(axes.flatten(), LANDCOVER_GROUPS):
        group_frame = analyzed[analyzed["landcover_group"] == group]
        if group_frame.empty:
            axis.axis("off")
            axis.set_title(group.title())
            continue
        if len(group_frame) > SCATTER_SAMPLE_MAX:
            indices = rng.choice(group_frame.index.to_numpy(), size=SCATTER_SAMPLE_MAX, replace=False)
            plot_frame = group_frame.loc[np.sort(indices)]
        else:
            plot_frame = group_frame
        scatter = axis.scatter(
            plot_frame["pc1"],
            plot_frame["pc2"],
            c=plot_frame["cluster_id"],
            cmap="tab10",
            s=10,
            alpha=0.55,
            linewidths=0,
        )
        flagged = plot_frame[plot_frame["is_outlier"]]
        if not flagged.empty:
            axis.scatter(
                flagged["pc1"],
                flagged["pc2"],
                facecolors="none",
                edgecolors="#cb181d",
                s=36,
                linewidths=0.8,
            )
        axis.set_title(f"{group.title()} PCA clusters")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    _finalize_plot(output_path)


def plot_cluster_signatures(plot_sample: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    curve_columns = [f"nm_{wavelength}" for wavelength in PLOT_WAVELENGTHS]
    for axis, group in zip(axes.flatten(), LANDCOVER_GROUPS):
        group_frame = plot_sample[plot_sample["landcover_group"] == group]
        if group_frame.empty:
            axis.axis("off")
            axis.set_title(group.title())
            continue
        overall = np.nanmedian(group_frame[curve_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float), axis=0)
        axis.plot(PLOT_WAVELENGTHS, overall, color="black", linestyle="--", linewidth=1.5, label="Group median")
        for cluster_id in sorted(group_frame["cluster_id"].unique()):
            cluster_frame = group_frame[group_frame["cluster_id"] == cluster_id]
            if cluster_frame.empty:
                continue
            cluster_matrix = cluster_frame[curve_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid_columns = ~np.isnan(cluster_matrix).all(axis=0)
            if not valid_columns.any():
                continue
            cluster_curve = np.full(cluster_matrix.shape[1], np.nan)
            cluster_curve[valid_columns] = np.nanmedian(cluster_matrix[:, valid_columns], axis=0)
            axis.plot(PLOT_WAVELENGTHS, cluster_curve, linewidth=1.8, label=f"Cluster {cluster_id}")
        axis.set_title(f"{group.title()} cluster medians")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Reflectance")
        axis.legend(fontsize=8)
    _finalize_plot(output_path)


def plot_outlier_rates(group_summary: pd.DataFrame, output_path: Path) -> None:
    summary = group_summary[group_summary["landcover_group"].isin(LANDCOVER_GROUPS)].copy()
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.bar(summary["landcover_group"], summary["outlier_rate"], color=[GROUP_COLORS[group] for group in summary["landcover_group"]])
    axis.set_title("Outlier Rate by Landcover Group")
    axis.set_ylabel("Outlier fraction")
    axis.set_ylim(0, max(0.05, summary["outlier_rate"].max() * 1.15))
    _finalize_plot(output_path)


def _finalize_plot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze normalized spectra by landcover group.")
    parser.add_argument("--manifest", default="manifests/sources.csv")
    parser.add_argument("--normalized-root", required=True)
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    normalized_root = Path(args.normalized_root)
    output_root = Path(args.output_root) if args.output_root else normalized_root / "landcover_analysis"
    tabular_root = resolve_tabular_root(normalized_root)

    manifest_map = load_manifest_map(manifest_path)
    labels = build_labels(tabular_root / "spectra_metadata.csv", manifest_map)
    save_table(labels, output_root / "landcover_labels.csv")

    classified = labels[labels["landcover_group"].isin(LANDCOVER_GROUPS)].copy()

    feature_columns = [f"nm_{wavelength}" for wavelength in FEATURE_WAVELENGTHS]
    spectra = read_spectra_columns(tabular_root / "normalized_spectra.csv", feature_columns)
    features = spectra.merge(classified, on=["source_id", "spectrum_id", "sample_name"], how="inner")
    analyzed_frames = []
    for group in LANDCOVER_GROUPS:
        group_frame = features[features["landcover_group"] == group].copy()
        if group_frame.empty:
            continue
        analyzed_frames.append(analyze_group_features(group_frame, group))
    analyzed = pd.concat(analyzed_frames, ignore_index=True)

    plot_sample = load_plot_sample(tabular_root / "normalized_spectra.csv", analyzed)

    outliers = analyzed[analyzed["is_outlier"]].sort_values("outlier_score", ascending=False).copy()
    save_table(outliers, output_root / "outlier_flags.csv")

    classification_counts = labels["landcover_group"].replace("", "unclassified").value_counts()
    group_summary_rows = []
    for group in LANDCOVER_GROUPS:
        group_frame = analyzed[analyzed["landcover_group"] == group]
        if group_frame.empty:
            group_summary_rows.append(
                {
                    "landcover_group": group,
                    "classified_spectra": 0,
                    "source_count": 0,
                    "cluster_count": 0,
                    "flagged_outliers": 0,
                    "outlier_rate": 0.0,
                }
            )
            continue
        group_summary_rows.append(
            {
                "landcover_group": group,
                "classified_spectra": int(len(group_frame)),
                "source_count": int(group_frame["source_id"].nunique()),
                "cluster_count": int(group_frame["cluster_id"].nunique()),
                "flagged_outliers": int(group_frame["is_outlier"].sum()),
                "outlier_rate": float(group_frame["is_outlier"].mean()),
            }
        )
    group_summary = pd.DataFrame(group_summary_rows)
    save_table(group_summary, output_root / "landcover_group_summary.csv")

    source_summary = (
        analyzed.groupby(["landcover_group", "source_id"], as_index=False)
        .agg(
            spectra=("spectrum_id", "count"),
            flagged_outliers=("is_outlier", "sum"),
            max_outlier_score=("outlier_score", "max"),
        )
        .sort_values(["landcover_group", "flagged_outliers", "spectra"], ascending=[True, False, False])
    )
    source_summary["outlier_rate"] = source_summary["flagged_outliers"] / source_summary["spectra"]
    save_table(source_summary, output_root / "source_outlier_summary.csv")

    plots_dir = output_root / "plots"
    plot_classification_counts(labels, plots_dir / "classification_counts.png")
    plot_typical_signatures(plot_sample, plots_dir / "typical_signatures.png")
    plot_cluster_scatter(analyzed, plots_dir / "cluster_scatter.png")
    plot_cluster_signatures(plot_sample, plots_dir / "cluster_signatures.png")
    plot_outlier_rates(group_summary, plots_dir / "outlier_rates.png")

    summary = {
        "input_root": str(normalized_root),
        "tabular_root": str(tabular_root),
        "output_root": str(output_root),
        "classified_spectra": int(len(classified)),
        "unclassified_spectra": int(classification_counts.get("unclassified", 0)),
        "landcover_counts": {group: int(classification_counts.get(group, 0)) for group in LANDCOVER_GROUPS},
        "flagged_outliers": int(outliers.shape[0]),
        "plots": {
            "classification_counts": str(plots_dir / "classification_counts.png"),
            "typical_signatures": str(plots_dir / "typical_signatures.png"),
            "cluster_scatter": str(plots_dir / "cluster_scatter.png"),
            "cluster_signatures": str(plots_dir / "cluster_signatures.png"),
            "outlier_rates": str(plots_dir / "outlier_rates.png"),
        },
    }
    summary_path = output_root / "landcover_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
