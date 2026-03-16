#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


CURATED_SOURCE_IDS = [
    "antarctic_vegetation_speclib",
    "branch_tree_spectra_boreal_temperate",
    "bssl",
    "cabo_leaf_v2",
    "drylands_emit",
    "ecostress_v1",
    "emit_adjusted_snow_liquids",
    "emit_adjusted_surface",
    "emit_adjusted_vegetation",
    "emit_l2a_surface",
    "ghisacasia_v001",
    "hsdos",
    "hyspiri_ground_targets",
    "montesinho_plants",
    "natural_snow_twigs",
    "neon_field_spectra",
    "ngee_arctic_2018",
    "ngee_arctic_leaf_reflectance_barrow_2013",
    "ngee_arctic_leaf_reflectance_kougarok_2016",
    "ngee_arctic_leaf_reflectance_transmittance_barrow_2014_2016",
    "ossl",
    "santa_barbara_urban_reflectance",
    "sispec",
    "slum",
    "understory_estonia_czech",
    "understory_icos_europe",
    "usgs_v7",
    "warm_roof",
]

MANUAL_SOURCE_IDS = {"ghisacasia_v001", "usgs_v7"}
ECOSTRESS_SOURCE_ID = "ecostress_v1"
MANUAL_SOURCE_PATHS = {
    "ghisacasia_v001": "GHISACASIA_Spectroradiometer_2006_001.xlsx",
    "montesinho_plants": "Montesinho Natural Park",
    "usgs_v7": "ASCIIdata_splib07b_cvASD",
}


def run_command(command: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    src_root = str(cwd / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing else f"{src_root}{os.pathsep}{existing}"
    if extra_env:
        env.update(extra_env)
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def run_command_allow_failure(
    command: list[str],
    cwd: Path,
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> tuple[str, str]:
    env = os.environ.copy()
    src_root = str(cwd / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing else f"{src_root}{os.pathsep}{existing}"
    if extra_env:
        env.update(extra_env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return "ok", completed.stdout
    except subprocess.TimeoutExpired as exc:
        combined = exc.stdout or ""
        if exc.stderr:
            combined += exc.stderr
        return "timeout", str(combined)
    except subprocess.CalledProcessError as exc:
        return "error", exc.stdout or ""


def load_manifest_lookup(manifest_path: Path) -> dict[str, dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["source_id"]: row for row in rows}


def write_seed_fetch_result(source: dict[str, str], source_dir: Path, note: str) -> None:
    payload = {
        "source_id": source["source_id"],
        "source_name": source["name"],
        "fetch_adapter": source["fetch_adapter"],
        "fetch_mode": "assets",
        "status": "downloaded",
        "landing_url": source["landing_url"],
        "started_at": "",
        "finished_at": "",
        "notes": [note],
        "artifacts": [],
    }
    (source_dir / "fetch-result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_manual_source_tree(source_dir: Path, source: dict[str, str], source_path: Path, note: str) -> None:
    if source_dir.exists():
        shutil.rmtree(source_dir)
    data_dir = source_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / source_path.name
    if source_path.is_dir():
        shutil.copytree(source_path, target)
    else:
        shutil.copy2(source_path, target)
    write_seed_fetch_result(source, source_dir, note)


def seed_source_from_manual_root(
    source_id: str,
    raw_sources_root: Path,
    manifest_lookup: dict[str, dict[str, str]],
    manual_root: Path,
) -> Path | None:
    relative_path = MANUAL_SOURCE_PATHS.get(source_id)
    if not relative_path:
        return None
    manual_source = manual_root / relative_path
    if not manual_source.exists():
        return None
    copy_manual_source_tree(
        raw_sources_root / source_id,
        manifest_lookup[source_id],
        manual_source,
        f"Seeded from manual source path {manual_source}.",
    )
    return manual_source


def install_manual_sources(raw_sources_root: Path, manifest_lookup: dict[str, dict[str, str]], manual_root: Path) -> dict[str, str]:
    installed: dict[str, str] = {}
    for source_id in sorted(MANUAL_SOURCE_IDS):
        manual_source = seed_source_from_manual_root(source_id, raw_sources_root, manifest_lookup, manual_root)
        if manual_source is not None:
            installed[source_id] = str(manual_source)
    return installed


def read_fetch_statuses(raw_sources_root: Path, source_ids: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for source_id in source_ids:
        fetch_result_path = raw_sources_root / source_id / "fetch-result.json"
        if not fetch_result_path.exists():
            statuses[source_id] = "missing"
            continue
        payload = json.loads(fetch_result_path.read_text(encoding="utf-8"))
        statuses[source_id] = str(payload.get("status", "missing"))
    return statuses


def source_has_downloaded_fetch_result(raw_sources_root: Path, source_id: str) -> bool:
    fetch_result_path = raw_sources_root / source_id / "fetch-result.json"
    if not fetch_result_path.exists():
        return False
    payload = json.loads(fetch_result_path.read_text(encoding="utf-8"))
    return str(payload.get("status")) == "downloaded"


def copy_existing_source(source_id: str, raw_sources_root: Path, fallback_roots: list[Path]) -> Path | None:
    target_dir = raw_sources_root / source_id
    for fallback_root in fallback_roots:
        candidate = fallback_root / source_id
        if not candidate.exists() or not candidate.is_dir():
            continue
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(candidate, target_dir)
        return candidate
    return None


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a fresh raw-to-SIAC rebuild using public downloads plus manual/auth-only seed data."
    )
    parser.add_argument("--manifest", default="manifests/sources.csv")
    parser.add_argument("--manual-root", default="build/mannual_download_data")
    parser.add_argument(
        "--fallback-raw-roots",
        default="build/local_sources_full_raw,build/local_sources_vegetation_all,build/local_sources",
    )
    parser.add_argument("--raw-sources-root", default="build/local_sources_full_raw")
    parser.add_argument("--pipeline-root", default="build/real_siac_pipeline_full_raw")
    parser.add_argument(
        "--output-root",
        default="build/siac_spectral_library_real_full_raw_no_ghisacasia_no_understory_no_santa37",
    )
    parser.add_argument("--exclude-source-ids", default="ghisacasia_v001,understory_estonia_czech")
    parser.add_argument("--exclude-spectra-csv", default="manifests/siac_excluded_spectra.csv")
    parser.add_argument("--max-per-source", type=int, default=100)
    parser.add_argument("--max-per-landcover", type=int, default=100)
    parser.add_argument("--top-n-sources", type=int, default=18)
    parser.add_argument("--min-start-nm", type=float, default=410.0)
    parser.add_argument("--min-end-nm", type=float, default=2400.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    manual_root = (repo_root / args.manual_root).resolve()
    fallback_roots = [
        (repo_root / value).resolve()
        for value in str(args.fallback_raw_roots).split(",")
        if value.strip()
    ]
    raw_sources_root = (repo_root / args.raw_sources_root).resolve()
    pipeline_root = (repo_root / args.pipeline_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    exclude_spectra_csv = (repo_root / args.exclude_spectra_csv).resolve() if args.exclude_spectra_csv else None

    manifest_lookup = load_manifest_lookup(manifest_path)
    missing_manifest_sources = [source_id for source_id in CURATED_SOURCE_IDS if source_id not in manifest_lookup]
    if missing_manifest_sources:
        raise SystemExit(f"Missing sources in manifest: {', '.join(missing_manifest_sources)}")

    if raw_sources_root.exists():
        shutil.rmtree(raw_sources_root)
    raw_sources_root.mkdir(parents=True, exist_ok=True)
    if pipeline_root.exists():
        shutil.rmtree(pipeline_root)
    pipeline_root.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        shutil.rmtree(output_root)

    public_source_ids = [source_id for source_id in CURATED_SOURCE_IDS if source_id not in MANUAL_SOURCE_IDS]
    fetch_logs_dir = pipeline_root / "00_fetch_logs"
    fetch_logs_dir.mkdir(parents=True, exist_ok=True)
    fetch_rows: list[dict[str, str]] = []
    for source_id in public_source_ids:
        source_dir = raw_sources_root / source_id
        if source_dir.exists():
            shutil.rmtree(source_dir)

        cached_source = copy_existing_source(source_id, raw_sources_root, fallback_roots)
        if cached_source is not None:
            (fetch_logs_dir / f"{source_id}.log").write_text(
                f"Skipped network fetch; seeded from cached source tree {cached_source}.\n",
                encoding="utf-8",
            )
            fetch_rows.append(
                {
                    "source_id": source_id,
                    "command_status": "skipped_cached",
                    "final_status": "downloaded" if source_has_downloaded_fetch_result(raw_sources_root, source_id) else "missing",
                    "fallback_source": str(cached_source),
                }
            )
            continue

        manual_source = seed_source_from_manual_root(source_id, raw_sources_root, manifest_lookup, manual_root)
        if manual_source is not None:
            (fetch_logs_dir / f"{source_id}.log").write_text(
                f"Skipped network fetch; seeded from manual source path {manual_source}.\n",
                encoding="utf-8",
            )
            fetch_rows.append(
                {
                    "source_id": source_id,
                    "command_status": "skipped_manual_bundle",
                    "final_status": "downloaded",
                    "fallback_source": str(manual_source),
                }
            )
            continue

        extra_env = {"ECOSTRESS_DOWNLOAD_WORKERS": "16"} if source_id == ECOSTRESS_SOURCE_ID else None
        timeout = 1800 if source_id == ECOSTRESS_SOURCE_ID else 600
        status, output = run_command_allow_failure(
            [
                sys.executable,
                "-m",
                "spectral_library.cli",
                "internal",
                "fetch-source",
                "--manifest",
                str(manifest_path),
                "--source-id",
                source_id,
                "--fetch-mode",
                "assets",
                "--output-root",
                str(raw_sources_root),
            ],
            cwd=repo_root,
            extra_env=extra_env,
            timeout=timeout,
        )
        (fetch_logs_dir / f"{source_id}.log").write_text(output, encoding="utf-8", errors="replace")

        fallback_source = None
        if not source_has_downloaded_fetch_result(raw_sources_root, source_id):
            fallback_source = copy_existing_source(source_id, raw_sources_root, fallback_roots)
        if fallback_source is None and not source_has_downloaded_fetch_result(raw_sources_root, source_id):
            manual_source = seed_source_from_manual_root(source_id, raw_sources_root, manifest_lookup, manual_root)
            if manual_source is not None:
                fallback_source = manual_source

        fetch_rows.append(
            {
                "source_id": source_id,
                "command_status": status,
                "final_status": "downloaded" if source_has_downloaded_fetch_result(raw_sources_root, source_id) else "missing",
                "fallback_source": str(fallback_source) if fallback_source else "",
            }
        )

    installed_manual = install_manual_sources(raw_sources_root, manifest_lookup, manual_root)

    normalized_root = pipeline_root / "01_normalized_raw"
    run_command(
        [
            sys.executable,
            "-m",
            "spectral_library.cli",
            "internal",
            "normalize-sources",
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(raw_sources_root),
            "--output-root",
            str(normalized_root),
            "--source-ids",
            ",".join(CURATED_SOURCE_IDS),
        ],
        cwd=repo_root,
    )

    cov80_root = pipeline_root / "02_cov80"
    run_command(
        [
            sys.executable,
            "-m",
            "spectral_library.cli",
            "internal",
            "filter-coverage",
            "--normalized-root",
            str(normalized_root),
            "--output-root",
            str(cov80_root),
            "--min-coverage",
            "0.8",
        ],
        cwd=repo_root,
    )

    emit_santa_root = pipeline_root / "03_emit_santa_fixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "repair_emit_santa_artifacts.py"),
            "--base-root",
            str(cov80_root),
            "--output-root",
            str(emit_santa_root),
        ],
        cwd=repo_root,
    )

    ghisacasia_root = pipeline_root / "04_ghisacasia_fixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "repair_ghisacasia_artifacts.py"),
            "--base-root",
            str(emit_santa_root),
            "--output-root",
            str(ghisacasia_root),
        ],
        cwd=repo_root,
    )

    visible_root = pipeline_root / "05_visible_fixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "robust_visible_outlier_fix.py"),
            "--base-root",
            str(ghisacasia_root),
            "--output-root",
            str(visible_root),
        ],
        cwd=repo_root,
    )

    absorption_root = pipeline_root / "06_absorption_smoothed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "curate_source_absorption_rules.py"),
            "--base-root",
            str(visible_root),
            "--output-root",
            str(absorption_root),
        ],
        cwd=repo_root,
    )

    start_tag = str(int(args.min_start_nm))
    end_tag = str(int(args.min_end_nm))
    native_root = pipeline_root / f"07_native_{start_tag}_{end_tag}"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "filter_by_native_range.py"),
            "--base-root",
            str(absorption_root),
            "--output-root",
            str(native_root),
            "--min-start-nm",
            str(args.min_start_nm),
            "--min-end-nm",
            str(args.min_end_nm),
        ],
        cwd=repo_root,
    )

    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "landcover_analysis.py"),
            "--manifest",
            str(manifest_path),
            "--normalized-root",
            str(native_root),
        ],
        cwd=repo_root,
    )

    vegetation_outlier_root = pipeline_root / "08_vegetation_outliers_fixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "repair_vegetation_outliers.py"),
            "--base-root",
            str(native_root),
            "--output-root",
            str(vegetation_outlier_root),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "landcover_analysis.py"),
            "--manifest",
            str(manifest_path),
            "--normalized-root",
            str(vegetation_outlier_root),
        ],
        cwd=repo_root,
    )

    vegetation_water_root = pipeline_root / "09_vegetation_waterfixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "repair_vegetation_water_band_spikes.py"),
            "--base-root",
            str(vegetation_outlier_root),
            "--output-root",
            str(vegetation_water_root),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "landcover_analysis.py"),
            "--manifest",
            str(manifest_path),
            "--normalized-root",
            str(vegetation_water_root),
        ],
        cwd=repo_root,
    )

    subsetfiltered_root = pipeline_root / "10_subsetfiltered"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "filter_curated_subset_rules.py"),
            "--base-root",
            str(vegetation_water_root),
            "--output-root",
            str(subsetfiltered_root),
        ],
        cwd=repo_root,
    )

    final_normalized_root = pipeline_root / "11_source_artifacts_fixed"
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "repair_remaining_source_artifacts.py"),
            "--base-root",
            str(subsetfiltered_root),
            "--output-root",
            str(final_normalized_root),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "landcover_analysis.py"),
            "--manifest",
            str(manifest_path),
            "--normalized-root",
            str(final_normalized_root),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            "-m",
            "spectral_library.cli",
            "internal",
            "plot-quality",
            "--normalized-root",
            str(final_normalized_root),
            "--output-root",
            str(final_normalized_root / "plots" / "quality"),
        ],
        cwd=repo_root,
    )

    run_command(
        [
            sys.executable,
            "-m",
            "spectral_library.cli",
            "internal",
            "build-siac-library",
            "--manifest",
            str(manifest_path),
            "--normalized-root",
            str(final_normalized_root),
            "--output-root",
            str(output_root),
            "--exclude-source-ids",
            args.exclude_source_ids,
            "--exclude-spectra-csv",
            str(exclude_spectra_csv) if exclude_spectra_csv else "",
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_siac_visualizations.py"),
            "--root",
            str(output_root),
            "--output-root",
            str(output_root / "plots"),
            "--max-per-source",
            str(args.max_per_source),
            "--max-per-landcover",
            str(args.max_per_landcover),
            "--top-n-sources",
            str(args.top_n_sources),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "full_processed_spectra_review.py"),
            "--base-root",
            str(output_root),
            "--output-root",
            str(output_root / "full_review"),
        ],
        cwd=repo_root,
    )

    fetch_statuses = read_fetch_statuses(raw_sources_root, CURATED_SOURCE_IDS)
    summary = {
        "manifest_path": str(manifest_path),
        "manual_root": str(manual_root),
        "raw_sources_root": str(raw_sources_root),
        "pipeline_root": str(pipeline_root),
        "final_normalized_root": str(final_normalized_root),
        "output_root": str(output_root),
        "exclude_spectra_csv": str(exclude_spectra_csv) if exclude_spectra_csv else "",
        "min_start_nm": args.min_start_nm,
        "min_end_nm": args.min_end_nm,
        "curated_source_ids": CURATED_SOURCE_IDS,
        "public_source_ids": public_source_ids,
        "manual_source_ids": sorted(MANUAL_SOURCE_IDS),
        "fallback_roots": [str(path) for path in fallback_roots],
        "fetch_rows": fetch_rows,
        "installed_manual_sources": installed_manual,
        "fetch_status_counts": dict(Counter(fetch_statuses.values())),
        "fetch_statuses": fetch_statuses,
        "stages": {
            "normalized_root": str(normalized_root),
            "cov80_root": str(cov80_root),
            "emit_santa_root": str(emit_santa_root),
            "ghisacasia_root": str(ghisacasia_root),
            "visible_root": str(visible_root),
            "absorption_root": str(absorption_root),
            "native_root": str(native_root),
            "vegetation_outlier_root": str(vegetation_outlier_root),
            "vegetation_water_root": str(vegetation_water_root),
            "subsetfiltered_root": str(subsetfiltered_root),
            "final_normalized_root": str(final_normalized_root),
        },
        "subsetfiltered_summary": read_json_if_exists(subsetfiltered_root / "curated_subset_filter_summary.json"),
        "final_normalized_summary": read_json_if_exists(final_normalized_root / "repair_summary.json"),
        "final_package_summary": read_json_if_exists(output_root / "build_summary.json"),
        "full_review_summary": read_json_if_exists(output_root / "full_review" / "review_summary.json"),
    }
    (pipeline_root / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
