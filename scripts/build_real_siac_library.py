#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical end-to-end SIAC build with cache fallback for existing raw source trees."
    )
    parser.add_argument("--manifest", default="manifests/sources.csv")
    parser.add_argument("--manual-root", default="build/mannual_download_data")
    parser.add_argument(
        "--fallback-raw-roots",
        default="build/local_sources_full_raw,build/local_sources_vegetation_all,build/local_sources",
    )
    parser.add_argument("--raw-sources-root", default="build/local_sources_full_raw_cached")
    parser.add_argument("--pipeline-root", default="build/real_siac_pipeline_full_raw_cached")
    parser.add_argument(
        "--output-root",
        default="build/siac_spectral_library_real_full_raw_cached_no_ghisacasia_no_understory_no_santa37",
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
    command = [
        sys.executable,
        str(repo_root / "scripts" / "build_real_siac_library_from_scratch.py"),
        "--manifest",
        args.manifest,
        "--manual-root",
        args.manual_root,
        "--fallback-raw-roots",
        args.fallback_raw_roots,
        "--raw-sources-root",
        args.raw_sources_root,
        "--pipeline-root",
        args.pipeline_root,
        "--output-root",
        args.output_root,
        "--exclude-source-ids",
        args.exclude_source_ids,
        "--exclude-spectra-csv",
        args.exclude_spectra_csv,
        "--max-per-source",
        str(args.max_per_source),
        "--max-per-landcover",
        str(args.max_per_landcover),
        "--top-n-sources",
        str(args.top_n_sources),
        "--min-start-nm",
        str(args.min_start_nm),
        "--min-end-nm",
        str(args.min_end_nm),
    ]

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root if not existing else f"{src_root}{os.pathsep}{existing}"
    subprocess.run(command, cwd=str(repo_root), check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
