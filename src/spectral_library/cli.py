from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch import fetch_batch, tidy_source_directory
from .build_db import assemble_catalog
from .coverage_filter import filter_normalized_by_coverage
from .fetchers import get_fetcher
from .manifest import filter_sources, load_manifest, manifest_sha256, split_csv_arg
from .normalize import normalize_sources
from .quality_plots import generate_quality_plots
from .siac import build_siac_library


DEFAULT_MANIFEST = Path("manifests/sources.csv")
DEFAULT_USER_AGENT = "spectral-library-db/0.1.0"


def cmd_plan_matrix(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    records = load_manifest(manifest_path)
    selected = filter_sources(
        records,
        source_ids=split_csv_arg(args.source_ids),
        tiers=split_csv_arg(args.tiers),
        statuses=split_csv_arg(args.statuses),
        adapters=split_csv_arg(args.adapters),
    )
    if args.limit > 0:
        selected = selected[: args.limit]

    payload = {
        "include": [record.to_matrix_row() for record in selected],
        "manifest_sha256": manifest_sha256(manifest_path),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_fetch_source(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    records = load_manifest(manifest_path)
    matches = [record for record in records if record.source_id == args.source_id]
    if not matches:
        raise SystemExit(f"Unknown source_id: {args.source_id}")

    source = matches[0]
    output_dir = Path(args.output_root) / source.source_id
    fetcher = get_fetcher(source.fetch_adapter)
    result = fetcher(source, output_dir, args.fetch_mode, args.user_agent)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "fetch-result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"source_id": source.source_id, "status": result.status, "path": str(result_path)}))
    return 0


def cmd_assemble_database(args: argparse.Namespace) -> int:
    summary = assemble_catalog(Path(args.manifest), Path(args.results_root), Path(args.output_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_fetch_batch(args: argparse.Namespace) -> int:
    seed_roots = [Path(value) for value in split_csv_arg(args.seed_roots)]
    summary = fetch_batch(
        Path(args.manifest),
        Path(args.output_root),
        fetch_mode=args.fetch_mode,
        source_ids=split_csv_arg(args.source_ids),
        tiers=split_csv_arg(args.tiers),
        statuses=split_csv_arg(args.statuses),
        adapters=split_csv_arg(args.adapters),
        user_agent=args.user_agent,
        continue_on_error=args.continue_on_error,
        seed_roots=seed_roots,
        clean_output=args.clean_output,
        tidy_downloads=not args.no_tidy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_tidy_results(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root)
    rows = []
    for source_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        rows.append({"source_id": source_dir.name, **tidy_source_directory(source_dir)})
    print(json.dumps({"sources": rows}, indent=2, sort_keys=True))
    return 0


def cmd_normalize_sources(args: argparse.Namespace) -> int:
    summary = normalize_sources(
        Path(args.manifest),
        Path(args.results_root),
        Path(args.output_root),
        source_ids=split_csv_arg(args.source_ids),
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_plot_quality(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root) if args.output_root else None
    summary = generate_quality_plots(
        Path(args.normalized_root),
        output_root,
        top_n_sources=args.top_n_sources,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_filter_coverage(args: argparse.Namespace) -> int:
    summary = filter_normalized_by_coverage(
        Path(args.normalized_root),
        Path(args.output_root),
        min_coverage=args.min_coverage,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_build_siac_library(args: argparse.Namespace) -> int:
    summary = build_siac_library(
        Path(args.manifest),
        Path(args.normalized_root),
        Path(args.output_root),
        exclude_source_ids=split_csv_arg(args.exclude_source_ids),
        exclude_spectra_csv=Path(args.exclude_spectra_csv) if args.exclude_spectra_csv else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectral-library")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan-matrix", help="Create a GitHub Actions matrix from the manifest.")
    plan_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    plan_parser.add_argument("--source-ids", default="")
    plan_parser.add_argument("--tiers", default="")
    plan_parser.add_argument("--statuses", default="")
    plan_parser.add_argument("--adapters", default="")
    plan_parser.add_argument("--limit", type=int, default=0)
    plan_parser.add_argument("--output", default="")
    plan_parser.set_defaults(func=cmd_plan_matrix)

    fetch_parser = subparsers.add_parser("fetch-source", help="Fetch one source from the manifest.")
    fetch_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    fetch_parser.add_argument("--source-id", required=True)
    fetch_parser.add_argument("--output-root", default="build/sources")
    fetch_parser.add_argument("--fetch-mode", choices=["metadata", "assets"], default="metadata")
    fetch_parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    fetch_parser.set_defaults(func=cmd_fetch_source)

    batch_parser = subparsers.add_parser("fetch-batch", help="Fetch multiple sources and tidy their output directories.")
    batch_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    batch_parser.add_argument("--output-root", default="build/local_sources")
    batch_parser.add_argument("--fetch-mode", choices=["metadata", "assets"], default="assets")
    batch_parser.add_argument("--source-ids", default="")
    batch_parser.add_argument("--tiers", default="")
    batch_parser.add_argument("--statuses", default="")
    batch_parser.add_argument("--adapters", default="")
    batch_parser.add_argument("--seed-roots", default="build/live_metadata_sources")
    batch_parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    batch_parser.add_argument("--continue-on-error", action="store_true")
    batch_parser.add_argument("--clean-output", action="store_true")
    batch_parser.add_argument("--no-tidy", action="store_true")
    batch_parser.set_defaults(func=cmd_fetch_batch)

    assemble_parser = subparsers.add_parser("assemble-database", help="Assemble the catalog database from fetch outputs.")
    assemble_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    assemble_parser.add_argument("--results-root", default="build/sources")
    assemble_parser.add_argument("--output-root", default="build/assembled")
    assemble_parser.set_defaults(func=cmd_assemble_database)

    tidy_parser = subparsers.add_parser("tidy-results", help="Reorganize fetched source directories into metadata/docs/data.")
    tidy_parser.add_argument("--results-root", default="build/local_sources")
    tidy_parser.set_defaults(func=cmd_tidy_results)

    normalize_parser = subparsers.add_parser(
        "normalize-sources",
        help="Normalize downloaded spectra onto the shared 400-2500 nm, 1 nm grid.",
    )
    normalize_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    normalize_parser.add_argument("--results-root", default="build/local_sources")
    normalize_parser.add_argument("--output-root", default="build/normalized")
    normalize_parser.add_argument("--source-ids", default="")
    normalize_parser.add_argument("--limit", type=int, default=0)
    normalize_parser.set_defaults(func=cmd_normalize_sources)

    plot_parser = subparsers.add_parser(
        "plot-quality",
        help="Generate QA plots from normalized tabular outputs.",
    )
    plot_parser.add_argument("--normalized-root", default="build/normalized")
    plot_parser.add_argument("--output-root", default="")
    plot_parser.add_argument("--top-n-sources", type=int, default=20)
    plot_parser.set_defaults(func=cmd_plot_quality)

    filter_parser = subparsers.add_parser(
        "filter-coverage",
        help="Retain only normalized spectra above a minimum grid coverage threshold.",
    )
    filter_parser.add_argument("--normalized-root", default="build/normalized")
    filter_parser.add_argument("--output-root", required=True)
    filter_parser.add_argument("--min-coverage", type=float, default=0.8)
    filter_parser.set_defaults(func=cmd_filter_coverage)

    siac_parser = subparsers.add_parser(
        "build-siac-library",
        help="Build the SIAC-oriented spectral library package from a normalized dataset.",
    )
    siac_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    siac_parser.add_argument("--normalized-root", default="build/normalized_rebuild_v9_final")
    siac_parser.add_argument("--output-root", default="build/siac_spectral_library_v1")
    siac_parser.add_argument("--exclude-source-ids", default="")
    siac_parser.add_argument("--exclude-spectra-csv", default="")
    siac_parser.set_defaults(func=cmd_build_siac_library)

    return parser


def main_with_args(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def main() -> int:
    return main_with_args()


if __name__ == "__main__":
    raise SystemExit(main())
