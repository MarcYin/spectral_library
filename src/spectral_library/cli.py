from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ._version import __version__
from .batch import fetch_batch, tidy_source_directory
from .build_db import assemble_catalog
from .coverage_filter import filter_normalized_by_coverage
from .fetchers import get_fetcher
from .manifest import filter_sources, load_manifest, manifest_sha256, split_csv_arg
from .mapping import SUPPORTED_OUTPUT_MODES, SpectralLibraryError, SpectralMapper, benchmark_mapping, prepare_mapping_library
from .normalize import normalize_sources
from .quality_plots import generate_quality_plots
from .siac import build_siac_library


DEFAULT_MANIFEST = Path("manifests/sources.csv")
DEFAULT_USER_AGENT = f"spectral-library/{__version__}"


def _split_repeated_csv_arg(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(split_csv_arg(value))
    return items


def _parse_valid_cell(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise SpectralLibraryError(
        "invalid_input_csv",
        "Input reflectance CSV valid column must be boolean-like.",
        context={"value": value},
    )


def _parse_reflectance_cell(
    value: str | None,
    *,
    path: Path,
    band_id: str | None = None,
    column: str | None = None,
) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        context: dict[str, object] = {"path": str(path), "value": value}
        if band_id is not None:
            context["band_id"] = band_id
        if column is not None:
            context["column"] = column
        raise SpectralLibraryError(
            "invalid_input_csv",
            "Input reflectance CSV values must be numeric.",
            context=context,
        ) from exc


def _load_reflectance_input(path: Path) -> tuple[dict[str, float], dict[str, bool] | None]:
    if not path.exists():
        raise SpectralLibraryError("missing_input_file", "Input reflectance CSV does not exist.", context={"path": str(path)})

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SpectralLibraryError(
                "invalid_input_csv",
                "Input reflectance CSV must include a header row.",
                context={"path": str(path)},
            )

        if {"band_id", "reflectance"}.issubset(fieldnames):
            reflectance: dict[str, float] = {}
            valid_mask: dict[str, bool] = {}
            has_valid = "valid" in fieldnames
            for row in reader:
                band_id = (row.get("band_id") or "").strip()
                if not band_id:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Input reflectance CSV band_id values must be non-empty.",
                        context={"path": str(path)},
                    )
                if band_id in reflectance:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Input reflectance CSV must not contain duplicate band_id rows.",
                        context={"path": str(path), "band_id": band_id},
                    )
                reflectance[band_id] = _parse_reflectance_cell(
                    row.get("reflectance"),
                    path=path,
                    band_id=band_id,
                )
                if has_valid:
                    valid_mask[band_id] = _parse_valid_cell(row.get("valid"))
            if not reflectance:
                raise SpectralLibraryError(
                    "invalid_input_csv",
                    "Input reflectance CSV did not contain any band rows.",
                    context={"path": str(path)},
                )
            return reflectance, valid_mask if has_valid else None

        rows = list(reader)
        if len(rows) != 1:
            raise SpectralLibraryError(
                "invalid_input_csv",
                "Wide-format input reflectance CSV must contain exactly one data row.",
                context={"path": str(path), "row_count": len(rows)},
            )
        reflectance = {
            fieldname: _parse_reflectance_cell(rows[0].get(fieldname), path=path, column=fieldname)
            for fieldname in fieldnames
            if rows[0].get(fieldname) is not None and str(rows[0][fieldname]).strip()
        }
        if not reflectance:
            raise SpectralLibraryError(
                "invalid_input_csv",
                "Input reflectance CSV did not contain any usable band values.",
                context={"path": str(path)},
            )
        return reflectance, None


def _write_mapping_output(
    mapper: SpectralMapper,
    result: object,
    *,
    output_mode: str,
    target_sensor: str | None,
    output_path: Path,
) -> int:
    from .mapping import MappingResult

    if not isinstance(result, MappingResult):
        raise SpectralLibraryError("invalid_result", "Unexpected mapping result type.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if output_mode == "target_sensor":
            if result.target_reflectance is None:
                raise SpectralLibraryError("invalid_result", "Mapping result is missing target_reflectance.")
            schema = mapper.get_sensor_schema(target_sensor or "")
            segment_by_band = {band.band_id: band.segment for band in schema.bands}
            writer = csv.DictWriter(handle, fieldnames=["band_id", "segment", "reflectance"])
            writer.writeheader()
            for band_id, reflectance in zip(result.target_band_ids, result.target_reflectance):
                writer.writerow(
                    {
                        "band_id": band_id,
                        "segment": segment_by_band.get(band_id, ""),
                        "reflectance": float(reflectance),
                    }
                )
            return len(result.target_band_ids)

        if output_mode == "vnir_spectrum":
            reflectance = result.reconstructed_vnir
        elif output_mode == "swir_spectrum":
            reflectance = result.reconstructed_swir
        else:
            reflectance = result.reconstructed_full_spectrum
        if reflectance is None or result.reconstructed_wavelength_nm is None:
            raise SpectralLibraryError("invalid_result", "Mapping result is missing reconstructed spectrum output.")

        writer = csv.DictWriter(handle, fieldnames=["wavelength_nm", "reflectance"])
        writer.writeheader()
        for wavelength_nm, value in zip(result.reconstructed_wavelength_nm, reflectance):
            writer.writerow({"wavelength_nm": int(wavelength_nm), "reflectance": float(value)})
        return int(len(reflectance))


def _emit_cli_error(error: SpectralLibraryError, *, command: str | None, json_errors: bool) -> None:
    if json_errors:
        print(json.dumps(error.to_dict(command=command), indent=2, sort_keys=True), file=sys.stderr)
        return
    print(f"{error.code}: {error.message}", file=sys.stderr)


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


def cmd_prepare_mapping_library(args: argparse.Namespace) -> int:
    manifest = prepare_mapping_library(
        Path(args.siac_root),
        Path(args.srf_root),
        Path(args.output_root),
        _split_repeated_csv_arg(args.source_sensor),
        dtype=args.dtype,
    )
    payload = manifest.to_dict()
    payload["output_root"] = str(Path(args.output_root))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_map_reflectance(args: argparse.Namespace) -> int:
    mapper = SpectralMapper(Path(args.prepared_root))
    reflectance, valid_mask = _load_reflectance_input(Path(args.input))
    result = mapper.map_reflectance(
        source_sensor=args.source_sensor,
        reflectance=reflectance,
        valid_mask=valid_mask,
        output_mode=args.output_mode,
        target_sensor=args.target_sensor or None,
        k=args.k,
        min_valid_bands=args.min_valid_bands,
    )
    output_path = Path(args.output)
    written_rows = _write_mapping_output(
        mapper,
        result,
        output_mode=args.output_mode,
        target_sensor=args.target_sensor or None,
        output_path=output_path,
    )
    payload = result.to_summary_dict()
    payload.update(
        {
            "source_sensor": args.source_sensor,
            "target_sensor": args.target_sensor or None,
            "output_mode": args.output_mode,
            "output_path": str(output_path),
            "written_rows": written_rows,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_benchmark_mapping(args: argparse.Namespace) -> int:
    report = benchmark_mapping(
        Path(args.prepared_root),
        args.source_sensor,
        args.target_sensor,
        k=args.k,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "prepared_root": str(Path(args.prepared_root)),
                "report": str(report_path),
                "test_rows": report["test_rows"],
                "train_rows": report["train_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectral-library")
    parser.add_argument("--json-errors", action="store_true", help="Emit machine-readable JSON errors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_mapping_parser = subparsers.add_parser(
        "prepare-mapping-library",
        help="Build a prepared runtime layer for retrieval-based spectral mapping.",
    )
    prepare_mapping_parser.add_argument("--siac-root", required=True)
    prepare_mapping_parser.add_argument("--srf-root", required=True)
    prepare_mapping_parser.add_argument("--source-sensor", action="append", required=True)
    prepare_mapping_parser.add_argument("--output-root", required=True)
    prepare_mapping_parser.add_argument("--dtype", default="float32")
    prepare_mapping_parser.set_defaults(func=cmd_prepare_mapping_library)

    map_parser = subparsers.add_parser(
        "map-reflectance",
        help="Map source-sensor reflectance to a target sensor or reconstructed spectral output.",
    )
    map_parser.add_argument("--prepared-root", required=True)
    map_parser.add_argument("--source-sensor", required=True)
    map_parser.add_argument("--target-sensor", default="")
    map_parser.add_argument("--input", required=True)
    map_parser.add_argument("--output-mode", choices=list(SUPPORTED_OUTPUT_MODES), required=True)
    map_parser.add_argument("--k", type=int, default=10)
    map_parser.add_argument("--min-valid-bands", type=int, default=1)
    map_parser.add_argument("--output", required=True)
    map_parser.set_defaults(func=cmd_map_reflectance)

    benchmark_parser = subparsers.add_parser(
        "benchmark-mapping",
        help="Benchmark retrieval-based mapping against a regression baseline on held-out library spectra.",
    )
    benchmark_parser.add_argument("--prepared-root", required=True)
    benchmark_parser.add_argument("--source-sensor", required=True)
    benchmark_parser.add_argument("--target-sensor", required=True)
    benchmark_parser.add_argument("--k", type=int, default=10)
    benchmark_parser.add_argument("--test-fraction", type=float, default=0.2)
    benchmark_parser.add_argument("--random-seed", type=int, default=0)
    benchmark_parser.add_argument("--report", required=True)
    benchmark_parser.set_defaults(func=cmd_benchmark_mapping)

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
    try:
        return args.func(args)
    except SpectralLibraryError as error:
        _emit_cli_error(error, command=args.command, json_errors=bool(args.json_errors))
        return 2


def main() -> int:
    return main_with_args()


if __name__ == "__main__":
    raise SystemExit(main())
