from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from ._version import __version__
from .batch import fetch_batch, tidy_source_directory
from .build_db import assemble_catalog
from .coverage_filter import filter_normalized_by_coverage
from .fetchers import get_fetcher
from .manifest import filter_sources, load_manifest, manifest_sha256, split_csv_arg
from .mapping import (
    CANONICAL_WAVELENGTHS,
    SUPPORTED_KNN_BACKENDS,
    SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS,
    SUPPORTED_NEIGHBOR_ESTIMATORS,
    SUPPORTED_OUTPUT_MODES,
    SWIR_WAVELENGTHS,
    VNIR_WAVELENGTHS,
    BatchMappingResult,
    SpectralLibraryError,
    SpectralMapper,
    _attach_sample_context,
    _default_sample_id,
    benchmark_mapping,
    prepare_mapping_library,
    validate_prepared_library,
)
from .normalize import normalize_sources
from .quality_plots import generate_quality_plots
from .library_package import build_library_package


DEFAULT_MANIFEST = Path("manifests/sources.csv")
DEFAULT_USER_AGENT = f"spectral-library/{__version__}"
PUBLIC_COMMANDS = (
    "prepare-mapping-library",
    "download-prepared-library",
    "map-reflectance",
    "map-reflectance-batch",
    "benchmark-mapping",
    "validate-prepared-library",
)
INTERNAL_COMMANDS = (
    "plan-matrix",
    "fetch-source",
    "fetch-batch",
    "assemble-database",
    "tidy-results",
    "normalize-sources",
    "plot-quality",
    "filter-coverage",
    "build-library-package",
)
LEGACY_INTERNAL_COMMANDS = ("build-siac-library",)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_cli_log(
    args: argparse.Namespace,
    *,
    command: str,
    event: str,
    context: dict[str, object] | None = None,
    level: str = "info",
) -> None:
    if not bool(getattr(args, "json_logs", False)):
        return
    payload: dict[str, object] = {
        "command": command,
        "event": event,
        "level": level,
        "timestamp": _utc_timestamp(),
    }
    if context:
        payload["context"] = context
    started_at = getattr(args, "_cli_started_monotonic", None)
    if started_at is not None and event != "command_started":
        payload["elapsed_ms"] = int(round((time.monotonic() - float(started_at)) * 1000.0))
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


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


def _load_batch_reflectance_input(
    path: Path,
) -> tuple[tuple[str, ...], list[dict[str, float]], list[dict[str, bool] | None], tuple[str | None, ...]]:
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

        if {"sample_id", "band_id", "reflectance"}.issubset(fieldnames):
            sample_ids: list[str] = []
            sample_indices: dict[str, int] = {}
            reflectance_by_sample: dict[str, dict[str, float]] = {}
            valid_by_sample: dict[str, dict[str, bool]] = {}
            exclude_row_id_by_sample: dict[str, str | None] = {}
            has_valid = "valid" in fieldnames
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Batch long-format input requires non-empty sample_id values.",
                        context={"path": str(path)},
                    )
                if sample_id not in sample_indices:
                    sample_indices[sample_id] = len(sample_ids)
                    sample_ids.append(sample_id)
                    reflectance_by_sample[sample_id] = {}
                    valid_by_sample[sample_id] = {}
                    exclude_row_id_by_sample[sample_id] = None
                sample_index = sample_indices[sample_id]
                exclude_row_id = (row.get("exclude_row_id") or "").strip() or None
                if exclude_row_id_by_sample[sample_id] != exclude_row_id and exclude_row_id_by_sample[sample_id] is not None:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Batch long-format exclude_row_id values must remain consistent within each sample.",
                        context={"path": str(path), "sample_id": sample_id, "sample_index": sample_index},
                    )
                if exclude_row_id_by_sample[sample_id] is None:
                    exclude_row_id_by_sample[sample_id] = exclude_row_id
                band_id = (row.get("band_id") or "").strip()
                if not band_id:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Input reflectance CSV band_id values must be non-empty.",
                        context={"path": str(path), "sample_id": sample_id, "sample_index": sample_index},
                    )
                if band_id in reflectance_by_sample[sample_id]:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Batch input CSV must not contain duplicate band_id rows within a sample.",
                        context={"path": str(path), "sample_id": sample_id, "sample_index": sample_index, "band_id": band_id},
                    )
                try:
                    reflectance_by_sample[sample_id][band_id] = _parse_reflectance_cell(
                        row.get("reflectance"),
                        path=path,
                        band_id=band_id,
                    )
                    if has_valid:
                        valid_by_sample[sample_id][band_id] = _parse_valid_cell(row.get("valid"))
                except SpectralLibraryError as error:
                    raise _attach_sample_context(error, sample_id=sample_id, sample_index=sample_index) from error

            if not sample_ids:
                raise SpectralLibraryError(
                    "invalid_input_csv",
                    "Batch input reflectance CSV did not contain any sample rows.",
                    context={"path": str(path)},
                )

            return (
                tuple(sample_ids),
                [reflectance_by_sample[sample_id] for sample_id in sample_ids],
                [valid_by_sample[sample_id] if has_valid else None for sample_id in sample_ids],
                tuple(exclude_row_id_by_sample[sample_id] for sample_id in sample_ids),
            )

        rows = list(reader)
        if not rows:
            raise SpectralLibraryError(
                "invalid_input_csv",
                "Batch input reflectance CSV did not contain any sample rows.",
                context={"path": str(path)},
            )

        valid_columns = {
            fieldname[6:]: fieldname
            for fieldname in fieldnames
            if fieldname.startswith("valid_") and len(fieldname) > 6
        }
        reflectance_columns = [
            fieldname
            for fieldname in fieldnames
            if fieldname not in {"sample_id", "exclude_row_id"} and fieldname not in valid_columns.values()
        ]
        if not reflectance_columns:
            raise SpectralLibraryError(
                "invalid_input_csv",
                "Wide-format batch input must include at least one reflectance band column.",
                context={"path": str(path)},
            )
        for band_id, column_name in valid_columns.items():
            if band_id not in reflectance_columns:
                raise SpectralLibraryError(
                    "invalid_input_csv",
                    "Wide-format valid_<band_id> columns must match reflectance band columns.",
                    context={"path": str(path), "column": column_name},
                )

        sample_ids: list[str] = []
        reflectance_rows: list[dict[str, float]] = []
        valid_rows: list[dict[str, bool] | None] = []
        exclude_row_ids: list[str | None] = []
        for row_index, row in enumerate(rows):
            if "sample_id" in fieldnames:
                sample_id = (row.get("sample_id") or "").strip()
                if not sample_id:
                    raise SpectralLibraryError(
                        "invalid_input_csv",
                        "Wide-format batch input sample_id values must be non-empty when the column is present.",
                        context={"path": str(path), "sample_index": row_index},
                    )
            else:
                sample_id = _default_sample_id(row_index)
            if sample_id in sample_ids:
                raise SpectralLibraryError(
                    "invalid_input_csv",
                    "Batch input sample_id values must be unique.",
                    context={"path": str(path), "sample_id": sample_id, "sample_index": row_index},
                )

            reflectance: dict[str, float] = {}
            valid_mask: dict[str, bool] = {}
            for band_id in reflectance_columns:
                value = row.get(band_id)
                if value is not None and str(value).strip():
                    try:
                        reflectance[band_id] = _parse_reflectance_cell(value, path=path, column=band_id)
                    except SpectralLibraryError as error:
                        raise _attach_sample_context(error, sample_id=sample_id, sample_index=row_index) from error
                valid_column = valid_columns.get(band_id)
                if valid_column is not None:
                    try:
                        valid_mask[band_id] = _parse_valid_cell(row.get(valid_column))
                    except SpectralLibraryError as error:
                        raise _attach_sample_context(error, sample_id=sample_id, sample_index=row_index) from error

            if not reflectance:
                raise SpectralLibraryError(
                    "invalid_input_csv",
                    "Batch input sample rows must contain at least one usable band value.",
                    context={"path": str(path), "sample_id": sample_id, "sample_index": row_index},
                )

            sample_ids.append(sample_id)
            reflectance_rows.append(reflectance)
            valid_rows.append(valid_mask or None)
            exclude_row_ids.append((row.get("exclude_row_id") or "").strip() or None)

        return tuple(sample_ids), reflectance_rows, valid_rows, tuple(exclude_row_ids)


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
            writer = csv.DictWriter(handle, fieldnames=["band_id", "segment", "reflectance"], lineterminator="\n")
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

        writer = csv.DictWriter(handle, fieldnames=["wavelength_nm", "reflectance"], lineterminator="\n")
        writer.writeheader()
        for wavelength_nm, value in zip(result.reconstructed_wavelength_nm, reflectance):
            writer.writerow({"wavelength_nm": int(wavelength_nm), "reflectance": float(value)})
        return int(len(reflectance))


def _write_batch_mapping_output(
    mapper: SpectralMapper,
    result: BatchMappingResult,
    *,
    output_mode: str,
    target_sensor: str | None,
    output_path: Path,
) -> tuple[int, tuple[str, ...]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        if output_mode == "target_sensor":
            schema = mapper.get_sensor_schema(target_sensor or "")
            output_columns = schema.band_ids()
            fieldnames = ["sample_id", *output_columns]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for sample_id, sample_result in zip(result.sample_ids, result.results):
                if sample_result.target_reflectance is None:
                    raise SpectralLibraryError(
                        "invalid_result",
                        "Batch mapping result is missing target_reflectance.",
                        context={"sample_id": sample_id},
                    )
                row = {fieldname: "" for fieldname in fieldnames}
                row["sample_id"] = sample_id
                for band_id, reflectance in zip(sample_result.target_band_ids, sample_result.target_reflectance):
                    if band_id not in output_columns:
                        raise SpectralLibraryError(
                            "invalid_result",
                            "Batch mapping result emitted an unexpected target band.",
                            context={"sample_id": sample_id, "band_id": band_id},
                        )
                    row[band_id] = float(reflectance)
                writer.writerow(row)
            return len(result.results), output_columns

        if output_mode == "vnir_spectrum":
            output_columns = tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in VNIR_WAVELENGTHS)
            getter = lambda mapping_result: mapping_result.reconstructed_vnir
        elif output_mode == "swir_spectrum":
            output_columns = tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in SWIR_WAVELENGTHS)
            getter = lambda mapping_result: mapping_result.reconstructed_swir
        else:
            output_columns = tuple(f"nm_{int(wavelength_nm)}" for wavelength_nm in CANONICAL_WAVELENGTHS)
            getter = lambda mapping_result: mapping_result.reconstructed_full_spectrum

        fieldnames = ["sample_id", *output_columns]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample_id, sample_result in zip(result.sample_ids, result.results):
            reflectance = getter(sample_result)
            if reflectance is None:
                raise SpectralLibraryError(
                    "invalid_result",
                    "Batch mapping result is missing reconstructed spectral output.",
                    context={"sample_id": sample_id, "output_mode": output_mode},
                )
            if len(reflectance) != len(output_columns):
                raise SpectralLibraryError(
                    "invalid_result",
                    "Batch mapping result emitted a spectral output with the wrong width.",
                    context={
                        "sample_id": sample_id,
                        "output_mode": output_mode,
                        "expected_length": len(output_columns),
                        "actual_length": len(reflectance),
                    },
                )
            row = {"sample_id": sample_id}
            row.update({column: float(value) for column, value in zip(output_columns, reflectance)})
            writer.writerow(row)
        return len(result.results), output_columns


def _json_cell(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _neighbor_review_rows_for_result(sample_id: str, result: object) -> list[dict[str, object]]:
    from .mapping import MappingResult

    if not isinstance(result, MappingResult):
        raise SpectralLibraryError("invalid_result", "Unexpected mapping result type while writing neighbor review output.")

    diagnostics = dict(result.diagnostics)
    segments = diagnostics.get("segments")
    if not isinstance(segments, Mapping):
        raise SpectralLibraryError("invalid_result", "Mapping diagnostics are missing segment details.")

    base_fields = {
        "sample_id": sample_id,
        "source_sensor": diagnostics.get("source_sensor"),
        "target_sensor": diagnostics.get("target_sensor"),
        "output_mode": diagnostics.get("output_mode"),
        "neighbor_estimator": diagnostics.get("neighbor_estimator"),
        "knn_backend": diagnostics.get("knn_backend"),
        "knn_eps": diagnostics.get("knn_eps"),
        "k": diagnostics.get("k"),
    }
    rows: list[dict[str, object]] = []
    for segment, segment_payload_obj in segments.items():
        if not isinstance(segment_payload_obj, Mapping):
            raise SpectralLibraryError("invalid_result", "Segment diagnostics payload must be an object.")
        segment_payload = dict(segment_payload_obj)
        neighbor_ids = [str(value) for value in segment_payload.get("neighbor_ids", [])]  # type: ignore[arg-type]
        neighbor_distances = [float(value) for value in segment_payload.get("neighbor_distances", [])]  # type: ignore[arg-type]
        neighbor_weights = [float(value) for value in segment_payload.get("neighbor_weights", [])]  # type: ignore[arg-type]
        neighbor_band_values = list(segment_payload.get("neighbor_band_values", []))  # type: ignore[arg-type]
        common_fields = {
            **base_fields,
            "segment": str(segment),
            "segment_status": segment_payload.get("status"),
            "valid_band_count": segment_payload.get("valid_band_count"),
            "source_fit_rmse": segment_payload.get("source_fit_rmse"),
            "query_band_ids": _json_cell(segment_payload.get("query_band_ids", [])),
            "query_band_values": _json_cell(segment_payload.get("query_band_values", [])),
            "query_valid_mask": _json_cell(segment_payload.get("query_valid_mask", [])),
        }
        if not neighbor_ids:
            rows.append(
                {
                    **common_fields,
                    "rank": "",
                    "neighbor_id": "",
                    "neighbor_distance": "",
                    "neighbor_weight": "",
                    "neighbor_band_values": _json_cell([]),
                }
            )
            continue
        for rank, neighbor_id in enumerate(neighbor_ids, start=1):
            rows.append(
                {
                    **common_fields,
                    "rank": rank,
                    "neighbor_id": neighbor_id,
                    "neighbor_distance": neighbor_distances[rank - 1] if rank - 1 < len(neighbor_distances) else "",
                    "neighbor_weight": neighbor_weights[rank - 1] if rank - 1 < len(neighbor_weights) else "",
                    "neighbor_band_values": _json_cell(
                        neighbor_band_values[rank - 1] if rank - 1 < len(neighbor_band_values) else []
                    ),
                }
            )
    return rows


def _write_neighbor_review_output(path: Path, rows: Sequence[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_sensor",
        "target_sensor",
        "output_mode",
        "neighbor_estimator",
        "knn_backend",
        "knn_eps",
        "k",
        "segment",
        "segment_status",
        "valid_band_count",
        "source_fit_rmse",
        "query_band_ids",
        "query_band_values",
        "query_valid_mask",
        "rank",
        "neighbor_id",
        "neighbor_distance",
        "neighbor_weight",
        "neighbor_band_values",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


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


def cmd_build_library_package(args: argparse.Namespace) -> int:
    summary = build_library_package(
        Path(args.manifest),
        Path(args.normalized_root),
        Path(args.output_root),
        exclude_source_ids=split_csv_arg(args.exclude_source_ids),
        exclude_spectra_csv=Path(args.exclude_spectra_csv) if args.exclude_spectra_csv else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _configure_library_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--normalized-root", default="build/normalized_rebuild_v9_final")
    parser.add_argument("--output-root", default="build/siac_spectral_library_v1")
    parser.add_argument("--exclude-source-ids", default="")
    parser.add_argument("--exclude-spectra-csv", default="")
    parser.set_defaults(func=cmd_build_library_package)


def cmd_prepare_mapping_library(args: argparse.Namespace) -> int:
    knn_index_backends = _split_repeated_csv_arg(args.knn_index_backend)
    _emit_cli_log(
        args,
        command="prepare-mapping-library",
        event="command_started",
        context={
            "dtype": args.dtype,
            "knn_index_backends": knn_index_backends,
            "output_root": str(Path(args.output_root)),
            "siac_root": str(Path(args.siac_root)),
            "source_sensors": _split_repeated_csv_arg(args.source_sensor),
            "srf_root": str(Path(args.srf_root)),
        },
    )
    manifest = prepare_mapping_library(
        Path(args.siac_root),
        Path(args.srf_root),
        Path(args.output_root),
        _split_repeated_csv_arg(args.source_sensor),
        dtype=args.dtype,
        knn_index_backends=knn_index_backends,
    )
    payload = manifest.to_dict()
    payload["output_root"] = str(Path(args.output_root))
    _emit_cli_log(
        args,
        command="prepare-mapping-library",
        event="command_completed",
        context={
            "knn_index_backends": knn_index_backends,
            "output_root": str(Path(args.output_root)),
            "row_count": manifest.row_count,
            "source_sensors": list(manifest.source_sensors),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_map_reflectance(args: argparse.Namespace) -> int:
    exclude_row_ids = _split_repeated_csv_arg(args.exclude_row_id)
    exclude_sample_names = _split_repeated_csv_arg(args.exclude_sample_name)
    _emit_cli_log(
        args,
        command="map-reflectance",
        event="command_started",
        context={
            "diagnostics_output": str(Path(args.diagnostics_output)) if args.diagnostics_output else None,
            "input_path": str(Path(args.input)),
            "k": args.k,
            "knn_backend": args.knn_backend,
            "knn_eps": args.knn_eps,
            "min_valid_bands": args.min_valid_bands,
            "neighbor_estimator": args.neighbor_estimator,
            "neighbor_review_output": str(Path(args.neighbor_review_output)) if args.neighbor_review_output else None,
            "output_mode": args.output_mode,
            "output_path": str(Path(args.output)),
            "prepared_root": str(Path(args.prepared_root)),
            "source_sensor": args.source_sensor,
            "target_sensor": args.target_sensor or None,
        },
    )
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
        neighbor_estimator=args.neighbor_estimator,
        knn_backend=args.knn_backend,
        knn_eps=args.knn_eps,
        exclude_row_ids=exclude_row_ids,
        exclude_sample_names=exclude_sample_names,
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
            "excluded_row_ids": exclude_row_ids,
            "excluded_sample_names": exclude_sample_names,
            "neighbor_estimator": args.neighbor_estimator,
            "knn_backend": args.knn_backend,
            "knn_eps": args.knn_eps,
        }
    )
    if args.diagnostics_output:
        diagnostics_path = Path(args.diagnostics_output)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(result.to_summary_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["diagnostics_output"] = str(diagnostics_path)
    if args.neighbor_review_output:
        neighbor_review_path = Path(args.neighbor_review_output)
        review_row_count = _write_neighbor_review_output(
            neighbor_review_path,
            _neighbor_review_rows_for_result("sample_000001", result),
        )
        payload["neighbor_review_output"] = str(neighbor_review_path)
        payload["neighbor_review_row_count"] = int(review_row_count)
    _emit_cli_log(
        args,
        command="map-reflectance",
        event="command_completed",
        context={
            "diagnostics_output": payload.get("diagnostics_output"),
            "neighbor_review_output": payload.get("neighbor_review_output"),
            "output_mode": args.output_mode,
            "output_path": str(output_path),
            "segment_statuses": {
                segment: segment_payload["status"]
                for segment, segment_payload in result.diagnostics["segments"].items()  # type: ignore[index]
            },
            "written_rows": written_rows,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_map_reflectance_batch(args: argparse.Namespace) -> int:
    exclude_row_ids = _split_repeated_csv_arg(args.exclude_row_id)
    exclude_sample_names = _split_repeated_csv_arg(args.exclude_sample_name)
    _emit_cli_log(
        args,
        command="map-reflectance-batch",
        event="command_started",
        context={
            "diagnostics_output": str(Path(args.diagnostics_output)) if args.diagnostics_output else None,
            "input_path": str(Path(args.input)),
            "k": args.k,
            "knn_backend": args.knn_backend,
            "knn_eps": args.knn_eps,
            "min_valid_bands": args.min_valid_bands,
            "neighbor_review_output": str(Path(args.neighbor_review_output)) if args.neighbor_review_output else None,
            "output_mode": args.output_mode,
            "output_path": str(Path(args.output)),
            "prepared_root": str(Path(args.prepared_root)),
            "self_exclude_sample_id": bool(args.self_exclude_sample_id),
            "neighbor_estimator": args.neighbor_estimator,
            "source_sensor": args.source_sensor,
            "target_sensor": args.target_sensor or None,
        },
    )
    mapper = SpectralMapper(Path(args.prepared_root))
    sample_ids, reflectance_rows, valid_mask_rows, input_exclude_row_ids = _load_batch_reflectance_input(Path(args.input))
    result = mapper.map_reflectance_batch(
        source_sensor=args.source_sensor,
        reflectance_rows=reflectance_rows,
        valid_mask_rows=valid_mask_rows,
        sample_ids=sample_ids,
        output_mode=args.output_mode,
        target_sensor=args.target_sensor or None,
        k=args.k,
        min_valid_bands=args.min_valid_bands,
        neighbor_estimator=args.neighbor_estimator,
        knn_backend=args.knn_backend,
        knn_eps=args.knn_eps,
        exclude_row_ids=exclude_row_ids,
        exclude_sample_names=exclude_sample_names,
        exclude_row_ids_per_sample=input_exclude_row_ids,
        self_exclude_sample_id=bool(args.self_exclude_sample_id),
    )
    output_path = Path(args.output)
    written_rows, output_columns = _write_batch_mapping_output(
        mapper,
        result,
        output_mode=args.output_mode,
        target_sensor=args.target_sensor or None,
        output_path=output_path,
    )

    payload: dict[str, object] = {
        "source_sensor": args.source_sensor,
        "target_sensor": args.target_sensor or None,
        "output_mode": args.output_mode,
        "output_path": str(output_path),
        "output_columns": list(output_columns),
        "sample_count": len(result.results),
        "written_rows": written_rows,
        "excluded_row_ids": exclude_row_ids,
        "excluded_sample_names": exclude_sample_names,
        "neighbor_estimator": args.neighbor_estimator,
        "knn_backend": args.knn_backend,
        "knn_eps": args.knn_eps,
        "self_exclude_sample_id": bool(args.self_exclude_sample_id),
        "input_exclude_row_id_column": any(value is not None for value in input_exclude_row_ids),
    }
    if args.diagnostics_output:
        diagnostics_path = Path(args.diagnostics_output)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(result.to_summary_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["diagnostics_output"] = str(diagnostics_path)
    if args.neighbor_review_output:
        neighbor_review_path = Path(args.neighbor_review_output)
        review_rows: list[dict[str, object]] = []
        for sample_id, sample_result in zip(result.sample_ids, result.results):
            review_rows.extend(_neighbor_review_rows_for_result(sample_id, sample_result))
        review_row_count = _write_neighbor_review_output(neighbor_review_path, review_rows)
        payload["neighbor_review_output"] = str(neighbor_review_path)
        payload["neighbor_review_row_count"] = int(review_row_count)
    _emit_cli_log(
        args,
        command="map-reflectance-batch",
        event="command_completed",
        context={
            "diagnostics_output": payload.get("diagnostics_output"),
            "neighbor_review_output": payload.get("neighbor_review_output"),
            "output_columns": list(output_columns),
            "output_path": str(output_path),
            "sample_count": len(result.results),
            "written_rows": written_rows,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_benchmark_mapping(args: argparse.Namespace) -> int:
    _emit_cli_log(
        args,
        command="benchmark-mapping",
        event="command_started",
        context={
            "k": args.k,
            "knn_backend": args.knn_backend,
            "knn_eps": args.knn_eps,
            "max_test_rows": args.max_test_rows if args.max_test_rows > 0 else None,
            "prepared_root": str(Path(args.prepared_root)),
            "random_seed": args.random_seed,
            "report": str(Path(args.report)),
            "neighbor_estimator": args.neighbor_estimator,
            "source_sensor": args.source_sensor,
            "target_sensor": args.target_sensor,
            "test_fraction": args.test_fraction,
        },
    )
    report = benchmark_mapping(
        Path(args.prepared_root),
        args.source_sensor,
        args.target_sensor,
        k=args.k,
        test_fraction=args.test_fraction,
        max_test_rows=(args.max_test_rows if args.max_test_rows > 0 else None),
        random_seed=args.random_seed,
        neighbor_estimator=args.neighbor_estimator,
        knn_backend=args.knn_backend,
        knn_eps=args.knn_eps,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit_cli_log(
        args,
        command="benchmark-mapping",
        event="command_completed",
        context={
            "neighbor_estimator": args.neighbor_estimator,
            "knn_backend": args.knn_backend,
            "knn_eps": args.knn_eps,
            "max_test_rows": args.max_test_rows if args.max_test_rows > 0 else None,
            "report": str(report_path),
            "test_rows": report["test_rows"],
            "train_rows": report["train_rows"],
        },
    )
    print(
        json.dumps(
            {
                "prepared_root": str(Path(args.prepared_root)),
                "report": str(report_path),
                "neighbor_estimator": args.neighbor_estimator,
                "knn_backend": args.knn_backend,
                "knn_eps": args.knn_eps,
                "max_test_rows": args.max_test_rows if args.max_test_rows > 0 else None,
                "test_rows": report["test_rows"],
                "train_rows": report["train_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_validate_prepared_library(args: argparse.Namespace) -> int:
    _emit_cli_log(
        args,
        command="validate-prepared-library",
        event="command_started",
        context={
            "checksums_verified": not args.no_verify_checksums,
            "prepared_root": str(Path(args.prepared_root)),
        },
    )
    manifest = validate_prepared_library(
        Path(args.prepared_root),
        verify_checksums=not args.no_verify_checksums,
    )
    payload = manifest.to_dict()
    payload["prepared_root"] = str(Path(args.prepared_root))
    payload["checksums_verified"] = not args.no_verify_checksums
    _emit_cli_log(
        args,
        command="validate-prepared-library",
        event="command_completed",
        context={
            "checksums_verified": not args.no_verify_checksums,
            "prepared_root": str(Path(args.prepared_root)),
            "row_count": manifest.row_count,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_download_prepared_library(args: argparse.Namespace) -> int:
    from .runtime_download import RuntimeDownloadError, download_prepared_library

    _emit_cli_log(
        args,
        command="download-prepared-library",
        event="command_started",
        context={"output_root": str(Path(args.output_root))},
    )
    try:
        output_root = download_prepared_library(
            Path(args.output_root),
            url=args.url or None,
            tag=args.tag or None,
            sha256=args.sha256 or None,
            verify_after_extract=not args.no_verify,
        )
    except RuntimeDownloadError as exc:
        raise SpectralLibraryError(
            "download_failed",
            str(exc),
            context={"output_root": str(Path(args.output_root))},
        ) from exc
    _emit_cli_log(
        args,
        command="download-prepared-library",
        event="command_completed",
        context={"output_root": str(output_root)},
    )
    return 0


def _build_base_parser(*, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--json-errors", action="store_true", help="Emit machine-readable JSON errors.")
    parser.add_argument("--json-logs", action="store_true", help="Emit structured JSON log events to stderr.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _add_public_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    prepare_mapping_parser = subparsers.add_parser(
        "prepare-mapping-library",
        help="Build a prepared runtime layer for retrieval-based spectral mapping.",
    )
    prepare_mapping_parser.add_argument("--siac-root", required=True)
    prepare_mapping_parser.add_argument("--srf-root", required=True)
    prepare_mapping_parser.add_argument("--source-sensor", action="append", required=True)
    prepare_mapping_parser.add_argument("--output-root", required=True)
    prepare_mapping_parser.add_argument("--dtype", default="float32")
    prepare_mapping_parser.add_argument(
        "--knn-index-backend",
        action="append",
        default=[],
        choices=list(SUPPORTED_PERSISTED_KNN_INDEX_BACKENDS),
    )
    prepare_mapping_parser.set_defaults(func=cmd_prepare_mapping_library)

    download_parser = subparsers.add_parser(
        "download-prepared-library",
        help="Download a pre-built prepared runtime from a GitHub Release or URL.",
    )
    download_parser.add_argument(
        "--output-root", required=True,
        help="Local directory to extract the runtime into.",
    )
    download_parser.add_argument(
        "--url", default="",
        help="Direct URL to a runtime .tar.gz archive (skips GitHub Release lookup).",
    )
    download_parser.add_argument(
        "--tag", default="",
        help="GitHub Release tag to download from (e.g. v0.2.0). Defaults to latest.",
    )
    download_parser.add_argument(
        "--sha256", default="",
        help="Expected SHA-256 hex digest for the archive.",
    )
    download_parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip runtime validation after extraction.",
    )
    download_parser.set_defaults(func=cmd_download_prepared_library)

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
    map_parser.add_argument("--neighbor-estimator", choices=list(SUPPORTED_NEIGHBOR_ESTIMATORS), default="mean")
    map_parser.add_argument("--knn-backend", choices=list(SUPPORTED_KNN_BACKENDS), default="numpy")
    map_parser.add_argument("--knn-eps", type=float, default=0.0)
    map_parser.add_argument("--exclude-row-id", action="append", default=[])
    map_parser.add_argument("--exclude-sample-name", action="append", default=[])
    map_parser.add_argument("--output", required=True)
    map_parser.add_argument("--diagnostics-output", default="")
    map_parser.add_argument("--neighbor-review-output", default="")
    map_parser.set_defaults(func=cmd_map_reflectance)

    batch_map_parser = subparsers.add_parser(
        "map-reflectance-batch",
        help="Map multiple source-sensor reflectance samples from one CSV input file.",
    )
    batch_map_parser.add_argument("--prepared-root", required=True)
    batch_map_parser.add_argument("--source-sensor", required=True)
    batch_map_parser.add_argument("--target-sensor", default="")
    batch_map_parser.add_argument("--input", required=True)
    batch_map_parser.add_argument("--output-mode", choices=list(SUPPORTED_OUTPUT_MODES), required=True)
    batch_map_parser.add_argument("--k", type=int, default=10)
    batch_map_parser.add_argument("--min-valid-bands", type=int, default=1)
    batch_map_parser.add_argument("--neighbor-estimator", choices=list(SUPPORTED_NEIGHBOR_ESTIMATORS), default="mean")
    batch_map_parser.add_argument("--knn-backend", choices=list(SUPPORTED_KNN_BACKENDS), default="numpy")
    batch_map_parser.add_argument("--knn-eps", type=float, default=0.0)
    batch_map_parser.add_argument("--exclude-row-id", action="append", default=[])
    batch_map_parser.add_argument("--exclude-sample-name", action="append", default=[])
    batch_map_parser.add_argument("--self-exclude-sample-id", action="store_true")
    batch_map_parser.add_argument("--output", required=True)
    batch_map_parser.add_argument("--diagnostics-output", default="")
    batch_map_parser.add_argument("--neighbor-review-output", default="")
    batch_map_parser.set_defaults(func=cmd_map_reflectance_batch)

    benchmark_parser = subparsers.add_parser(
        "benchmark-mapping",
        help="Benchmark retrieval-based mapping against a regression baseline on held-out library spectra.",
    )
    benchmark_parser.add_argument("--prepared-root", required=True)
    benchmark_parser.add_argument("--source-sensor", required=True)
    benchmark_parser.add_argument("--target-sensor", required=True)
    benchmark_parser.add_argument("--k", type=int, default=10)
    benchmark_parser.add_argument("--test-fraction", type=float, default=0.2)
    benchmark_parser.add_argument("--max-test-rows", type=int, default=0)
    benchmark_parser.add_argument("--random-seed", type=int, default=0)
    benchmark_parser.add_argument("--neighbor-estimator", choices=list(SUPPORTED_NEIGHBOR_ESTIMATORS), default="mean")
    benchmark_parser.add_argument("--knn-backend", choices=list(SUPPORTED_KNN_BACKENDS), default="numpy")
    benchmark_parser.add_argument("--knn-eps", type=float, default=0.0)
    benchmark_parser.add_argument("--report", required=True)
    benchmark_parser.set_defaults(func=cmd_benchmark_mapping)

    validate_parser = subparsers.add_parser(
        "validate-prepared-library",
        help="Validate a prepared mapping runtime root and optionally verify its checksums.",
    )
    validate_parser.add_argument("--prepared-root", required=True)
    validate_parser.add_argument("--no-verify-checksums", action="store_true")
    validate_parser.set_defaults(func=cmd_validate_prepared_library)


def _add_internal_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    visible: bool,
    include_legacy_aliases: bool = False,
) -> None:
    plan_parser = subparsers.add_parser(
        "plan-matrix",
        help="Create a GitHub Actions matrix from the manifest." if visible else argparse.SUPPRESS,
    )
    plan_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    plan_parser.add_argument("--source-ids", default="")
    plan_parser.add_argument("--tiers", default="")
    plan_parser.add_argument("--statuses", default="")
    plan_parser.add_argument("--adapters", default="")
    plan_parser.add_argument("--limit", type=int, default=0)
    plan_parser.add_argument("--output", default="")
    plan_parser.set_defaults(func=cmd_plan_matrix)

    fetch_parser = subparsers.add_parser(
        "fetch-source",
        help="Fetch one source from the manifest." if visible else argparse.SUPPRESS,
    )
    fetch_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    fetch_parser.add_argument("--source-id", required=True)
    fetch_parser.add_argument("--output-root", default="build/sources")
    fetch_parser.add_argument("--fetch-mode", choices=["metadata", "assets"], default="metadata")
    fetch_parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    fetch_parser.set_defaults(func=cmd_fetch_source)

    batch_parser = subparsers.add_parser(
        "fetch-batch",
        help="Fetch multiple sources and tidy their output directories." if visible else argparse.SUPPRESS,
    )
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

    assemble_parser = subparsers.add_parser(
        "assemble-database",
        help="Assemble the catalog database from fetch outputs." if visible else argparse.SUPPRESS,
    )
    assemble_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    assemble_parser.add_argument("--results-root", default="build/sources")
    assemble_parser.add_argument("--output-root", default="build/assembled")
    assemble_parser.set_defaults(func=cmd_assemble_database)

    tidy_parser = subparsers.add_parser(
        "tidy-results",
        help="Reorganize fetched source directories into metadata/docs/data." if visible else argparse.SUPPRESS,
    )
    tidy_parser.add_argument("--results-root", default="build/local_sources")
    tidy_parser.set_defaults(func=cmd_tidy_results)

    normalize_parser = subparsers.add_parser(
        "normalize-sources",
        help="Normalize downloaded spectra onto the shared 400-2500 nm, 1 nm grid." if visible else argparse.SUPPRESS,
    )
    normalize_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    normalize_parser.add_argument("--results-root", default="build/local_sources")
    normalize_parser.add_argument("--output-root", default="build/normalized")
    normalize_parser.add_argument("--source-ids", default="")
    normalize_parser.add_argument("--limit", type=int, default=0)
    normalize_parser.set_defaults(func=cmd_normalize_sources)

    plot_parser = subparsers.add_parser(
        "plot-quality",
        help="Generate QA plots from normalized tabular outputs." if visible else argparse.SUPPRESS,
    )
    plot_parser.add_argument("--normalized-root", default="build/normalized")
    plot_parser.add_argument("--output-root", default="")
    plot_parser.add_argument("--top-n-sources", type=int, default=20)
    plot_parser.set_defaults(func=cmd_plot_quality)

    filter_parser = subparsers.add_parser(
        "filter-coverage",
        help="Retain only normalized spectra above a minimum grid coverage threshold." if visible else argparse.SUPPRESS,
    )
    filter_parser.add_argument("--normalized-root", default="build/normalized")
    filter_parser.add_argument("--output-root", required=True)
    filter_parser.add_argument("--min-coverage", type=float, default=0.8)
    filter_parser.set_defaults(func=cmd_filter_coverage)

    package_parser = subparsers.add_parser(
        "build-library-package",
        help="Build the packaged spectral-library export from a normalized dataset." if visible else argparse.SUPPRESS,
    )
    _configure_library_package_parser(package_parser)

    if include_legacy_aliases:
        legacy_package_parser = subparsers.add_parser("build-siac-library", help=argparse.SUPPRESS)
        _configure_library_package_parser(legacy_package_parser)


def build_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser(prog="spectral-library")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_public_subparsers(subparsers)
    return parser


def build_internal_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser(prog="spectral-library-internal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_internal_subparsers(subparsers, visible=True)
    return parser


def _build_internal_dispatch_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser(prog="spectral-library-internal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_internal_subparsers(subparsers, visible=True, include_legacy_aliases=True)
    return parser


def _build_legacy_dispatch_parser() -> argparse.ArgumentParser:
    parser = _build_base_parser(prog="spectral-library")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_public_subparsers(subparsers)
    _add_internal_subparsers(subparsers, visible=False, include_legacy_aliases=True)
    return parser


def _first_command_position(argv: Sequence[str]) -> tuple[int | None, str | None]:
    for index, token in enumerate(argv):
        if not str(token).startswith("-"):
            return index, str(token)
    return None, None


def _internal_parser_for_argv(argv: Sequence[str]) -> argparse.ArgumentParser:
    _, command_token = _first_command_position(argv)
    if command_token in LEGACY_INTERNAL_COMMANDS:
        return _build_internal_dispatch_parser()
    return build_internal_parser()


def _dispatch_parser_for_argv(argv: Sequence[str]) -> tuple[argparse.ArgumentParser, list[str]]:
    argv_list = list(argv)
    command_index, command_token = _first_command_position(argv_list)
    if command_token == "internal":
        if command_index is None:
            return build_internal_parser(), argv_list
        internal_argv = argv_list[:command_index] + argv_list[command_index + 1 :]
        return _internal_parser_for_argv(internal_argv), internal_argv
    if command_token in INTERNAL_COMMANDS or command_token in LEGACY_INTERNAL_COMMANDS:
        return _build_legacy_dispatch_parser(), argv_list
    return build_parser(), argv_list


def _run_parser(parser: argparse.ArgumentParser, argv: list[str]) -> int:
    args = parser.parse_args(argv)
    args._cli_started_monotonic = time.monotonic()
    command_name = str(getattr(args, "command", ""))
    try:
        return args.func(args)
    except SpectralLibraryError as error:
        _emit_cli_log(
            args,
            command=command_name,
            event="command_failed",
            level="error",
            context={
                "error_code": error.code,
                "message": error.message,
                **({"context": error.context} if error.context else {}),
            },
        )
        _emit_cli_error(error, command=command_name, json_errors=bool(args.json_errors))
        return 2


def main_with_args(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser, dispatched_argv = _dispatch_parser_for_argv(argv_list)
    return _run_parser(parser, dispatched_argv)


def main_internal_with_args(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    return _run_parser(_internal_parser_for_argv(argv_list), argv_list)


def main() -> int:
    return main_with_args()


def main_internal() -> int:
    return main_internal_with_args()


if __name__ == "__main__":
    raise SystemExit(main())
