from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from spectral_library import __version__
from spectral_library import cli
from spectral_library.build_db import _load_fetch_results, assemble_catalog
from spectral_library.manifest import SourceRecord


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SourceRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def manifest_row(**overrides: str) -> dict[str, str]:
    row = {
        "source_id": "src1",
        "name": "Source 1",
        "section": "direct_public",
        "subsection": "soil",
        "spectral_type": "Soil",
        "coverage": "400-2500 nm",
        "resource_type": "Direct library",
        "provider": "manual",
        "landing_url": "https://example.com/src1",
        "download_url": "",
        "fetch_adapter": "manual_portal",
        "auth_mode": "manual_review",
        "expected_format": "csv",
        "tier": "tier1",
        "priority": "high",
        "ingest_role": "primary_raw",
        "normalization_eligibility": "eligible_full",
        "status": "planned",
        "notes": "note",
    }
    row.update(overrides)
    return row


def write_fetch_result(path: Path, source_id: str, artifact_note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": source_id,
        "source_name": f"Source {source_id}",
        "fetch_adapter": "manual_portal",
        "fetch_mode": "metadata",
        "status": "manual_review",
        "landing_url": f"https://example.com/{source_id}",
        "started_at": "2024-01-01T00:00:00+00:00",
        "finished_at": "2024-01-01T00:00:01+00:00",
        "notes": ["note-a", "note-b"],
        "artifacts": [
            {
                "artifact_id": "artifact-1",
                "kind": "metadata",
                "url": "",
                "path": str(path.parent / "artifact.json"),
                "media_type": "application/json",
                "size_bytes": 12,
                "sha256": "deadbeef",
                "status": "written",
                "note": artifact_note,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class BuildDatabaseTests(unittest.TestCase):
    def test_load_fetch_results_handles_missing_and_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows, artifacts = _load_fetch_results(root / "missing")
            self.assertEqual(rows, [])
            self.assertEqual(artifacts, [])

            write_fetch_result(root / "src1" / "fetch-result.json", "src1", "extra")
            rows, artifacts = _load_fetch_results(root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["notes"], "note-a | note-b")
            self.assertEqual(artifacts[0]["note"], "extra")

    def test_assemble_catalog_writes_outputs_and_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            results_root = root / "results"
            output_root = root / "assembled"
            write_manifest(
                manifest_path,
                [
                    manifest_row(source_id="src1", name="Source 1"),
                    manifest_row(source_id="src2", name="Source 2", status="review_required"),
                ],
            )
            write_fetch_result(results_root / "src1" / "fetch-result.json", "src1")

            summary = assemble_catalog(manifest_path, results_root, output_root)

            self.assertEqual(summary["manifest_sources"], 2)
            self.assertEqual(summary["fetched_sources"], 1)
            self.assertTrue((output_root / "db" / "catalog.duckdb").exists())
            self.assertTrue((output_root / "parquet" / "sources.parquet").exists())
            self.assertTrue((output_root / "tabular" / "fetch_results.csv").exists())

            connection = duckdb.connect(str(output_root / "db" / "catalog.duckdb"))
            try:
                rows = connection.execute(
                    "SELECT source_id, fetch_status, artifact_count FROM source_build_status ORDER BY source_id"
                ).fetchall()
            finally:
                connection.close()

            self.assertEqual(rows[0], ("src1", "manual_review", 1))
            self.assertEqual(rows[1], ("src2", "not_fetched", 0))


class CliCommandTests(unittest.TestCase):
    def test_plan_matrix_command_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            output_path = root / "plan" / "matrix.json"
            write_manifest(
                manifest_path,
                [
                    manifest_row(source_id="src1", fetch_adapter="manual_portal"),
                    manifest_row(source_id="src2", fetch_adapter="manual_portal", status="review_required"),
                ],
            )
            args = cli.build_internal_parser().parse_args(
                [
                    "plan-matrix",
                    "--manifest",
                    str(manifest_path),
                    "--statuses",
                    "planned",
                    "--adapters",
                    "manual_portal",
                    "--limit",
                    "1",
                    "--output",
                    str(output_path),
                ]
            )
            result = args.func(args)

            self.assertEqual(result, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["include"]), 1)
            self.assertEqual(payload["include"][0]["source_id"], "src1")

    def test_plan_matrix_command_prints_stdout_without_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            write_manifest(manifest_path, [manifest_row(source_id="src1", fetch_adapter="manual_portal")])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "plan-matrix",
                        "--manifest",
                        str(manifest_path),
                        "--adapters",
                        "manual_portal",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["include"][0]["source_id"], "src1")

    def test_fetch_source_command_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            output_root = root / "results"
            write_manifest(manifest_path, [manifest_row(source_id="src1", fetch_adapter="manual_portal")])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "fetch-source",
                        "--manifest",
                        str(manifest_path),
                        "--source-id",
                        "src1",
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            result_path = output_root / "src1" / "fetch-result.json"
            self.assertTrue(result_path.exists())
            printed = json.loads(stdout.getvalue())
            self.assertEqual(printed["source_id"], "src1")

    def test_fetch_source_unknown_id_raises_system_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            write_manifest(manifest_path, [manifest_row(source_id="src1")])

            with self.assertRaises(SystemExit):
                cli.main_with_args(
                    [
                        "fetch-source",
                        "--manifest",
                        str(manifest_path),
                        "--source-id",
                        "missing",
                    ]
                )

    def test_assemble_database_command_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            results_root = root / "results"
            output_root = root / "assembled"
            write_manifest(manifest_path, [manifest_row(source_id="src1")])
            write_fetch_result(results_root / "src1" / "fetch-result.json", "src1")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "assemble-database",
                        "--manifest",
                        str(manifest_path),
                        "--results-root",
                        str(results_root),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["fetched_sources"], 1)
            self.assertTrue((output_root / "db" / "catalog.duckdb").exists())

    def test_fetch_batch_command_prints_summary(self) -> None:
        with patch.object(cli, "fetch_batch", return_value={"selected_sources": 2}) as mock_fetch_batch:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "fetch-batch",
                        "--manifest",
                        "manifests/sources.csv",
                        "--source-ids",
                        "src1,src2",
                        "--continue-on-error",
                        "--clean-output",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["selected_sources"], 2)
            self.assertTrue(mock_fetch_batch.called)

    def test_tidy_results_command_prints_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "src1"
            source_dir.mkdir()
            (source_dir / "fetch-result.json").write_text(
                json.dumps(
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "fetch_adapter": "manual_portal",
                        "fetch_mode": "metadata",
                        "status": "manual_review",
                        "landing_url": "https://example.com/src1",
                        "started_at": "2024-01-01T00:00:00+00:00",
                        "finished_at": "2024-01-01T00:00:01+00:00",
                        "notes": [],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            (source_dir / "manual_review.json").write_text("{}", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(["tidy-results", "--results-root", str(root)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["sources"][0]["source_id"], "src1")
            self.assertTrue((source_dir / "metadata" / "manual_review.json").exists())

    def test_normalize_sources_command_prints_summary(self) -> None:
        with patch.object(cli, "normalize_sources", return_value={"normalized_spectra": 3}) as mock_normalize_sources:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "normalize-sources",
                        "--manifest",
                        "manifests/sources.csv",
                        "--results-root",
                        "build/local_sources",
                        "--output-root",
                        "build/normalized",
                        "--source-ids",
                        "src1,src2",
                        "--limit",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["normalized_spectra"], 3)
            self.assertTrue(mock_normalize_sources.called)

    def test_plot_quality_command_prints_summary(self) -> None:
        with patch.object(cli, "generate_quality_plots", return_value={"source_count": 4}) as mock_generate_quality_plots:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "plot-quality",
                        "--normalized-root",
                        "build/normalized_full_pass_with_usgs",
                        "--output-root",
                        "build/normalized_full_pass_with_usgs/plots",
                        "--top-n-sources",
                        "12",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["source_count"], 4)
            self.assertTrue(mock_generate_quality_plots.called)

    def test_filter_coverage_command_prints_summary(self) -> None:
        with patch.object(cli, "filter_normalized_by_coverage", return_value={"retained_spectra": 7}) as mock_filter:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "filter-coverage",
                        "--normalized-root",
                        "build/normalized_full_pass_with_usgs",
                        "--output-root",
                        "build/normalized_full_pass_with_usgs_cov80",
                        "--min-coverage",
                        "0.8",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["retained_spectra"], 7)
            self.assertTrue(mock_filter.called)

    def test_build_library_package_command_prints_summary(self) -> None:
        with patch.object(cli, "build_library_package", return_value={"classified_spectra": 9}) as mock_build_library_package:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "build-library-package",
                        "--manifest",
                        "manifests/sources.csv",
                        "--normalized-root",
                        "build/normalized_rebuild_v9_final",
                        "--output-root",
                        "build/siac_spectral_library_v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["classified_spectra"], 9)
            self.assertTrue(mock_build_library_package.called)
            self.assertEqual(mock_build_library_package.call_args.kwargs["exclude_source_ids"], [])
            self.assertIsNone(mock_build_library_package.call_args.kwargs["exclude_spectra_csv"])

    def test_build_siac_library_command_alias_prints_summary(self) -> None:
        with patch.object(cli, "build_library_package", return_value={"classified_spectra": 9}) as mock_build_library_package:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "build-siac-library",
                        "--manifest",
                        "manifests/sources.csv",
                        "--normalized-root",
                        "build/normalized_rebuild_v9_final",
                        "--output-root",
                        "build/siac_spectral_library_v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["classified_spectra"], 9)
            self.assertTrue(mock_build_library_package.called)

    def test_internal_build_siac_library_command_alias_prints_summary(self) -> None:
        with patch.object(cli, "build_library_package", return_value={"classified_spectra": 9}) as mock_build_library_package:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "internal",
                        "build-siac-library",
                        "--manifest",
                        "manifests/sources.csv",
                        "--normalized-root",
                        "build/normalized_rebuild_v9_final",
                        "--output-root",
                        "build/siac_spectral_library_v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["classified_spectra"], 9)
            self.assertTrue(mock_build_library_package.called)

    def test_build_library_package_command_passes_excluded_sources(self) -> None:
        with patch.object(cli, "build_library_package", return_value={"classified_spectra": 8}) as mock_build_library_package:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "build-library-package",
                        "--manifest",
                        "manifests/sources.csv",
                        "--normalized-root",
                        "build/normalized_rebuild_v22_native_410_2490",
                        "--output-root",
                        "build/siac_spectral_library_v14_native_410_2490_no_ghisacasia",
                        "--exclude-source-ids",
                        "ghisacasia_v001",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["classified_spectra"], 8)
            self.assertEqual(mock_build_library_package.call_args.kwargs["exclude_source_ids"], ["ghisacasia_v001"])

    def test_build_library_package_command_passes_excluded_spectra_csv(self) -> None:
        with patch.object(cli, "build_library_package", return_value={"classified_spectra": 7}) as mock_build_library_package:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "build-library-package",
                        "--manifest",
                        "manifests/sources.csv",
                        "--normalized-root",
                        "build/normalized_rebuild_v22_native_410_2490",
                        "--output-root",
                        "build/siac_spectral_library_v14_native_410_2490_no_ghisacasia",
                        "--exclude-spectra-csv",
                        "manifests/siac_excluded_spectra.csv",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["classified_spectra"], 7)
            self.assertEqual(
                mock_build_library_package.call_args.kwargs["exclude_spectra_csv"],
                Path("manifests/siac_excluded_spectra.csv"),
            )

    def test_main_without_subcommand_raises_system_exit(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main_with_args([])

    def test_main_delegates_to_main_with_args(self) -> None:
        with patch.object(cli, "main_with_args", return_value=7) as mock_main_with_args:
            self.assertEqual(cli.main(), 7)
        mock_main_with_args.assert_called_once_with()

    def test_build_parser_defaults_and_version_import(self) -> None:
        parser = cli.build_internal_parser()
        args = parser.parse_args(["plan-matrix"])
        self.assertEqual(args.manifest, str(cli.DEFAULT_MANIFEST))
        self.assertEqual(__version__, "0.2.0")

    def test_internal_parser_hides_legacy_build_siac_library_alias_from_help(self) -> None:
        parser = cli.build_internal_parser()
        subparsers_action = next(
            action for action in parser._actions if isinstance(action, cli.argparse._SubParsersAction)
        )
        self.assertNotIn("build-siac-library", subparsers_action.choices)


if __name__ == "__main__":
    unittest.main()
