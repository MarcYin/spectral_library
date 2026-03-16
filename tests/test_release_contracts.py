from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path

import spectral_library
from spectral_library import (
    BatchMappingResult,
    MappingResult,
    PreparedLibraryCompatibilityError,
    PreparedLibraryManifest,
    SensorSRFSchema,
    SpectralMapper,
    benchmark_mapping,
    cli,
    prepare_mapping_library,
    validate_prepared_library,
)
from tests.test_mapping import _prepare_fixture


class PublicApiContractTests(unittest.TestCase):
    def test_public_exports_and_type_marker_are_stable(self) -> None:
        expected_exports = {
            "__version__",
            "BatchMappingResult",
            "MappingInputError",
            "MappingResult",
            "PreparedLibraryCompatibilityError",
            "PreparedLibraryManifest",
            "PreparedLibraryValidationError",
            "SensorSRFSchema",
            "SpectralLibraryError",
            "SpectralMapper",
            "benchmark_mapping",
            "prepare_mapping_library",
            "validate_prepared_library",
        }
        self.assertTrue(expected_exports.issubset(set(spectral_library.__all__)))
        self.assertTrue((Path(spectral_library.__file__).resolve().parent / "py.typed").exists())

    def test_public_signatures_are_stable(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(prepare_mapping_library).parameters),
            ("siac_root", "srf_root", "output_root", "source_sensors", "dtype", "knn_index_backends"),
        )
        self.assertEqual(
            tuple(inspect.signature(validate_prepared_library).parameters),
            ("prepared_root", "verify_checksums"),
        )
        self.assertEqual(
            tuple(inspect.signature(benchmark_mapping).parameters),
            (
                "prepared_root",
                "source_sensor",
                "target_sensor",
                "k",
                "test_fraction",
                "max_test_rows",
                "random_seed",
                "neighbor_estimator",
                "knn_backend",
                "knn_eps",
            ),
        )
        self.assertEqual(
            tuple(inspect.signature(SpectralMapper).parameters),
            ("prepared_root", "verify_checksums"),
        )
        self.assertEqual(
            tuple(inspect.signature(SpectralMapper.map_reflectance).parameters),
            (
                "self",
                "source_sensor",
                "reflectance",
                "valid_mask",
                "output_mode",
                "target_sensor",
                "k",
                "min_valid_bands",
                "neighbor_estimator",
                "knn_backend",
                "knn_eps",
                "exclude_row_ids",
                "exclude_sample_names",
            ),
        )
        self.assertEqual(
            tuple(inspect.signature(SpectralMapper.map_reflectance_batch).parameters),
            (
                "self",
                "source_sensor",
                "reflectance_rows",
                "valid_mask_rows",
                "sample_ids",
                "output_mode",
                "target_sensor",
                "k",
                "min_valid_bands",
                "neighbor_estimator",
                "knn_backend",
                "knn_eps",
                "exclude_row_ids",
                "exclude_sample_names",
                "exclude_row_ids_per_sample",
                "self_exclude_sample_id",
            ),
        )

    def test_public_dataclass_fields_are_stable(self) -> None:
        self.assertEqual(
            tuple(MappingResult.__dataclass_fields__),
            (
                "target_reflectance",
                "target_band_ids",
                "reconstructed_vnir",
                "reconstructed_swir",
                "reconstructed_full_spectrum",
                "reconstructed_wavelength_nm",
                "neighbor_ids_by_segment",
                "neighbor_distances_by_segment",
                "segment_outputs",
                "segment_valid_band_counts",
                "diagnostics",
            ),
        )
        self.assertEqual(tuple(BatchMappingResult.__dataclass_fields__), ("sample_ids", "results"))
        self.assertEqual(
            tuple(PreparedLibraryManifest.__dataclass_fields__),
            (
                "schema_version",
                "package_version",
                "source_siac_root",
                "source_siac_build_id",
                "prepared_at",
                "source_sensors",
                "supported_output_modes",
                "row_count",
                "vnir_wavelength_range_nm",
                "swir_wavelength_range_nm",
                "array_dtype",
                "file_checksums",
                "knn_index_artifacts",
                "interpolation_summary",
            ),
        )
        self.assertEqual(tuple(SensorSRFSchema.__dataclass_fields__), ("sensor_id", "bands"))


class CliContractTests(unittest.TestCase):
    def test_cli_parser_exposes_public_commands_and_core_flags(self) -> None:
        parser = cli.build_parser()
        self.assertIn("--json-errors", parser._option_string_actions)
        self.assertIn("--json-logs", parser._option_string_actions)
        subparsers_action = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        command_parsers = subparsers_action.choices

        self.assertTrue(
            {
                "prepare-mapping-library",
                "map-reflectance",
                "map-reflectance-batch",
                "benchmark-mapping",
                "validate-prepared-library",
            }.issubset(set(command_parsers))
        )
        self.assertFalse(set(cli.INTERNAL_COMMANDS) & set(command_parsers))

        prepare_options = {option for action in command_parsers["prepare-mapping-library"]._actions for option in action.option_strings}
        map_options = {option for action in command_parsers["map-reflectance"]._actions for option in action.option_strings}
        batch_options = {option for action in command_parsers["map-reflectance-batch"]._actions for option in action.option_strings}
        benchmark_options = {option for action in command_parsers["benchmark-mapping"]._actions for option in action.option_strings}
        backend_choices = set(command_parsers["map-reflectance"]._option_string_actions["--knn-backend"].choices)

        self.assertTrue(
            {"--siac-root", "--srf-root", "--source-sensor", "--output-root", "--knn-index-backend"}.issubset(
                prepare_options
            )
        )
        self.assertTrue(
            {
                "--prepared-root",
                "--source-sensor",
                "--target-sensor",
                "--input",
                "--output-mode",
                "--k",
                "--neighbor-estimator",
                "--knn-backend",
                "--knn-eps",
                "--output",
                "--diagnostics-output",
                "--neighbor-review-output",
                "--exclude-row-id",
                "--exclude-sample-name",
            }.issubset(map_options)
        )
        self.assertTrue(
            {
                "--prepared-root",
                "--source-sensor",
                "--target-sensor",
                "--input",
                "--output-mode",
                "--k",
                "--neighbor-estimator",
                "--knn-backend",
                "--knn-eps",
                "--output",
                "--diagnostics-output",
                "--neighbor-review-output",
                "--exclude-row-id",
                "--exclude-sample-name",
                "--self-exclude-sample-id",
            }.issubset(batch_options)
        )
        self.assertTrue(
            {
                "--prepared-root",
                "--source-sensor",
                "--target-sensor",
                "--neighbor-estimator",
                "--knn-backend",
                "--knn-eps",
                "--max-test-rows",
                "--report",
            }.issubset(
                benchmark_options
            )
        )
        self.assertTrue({"numpy", "scipy_ckdtree", "faiss", "pynndescent", "scann"}.issubset(backend_choices))

    def test_internal_cli_parser_exposes_retained_maintainer_commands(self) -> None:
        parser = cli.build_internal_parser()
        subparsers_action = next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        )
        command_parsers = subparsers_action.choices
        self.assertTrue(set(cli.INTERNAL_COMMANDS).issubset(set(command_parsers)))
        self.assertFalse(set(cli.PUBLIC_COMMANDS) & set(command_parsers))

    def test_cli_json_error_envelope_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-errors",
                        "validate-prepared-library",
                        "--prepared-root",
                        str(Path(tmpdir) / "missing"),
                    ]
                )
            self.assertEqual(exit_code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(set(payload), {"command", "context", "error_code", "message"})
            self.assertEqual(payload["command"], "validate-prepared-library")
            self.assertEqual(payload["error_code"], "invalid_prepared_library")

    def test_cli_json_log_envelope_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-logs",
                        "validate-prepared-library",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                    ]
                )
            self.assertEqual(exit_code, 0)
            payloads = [json.loads(line) for line in stderr.getvalue().splitlines() if line.strip()]
            self.assertEqual([payload["event"] for payload in payloads], ["command_started", "command_completed"])
            self.assertEqual(set(payloads[0]), {"command", "context", "event", "level", "timestamp"})
            self.assertEqual(set(payloads[1]), {"command", "context", "elapsed_ms", "event", "level", "timestamp"})
            self.assertEqual(payloads[0]["command"], "validate-prepared-library")
            self.assertEqual(payloads[1]["command"], "validate-prepared-library")
            self.assertEqual(payloads[0]["level"], "info")
            self.assertEqual(payloads[1]["level"], "info")
            self.assertIsInstance(payloads[1]["elapsed_ms"], int)
            self.assertGreaterEqual(payloads[1]["elapsed_ms"], 0)

    def test_cli_json_log_failure_event_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-logs",
                        "--json-errors",
                        "validate-prepared-library",
                        "--prepared-root",
                        str(Path(tmpdir) / "missing"),
                    ]
                )
            self.assertEqual(exit_code, 2)
            stderr_text = stderr.getvalue()
            first_newline = stderr_text.find("\n")
            second_newline = stderr_text.find("\n", first_newline + 1)
            self.assertGreaterEqual(first_newline, 0)
            self.assertGreaterEqual(second_newline, 0)
            started_payload = json.loads(stderr_text[:first_newline])
            failed_payload = json.loads(stderr_text[first_newline + 1 : second_newline])
            error_payload = json.loads(stderr_text[second_newline + 1 :])
            self.assertEqual(set(started_payload), {"command", "context", "event", "level", "timestamp"})
            self.assertEqual(set(failed_payload), {"command", "context", "elapsed_ms", "event", "level", "timestamp"})
            self.assertEqual(started_payload["event"], "command_started")
            self.assertEqual(started_payload["level"], "info")
            self.assertEqual(failed_payload["command"], "validate-prepared-library")
            self.assertEqual(failed_payload["event"], "command_failed")
            self.assertEqual(failed_payload["level"], "error")
            self.assertEqual(failed_payload["context"]["error_code"], "invalid_prepared_library")
            self.assertIsInstance(failed_payload["elapsed_ms"], int)
            self.assertGreaterEqual(failed_payload["elapsed_ms"], 0)
            self.assertEqual(error_payload["command"], "validate-prepared-library")
            self.assertEqual(error_payload["error_code"], "invalid_prepared_library")


class PreparedRuntimeContractTests(unittest.TestCase):
    def test_manifest_and_checksum_payload_keys_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, manifest = _prepare_fixture(Path(tmpdir))

            manifest_payload = json.loads((fixture["prepared_root"] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(manifest_payload),
                {
                    "schema_version",
                    "package_version",
                    "source_siac_root",
                    "source_siac_build_id",
                    "prepared_at",
                    "source_sensors",
                    "supported_output_modes",
                    "row_count",
                    "vnir_wavelength_range_nm",
                    "swir_wavelength_range_nm",
                    "array_dtype",
                    "file_checksums",
                    "knn_index_artifacts",
                    "interpolation_summary",
                },
            )
            self.assertEqual(manifest_payload["schema_version"], "1.2.0")
            self.assertEqual(set(manifest_payload["file_checksums"]), set(manifest.file_checksums))
            self.assertEqual(manifest_payload["knn_index_artifacts"], manifest.knn_index_artifacts)
            self.assertEqual(set(manifest_payload["interpolation_summary"]), set(manifest.interpolation_summary))

            checksums_payload = json.loads((fixture["prepared_root"] / "checksums.json").read_text(encoding="utf-8"))
            self.assertEqual(set(checksums_payload), {"files", "schema_version"})
            self.assertEqual(checksums_payload["schema_version"], "1.2.0")
            self.assertIn("manifest.json", checksums_payload["files"])
            self.assertIn("mapping_metadata.parquet", checksums_payload["files"])
            self.assertIn("sensor_schema.json", checksums_payload["files"])
            self.assertIn("hyperspectral_vnir.npy", checksums_payload["files"])
            self.assertIn("hyperspectral_swir.npy", checksums_payload["files"])

    def test_prepared_runtime_compatibility_error_is_public_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            manifest_path = fixture["prepared_root"] / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["schema_version"] = "2.0.0"
            manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            with self.assertRaises(PreparedLibraryCompatibilityError):
                SpectralMapper(fixture["prepared_root"])

    def test_validate_prepared_library_round_trips_public_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, manifest = _prepare_fixture(Path(tmpdir))
            validated = validate_prepared_library(fixture["prepared_root"])
            self.assertEqual(validated.to_dict(), manifest.to_dict())


if __name__ == "__main__":
    unittest.main()
