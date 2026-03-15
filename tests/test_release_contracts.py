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
            ("siac_root", "srf_root", "output_root", "source_sensors", "dtype"),
        )
        self.assertEqual(
            tuple(inspect.signature(validate_prepared_library).parameters),
            ("prepared_root", "verify_checksums"),
        )
        self.assertEqual(
            tuple(inspect.signature(benchmark_mapping).parameters),
            ("prepared_root", "source_sensor", "target_sensor", "k", "test_fraction", "random_seed"),
        )
        self.assertEqual(
            tuple(inspect.signature(SpectralMapper).parameters),
            ("prepared_root", "verify_checksums"),
        )
        self.assertEqual(
            tuple(inspect.signature(SpectralMapper.map_reflectance).parameters),
            ("self", "source_sensor", "reflectance", "valid_mask", "output_mode", "target_sensor", "k", "min_valid_bands"),
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
            ),
        )
        self.assertEqual(tuple(SensorSRFSchema.__dataclass_fields__), ("sensor_id", "bands"))


class CliContractTests(unittest.TestCase):
    def test_cli_parser_exposes_public_commands_and_core_flags(self) -> None:
        parser = cli.build_parser()
        self.assertIn("--json-errors", parser._option_string_actions)
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

        prepare_options = {option for action in command_parsers["prepare-mapping-library"]._actions for option in action.option_strings}
        map_options = {option for action in command_parsers["map-reflectance"]._actions for option in action.option_strings}
        batch_options = {option for action in command_parsers["map-reflectance-batch"]._actions for option in action.option_strings}
        benchmark_options = {option for action in command_parsers["benchmark-mapping"]._actions for option in action.option_strings}

        self.assertTrue({"--siac-root", "--srf-root", "--source-sensor", "--output-root"}.issubset(prepare_options))
        self.assertTrue({"--prepared-root", "--source-sensor", "--target-sensor", "--input", "--output-mode", "--k", "--output"}.issubset(map_options))
        self.assertTrue({"--prepared-root", "--source-sensor", "--target-sensor", "--input", "--output-mode", "--k", "--output", "--diagnostics-output"}.issubset(batch_options))
        self.assertTrue({"--prepared-root", "--source-sensor", "--target-sensor", "--report"}.issubset(benchmark_options))

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
                },
            )
            self.assertEqual(manifest_payload["schema_version"], "1.0.0")
            self.assertEqual(set(manifest_payload["file_checksums"]), set(manifest.file_checksums))

            checksums_payload = json.loads((fixture["prepared_root"] / "checksums.json").read_text(encoding="utf-8"))
            self.assertEqual(set(checksums_payload), {"files", "schema_version"})
            self.assertEqual(checksums_payload["schema_version"], "1.0.0")
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
