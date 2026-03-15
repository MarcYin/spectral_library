from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np

import spectral_library.mapping as mapping_module
from spectral_library import (
    MappingInputError,
    PreparedLibraryCompatibilityError,
    PreparedLibraryManifest,
    PreparedLibraryValidationError,
    SensorSRFSchema,
    SpectralMapper,
    benchmark_mapping,
    cli,
    prepare_mapping_library,
)
from spectral_library.mapping import PreparedLibraryBuildError, SensorSchemaError


WAVELENGTHS = list(range(400, 2501))
NM_COLUMNS = [f"nm_{wavelength}" for wavelength in WAVELENGTHS]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _spectrum_values(vnir: float, overlap: float, swir: float) -> dict[str, float]:
    row: dict[str, float] = {}
    for wavelength in WAVELENGTHS:
        if wavelength < 900:
            value = vnir
        elif wavelength <= 1000:
            value = overlap
        else:
            value = swir
        row[f"nm_{wavelength}"] = value
    return row


def _build_fixture(root: Path) -> dict[str, Path]:
    siac_root = root / "siac"
    srf_root = root / "srfs"
    prepared_root = root / "prepared"

    metadata_rows = [
        {
            "source_id": "fixture_source",
            "spectrum_id": "base",
            "sample_name": "base",
            "source_name": "Fixture Source",
            "landcover_group": "soil",
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "vnir_high",
            "sample_name": "vnir_high",
            "source_name": "Fixture Source",
            "landcover_group": "soil",
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "swir_high",
            "sample_name": "swir_high",
            "source_name": "Fixture Source",
            "landcover_group": "soil",
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "mid",
            "sample_name": "mid",
            "source_name": "Fixture Source",
            "landcover_group": "soil",
        },
    ]
    spectra_rows = [
        {
            "source_id": "fixture_source",
            "spectrum_id": "base",
            "sample_name": "base",
            **_spectrum_values(0.15, 0.25, 0.25),
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "vnir_high",
            "sample_name": "vnir_high",
            **_spectrum_values(0.80, 0.40, 0.20),
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "swir_high",
            "sample_name": "swir_high",
            **_spectrum_values(0.10, 0.90, 0.90),
        },
        {
            "source_id": "fixture_source",
            "spectrum_id": "mid",
            "sample_name": "mid",
            **_spectrum_values(0.60, 0.60, 0.60),
        },
    ]

    _write_csv(
        siac_root / "tabular" / "siac_spectra_metadata.csv",
        ["source_id", "spectrum_id", "sample_name", "source_name", "landcover_group"],
        metadata_rows,
    )
    _write_csv(
        siac_root / "tabular" / "siac_normalized_spectra.csv",
        ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
        spectra_rows,
    )

    source_sensor = {
        "sensor_id": "sensor_a",
        "bands": [
            {
                "band_id": "blue",
                "segment": "vnir",
                "wavelength_nm": [445.0, 450.0, 455.0],
                "rsr": [0.2, 1.0, 0.2],
            },
            {
                "band_id": "swir",
                "segment": "swir",
                "wavelength_nm": [1595.0, 1600.0, 1605.0],
                "rsr": [0.2, 1.0, 0.2],
            },
        ],
    }
    target_sensor = {
        "sensor_id": "sensor_b",
        "bands": [
            {
                "band_id": "target_vnir",
                "segment": "vnir",
                "wavelength_nm": [495.0, 500.0, 505.0],
                "rsr": [0.2, 1.0, 0.2],
            },
            {
                "band_id": "target_swir",
                "segment": "swir",
                "wavelength_nm": [1695.0, 1700.0, 1705.0],
                "rsr": [0.2, 1.0, 0.2],
            },
        ],
    }
    srf_root.mkdir(parents=True, exist_ok=True)
    (srf_root / "sensor_a.json").write_text(json.dumps(source_sensor, indent=2) + "\n", encoding="utf-8")
    (srf_root / "sensor_b.json").write_text(json.dumps(target_sensor, indent=2) + "\n", encoding="utf-8")

    return {
        "siac_root": siac_root,
        "srf_root": srf_root,
        "prepared_root": prepared_root,
    }


def _prepare_fixture(root: Path) -> tuple[dict[str, Path], PreparedLibraryManifest]:
    fixture = _build_fixture(root)
    manifest = prepare_mapping_library(
        fixture["siac_root"],
        fixture["srf_root"],
        fixture["prepared_root"],
        ["sensor_a"],
    )
    return fixture, manifest


class MappingWorkflowTests(unittest.TestCase):
    def test_prepare_mapping_library_writes_runtime_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, manifest = _prepare_fixture(Path(tmpdir))

            self.assertEqual(manifest.schema_version, "1.0.0")
            self.assertEqual(manifest.source_sensors, ("sensor_a",))
            self.assertEqual(manifest.row_count, 4)
            self.assertEqual(manifest.supported_output_modes, ("target_sensor", "vnir_spectrum", "swir_spectrum", "full_spectrum"))
            self.assertTrue((fixture["prepared_root"] / "manifest.json").exists())
            self.assertTrue((fixture["prepared_root"] / "mapping_metadata.parquet").exists())
            self.assertTrue((fixture["prepared_root"] / "sensor_schema.json").exists())
            self.assertTrue((fixture["prepared_root"] / "checksums.json").exists())

            self.assertEqual(np.load(fixture["prepared_root"] / "hyperspectral_vnir.npy").shape, (4, 601))
            self.assertEqual(np.load(fixture["prepared_root"] / "hyperspectral_swir.npy").shape, (4, 1601))
            self.assertEqual(np.load(fixture["prepared_root"] / "source_sensor_a_vnir.npy").shape, (4, 1))
            self.assertEqual(np.load(fixture["prepared_root"] / "source_sensor_a_swir.npy").shape, (4, 1))

            connection = duckdb.connect()
            try:
                row_count = connection.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [str(fixture["prepared_root"] / "mapping_metadata.parquet")],
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(row_count, 4)

            round_trip_manifest = PreparedLibraryManifest.from_json(fixture["prepared_root"] / "manifest.json")
            self.assertEqual(round_trip_manifest.to_dict(), manifest.to_dict())
            checksums = json.loads((fixture["prepared_root"] / "checksums.json").read_text(encoding="utf-8"))
            self.assertIn("manifest.json", checksums["files"])
            self.assertEqual(checksums["files"]["hyperspectral_vnir.npy"], manifest.file_checksums["hyperspectral_vnir.npy"])

    def test_spectral_mapper_identity_retrieval_matches_when_source_equals_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
            )

            self.assertTrue(np.allclose(result.target_reflectance, np.array([0.80, 0.20])))
            self.assertEqual(result.neighbor_ids_by_segment["vnir"], ("fixture_source:vnir_high",))
            self.assertEqual(result.neighbor_ids_by_segment["swir"], ("fixture_source:vnir_high",))

    def test_full_spectrum_output_blends_vnir_and_swir_segment_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="full_spectrum",
                k=1,
            )

            self.assertEqual(result.neighbor_ids_by_segment["vnir"], ("fixture_source:vnir_high",))
            self.assertEqual(result.neighbor_ids_by_segment["swir"], ("fixture_source:swir_high",))
            self.assertIsNotNone(result.reconstructed_full_spectrum)
            full_spectrum = result.reconstructed_full_spectrum
            assert full_spectrum is not None
            self.assertAlmostEqual(full_spectrum[450 - 400], 0.80, places=6)
            self.assertAlmostEqual(full_spectrum[900 - 400], 0.40, places=6)
            self.assertAlmostEqual(full_spectrum[950 - 400], 0.65, places=6)
            self.assertAlmostEqual(full_spectrum[1000 - 400], 0.90, places=6)
            self.assertAlmostEqual(full_spectrum[1600 - 400], 0.90, places=6)

    def test_full_spectrum_requires_both_segments_when_valid_mask_removes_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.80, "swir": 0.90},
                    valid_mask={"blue": True, "swir": False},
                    output_mode="full_spectrum",
                    k=1,
                )

    def test_benchmark_mapping_writes_target_and_spectral_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            report = benchmark_mapping(
                fixture["prepared_root"],
                "sensor_a",
                "sensor_b",
                k=1,
                test_fraction=0.25,
                random_seed=0,
            )

            self.assertEqual(report["source_sensor_id"], "sensor_a")
            self.assertEqual(report["target_sensor_id"], "sensor_b")
            self.assertEqual(report["target_sensor"]["band_ids"], ["target_vnir", "target_swir"])
            self.assertIn("retrieval", report["full_spectrum"])
            self.assertIn("regression_baseline", report["target_sensor"])

    def test_segment_isolation_keeps_swir_output_stable_when_only_vnir_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            base = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.15, "swir": 0.90},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )
            changed_vnir = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            assert base.target_reflectance is not None
            assert changed_vnir.target_reflectance is not None
            self.assertNotEqual(base.neighbor_ids_by_segment["vnir"], changed_vnir.neighbor_ids_by_segment["vnir"])
            self.assertEqual(base.neighbor_ids_by_segment["swir"], changed_vnir.neighbor_ids_by_segment["swir"])
            self.assertNotEqual(float(base.target_reflectance[0]), float(changed_vnir.target_reflectance[0]))
            self.assertEqual(float(base.target_reflectance[1]), float(changed_vnir.target_reflectance[1]))

    def test_target_sensor_output_can_emit_only_successful_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                valid_mask={"blue": True, "swir": False},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            self.assertEqual(result.target_band_ids, ("target_vnir",))
            assert result.target_reflectance is not None
            self.assertEqual(result.target_reflectance.shape, (1,))
            self.assertEqual(result.diagnostics["segments"]["swir"]["status"], "insufficient_valid_bands")

    def test_vnir_and_swir_output_modes_return_expected_wavelength_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            vnir_result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="vnir_spectrum",
                k=1,
            )
            swir_result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="swir_spectrum",
                k=1,
            )

            self.assertEqual(int(vnir_result.reconstructed_wavelength_nm[0]), 400)
            self.assertEqual(int(vnir_result.reconstructed_wavelength_nm[-1]), 1000)
            self.assertEqual(int(swir_result.reconstructed_wavelength_nm[0]), 900)
            self.assertEqual(int(swir_result.reconstructed_wavelength_nm[-1]), 2500)
            self.assertIsNotNone(vnir_result.reconstructed_vnir)
            self.assertIsNotNone(swir_result.reconstructed_swir)

    def test_map_reflectance_accepts_array_input_and_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper._map_reflectance_internal(
                source_sensor="sensor_a",
                reflectance=np.array([0.80, 0.20]),
                valid_mask=np.array([True, True]),
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=2,
                min_valid_bands=1,
                candidate_row_indices=[0, 1],
            )

            self.assertEqual(set(result.neighbor_ids_by_segment["vnir"]), {"fixture_source:base", "fixture_source:vnir_high"})
            self.assertEqual(set(result.neighbor_ids_by_segment["swir"]), {"fixture_source:base", "fixture_source:vnir_high"})

    def test_map_reflectance_validates_shapes_and_required_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(source_sensor="sensor_a", reflectance=[0.1], output_mode="vnir_spectrum")
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    valid_mask=[True],
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.1, "swir": 0.2},
                    valid_mask=[True, True],
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(source_sensor="sensor_a", reflectance=[0.1, 0.2], output_mode="target_sensor")
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    output_mode="vnir_spectrum",
                    k=0,
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    output_mode="vnir_spectrum",
                    min_valid_bands=0,
                )

    def test_map_reflectance_validates_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper._map_reflectance_internal(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    valid_mask=None,
                    output_mode="vnir_spectrum",
                    target_sensor=None,
                    k=1,
                    min_valid_bands=1,
                    candidate_row_indices=[],
                )
            with self.assertRaises(MappingInputError):
                mapper._map_reflectance_internal(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    valid_mask=None,
                    output_mode="vnir_spectrum",
                    target_sensor=None,
                    k=1,
                    min_valid_bands=1,
                    candidate_row_indices=[-1],
                )

    def test_benchmark_mapping_validates_test_fraction_and_minimum_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            with self.assertRaises(MappingInputError):
                benchmark_mapping(fixture["prepared_root"], "sensor_a", "sensor_b", test_fraction=0.0)

            one_row_root = Path(tmpdir) / "one_row"
            one_row_fixture = _build_fixture(one_row_root)
            metadata_path = one_row_fixture["siac_root"] / "tabular" / "siac_spectra_metadata.csv"
            spectra_path = one_row_fixture["siac_root"] / "tabular" / "siac_normalized_spectra.csv"
            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                metadata_rows = list(csv.DictReader(handle))
            with spectra_path.open("r", encoding="utf-8", newline="") as handle:
                spectra_rows = list(csv.DictReader(handle))
            _write_csv(metadata_path, list(metadata_rows[0].keys()), metadata_rows[:1])
            _write_csv(spectra_path, list(spectra_rows[0].keys()), spectra_rows[:1])
            prepare_mapping_library(
                one_row_fixture["siac_root"],
                one_row_fixture["srf_root"],
                one_row_fixture["prepared_root"],
                ["sensor_a"],
            )
            with self.assertRaises(MappingInputError):
                benchmark_mapping(one_row_fixture["prepared_root"], "sensor_a", "sensor_b", test_fraction=0.5)

    def test_benchmark_mapping_handles_high_test_fraction_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            report = benchmark_mapping(
                fixture["prepared_root"],
                "sensor_a",
                "sensor_b",
                k=1,
                test_fraction=0.99,
                random_seed=0,
            )
            self.assertEqual(report["train_rows"], 1)
            self.assertEqual(report["test_rows"], 3)


class MappingValidationTests(unittest.TestCase):
    def test_base_error_to_dict_and_string(self) -> None:
        error = mapping_module.SpectralLibraryError("example", "Example error", context={"a": 1})
        self.assertEqual(str(error), "Example error")
        self.assertEqual(error.to_dict(command="map-reflectance")["command"], "map-reflectance")
        self.assertEqual(error.to_dict()["context"], {"a": 1})

    def test_optional_float_and_output_mode_helpers_validate_inputs(self) -> None:
        self.assertIsNone(mapping_module._optional_float(None))
        self.assertIsNone(mapping_module._optional_float(""))
        self.assertEqual(mapping_module._optional_float("1.5"), 1.5)
        with self.assertRaises(MappingInputError):
            mapping_module._ensure_supported_output_mode("unknown")
        self.assertEqual(mapping_module._normalized_source_sensors(["sensor_a", "sensor_a", " sensor_b "]), ["sensor_a", "sensor_b"])
        with self.assertRaises(PreparedLibraryBuildError):
            mapping_module._normalized_source_sensors([])

    def test_sensor_schema_round_trip_and_band_sorting(self) -> None:
        schema = SensorSRFSchema.from_dict(
            {
                "sensor_id": "sorted_sensor",
                "bands": [
                    {
                        "band_id": "b1",
                        "segment": "vnir",
                        "wavelength_nm": [455.0, 445.0, 450.0],
                        "rsr": [0.1, 0.2, 1.0],
                    }
                ],
            }
        )

        self.assertEqual(schema.band_ids(), ("b1",))
        self.assertEqual(schema.bands[0].wavelength_nm, (445.0, 450.0, 455.0))
        self.assertEqual(schema.to_dict()["sensor_id"], "sorted_sensor")

    def test_sensor_schema_validation_is_detailed(self) -> None:
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="",
                segment="vnir",
                wavelength_nm=(445.0,),
                rsr=(1.0,),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="mid",
                wavelength_nm=(445.0,),
                rsr=(1.0,),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(445.0, 450.0),
                rsr=(1.0,),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(),
                rsr=(),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(445.0,),
                rsr=(float("nan"),),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(450.0, 445.0),
                rsr=(1.0, 0.5),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(445.0,),
                rsr=(0.0,),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition(
                band_id="b1",
                segment="vnir",
                wavelength_nm=(1200.0,),
                rsr=(1.0,),
            )
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict({})
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict({"band_id": "b1"})
        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict({"band_id": "b1", "segment": "vnir"})

        band = mapping_module.SensorBandDefinition.from_dict(
            {
                "band_id": "b1",
                "segment": "vnir",
                "wavelength_nm": [445.0, 450.0],
                "rsr": [0.2, 1.0],
                "center_nm": 450.0,
                "fwhm_nm": 10.0,
                "support_min_nm": "",
                "support_max_nm": "",
            }
        )
        band_payload = band.to_dict()
        self.assertEqual(band_payload["center_nm"], 450.0)
        self.assertEqual(band_payload["fwhm_nm"], 10.0)
        self.assertIn("support_min_nm", band_payload)
        self.assertIn("support_max_nm", band_payload)
        valid_schema = SensorSRFSchema(sensor_id="s_valid", bands=(band,))

        with self.assertRaises(SensorSchemaError):
            SensorSRFSchema(sensor_id="", bands=(band,))
        with self.assertRaises(SensorSchemaError):
            SensorSRFSchema(sensor_id="s1", bands=())
        with self.assertRaises(SensorSchemaError):
            SensorSRFSchema(sensor_id="s1", bands=(band, band))
        with self.assertRaises(SensorSchemaError):
            SensorSRFSchema.from_dict({"bands": []})
        with self.assertRaises(SensorSchemaError):
            SensorSRFSchema.from_dict({"sensor_id": "s1"})
        with self.assertRaises(SensorSchemaError):
            valid_schema.bands_for_segment("bad")

    def test_prepare_mapping_library_rejects_invalid_dtype_and_missing_source_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            with self.assertRaises(PreparedLibraryBuildError):
                prepare_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                    dtype="int32",
                )
            with self.assertRaises(SensorSchemaError):
                prepare_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["missing_sensor"],
                )

    def test_prepare_mapping_library_rejects_invalid_sensor_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            (fixture["srf_root"] / "sensor_a.json").write_text(
                json.dumps(
                    {
                        "sensor_id": "sensor_a",
                        "bands": [
                            {
                                "band_id": "blue",
                                "segment": "vnir",
                                "wavelength_nm": [1200.0, 1205.0],
                                "rsr": [1.0, 0.5],
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SensorSchemaError):
                prepare_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                )

    def test_load_sensor_schemas_and_payload_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(SensorSchemaError):
                mapping_module.load_sensor_schemas(root / "missing")

            empty_root = root / "empty"
            empty_root.mkdir()
            with self.assertRaises(SensorSchemaError):
                mapping_module.load_sensor_schemas(empty_root)

            bad_payload_path = root / "bad_payload.json"
            bad_payload_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaises(SensorSchemaError):
                mapping_module._load_sensor_payloads(bad_payload_path)

            non_list_path = root / "non_list.json"
            non_list_path.write_text(json.dumps({"sensors": {"bad": True}}), encoding="utf-8")
            with self.assertRaises(SensorSchemaError):
                mapping_module._load_sensor_payloads(non_list_path)

            dup_root = root / "dup"
            dup_root.mkdir()
            payload = {
                "sensor_id": "dup_sensor",
                "bands": [
                    {
                        "band_id": "b1",
                        "segment": "vnir",
                        "wavelength_nm": [445.0, 450.0],
                        "rsr": [0.2, 1.0],
                    }
                ],
            }
            (dup_root / "a.json").write_text(json.dumps(payload), encoding="utf-8")
            (dup_root / "b.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SensorSchemaError):
                mapping_module.load_sensor_schemas(dup_root)

    def test_prepare_mapping_library_rejects_missing_and_extra_siac_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            metadata_path = fixture["siac_root"] / "tabular" / "siac_spectra_metadata.csv"
            spectra_path = fixture["siac_root"] / "tabular" / "siac_normalized_spectra.csv"

            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                metadata_rows = list(csv.DictReader(handle))
            _write_csv(metadata_path, list(metadata_rows[0].keys()), metadata_rows[:-1])
            with self.assertRaises(PreparedLibraryBuildError):
                prepare_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                )

            fixture = _build_fixture(Path(tmpdir))
            spectra_path = fixture["siac_root"] / "tabular" / "siac_normalized_spectra.csv"
            with spectra_path.open("r", encoding="utf-8", newline="") as handle:
                spectra_rows = list(csv.DictReader(handle))
            spectra_rows.append(
                {
                    "source_id": "fixture_source",
                    "spectrum_id": "extra",
                    "sample_name": "extra",
                    **_spectrum_values(0.2, 0.2, 0.2),
                }
            )
            _write_csv(spectra_path, list(spectra_rows[0].keys()), spectra_rows)
            with self.assertRaises(PreparedLibraryBuildError):
                prepare_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                )

    def test_load_siac_rows_validation_is_detailed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(root / "missing_metadata.csv", root / "missing_spectra.csv", dtype=np.dtype("float32"))

            metadata_path = root / "metadata.csv"
            spectra_path = root / "spectra.csv"
            _write_csv(metadata_path, ["source_id", "spectrum_id", "sample_name"], [])
            _write_csv(spectra_path, ["source_id", "spectrum_id", "sample_name", "nm_400"], [])
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            _write_csv(metadata_path, ["source_id", "spectrum_id"], [{"source_id": "s1", "spectrum_id": "a"}])
            _write_csv(spectra_path, ["source_id", "spectrum_id", "sample_name", "nm_400"], [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a", "nm_400": 0.1}])
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            _write_csv(metadata_path, ["source_id", "spectrum_id", "sample_name"], [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a"}])
            _write_csv(spectra_path, ["source_id", "spectrum_id", "sample_name"], [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a"}])
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            bad_nm_columns = ["source_id", "spectrum_id", "sample_name", "nm_401", *[f"nm_{w}" for w in range(402, 2502)]]
            bad_nm_row = {"source_id": "s1", "spectrum_id": "a", "sample_name": "a", **{column: 0.1 for column in bad_nm_columns[3:]}}
            _write_csv(spectra_path, bad_nm_columns, [bad_nm_row])
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            full_row = {"source_id": "s1", "spectrum_id": "a", "sample_name": "a", **_spectrum_values(0.1, 0.1, 0.1)}
            _write_csv(
                spectra_path,
                ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
                [full_row, full_row],
            )
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            _write_csv(metadata_path, ["source_id", "spectrum_id", "sample_name"], [{"source_id": "s1", "spectrum_id": "missing", "sample_name": "missing"}])
            _write_csv(
                spectra_path,
                ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
                [full_row],
            )
            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

    def test_spectral_mapper_rejects_incompatible_manifest_and_missing_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            manifest_path = fixture["prepared_root"] / "manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            manifest_payload["schema_version"] = "2.0.0"
            manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryCompatibilityError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            os.remove(fixture["prepared_root"] / "source_sensor_a_vnir.npy")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

    def test_spectral_mapper_rejects_missing_prepared_files_and_unknown_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(root / "missing")

            fixture, _ = _prepare_fixture(root)
            os.remove(fixture["prepared_root"] / "sensor_schema.json")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(root)
            os.remove(fixture["prepared_root"] / "mapping_metadata.parquet")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(root)
            mapper = SpectralMapper(fixture["prepared_root"])
            with self.assertRaises(SensorSchemaError):
                mapper.get_sensor_schema("missing")

    def test_spectral_mapper_rejects_shape_mismatches_in_prepared_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            np.save(fixture["prepared_root"] / "hyperspectral_vnir.npy", np.zeros((4, 600), dtype=np.float32))
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            np.save(fixture["prepared_root"] / "source_sensor_a_swir.npy", np.zeros((4, 2), dtype=np.float32))
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

    def test_internal_mapper_validation_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": float("nan"), "swir": float("nan")},
                    output_mode="vnir_spectrum",
                )

            mapper._source_matrix_cache[("sensor_a", "vnir")] = np.zeros((mapper.manifest.row_count, 2), dtype=np.float64)
            with self.assertRaises(PreparedLibraryValidationError):
                mapper._retrieve_segment(
                    source_sensor="sensor_a",
                    segment="vnir",
                    query_values=np.array([0.1]),
                    valid_mask=np.array([True]),
                    k=1,
                    min_valid_bands=1,
                    candidate_row_indices=np.array([0, 1], dtype=np.int64),
                )

            mapper = SpectralMapper(fixture["prepared_root"])
            with self.assertRaises(MappingInputError):
                mapper._retrieve_segment(
                    source_sensor="sensor_a",
                    segment="vnir",
                    query_values=np.array([0.1]),
                    valid_mask=np.array([True]),
                    k=1,
                    min_valid_bands=1,
                    candidate_row_indices=np.array([], dtype=np.int64),
                )

            self.assertIs(mapper._source_queries("sensor_a"), mapper._source_queries("sensor_a"))
            self.assertEqual(mapper._simulate_target_sensor("sensor_b", {})[1], ())
            self.assertIsNone(mapper._simulate_target_sensor("sensor_b", {})[0])
            self.assertGreater(mapper._band_response("sensor_b", mapper.get_sensor_schema("sensor_b").bands[0], segment_only=False).size, 601)

            with patch.object(mapper, "_band_response", return_value=np.zeros(601, dtype=np.float64)):
                with self.assertRaises(SensorSchemaError):
                    mapper._simulate_target_sensor("sensor_b", {"vnir": np.ones(601, dtype=np.float64)})

            with patch.object(mapping_module, "_resample_band_response", return_value=np.zeros(601, dtype=np.float64)):
                with self.assertRaises(SensorSchemaError):
                    mapping_module._simulate_segment_matrix(
                        np.ones((2, 601), dtype=np.float32),
                        mapper.get_sensor_schema("sensor_a").bands_for_segment("vnir"),
                        dtype=np.dtype("float32"),
                    )

    def test_output_mode_validation_and_target_sensor_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            sensor_c = {
                "sensor_id": "sensor_c",
                "bands": [
                    {
                        "band_id": "only_swir",
                        "segment": "swir",
                        "wavelength_nm": [1695.0, 1700.0, 1705.0],
                        "rsr": [0.2, 1.0, 0.2],
                    }
                ],
            }
            (fixture["srf_root"] / "sensor_c.json").write_text(json.dumps(sensor_c, indent=2) + "\n", encoding="utf-8")
            prepare_mapping_library(
                fixture["siac_root"],
                fixture["srf_root"],
                fixture["prepared_root"],
                ["sensor_a"],
            )
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.80, "swir": 0.90},
                    valid_mask={"blue": False, "swir": True},
                    output_mode="vnir_spectrum",
                    k=1,
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.80, "swir": 0.90},
                    valid_mask={"blue": True, "swir": False},
                    output_mode="swir_spectrum",
                    k=1,
                )
            with self.assertRaises(MappingInputError):
                mapper._map_reflectance_internal(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.80, "swir": 0.90},
                    valid_mask=None,
                    output_mode="bad_mode",
                    target_sensor=None,
                    k=1,
                    min_valid_bands=1,
                    candidate_row_indices=None,
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.80, "swir": 0.90},
                    valid_mask={"blue": True, "swir": False},
                    output_mode="target_sensor",
                    target_sensor="sensor_c",
                    k=1,
                )

    def test_benchmark_mapping_error_branches_are_exercised(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            target_band_count = len(mapper._simulate_full_sensor_matrix("sensor_b")[1])

            with patch.object(
                mapping_module.SpectralMapper,
                "_map_reflectance_internal",
                return_value=mapping_module.MappingResult(),
            ):
                with self.assertRaises(PreparedLibraryValidationError):
                    benchmark_mapping(fixture["prepared_root"], "sensor_a", "sensor_b", k=1, test_fraction=0.5)

            with patch.object(
                mapping_module.SpectralMapper,
                "_map_reflectance_internal",
                return_value=mapping_module.MappingResult(
                    target_reflectance=np.zeros(target_band_count, dtype=np.float64),
                    reconstructed_vnir=np.zeros(len(mapping_module.VNIR_WAVELENGTHS), dtype=np.float64),
                    reconstructed_swir=np.zeros(len(mapping_module.SWIR_WAVELENGTHS), dtype=np.float64),
                ),
            ):
                with self.assertRaises(PreparedLibraryValidationError):
                    benchmark_mapping(fixture["prepared_root"], "sensor_a", "sensor_b", k=1, test_fraction=0.5)


class MappingCliTests(unittest.TestCase):
    def test_prepare_map_and_benchmark_commands_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = _build_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"
            report_path = root / "benchmark.json"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": 0.80},
                    {"band_id": "swir", "reflectance": 0.20},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                prepare_exit = cli.main_with_args(
                    [
                        "prepare-mapping-library",
                        "--siac-root",
                        str(fixture["siac_root"]),
                        "--srf-root",
                        str(fixture["srf_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--output-root",
                        str(fixture["prepared_root"]),
                    ]
                )
            self.assertEqual(prepare_exit, 0)
            self.assertTrue((fixture["prepared_root"] / "manifest.json").exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                map_exit = cli.main_with_args(
                    [
                        "map-reflectance",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--target-sensor",
                        "sensor_b",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "target_sensor",
                        "--k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(map_exit, 0)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["band_id"], "target_vnir")
            self.assertEqual(rows[1]["band_id"], "target_swir")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                benchmark_exit = cli.main_with_args(
                    [
                        "benchmark-mapping",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--target-sensor",
                        "sensor_b",
                        "--k",
                        "1",
                        "--test-fraction",
                        "0.25",
                        "--random-seed",
                        "0",
                        "--report",
                        str(report_path),
                    ]
                )
            self.assertEqual(benchmark_exit, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["target_sensor_id"], "sensor_b")
            self.assertEqual(report["target_sensor"]["band_ids"], ["target_vnir", "target_swir"])

    def test_map_reflectance_command_accepts_wide_input_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query_wide.csv"
            output_path = root / "reconstructed.csv"

            _write_csv(input_path, ["blue", "swir"], [{"blue": 0.80, "swir": 0.90}])

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "vnir_spectrum",
                        "--k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["wavelength_nm"], "400")
            self.assertEqual(rows[-1]["wavelength_nm"], "1000")

    def test_map_reflectance_command_supports_valid_column_and_json_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance", "valid"],
                [
                    {"band_id": "blue", "reflectance": 0.80, "valid": "true"},
                    {"band_id": "swir", "reflectance": 0.90, "valid": "false"},
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--target-sensor",
                        "sensor_b",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "target_sensor",
                        "--k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["band_id"] for row in rows], ["target_vnir"])

            _write_csv(
                input_path,
                ["band_id", "reflectance", "valid"],
                [
                    {"band_id": "blue", "reflectance": 0.80, "valid": "maybe"},
                    {"band_id": "swir", "reflectance": 0.90, "valid": "true"},
                ],
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                error_exit = cli.main_with_args(
                    [
                        "--json-errors",
                        "map-reflectance",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--target-sensor",
                        "sensor_b",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "target_sensor",
                        "--k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(error_exit, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["command"], "map-reflectance")
            self.assertEqual(payload["error_code"], "invalid_input_csv")

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": "not-a-number"},
                    {"band_id": "swir", "reflectance": 0.90},
                ],
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                numeric_error_exit = cli.main_with_args(
                    [
                        "--json-errors",
                        "map-reflectance",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--target-sensor",
                        "sensor_b",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "target_sensor",
                        "--k",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(numeric_error_exit, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["command"], "map-reflectance")
            self.assertEqual(payload["error_code"], "invalid_input_csv")
            self.assertEqual(payload["message"], "Input reflectance CSV values must be numeric.")
            self.assertEqual(payload["context"]["band_id"], "blue")
            self.assertEqual(payload["context"]["path"], str(input_path))


if __name__ == "__main__":
    unittest.main()
