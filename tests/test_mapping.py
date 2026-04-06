from __future__ import annotations

import contextlib
import csv
import io
import importlib.util
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import numpy as np

import spectral_library.mapping as mapping_module
from spectral_library import __version__ as PACKAGE_VERSION
from spectral_library.distribution import RuntimeDownloadError, default_prepared_runtime_root, resolve_prepared_library_root
from spectral_library.mapping.adapters import backends as backends_module
from spectral_library.mapping.adapters import sensors as schema_module
from spectral_library.mapping.engine import runtime as runtime_module
from spectral_library import (
    BandInput,
    BatchMappingArrayResult,
    BatchMappingResult,
    HyperspectralLibraryInput,
    LinearSpectralMapper,
    MappingInputError,
    PreparedRuntime,
    PreparedLibraryCompatibilityError,
    PreparedLibraryManifest,
    PreparedLibraryValidationError,
    SensorInput,
    SensorSRFSchema,
    SpectralMapper,
    benchmark_mapping,
    cli,
    build_mapping_library,
    build_mapping_runtime,
    coerce_sensor_input,
    validate_prepared_library,
)
from spectral_library.mapping import PreparedLibraryBuildError, SensorSchemaError


WAVELENGTHS = list(range(400, 2501))
NM_COLUMNS = [f"nm_{wavelength}" for wavelength in WAVELENGTHS]
ZARR_AVAILABLE = importlib.util.find_spec("zarr") is not None
RUST_ACCEL_AVAILABLE = importlib.util.find_spec("spectral_library._mapping_rust") is not None


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
        if wavelength < 800:
            value = vnir
        elif wavelength <= 1000:
            value = overlap
        else:
            value = swir
        row[f"nm_{wavelength}"] = value
    return row


def _custom_band_payload(
    *,
    band_id: str,
    segment: str,
    wavelength_nm: list[float],
    response: list[float],
) -> dict[str, object]:
    return {
        "band_id": band_id,
        "response_definition": {
            "kind": "sampled",
            "wavelength_nm": wavelength_nm,
            "response": response,
        },
        "extensions": {
            "spectral_library": {
                "segment": segment,
            }
        },
    }


def _custom_sensor_payload(*, sensor_id: str, bands: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_type": "rsrf_sensor_definition",
        "schema_version": "1.0.0",
        "sensor_id": sensor_id,
        "bands": bands,
    }


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

    source_sensor = _custom_sensor_payload(
        sensor_id="sensor_a",
        bands=[
            _custom_band_payload(
                band_id="blue",
                segment="vnir",
                wavelength_nm=[445.0, 450.0, 455.0],
                response=[0.2, 1.0, 0.2],
            ),
            _custom_band_payload(
                band_id="swir",
                segment="swir",
                wavelength_nm=[1595.0, 1600.0, 1605.0],
                response=[0.2, 1.0, 0.2],
            ),
        ],
    )
    target_sensor = _custom_sensor_payload(
        sensor_id="sensor_b",
        bands=[
            _custom_band_payload(
                band_id="target_vnir",
                segment="vnir",
                wavelength_nm=[495.0, 500.0, 505.0],
                response=[0.2, 1.0, 0.2],
            ),
            _custom_band_payload(
                band_id="target_swir",
                segment="swir",
                wavelength_nm=[1695.0, 1700.0, 1705.0],
                response=[0.2, 1.0, 0.2],
            ),
        ],
    )
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
    manifest = build_mapping_library(
        fixture["siac_root"],
        fixture["srf_root"],
        fixture["prepared_root"],
        ["sensor_a"],
    )
    return fixture, manifest


def _in_memory_library_input() -> HyperspectralLibraryInput:
    sample_ids = ["base", "vnir_high", "swir_high", "mid"]
    spectra = np.asarray(
        [
            [float(_spectrum_values(0.15, 0.25, 0.25)[column]) for column in NM_COLUMNS],
            [float(_spectrum_values(0.80, 0.40, 0.20)[column]) for column in NM_COLUMNS],
            [float(_spectrum_values(0.10, 0.90, 0.90)[column]) for column in NM_COLUMNS],
            [float(_spectrum_values(0.60, 0.60, 0.60)[column]) for column in NM_COLUMNS],
        ],
        dtype=np.float32,
    )
    return HyperspectralLibraryInput(
        wavelengths_nm=WAVELENGTHS,
        spectra=spectra,
        sample_ids=sample_ids,
        provenance_metadata=[
            {"source_id": "fixture_source", "spectrum_id": sample_id, "sample_name": sample_id}
            for sample_id in sample_ids
        ],
    )


class MappingWorkflowTests(unittest.TestCase):
    def test_finalize_output_path_restores_existing_store_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_path = root / "existing.zarr"
            temp_path = root / "staged.zarr"
            output_path.mkdir(parents=True, exist_ok=True)
            temp_path.mkdir(parents=True, exist_ok=True)
            (output_path / "marker.txt").write_text("old", encoding="utf-8")
            (temp_path / "marker.txt").write_text("new", encoding="utf-8")

            original_replace = Path.replace

            def _flaky_replace(self: Path, target: str | Path) -> Path:
                if Path(self) == temp_path and Path(target) == output_path:
                    raise OSError("simulated finalize failure")
                return original_replace(self, target)

            with patch.object(Path, "replace", new=_flaky_replace):
                with self.assertRaises(OSError):
                    mapping_module._finalize_output_path(temp_path, output_path)

            self.assertEqual((output_path / "marker.txt").read_text(encoding="utf-8"), "old")
            self.assertTrue(temp_path.exists())
            self.assertEqual(len(list(root.glob(".existing.zarr.bak-*"))), 0)

    def test_build_mapping_library_writes_runtime_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, manifest = _prepare_fixture(Path(tmpdir))

            self.assertEqual(manifest.schema_version, "3.0.0")
            self.assertEqual(manifest.source_sensors, ("sensor_a",))
            self.assertEqual(manifest.row_count, 4)
            self.assertEqual(manifest.supported_output_modes, ("target_sensor", "vnir_spectrum", "swir_spectrum", "full_spectrum"))
            self.assertEqual(
                manifest.interpolation_summary,
                {
                    "interpolated_row_count": 0,
                    "rows_with_leading_gaps": 0,
                    "rows_with_trailing_gaps": 0,
                    "rows_with_internal_gaps": 0,
                    "max_missing_count": 0,
                    "max_leading_gap_count": 0,
                    "max_trailing_gap_count": 0,
                    "max_internal_gap_count": 0,
                    "max_internal_gap_run_count": 0,
                },
            )
            self.assertEqual(manifest.knn_index_artifacts, {})
            self.assertTrue((fixture["prepared_root"] / "manifest.json").exists())
            self.assertTrue((fixture["prepared_root"] / "mapping_metadata.parquet").exists())
            self.assertTrue((fixture["prepared_root"] / "sensor_schema.json").exists())
            self.assertTrue((fixture["prepared_root"] / "checksums.json").exists())

            sensor_schema_payload = json.loads((fixture["prepared_root"] / "sensor_schema.json").read_text(encoding="utf-8"))
            first_band = sensor_schema_payload["sensors"][0]["bands"][0]
            self.assertEqual(sensor_schema_payload["schema_version"], "3.0.0")
            self.assertEqual(sensor_schema_payload["sensors"][0]["schema_type"], "rsrf_sensor_definition")
            self.assertEqual(sensor_schema_payload["sensors"][0]["schema_version"], "1.0.0")
            self.assertIn("response_definition", first_band)
            self.assertNotIn("wavelength_nm", first_band)
            self.assertNotIn("rsr", first_band)
            self.assertNotIn("segment", first_band)
            self.assertEqual(first_band["extensions"]["spectral_library"]["segment"], "vnir")

            self.assertEqual(np.load(fixture["prepared_root"] / "hyperspectral_vnir.npy").shape, (4, 601))
            self.assertEqual(np.load(fixture["prepared_root"] / "hyperspectral_swir.npy").shape, (4, 1701))
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

    def test_build_mapping_library_can_persist_faiss_indexes(self) -> None:
        class BuildFaissIndex:
            def __init__(self, dim: int, m: int) -> None:
                del dim, m
                self.data: np.ndarray | None = None
                self.hnsw = type("FakeHNSW", (), {})()

            def add(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float32)

        class BuildFaissModule:
            IndexHNSWFlat = BuildFaissIndex

            @staticmethod
            def write_index(index: BuildFaissIndex, path: str) -> None:
                with Path(path).open("wb") as handle:
                    pickle.dump(np.asarray(index.data, dtype=np.float32), handle)

            @staticmethod
            def read_index(path: str) -> BuildFaissIndex:
                with Path(path).open("rb") as handle:
                    data = pickle.load(handle)
                index = BuildFaissIndex(int(np.asarray(data).shape[1]), 32)
                index.add(np.asarray(data, dtype=np.float32))
                return index

        class QueryFaissIndex:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float32)

            def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
                query_array = np.asarray(query, dtype=np.float32)
                squared_distances = np.sum((self.data[None, :, :] - query_array[:, None, :]) ** 2, axis=2)
                ordered = np.argsort(squared_distances, axis=1)[:, :k]
                selected = np.take_along_axis(squared_distances, ordered, axis=1)
                return selected, ordered

        class QueryFaissModule:
            @staticmethod
            def read_index(path: str) -> QueryFaissIndex:
                with Path(path).open("rb") as handle:
                    data = pickle.load(handle)
                return QueryFaissIndex(np.asarray(data, dtype=np.float32))

            class IndexHNSWFlat:
                def __init__(self, *_: object) -> None:
                    raise AssertionError("Persisted FAISS test should not rebuild an in-memory index at query time.")

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            with patch.object(backends_module, "_load_faiss_module", return_value=BuildFaissModule):
                manifest = build_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                    knn_index_backends=["faiss"],
                )

            self.assertIn("faiss", manifest.knn_index_artifacts)
            self.assertEqual(
                manifest.knn_index_artifacts["faiss"]["sensor_a"]["vnir"],
                "knn_indexes/faiss_sensor_a_vnir.faiss",
            )
            self.assertTrue((fixture["prepared_root"] / "knn_indexes" / "faiss_sensor_a_vnir.faiss").exists())
            self.assertIn(
                "knn_indexes/faiss_sensor_a_vnir.faiss",
                manifest.file_checksums,
            )

            with patch.object(backends_module, "_load_faiss_module", return_value=QueryFaissModule):
                mapper = SpectralMapper(fixture["prepared_root"])
                result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="faiss",
                )

            assert result.target_reflectance is not None
            self.assertTrue(np.allclose(result.target_reflectance, np.array([0.79, 0.21]), atol=1e-4))
            self.assertEqual(result.diagnostics["knn_backend"], "faiss")

    def test_build_mapping_library_can_persist_scann_indexes_with_tiny_candidate_sets(self) -> None:
        trace: dict[str, bool] = {
            "score_brute_force": False,
        }

        class FakeScannSearcher:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float32)

            def serialize(self, path: str) -> None:
                target = Path(path)
                target.mkdir(parents=True, exist_ok=True)
                with (target / "data.pkl").open("wb") as handle:
                    pickle.dump(self.data, handle)

            def search_batched(self, query: np.ndarray, final_num_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
                query_array = np.asarray(query, dtype=np.float32)
                k = int(final_num_neighbors or 1)
                squared_distances = np.sum((self.data[None, :, :] - query_array[:, None, :]) ** 2, axis=2)
                ordered = np.argsort(squared_distances, axis=1)[:, :k]
                selected = np.take_along_axis(squared_distances, ordered, axis=1)
                return ordered, selected

        class FakeScannBuilder:
            def __init__(self, data: np.ndarray, k: int, metric: str) -> None:
                del k, metric
                self.data = np.asarray(data, dtype=np.float32)

            def tree(self, **_: object) -> "FakeScannBuilder":
                return self

            def score_ah(self, *args: object, **kwargs: object) -> "FakeScannBuilder":
                del args, kwargs
                raise AssertionError("Tiny persisted ScaNN test should skip score_ah.")

            def score_brute_force(self) -> "FakeScannBuilder":
                trace["score_brute_force"] = True
                return self

            def reorder(self, *_: object) -> "FakeScannBuilder":
                return self

            def build(self) -> FakeScannSearcher:
                return FakeScannSearcher(self.data)

        class FakeScannOps:
            @staticmethod
            def builder(data: np.ndarray, k: int, metric: str) -> FakeScannBuilder:
                return FakeScannBuilder(data, k, metric)

            @staticmethod
            def load_searcher(path: str) -> FakeScannSearcher:
                with (Path(path) / "data.pkl").open("rb") as handle:
                    data = pickle.load(handle)
                return FakeScannSearcher(np.asarray(data, dtype=np.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            with patch.object(backends_module, "_load_scann_ops", return_value=FakeScannOps):
                manifest = build_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                    knn_index_backends=["scann"],
                )

                self.assertIn("scann", manifest.knn_index_artifacts)
                self.assertEqual(
                    manifest.knn_index_artifacts["scann"]["sensor_a"]["vnir"],
                    "knn_indexes/scann_sensor_a_vnir",
                )
                self.assertTrue((fixture["prepared_root"] / "knn_indexes" / "scann_sensor_a_vnir").exists())

                mapper = SpectralMapper(fixture["prepared_root"])
                result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="scann",
                )

            assert result.target_reflectance is not None
            self.assertTrue(np.allclose(result.target_reflectance, np.array([0.79, 0.21]), atol=1e-4))
            self.assertEqual(result.diagnostics["knn_backend"], "scann")
            self.assertTrue(trace["score_brute_force"])

    def test_spectral_mapper_identity_retrieval_matches_when_source_equals_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
            )

            self.assertTrue(np.allclose(result.target_reflectance, np.array([0.80, 0.20])))
            self.assertEqual(result.neighbor_ids_by_segment["vnir"], ("fixture_source:vnir_high:vnir_high",))
            self.assertEqual(result.neighbor_ids_by_segment["swir"], ("fixture_source:vnir_high:vnir_high",))

    def test_candidate_row_indices_can_exclude_by_row_id_and_sample_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            by_sample = mapper.candidate_row_indices(exclude_sample_names=["vnir_high"])
            by_row_id = mapper.candidate_row_indices(exclude_row_ids=["fixture_source:vnir_high:vnir_high"])

            self.assertEqual(by_sample.tolist(), [0, 2, 3])
            self.assertEqual(by_row_id.tolist(), [0, 2, 3])

            with self.assertRaises(MappingInputError):
                mapper.candidate_row_indices(exclude_sample_names=["missing_sample"])
            with self.assertRaises(MappingInputError):
                mapper.candidate_row_indices(exclude_row_ids=["fixture_source:missing:missing"])

    def test_map_reflectance_can_exclude_rows_and_sample_names_via_public_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            by_row = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
                exclude_row_ids=["fixture_source:vnir_high:vnir_high"],
            )
            by_sample = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
                exclude_sample_names=["vnir_high"],
            )

            self.assertEqual(by_row.neighbor_ids_by_segment["vnir"], ("fixture_source:mid:mid",))
            self.assertEqual(by_sample.neighbor_ids_by_segment["vnir"], ("fixture_source:mid:mid",))
            self.assertEqual(by_row.neighbor_ids_by_segment["swir"], ("fixture_source:base:base",))
            self.assertEqual(by_sample.neighbor_ids_by_segment["swir"], ("fixture_source:base:base",))

    def test_swir_retrieval_query_includes_nir_bridge_band_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = _build_fixture(root)
            source_sensor = _custom_sensor_payload(
                sensor_id="sensor_overlap",
                bands=[
                    _custom_band_payload(
                        band_id="blue",
                        segment="vnir",
                        wavelength_nm=[445.0, 450.0, 455.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="nir",
                        segment="vnir",
                        wavelength_nm=[845.0, 850.0, 855.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="swir",
                        segment="swir",
                        wavelength_nm=[1595.0, 1600.0, 1605.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                ],
            )
            (fixture["srf_root"] / "sensor_overlap.json").write_text(
                json.dumps(source_sensor, indent=2) + "\n",
                encoding="utf-8",
            )
            build_mapping_library(
                fixture["siac_root"],
                fixture["srf_root"],
                fixture["prepared_root"],
                ["sensor_overlap"],
            )
            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_debug(
                source_sensor="sensor_overlap",
                reflectance={"blue": 0.80, "nir": 0.40, "swir": 0.20},
                output_mode="swir_spectrum",
                k=1,
            )

            self.assertEqual(
                result.diagnostics["segments"]["swir"]["query_band_ids"],
                ["nir", "swir"],
            )

    def test_full_spectrum_output_blends_vnir_and_swir_segment_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="full_spectrum",
                k=1,
            )

            self.assertEqual(result.neighbor_ids_by_segment["vnir"], ("fixture_source:vnir_high:vnir_high",))
            self.assertEqual(result.neighbor_ids_by_segment["swir"], ("fixture_source:swir_high:swir_high",))
            self.assertIsNotNone(result.reconstructed_full_spectrum)
            full_spectrum = result.reconstructed_full_spectrum
            assert full_spectrum is not None
            self.assertAlmostEqual(full_spectrum[450 - 400], 0.80, places=6)
            self.assertAlmostEqual(full_spectrum[800 - 400], 0.40, places=6)
            self.assertAlmostEqual(full_spectrum[900 - 400], 0.65, places=6)
            self.assertAlmostEqual(full_spectrum[950 - 400], 0.775, places=6)
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

    def test_distance_weighted_neighbor_estimator_is_available_without_changing_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            mean_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
            )
            weighted_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="distance_weighted_mean",
            )

            assert mean_result.target_reflectance is not None
            assert weighted_result.target_reflectance is not None
            self.assertTrue(np.allclose(mean_result.target_reflectance, np.array([0.70, 0.225])))
            self.assertTrue(np.allclose(weighted_result.target_reflectance, np.array([0.79, 0.21])))
            self.assertEqual(mean_result.diagnostics["neighbor_estimator"], "mean")
            self.assertEqual(weighted_result.diagnostics["neighbor_estimator"], "distance_weighted_mean")

    @unittest.skipUnless(RUST_ACCEL_AVAILABLE, "Rust mapping acceleration module is not available.")
    def test_rust_batch_combine_supports_all_estimators(self) -> None:
        source_matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        hyperspectral_rows = np.asarray(
            [
                [0.9, 0.1, 0.2, 0.3],
                [0.1, 0.9, 0.8, 0.7],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        )
        neighbor_indices = np.asarray([[0, 1, 2]], dtype=np.int64)
        neighbor_distances = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
        query_values = np.asarray([[0.8, 0.2]], dtype=np.float64)
        valid_indices = np.asarray([0, 1], dtype=np.int64)

        for estimator in mapping_module.SUPPORTED_NEIGHBOR_ESTIMATORS:
            with self.subTest(neighbor_estimator=estimator):
                reconstructed, weights, source_fit_rmse = mapping_module._combine_neighbor_spectra_batch_accel(
                    hyperspectral_rows=hyperspectral_rows,
                    source_matrix=source_matrix,
                    neighbor_indices=neighbor_indices,
                    neighbor_distances=neighbor_distances,
                    query_values=query_values,
                    valid_indices=valid_indices,
                    neighbor_estimator=estimator,
                )
                self.assertEqual(reconstructed.shape, (1, 4))
                self.assertEqual(weights.shape, (1, 3))
                self.assertEqual(source_fit_rmse.shape, (1,))
                self.assertAlmostEqual(float(np.sum(weights[0])), 1.0, places=6)
                self.assertGreaterEqual(float(source_fit_rmse[0]), 0.0)

    def test_simplex_mixture_neighbor_estimator_can_fit_convex_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            simplex_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="simplex_mixture",
            )

            assert simplex_result.target_reflectance is not None
            self.assertTrue(np.allclose(simplex_result.target_reflectance, np.array([0.79, 0.21]), atol=1e-4))
            self.assertEqual(simplex_result.diagnostics["neighbor_estimator"], "simplex_mixture")
            self.assertLess(simplex_result.diagnostics["segments"]["vnir"]["source_fit_rmse"], 1e-4)
            self.assertLess(simplex_result.diagnostics["segments"]["swir"]["source_fit_rmse"], 1e-4)
            self.assertAlmostEqual(sum(simplex_result.diagnostics["segments"]["vnir"]["neighbor_weights"]), 1.0, places=6)
            self.assertAlmostEqual(sum(simplex_result.diagnostics["segments"]["swir"]["neighbor_weights"]), 1.0, places=6)

    def test_segment_confidence_payload_batch_matches_scalar_helper(self) -> None:
        query_matrix = np.asarray(
            [
                [0.25, 0.75, 0.50],
                [0.10, 0.20, 0.30],
                [0.00, 0.00, 0.00],
            ],
            dtype=np.float64,
        )
        neighbor_distances = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.4, 0.8],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        neighbor_weights = np.asarray(
            [
                [0.2, 0.3, 0.5],
                [0.7, 0.2, 0.1],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        source_fit_rmse = np.asarray([0.05, 0.12, 0.0], dtype=np.float64)

        batch_scores, batch_components = mapping_module._segment_confidence_payload_batch(
            query_matrix=query_matrix,
            valid_band_count=3,
            total_band_count=4,
            neighbor_distance_matrix=neighbor_distances,
            neighbor_weight_matrix=neighbor_weights,
            source_fit_rmse=source_fit_rmse,
        )

        expected_scores = []
        expected_components = []
        for row_index in range(query_matrix.shape[0]):
            score, components = mapping_module._segment_confidence_payload(
                query_vector=query_matrix[row_index],
                valid_band_count=3,
                total_band_count=4,
                neighbor_distances=neighbor_distances[row_index],
                neighbor_weights=neighbor_weights[row_index],
                source_fit_rmse=float(source_fit_rmse[row_index]),
            )
            expected_scores.append(score)
            expected_components.append(components)

        self.assertTrue(np.allclose(batch_scores, np.asarray(expected_scores, dtype=np.float64)))
        self.assertEqual(batch_components, tuple(expected_components))

    def test_combine_neighbor_spectra_batch_uses_rust_module_when_available(self) -> None:
        hyperspectral_rows = np.asarray([[0.2, 0.3], [0.7, 0.8]], dtype=np.float32)
        source_rows = np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
        neighbor_indices = np.asarray([[0, 1]], dtype=np.int64)
        neighbor_distances = np.asarray([[0.1, 0.2]], dtype=np.float64)
        query_vectors = np.asarray([[0.25, 0.45]], dtype=np.float64)
        native_result = (
            np.asarray([[0.5, 0.6]], dtype=np.float64),
            np.asarray([[0.4, 0.6]], dtype=np.float64),
            np.asarray([0.07], dtype=np.float64),
        )

        with patch.object(mapping_module._rustaccel, "combine_neighbor_spectra_batch", return_value=native_result) as native_mock:
            reconstructed, weights, source_fit_rmse = mapping_module._combine_neighbor_spectra_batch_accel(
                hyperspectral_rows=hyperspectral_rows,
                source_matrix=source_rows,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                query_values=query_vectors,
                valid_indices=None,
                neighbor_estimator="mean",
            )

        native_mock.assert_called_once()
        self.assertTrue(np.allclose(reconstructed, native_result[0]))
        self.assertTrue(np.allclose(weights, native_result[1]))
        self.assertTrue(np.allclose(source_fit_rmse, native_result[2]))

    def test_finalize_target_sensor_batch_uses_rust_module(self) -> None:
        native_result = (
            np.asarray([[0.25, np.nan, 0.75]], dtype=np.float64),
            np.asarray([0], dtype=np.int32),
        )

        with patch.object(mapping_module._rustaccel, "finalize_target_sensor_batch", return_value=native_result) as native_mock:
            output_rows, status_codes = mapping_module._finalize_target_sensor_batch_accel(
                vnir_reconstructed=np.zeros((1, 2), dtype=np.float64),
                swir_reconstructed=np.zeros((1, 1), dtype=np.float64),
                vnir_success=np.asarray([True], dtype=bool),
                swir_success=np.asarray([False], dtype=bool),
                vnir_response_matrix=np.zeros((2, 2), dtype=np.float64),
                swir_response_matrix=np.zeros((1, 1), dtype=np.float64),
                vnir_output_indices=np.asarray([0, 2], dtype=np.int64),
                swir_output_indices=np.asarray([1], dtype=np.int64),
                output_width=3,
            )

        native_mock.assert_called_once()
        self.assertTrue(np.allclose(output_rows, native_result[0], equal_nan=True))
        self.assertTrue(np.array_equal(status_codes, native_result[1]))

    def test_stitch_target_sensor_segment_rows_combines_segment_outputs(self) -> None:
        output_rows, status_codes = mapping_module._stitch_target_sensor_segment_rows(
            vnir_rows=np.asarray([[0.2, 0.3], [9.0, 9.0]], dtype=np.float64),
            swir_rows=np.asarray([[0.8], [0.9]], dtype=np.float64),
            vnir_success=np.asarray([True, False], dtype=bool),
            swir_success=np.asarray([True, True], dtype=bool),
            vnir_output_indices=np.asarray([0, 2], dtype=np.int64),
            swir_output_indices=np.asarray([1], dtype=np.int64),
            output_width=3,
        )

        self.assertTrue(
            np.allclose(
                output_rows,
                np.asarray([[0.2, 0.8, 0.3], [np.nan, 0.9, np.nan]], dtype=np.float64),
                equal_nan=True,
            )
        )
        self.assertTrue(np.array_equal(status_codes, np.asarray([0, 0], dtype=np.int32)))

    def test_refine_and_combine_neighbor_spectra_batch_uses_rust_module(self) -> None:
        native_result = (
            np.asarray([[0.5, 0.6]], dtype=np.float64),
            np.asarray([0.07], dtype=np.float64),
        )

        with patch.object(
            mapping_module._rustaccel,
            "refine_and_combine_neighbor_spectra_batch",
            return_value=native_result,
        ) as native_mock:
            reconstructed, source_fit_rmse = mapping_module._refine_and_combine_neighbor_spectra_batch_accel(
                candidate_matrix=np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                candidate_row_indices=np.asarray([0, 1], dtype=np.int64),
                source_matrix=np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                hyperspectral_rows=np.asarray([[0.2, 0.3], [0.7, 0.8]], dtype=np.float32),
                query_values=np.asarray([[0.25, 0.45]], dtype=np.float64),
                local_candidate_indices=np.asarray([[0, 1]], dtype=np.int64),
                local_candidate_distances=np.asarray([[0.1, 0.2]], dtype=np.float64),
                valid_indices=None,
                k=1,
                neighbor_estimator="mean",
            )

        native_mock.assert_called_once()
        self.assertTrue(np.allclose(reconstructed, native_result[0]))
        self.assertTrue(np.allclose(source_fit_rmse, native_result[1]))

    def test_refine_and_combine_neighbor_spectra_batch_forwards_native_output_buffers(self) -> None:
        out_reconstructed = np.empty((1, 2), dtype=np.float64)
        out_source_fit_rmse = np.empty((1,), dtype=np.float64)

        def _fill_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
            self.assertIs(kwargs["out_reconstructed"], out_reconstructed)
            self.assertIs(kwargs["out_source_fit_rmse"], out_source_fit_rmse)
            kwargs["out_reconstructed"][:] = np.asarray([[0.9, 0.4]], dtype=np.float64)
            kwargs["out_source_fit_rmse"][:] = np.asarray([0.02], dtype=np.float64)
            return kwargs["out_reconstructed"], kwargs["out_source_fit_rmse"]

        with patch.object(
            mapping_module._rustaccel,
            "refine_and_combine_neighbor_spectra_batch",
            side_effect=_fill_outputs,
        ) as native_mock:
            reconstructed, source_fit_rmse = mapping_module._refine_and_combine_neighbor_spectra_batch_accel(
                candidate_matrix=np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                candidate_row_indices=np.asarray([0, 1], dtype=np.int64),
                source_matrix=np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
                hyperspectral_rows=np.asarray([[0.2, 0.3], [0.7, 0.8]], dtype=np.float32),
                query_values=np.asarray([[0.25, 0.45]], dtype=np.float64),
                local_candidate_indices=np.asarray([[0, 1]], dtype=np.int64),
                local_candidate_distances=np.asarray([[0.1, 0.2]], dtype=np.float64),
                valid_indices=None,
                k=1,
                neighbor_estimator="mean",
                out_reconstructed=out_reconstructed,
                out_source_fit_rmse=out_source_fit_rmse,
            )

        native_mock.assert_called_once()
        self.assertIs(reconstructed, out_reconstructed)
        self.assertIs(source_fit_rmse, out_source_fit_rmse)

    def test_scipy_ckdtree_knn_backend_can_match_numpy_backend(self) -> None:
        build_shapes: list[tuple[int, ...]] = []
        worker_values: list[int] = []

        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                build_shapes.append(tuple(np.asarray(data).shape))
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps
                worker_values.append(int(workers))
                query_array = np.asarray(query, dtype=np.float64)
                if query_array.ndim == 1:
                    distances = np.linalg.norm(self.data - query_array[None, :], axis=1)
                    ordered = np.argsort(distances)[:k]
                    return distances[ordered], ordered
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            numpy_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="simplex_mixture",
            )
            with patch.dict(os.environ, {"SPECTRAL_LIBRARY_SCIPY_WORKERS": "-1"}, clear=True):
                with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree):
                    scipy_result = mapper.map_reflectance_debug(
                        source_sensor="sensor_a",
                        reflectance={"blue": 0.79, "swir": 0.21},
                        output_mode="target_sensor",
                        target_sensor="sensor_b",
                        k=2,
                        neighbor_estimator="simplex_mixture",
                        knn_backend="scipy_ckdtree",
                        knn_eps=0.1,
                    )
                    mapper.map_reflectance_debug(
                        source_sensor="sensor_a",
                        reflectance={"blue": 0.79, "swir": 0.21},
                        output_mode="target_sensor",
                        target_sensor="sensor_b",
                        k=2,
                        neighbor_estimator="simplex_mixture",
                        knn_backend="scipy_ckdtree",
                        knn_eps=0.1,
                    )

            assert numpy_result.target_reflectance is not None
            assert scipy_result.target_reflectance is not None
            self.assertTrue(np.allclose(numpy_result.target_reflectance, scipy_result.target_reflectance, atol=1e-6))
            self.assertEqual(scipy_result.diagnostics["knn_backend"], "scipy_ckdtree")
            self.assertAlmostEqual(scipy_result.diagnostics["knn_eps"], 0.1)
            self.assertEqual(build_shapes, [(4, 1), (4, 1)])
            self.assertEqual(worker_values, [-1, -1, -1, -1])

    def test_scipy_ckdtree_knn_backend_matches_numpy_for_masked_query(self) -> None:
        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps, workers
                query_array = np.asarray(query, dtype=np.float64)
                if query_array.ndim == 1:
                    distances = np.linalg.norm(self.data - query_array[None, :], axis=1)
                    ordered = np.argsort(distances)[:k]
                    return distances[ordered], ordered
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            numpy_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                valid_mask={"blue": True, "swir": False},
                output_mode="vnir_spectrum",
                k=2,
            )
            with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree):
                scipy_result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    valid_mask={"blue": True, "swir": False},
                    output_mode="vnir_spectrum",
                    k=2,
                    knn_backend="scipy_ckdtree",
                )

            assert numpy_result.reconstructed_vnir is not None
            assert scipy_result.reconstructed_vnir is not None
            self.assertTrue(np.allclose(numpy_result.reconstructed_vnir, scipy_result.reconstructed_vnir, atol=1e-6))

    def test_scipy_ckdtree_batch_backend_matches_numpy_for_distinct_valid_masks(self) -> None:
        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps, workers
                query_array = np.asarray(query, dtype=np.float64)
                if query_array.ndim == 1:
                    distances = np.linalg.norm(self.data - query_array[None, :], axis=1)
                    ordered = np.argsort(distances)[:k]
                    return distances[ordered], ordered
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            numpy_result = mapper.map_reflectance_batch_debug(
                source_sensor="sensor_a",
                reflectance_rows=[
                    {"blue": 0.79, "swir": 0.21},
                    {"blue": 0.11, "swir": 0.89},
                ],
                valid_mask_rows=[
                    {"blue": True, "swir": False},
                    {"blue": False, "swir": True},
                ],
                sample_ids=["alpha", "beta"],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
            )
            with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree):
                scipy_result = mapper.map_reflectance_batch_debug(
                    source_sensor="sensor_a",
                    reflectance_rows=[
                        {"blue": 0.79, "swir": 0.21},
                        {"blue": 0.11, "swir": 0.89},
                    ],
                    valid_mask_rows=[
                        {"blue": True, "swir": False},
                        {"blue": False, "swir": True},
                    ],
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    knn_backend="scipy_ckdtree",
                )

            for numpy_row, scipy_row in zip(numpy_result.results, scipy_result.results):
                assert numpy_row.target_reflectance is not None
                assert scipy_row.target_reflectance is not None
                self.assertTrue(np.allclose(numpy_row.target_reflectance, scipy_row.target_reflectance, atol=1e-6))

    def test_scipy_ckdtree_exact_backend_skips_single_sample_python_rerank(self) -> None:
        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps, workers
                query_array = np.asarray(query, dtype=np.float64)
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree), patch.object(
                runtime_module,
                "_refine_neighbor_rows",
                side_effect=AssertionError("exact cKDTree path should not rerank in Python"),
            ):
                result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    knn_backend="scipy_ckdtree",
                    knn_eps=0.0,
                )

            assert result.target_reflectance is not None
            self.assertEqual(result.target_reflectance.shape, (2,))

    def test_faiss_pynndescent_and_scann_backends_can_match_numpy_backend(self) -> None:
        class FakeFaissIndex:
            def __init__(self, dim: int, m: int) -> None:
                del dim, m
                self.data: np.ndarray | None = None
                self.hnsw = type("FakeHNSW", (), {})()

            def add(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float32)

            def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
                assert self.data is not None
                query_array = np.asarray(query, dtype=np.float32)
                squared_distances = np.sum((self.data[None, :, :] - query_array[:, None, :]) ** 2, axis=2)
                ordered = np.argsort(squared_distances, axis=1)[:, :k]
                selected = np.take_along_axis(squared_distances, ordered, axis=1)
                return selected, ordered

        class FakeFaissModule:
            IndexHNSWFlat = FakeFaissIndex

        class FakeNNDescent:
            def __init__(self, data: np.ndarray, metric: str = "euclidean") -> None:
                del metric
                self.data = np.asarray(data, dtype=np.float32)

            def query(self, query: np.ndarray, k: int, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
                del epsilon
                query_array = np.asarray(query, dtype=np.float32)
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return ordered, selected

        class FakeScannSearcher:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float32)

            def search_batched(self, query: np.ndarray, final_num_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
                query_array = np.asarray(query, dtype=np.float32)
                k = int(final_num_neighbors or 1)
                squared_distances = np.sum((self.data[None, :, :] - query_array[:, None, :]) ** 2, axis=2)
                ordered = np.argsort(squared_distances, axis=1)[:, :k]
                selected = np.take_along_axis(squared_distances, ordered, axis=1)
                return ordered, selected

        class FakeScannBuilder:
            def __init__(self, data: np.ndarray, k: int, metric: str) -> None:
                del k, metric
                self.data = np.asarray(data, dtype=np.float32)

            def tree(self, **_: object) -> "FakeScannBuilder":
                return self

            def score_ah(self, *args: object, **kwargs: object) -> "FakeScannBuilder":
                del args, kwargs
                return self

            def reorder(self, *_: object) -> "FakeScannBuilder":
                return self

            def build(self) -> FakeScannSearcher:
                return FakeScannSearcher(self.data)

        class FakeScannOps:
            @staticmethod
            def builder(data: np.ndarray, k: int, metric: str) -> FakeScannBuilder:
                return FakeScannBuilder(data, k, metric)

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            numpy_result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="simplex_mixture",
            )

            with patch.object(backends_module, "_load_faiss_module", return_value=FakeFaissModule):
                faiss_result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="faiss",
                    knn_eps=0.2,
                )
            with patch.object(backends_module, "_load_pynndescent_class", return_value=FakeNNDescent):
                pynndescent_result = mapper.map_reflectance_batch_debug(
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.79, "swir": 0.21}, {"blue": 0.15, "swir": 0.25}],
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="pynndescent",
                    knn_eps=0.2,
                )
            with patch.object(backends_module, "_load_scann_ops", return_value=FakeScannOps()):
                scann_result = mapper.map_reflectance_debug(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="scann",
                    knn_eps=0.2,
                )

            assert numpy_result.target_reflectance is not None
            assert faiss_result.target_reflectance is not None
            assert scann_result.target_reflectance is not None
            self.assertTrue(np.allclose(numpy_result.target_reflectance, faiss_result.target_reflectance, atol=1e-6))
            self.assertTrue(np.allclose(numpy_result.target_reflectance, scann_result.target_reflectance, atol=1e-6))
            self.assertEqual(faiss_result.diagnostics["knn_backend"], "faiss")
            self.assertEqual(scann_result.diagnostics["knn_backend"], "scann")
            self.assertEqual(
                [result.diagnostics["knn_backend"] for result in pynndescent_result.results],
                ["pynndescent", "pynndescent"],
            )

    def test_knn_backend_validation_and_missing_scipy_error_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    knn_backend="unsupported_backend",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    knn_eps=-0.1,
                )
            with patch.object(
                backends_module,
                "_load_ckdtree_class",
                side_effect=MappingInputError(
                    "scipy_ckdtree backend requires scipy.",
                    context={"knn_backend": "scipy_ckdtree"},
                ),
            ):
                with self.assertRaises(MappingInputError):
                    mapper.map_reflectance(
                        source_sensor="sensor_a",
                        reflectance={"blue": 0.79, "swir": 0.21},
                        output_mode="target_sensor",
                        target_sensor="sensor_b",
                        k=2,
                        knn_backend="scipy_ckdtree",
                    )
            for loader_name, backend_name in (
                ("_load_faiss_module", "faiss"),
                ("_load_pynndescent_class", "pynndescent"),
                ("_load_scann_ops", "scann"),
            ):
                with patch.object(
                    backends_module,
                    loader_name,
                    side_effect=MappingInputError(
                        f"{backend_name} backend dependency is missing.",
                        context={"knn_backend": backend_name},
                    ),
                ):
                    with self.assertRaises(MappingInputError):
                        mapper.map_reflectance(
                            source_sensor="sensor_a",
                            reflectance={"blue": 0.79, "swir": 0.21},
                            output_mode="target_sensor",
                            target_sensor="sensor_b",
                            k=2,
                            knn_backend=backend_name,
                        )

    def test_mapping_diagnostics_include_query_and_neighbor_band_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            result = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            vnir_diag = result.diagnostics["segments"]["vnir"]
            swir_diag = result.diagnostics["segments"]["swir"]
            self.assertEqual(vnir_diag["query_band_ids"], ["blue"])
            self.assertEqual(vnir_diag["query_band_values"], [0.8])
            self.assertEqual(vnir_diag["query_valid_mask"], [True])
            self.assertEqual(vnir_diag["neighbor_ids"], ["fixture_source:vnir_high:vnir_high"])
            self.assertEqual(vnir_diag["neighbor_weights"], [1.0])
            self.assertAlmostEqual(vnir_diag["source_fit_rmse"], 0.0, places=6)
            self.assertGreaterEqual(vnir_diag["confidence_score"], 0.0)
            self.assertLessEqual(vnir_diag["confidence_score"], 1.0)
            self.assertIn("coverage", vnir_diag["confidence_components"])
            self.assertIn(vnir_diag["confidence_policy"]["band"], {"high", "medium", "low", "unavailable"})
            self.assertEqual(len(vnir_diag["neighbor_band_values"]), 1)
            self.assertAlmostEqual(vnir_diag["neighbor_band_values"][0][0], 0.8, places=6)
            self.assertEqual(swir_diag["query_band_ids"], ["swir"])
            self.assertEqual(swir_diag["query_band_values"], [0.2])
            self.assertEqual(swir_diag["query_valid_mask"], [True])
            self.assertEqual(swir_diag["neighbor_ids"], ["fixture_source:vnir_high:vnir_high"])
            self.assertEqual(swir_diag["neighbor_weights"], [1.0])
            self.assertAlmostEqual(swir_diag["source_fit_rmse"], 0.0, places=6)
            self.assertGreaterEqual(swir_diag["confidence_score"], 0.0)
            self.assertLessEqual(swir_diag["confidence_score"], 1.0)
            self.assertIn(swir_diag["confidence_policy"]["recommended_action"], {"accept", "manual_review", "reject"})
            self.assertEqual(len(swir_diag["neighbor_band_values"]), 1)
            self.assertAlmostEqual(swir_diag["neighbor_band_values"][0][0], 0.2, places=6)
            self.assertGreaterEqual(result.diagnostics["confidence_score"], 0.0)
            self.assertLessEqual(result.diagnostics["confidence_score"], 1.0)
            self.assertIn(result.diagnostics["confidence_policy"]["recommended_action"], {"accept", "manual_review", "reject"})
            self.assertEqual(result.diagnostics["knn_backend"], "numpy")
            self.assertAlmostEqual(result.diagnostics["knn_eps"], 0.0)

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
            self.assertEqual(report["neighbor_estimator"], "mean")
            self.assertEqual(report["target_sensor"]["band_ids"], ["target_vnir", "target_swir"])
            self.assertIn("retrieval", report["full_spectrum"])
            self.assertIn("regression_baseline", report["target_sensor"])

    def test_benchmark_mapping_supports_distance_weighted_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            report = benchmark_mapping(
                fixture["prepared_root"],
                "sensor_a",
                "sensor_b",
                k=2,
                test_fraction=0.25,
                random_seed=0,
                neighbor_estimator="distance_weighted_mean",
            )

            self.assertEqual(report["neighbor_estimator"], "distance_weighted_mean")
            self.assertIn("retrieval", report["target_sensor"])

    def test_benchmark_mapping_supports_simplex_mixture_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            report = benchmark_mapping(
                fixture["prepared_root"],
                "sensor_a",
                "sensor_b",
                k=2,
                test_fraction=0.25,
                random_seed=0,
                neighbor_estimator="simplex_mixture",
            )

            self.assertEqual(report["neighbor_estimator"], "simplex_mixture")
            self.assertIn("retrieval", report["target_sensor"])
            self.assertEqual(report["knn_backend"], "numpy")
            self.assertAlmostEqual(report["knn_eps"], 0.0)

    def test_benchmark_mapping_can_cap_held_out_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            report = benchmark_mapping(
                fixture["prepared_root"],
                "sensor_a",
                "sensor_b",
                k=1,
                test_fraction=0.75,
                max_test_rows=1,
                random_seed=0,
            )

            self.assertEqual(report["test_rows"], 1)
            self.assertEqual(report["max_test_rows"], 1)

    def test_compile_linear_mapper_matches_target_regression_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            linear_mapper = mapper.compile_linear_mapper(
                source_sensor="sensor_a",
                target_sensor="sensor_b",
                compile_chunk_size=2,
            )

            self.assertIsInstance(linear_mapper, LinearSpectralMapper)
            self.assertEqual(linear_mapper.source_band_ids, ("blue", "swir"))
            self.assertEqual(linear_mapper.output_band_ids, ("target_vnir", "target_swir"))

            source_queries = mapper._source_queries("sensor_a")
            target_truth, _ = mapper._simulate_full_sensor_matrix("sensor_b")
            design = np.column_stack([np.ones(source_queries.shape[0], dtype=np.float64), source_queries])
            coefficients, _, _, _ = np.linalg.lstsq(design, target_truth, rcond=None)
            expected = design @ coefficients

            out = np.empty(expected.shape, dtype=np.float64)
            mapped = linear_mapper.map_array(source_queries, out=out, chunk_size=1)
            self.assertIs(mapped, out)
            self.assertTrue(np.allclose(mapped, expected, atol=1e-6))

    def test_compile_linear_mapper_supports_full_spectrum_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            linear_mapper = mapper.compile_linear_mapper(
                source_sensor="sensor_a",
                output_mode="full_spectrum",
                compile_chunk_size=2,
            )

            source_queries = mapper._source_queries("sensor_a")
            full_truth = mapping_module._assemble_full_spectrum_batch(
                np.asarray(mapper._load_hyperspectral("vnir"), dtype=np.float64),
                np.asarray(mapper._load_hyperspectral("swir"), dtype=np.float64),
            )
            design = np.column_stack([np.ones(source_queries.shape[0], dtype=np.float64), source_queries])
            coefficients, _, _, _ = np.linalg.lstsq(design, full_truth, rcond=None)
            expected = design @ coefficients

            mapped = linear_mapper.map_array(source_queries, chunk_size=2)
            self.assertEqual(mapped.shape, (4, len(mapping_module.CANONICAL_WAVELENGTHS)))
            self.assertTrue(np.array_equal(linear_mapper.output_wavelength_nm, mapping_module.CANONICAL_WAVELENGTHS.astype(np.float64)))
            self.assertTrue(np.allclose(mapped, expected, atol=1e-6))

    def test_interleaved_sensor_band_order_is_preserved_for_queries_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))

            source_sensor = _custom_sensor_payload(
                sensor_id="sensor_a",
                bands=[
                    _custom_band_payload(
                        band_id="swir",
                        segment="swir",
                        wavelength_nm=[1595.0, 1600.0, 1605.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="blue",
                        segment="vnir",
                        wavelength_nm=[445.0, 450.0, 455.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                ],
            )
            target_sensor = _custom_sensor_payload(
                sensor_id="sensor_b",
                bands=[
                    _custom_band_payload(
                        band_id="target_swir",
                        segment="swir",
                        wavelength_nm=[1695.0, 1700.0, 1705.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="target_vnir",
                        segment="vnir",
                        wavelength_nm=[495.0, 500.0, 505.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                ],
            )
            (fixture["srf_root"] / "sensor_a.json").write_text(json.dumps(source_sensor, indent=2) + "\n", encoding="utf-8")
            (fixture["srf_root"] / "sensor_b.json").write_text(json.dumps(target_sensor, indent=2) + "\n", encoding="utf-8")
            build_mapping_library(
                fixture["siac_root"],
                fixture["srf_root"],
                fixture["prepared_root"],
                ["sensor_a"],
            )

            mapper = SpectralMapper(fixture["prepared_root"])
            source_queries = mapper._source_queries("sensor_a")

            self.assertEqual(mapper.get_sensor_schema("sensor_a").band_ids(), ("swir", "blue"))
            self.assertEqual(mapper.compile_linear_mapper(source_sensor="sensor_a", target_sensor="sensor_b").output_band_ids, ("target_swir", "target_vnir"))

            result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance=source_queries[0],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )
            self.assertEqual(result.target_band_ids, ("target_swir", "target_vnir"))

    def test_segment_isolation_keeps_swir_output_stable_when_only_vnir_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            base = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                reflectance={"blue": 0.15, "swir": 0.90},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )
            changed_vnir = mapper.map_reflectance_debug(
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

    def test_segment_isolation_keeps_vnir_output_stable_when_only_swir_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            base = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                target_sensor="sensor_b",
                reflectance={"blue": 0.80, "swir": 0.15},
                output_mode="target_sensor",
                k=1,
            )
            changed_swir = mapper.map_reflectance_debug(
                source_sensor="sensor_a",
                target_sensor="sensor_b",
                reflectance={"blue": 0.80, "swir": 0.90},
                output_mode="target_sensor",
                k=1,
            )

            assert base.target_reflectance is not None
            assert changed_swir.target_reflectance is not None
            self.assertEqual(base.neighbor_ids_by_segment["vnir"], changed_swir.neighbor_ids_by_segment["vnir"])
            self.assertNotEqual(base.neighbor_ids_by_segment["swir"], changed_swir.neighbor_ids_by_segment["swir"])
            self.assertEqual(float(base.target_reflectance[0]), float(changed_swir.target_reflectance[0]))
            self.assertNotEqual(float(base.target_reflectance[1]), float(changed_swir.target_reflectance[1]))

    def test_target_sensor_output_can_emit_only_successful_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_debug(
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

    def test_map_reflectance_batch_output_arrays_preserve_nan_for_missing_target_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            sample_ids, output_rows, source_fit_rmse, output_columns = mapper._map_reflectance_batch_output_arrays(
                source_sensor="sensor_a",
                reflectance_rows=[{"blue": 0.80, "swir": 0.90}],
                valid_mask_rows=[{"blue": True, "swir": False}],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            self.assertEqual(sample_ids, ("sample_000001",))
            self.assertEqual(output_columns, ("target_vnir", "target_swir"))
            self.assertEqual(output_rows.shape, (1, 2))
            self.assertTrue(np.isfinite(output_rows[0, 0]))
            self.assertTrue(np.isnan(output_rows[0, 1]))
            self.assertEqual(source_fit_rmse.shape, (1,))
            self.assertGreaterEqual(float(source_fit_rmse[0]), 0.0)

    def test_map_reflectance_batch_arrays_matches_dense_output_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            dense = mapper.map_reflectance_batch_arrays(
                source_sensor="sensor_a",
                reflectance_rows=[{"blue": 0.80, "swir": 0.90}],
                valid_mask_rows=[{"blue": True, "swir": False}],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )
            sample_ids, output_rows, source_fit_rmse, output_columns = mapper._map_reflectance_batch_output_arrays(
                source_sensor="sensor_a",
                reflectance_rows=[{"blue": 0.80, "swir": 0.90}],
                valid_mask_rows=[{"blue": True, "swir": False}],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            self.assertIsInstance(dense, BatchMappingArrayResult)
            self.assertEqual(dense.sample_ids, sample_ids)
            self.assertTrue(np.allclose(dense.reflectance, output_rows, equal_nan=True))
            self.assertTrue(np.allclose(dense.source_fit_rmse, source_fit_rmse))
            self.assertEqual(dense.output_columns, output_columns)
            self.assertIsNone(dense.wavelength_nm)
            self.assertTrue(np.isnan(dense.reflectance[0, 1]))

    def test_map_reflectance_batch_arrays_exposes_wavelength_axis_and_avoids_result_builders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_mapping_results_from_segment_retrievals_batch",
                side_effect=AssertionError("rich batch result assembly should not be used"),
            ), patch.object(
                mapping_module.SpectralMapper,
                "_build_mapping_result",
                side_effect=AssertionError("single-sample result builder should not be used"),
            ), patch.object(
                runtime_module,
                "_successful_segment_retrieval",
                side_effect=AssertionError("dense array path should not materialize rich retrieval objects"),
            ), patch.object(
                runtime_module,
                "_search_neighbor_rows",
                side_effect=AssertionError("dense array path should not use per-sample Python neighbor search"),
            ):
                dense = mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="full_spectrum",
                    k=1,
                )

            assert dense.wavelength_nm is not None
            self.assertEqual(dense.sample_ids, ("alpha", "beta"))
            self.assertEqual(dense.reflectance.shape, (2, 2101))
            self.assertEqual(dense.output_columns[0], "nm_400")
            self.assertEqual(dense.output_columns[-1], "nm_2500")
            self.assertEqual(int(dense.wavelength_nm[0]), 400)
            self.assertEqual(int(dense.wavelength_nm[-1]), 2500)
            self.assertEqual(dense.wavelength_nm.shape, (2101,))

    def test_map_reflectance_batch_arrays_exact_ckdtree_skips_separate_batch_rerank(self) -> None:
        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps, workers
                query_array = np.asarray(query, dtype=np.float64)
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree), patch.object(
                mapping_module._rustaccel,
                "refine_neighbor_rows_batch",
                side_effect=AssertionError("exact dense cKDTree path should not call the separate rerank kernel"),
            ):
                dense = mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="full_spectrum",
                    k=1,
                    knn_backend="scipy_ckdtree",
                    knn_eps=0.0,
                )

            self.assertEqual(dense.reflectance.shape, (2, 2101))

    def test_scipy_ckdtree_exact_masked_query_skips_python_rerank(self) -> None:
        class FakeKDTree:
            def __init__(self, data: np.ndarray) -> None:
                self.data = np.asarray(data, dtype=np.float64)

            def query(self, query: np.ndarray, k: int, eps: float = 0.0, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
                del eps, workers
                query_array = np.asarray(query, dtype=np.float64)
                if query_array.ndim == 1:
                    query_array = query_array.reshape(1, -1)
                distances = np.linalg.norm(self.data[None, :, :] - query_array[:, None, :], axis=2)
                ordered = np.argsort(distances, axis=1)[:, :k]
                selected = np.take_along_axis(distances, ordered, axis=1)
                return selected, ordered

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = _build_fixture(root)
            source_sensor = _custom_sensor_payload(
                sensor_id="sensor_overlap",
                bands=[
                    _custom_band_payload(
                        band_id="blue",
                        segment="vnir",
                        wavelength_nm=[445.0, 450.0, 455.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="nir",
                        segment="vnir",
                        wavelength_nm=[845.0, 850.0, 855.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                    _custom_band_payload(
                        band_id="swir",
                        segment="swir",
                        wavelength_nm=[1595.0, 1600.0, 1605.0],
                        response=[0.2, 1.0, 0.2],
                    ),
                ],
            )
            (fixture["srf_root"] / "sensor_overlap.json").write_text(
                json.dumps(source_sensor, indent=2) + "\n",
                encoding="utf-8",
            )
            prepared_root = root / "prepared"
            build_mapping_library(
                fixture["siac_root"],
                fixture["srf_root"],
                prepared_root,
                ["sensor_overlap"],
            )

            mapper = SpectralMapper(prepared_root)
            with patch.object(backends_module, "_load_ckdtree_class", return_value=FakeKDTree), patch.object(
                runtime_module,
                "_refine_neighbor_rows",
                side_effect=AssertionError("exact scipy cKDTree distances should bypass Python reranking"),
            ):
                result = mapper.map_reflectance_debug(
                    source_sensor="sensor_overlap",
                    reflectance={"blue": 0.80, "nir": 0.40, "swir": 0.20},
                    valid_mask={"blue": True, "nir": False, "swir": True},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    knn_backend="scipy_ckdtree",
                    knn_eps=0.0,
                )

            self.assertEqual(result.target_band_ids, ("target_vnir", "target_swir"))

    def test_map_reflectance_batch_arrays_reuses_output_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            reflectance_rows = np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64)
            valid_mask_rows = np.array([[True, True], [True, True]], dtype=bool)
            out = np.empty((2, 2101), dtype=np.float64)
            source_fit_out = np.empty((2,), dtype=np.float64)

            with patch.object(
                mapping_module,
                "_normalized_batch_rows",
                side_effect=AssertionError("NumPy batch inputs should stay on the ndarray fast path"),
            ), patch.object(
                mapping_module.SpectralMapper,
                "_coerced_batch_query_arrays",
                side_effect=AssertionError("NumPy batch inputs should not use per-row coercion"),
            ):
                dense = mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=reflectance_rows,
                    valid_mask_rows=valid_mask_rows,
                    sample_ids=["alpha", "beta"],
                    output_mode="full_spectrum",
                    k=1,
                    out=out,
                    source_fit_rmse_out=source_fit_out,
                )

            self.assertIs(dense.reflectance, out)
            self.assertIs(dense.source_fit_rmse, source_fit_out)
            self.assertEqual(dense.sample_ids, ("alpha", "beta"))

    def test_map_reflectance_batch_arrays_ndarray_reuses_output_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            reflectance_rows = np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64)
            valid_mask_rows = np.array([[True, True], [True, True]], dtype=bool)
            out = np.empty((2, 2101), dtype=np.float64)
            source_fit_out = np.empty((2,), dtype=np.float64)

            with patch.object(
                mapping_module,
                "_normalized_batch_rows",
                side_effect=AssertionError("strict ndarray API should not normalize row iterables"),
            ), patch.object(
                mapping_module.SpectralMapper,
                "_coerced_batch_query_arrays",
                side_effect=AssertionError("strict ndarray API should not run per-row coercion"),
            ):
                dense = mapper.map_reflectance_batch_arrays_ndarray(
                    source_sensor="sensor_a",
                    reflectance_rows=reflectance_rows,
                    valid_mask_rows=valid_mask_rows,
                    sample_ids=["alpha", "beta"],
                    output_mode="full_spectrum",
                    k=1,
                    out=out,
                    source_fit_rmse_out=source_fit_out,
                )

            self.assertIs(dense.reflectance, out)
            self.assertIs(dense.source_fit_rmse, source_fit_out)
            self.assertEqual(dense.reflectance.shape, (2, 2101))
            self.assertEqual(dense.sample_ids, ("alpha", "beta"))

    def test_map_reflectance_batch_rich_path_avoids_per_sample_python_reranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                runtime_module,
                "_search_neighbor_rows",
                side_effect=AssertionError("rich batch path should not use per-sample Python neighbor search"),
            ), patch.object(
                runtime_module,
                "_refine_neighbor_rows",
                side_effect=AssertionError("rich batch path should not use per-sample Python reranking"),
            ), patch.object(
                runtime_module,
                "_successful_segment_retrieval",
                side_effect=AssertionError("rich batch path should not materialize single-sample retrieval objects"),
            ), patch.object(
                mapping_module.SpectralMapper,
                "_dense_segment_output_batch",
                side_effect=AssertionError("rich batch path should not rebuild dense outputs from rich retrievals"),
            ), patch.object(
                runtime_module,
                "_segment_diagnostics_payload",
                side_effect=AssertionError("rich batch path should not reserialize retrieval diagnostics"),
            ):
                result = mapper.map_reflectance_batch_debug(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                )

            self.assertEqual(result.sample_ids, ("alpha", "beta"))
            self.assertEqual(len(result.results), 2)
            self.assertTrue(all(mapped.target_reflectance is not None for mapped in result.results))

    def test_map_reflectance_batch_arrays_target_sensor_matches_rich_batch_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            dense = mapper.map_reflectance_batch_arrays(
                source_sensor="sensor_a",
                reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                sample_ids=["alpha", "beta"],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )
            rich = mapper.map_reflectance_batch_debug(
                source_sensor="sensor_a",
                reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                sample_ids=["alpha", "beta"],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            expected_rows = np.vstack(
                [
                    np.asarray(rich.results[0].target_reflectance, dtype=np.float64),
                    np.asarray(rich.results[1].target_reflectance, dtype=np.float64),
                ]
            )
            self.assertTrue(np.allclose(dense.reflectance, expected_rows))

    def test_map_reflectance_batch_arrays_target_sensor_bypasses_rust_finalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module,
                "_finalize_target_sensor_batch_accel",
                side_effect=AssertionError("direct target-space batch path should not call the Rust finalizer"),
            ):
                dense = mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                )

            self.assertEqual(dense.reflectance.shape, (2, 2))

    def test_map_reflectance_batch_arrays_target_sensor_reuses_output_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])
            out = np.empty((2, 2), dtype=np.float64)
            source_fit_out = np.empty((2,), dtype=np.float64)

            with patch.object(
                mapping_module,
                "_finalize_target_sensor_batch_accel",
                side_effect=AssertionError("direct target-space batch path should not call the Rust finalizer"),
            ):
                dense = mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    out=out,
                    source_fit_rmse_out=source_fit_out,
                )

            self.assertIs(dense.reflectance, out)
            self.assertIs(dense.source_fit_rmse, source_fit_out)
            self.assertEqual(dense.reflectance.shape, (2, 2))

    def test_map_reflectance_batch_arrays_validates_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([0.10, 0.20]),
                    output_mode="vnir_spectrum",
                )

    def test_map_reflectance_batch_arrays_attaches_sample_context_on_output_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError) as raised:
                mapper.map_reflectance_batch_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.80, "swir": 0.90}],
                    valid_mask_rows=[{"blue": True, "swir": False}],
                    sample_ids=["alpha"],
                    output_mode="full_spectrum",
                    k=1,
                )

            self.assertEqual(raised.exception.context["sample_id"], "alpha")
            self.assertEqual(raised.exception.context["sample_index"], 0)

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
            self.assertEqual(int(swir_result.reconstructed_wavelength_nm[0]), 800)
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
                neighbor_estimator="mean",
                candidate_row_indices=[0, 1],
            )

            self.assertEqual(
                set(result.neighbor_ids_by_segment["vnir"]),
                {"fixture_source:base:base", "fixture_source:vnir_high:vnir_high"},
            )
            self.assertEqual(
                set(result.neighbor_ids_by_segment["swir"]),
                {"fixture_source:base:base", "fixture_source:vnir_high:vnir_high"},
            )

    def test_map_reflectance_default_omits_debug_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            assert result.target_reflectance is not None
            self.assertTrue(np.allclose(result.target_reflectance, np.array([0.80, 0.20])))
            self.assertEqual(result.neighbor_ids_by_segment, {})
            self.assertEqual(result.neighbor_distances_by_segment, {})
            self.assertEqual(result.segment_outputs, {})
            self.assertEqual(result.diagnostics, {})

    def test_map_reflectance_batch_returns_sample_aligned_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_batch(
                source_sensor="sensor_a",
                reflectance_rows=[
                    {"blue": 0.80, "swir": 0.20},
                    {"blue": 0.10, "swir": 0.90},
                ],
                sample_ids=["alpha", "beta"],
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=1,
            )

            self.assertIsInstance(result, BatchMappingArrayResult)
            self.assertEqual(result.sample_ids, ("alpha", "beta"))
            self.assertEqual(result.output_columns, ("target_vnir", "target_swir"))
            self.assertTrue(np.allclose(result.reflectance[0], np.array([0.80, 0.20])))
            self.assertTrue(np.allclose(result.reflectance[1], np.array([0.10, 0.90])))

    def test_map_reflectance_batch_uses_batched_target_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_simulate_target_sensor",
                side_effect=AssertionError("per-sample target resampling should not be used"),
            ):
                result = mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[
                        {"blue": 0.80, "swir": 0.20},
                        {"blue": 0.10, "swir": 0.90},
                    ],
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                )

            self.assertTrue(np.allclose(result.reflectance[0], np.array([0.80, 0.20])))
            self.assertTrue(np.allclose(result.reflectance[1], np.array([0.10, 0.90])))

    def test_map_reflectance_batch_uses_batched_full_spectrum_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module,
                "_assemble_full_spectrum",
                side_effect=AssertionError("per-sample full-spectrum assembly should not be used"),
            ):
                result = mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[
                        {"blue": 0.80, "swir": 0.20},
                        {"blue": 0.10, "swir": 0.90},
                    ],
                    sample_ids=["alpha", "beta"],
                    output_mode="full_spectrum",
                    k=1,
                )

            self.assertEqual(result.reflectance.shape, (2, len(mapping_module.CANONICAL_WAVELENGTHS)))

    def test_map_reflectance_batch_supports_public_exclusion_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_batch_debug(
                source_sensor="sensor_a",
                reflectance_rows=[
                    {"blue": 0.80, "swir": 0.20},
                    {"blue": 0.15, "swir": 0.25},
                ],
                sample_ids=["alpha", "base"],
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
                exclude_sample_names=["swir_high"],
                exclude_row_ids_per_sample=[
                    "fixture_source:vnir_high:vnir_high",
                    None,
                ],
                self_exclude_sample_id=True,
            )

            self.assertEqual(result.results[0].neighbor_ids_by_segment["vnir"], ("fixture_source:mid:mid",))
            self.assertEqual(result.results[0].neighbor_ids_by_segment["swir"], ("fixture_source:base:base",))
            self.assertEqual(result.results[1].neighbor_ids_by_segment["vnir"], ("fixture_source:mid:mid",))
            self.assertEqual(result.results[1].neighbor_ids_by_segment["swir"], ("fixture_source:vnir_high:vnir_high",))

    def test_map_reflectance_batch_avoids_single_sample_mapping_path_for_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_map_reflectance_internal",
                side_effect=AssertionError("single-sample mapping path should not be used"),
            ):
                result = mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.15, "swir": 0.25}],
                    sample_ids=["base"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    self_exclude_sample_id=True,
                )

            self.assertEqual(result.sample_ids, ("base",))
            self.assertEqual(result.reflectance.shape, (1, 2))

    def test_map_reflectance_batch_output_arrays_avoid_single_sample_retrieval_path_for_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_retrieve_segment",
                side_effect=AssertionError("single-sample retrieval path should not be used"),
            ):
                sample_ids, output_rows, source_fit_rmse, output_columns = mapper._map_reflectance_batch_output_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.15, "swir": 0.25}],
                    sample_ids=["base"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    self_exclude_sample_id=True,
                )

            self.assertEqual(sample_ids, ("base",))
            self.assertEqual(output_columns, ("target_vnir", "target_swir"))
            self.assertEqual(output_rows.shape, (1, 2))
            self.assertEqual(source_fit_rmse.shape, (1,))

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_to_zarr_avoids_single_sample_result_path_for_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            mapper = SpectralMapper(fixture["prepared_root"])
            output_path = root / "batch_exclusion.zarr"

            with patch.object(
                mapping_module.SpectralMapper,
                "_map_reflectance_internal",
                side_effect=AssertionError("single-sample result path should not be used"),
            ):
                summary = mapper.map_reflectance_batch_to_zarr(
                    zarr_path=output_path,
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.15, "swir": 0.25}],
                    sample_ids=["base"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    self_exclude_sample_id=True,
                )

            self.assertEqual(summary["sample_count"], 1)
            self.assertTrue(output_path.exists())

    def test_map_reflectance_batch_output_arrays_exclusions_use_grouped_batch_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_retrieve_segment",
                side_effect=AssertionError("single-sample retrieval path should not be used"),
            ):
                with patch.object(mapper, "_retrieve_segment_dense_batch", wraps=mapper._retrieve_segment_dense_batch) as batch_mock:
                    sample_ids, output_rows, source_fit_rmse, output_columns = mapper._map_reflectance_batch_output_arrays(
                        source_sensor="sensor_a",
                        reflectance_rows=[
                            {"blue": 0.80, "swir": 0.20},
                            {"blue": 0.15, "swir": 0.25},
                        ],
                        sample_ids=["alpha", "beta"],
                        output_mode="target_sensor",
                        target_sensor="sensor_b",
                        k=1,
                        exclude_row_ids_per_sample=[
                            "fixture_source:vnir_high:vnir_high",
                            "fixture_source:vnir_high:vnir_high",
                        ],
                    )

            self.assertEqual(sample_ids, ("alpha", "beta"))
            self.assertEqual(output_columns, ("target_vnir", "target_swir"))
            self.assertEqual(output_rows.shape, (2, 2))
            self.assertEqual(source_fit_rmse.shape, (2,))
            self.assertEqual(batch_mock.call_count, 2)

    def test_map_reflectance_batch_output_arrays_avoid_single_sample_retrieval_for_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_retrieve_segment",
                side_effect=AssertionError("single-sample segment retrieval should not be used"),
            ):
                sample_ids, output_rows, source_fit_rmse, output_columns = mapper._map_reflectance_batch_output_arrays(
                    source_sensor="sensor_a",
                    reflectance_rows=[{"blue": 0.15, "swir": 0.25}],
                    sample_ids=["base"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                    self_exclude_sample_id=True,
                )

            self.assertEqual(sample_ids, ("base",))
            self.assertEqual(output_columns, ("target_vnir", "target_swir"))
            self.assertEqual(output_rows.shape, (1, 2))
            self.assertEqual(source_fit_rmse.shape, (1,))

    def test_map_reflectance_batch_matches_single_sample_results_for_array_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            batch = mapper.map_reflectance_batch(
                source_sensor="sensor_a",
                reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                sample_ids=["alpha", "beta"],
                output_mode="full_spectrum",
                k=1,
            )
            single_alpha = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance=[0.80, 0.20],
                valid_mask=[True, True],
                output_mode="full_spectrum",
                k=1,
            )
            single_beta = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance=[0.10, 0.90],
                valid_mask=[True, True],
                output_mode="full_spectrum",
                k=1,
            )

            assert single_alpha.reconstructed_full_spectrum is not None
            assert single_beta.reconstructed_full_spectrum is not None
            self.assertTrue(np.allclose(batch.reflectance[0], single_alpha.reconstructed_full_spectrum))
            self.assertTrue(np.allclose(batch.reflectance[1], single_beta.reconstructed_full_spectrum))

    def test_map_reflectance_batch_avoids_single_sample_result_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with patch.object(
                mapping_module.SpectralMapper,
                "_build_mapping_result",
                side_effect=AssertionError("single-sample result builder should not be used"),
            ):
                batch = mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float64),
                    valid_mask_rows=np.array([[True, True], [True, True]], dtype=bool),
                    sample_ids=["alpha", "beta"],
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=1,
                )

            self.assertEqual(batch.sample_ids, ("alpha", "beta"))
            self.assertTrue(np.allclose(batch.reflectance[0], np.array([0.80, 0.20])))
            self.assertTrue(np.allclose(batch.reflectance[1], np.array([0.10, 0.90])))

    def test_map_reflectance_batch_validates_shapes_and_attaches_sample_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows={"blue": 0.10, "swir": 0.20},
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([0.10, 0.20]),
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[],
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=np.array([[0.10, 0.20]]),
                    valid_mask_rows=np.array([[True]]),
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[[0.10, 0.20]],
                    valid_mask_rows={"blue": True},
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[[0.10, 0.20]],
                    valid_mask_rows="bad-mask",
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows="bad-input",
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[[0.10, 0.20]],
                    sample_ids=["alpha"],
                    exclude_row_ids_per_sample={"beta": "fixture_source:base:base"},
                    output_mode="vnir_spectrum",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[[0.10, 0.20]],
                    sample_ids=["alpha"],
                    exclude_row_ids_per_sample=[],
                    output_mode="vnir_spectrum",
                )

            with self.assertRaises(MappingInputError) as error_context:
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[
                        {"blue": 0.80, "swir": 0.20},
                        {"blue": float("nan"), "swir": float("nan")},
                    ],
                    sample_ids=["ok", "bad"],
                    output_mode="vnir_spectrum",
                )
            self.assertEqual(error_context.exception.context["sample_id"], "bad")
            self.assertEqual(error_context.exception.context["sample_index"], 1)

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
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance=[0.1, 0.2],
                    output_mode="vnir_spectrum",
                    neighbor_estimator="unsupported",
                )
            with self.assertRaises(MappingInputError):
                mapper.map_reflectance_batch(
                    source_sensor="sensor_a",
                    reflectance_rows=[[0.1, 0.2]],
                    output_mode="vnir_spectrum",
                    neighbor_estimator="unsupported",
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
                    neighbor_estimator="mean",
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
                    neighbor_estimator="mean",
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
            build_mapping_library(
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
        self.assertEqual(mapping_module._normalized_sample_ids(None, sample_count=2), ("sample_000001", "sample_000002"))
        wrapped = mapping_module._attach_sample_context(
            mapping_module.SpectralLibraryError("example", "Example error"),
            sample_id="alpha",
            sample_index=0,
        )
        self.assertEqual(wrapped.context["sample_id"], "alpha")
        with self.assertRaises(PreparedLibraryBuildError):
            mapping_module._normalized_source_sensors([])
        with self.assertRaises(MappingInputError):
            mapping_module._normalized_sample_ids(None, sample_count=0)
        with self.assertRaises(MappingInputError):
            mapping_module._normalized_sample_ids(["alpha"], sample_count=2)
        with self.assertRaises(MappingInputError):
            mapping_module._normalized_sample_ids(["alpha", "alpha"], sample_count=2)
        with self.assertRaises(MappingInputError):
            mapping_module._normalized_sample_ids(["alpha", ""], sample_count=2)

    def test_sensor_schema_round_trip_and_band_sorting(self) -> None:
        schema = SensorSRFSchema.from_dict(
            _custom_sensor_payload(
                sensor_id="sorted_sensor",
                bands=[
                    _custom_band_payload(
                        band_id="b1",
                        segment="vnir",
                        wavelength_nm=[445.0, 450.0, 455.0],
                        response=[0.2, 1.0, 0.1],
                    )
                ],
            )
        )

        self.assertEqual(schema.band_ids(), ("b1",))
        self.assertEqual(schema.bands[0].wavelength_nm, (445.0, 450.0, 455.0))
        self.assertEqual(schema.to_dict()["sensor_id"], "sorted_sensor")
        self.assertEqual(schema.to_dict()["schema_type"], "rsrf_sensor_definition")

    def test_sensor_band_definition_accepts_rsrf_custom_response_definitions(self) -> None:
        sampled_band = mapping_module.SensorBandDefinition.from_dict(
            _custom_band_payload(
                band_id="b_curve",
                segment="vnir",
                wavelength_nm=[445.0, 450.0, 455.0],
                response=[0.2, 1.0, 0.1],
            )
        )
        self.assertEqual(sampled_band.wavelength_nm, (445.0, 450.0, 455.0))
        self.assertEqual(sampled_band.rsr, (0.2, 1.0, 0.1))

        realized_band = mapping_module.SensorBandDefinition.from_dict(
            {
                "band_id": "b_spec",
                "response_definition": {
                    "kind": "band_spec",
                    "center_wavelength_nm": 550.0,
                    "fwhm_nm": 20.0,
                },
                "extensions": {"spectral_library": {"segment": "vnir"}},
            }
        )
        self.assertAlmostEqual(realized_band.center_nm or 0.0, 550.0)
        self.assertAlmostEqual(realized_band.fwhm_nm or 0.0, 20.0)
        self.assertGreater(len(realized_band.wavelength_nm), 2)
        self.assertGreater(realized_band.support_max_nm or 0.0, realized_band.support_min_nm or 0.0)

        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict(
                _custom_band_payload(
                    band_id="b_curve_unsorted",
                    segment="vnir",
                    wavelength_nm=[455.0, 445.0, 450.0],
                    response=[0.1, 0.2, 1.0],
                )
            )

        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict(
                {
                    "band_id": "b_alias",
                    "center_nm": 650.0,
                    "fwhm": 30.0,
                    "extensions": {"spectral_library": {"segment": "vnir"}},
                }
            )

        with self.assertRaises(SensorSchemaError):
            mapping_module.SensorBandDefinition.from_dict(
                {
                    "band_id": "b_spec_missing_wrapper",
                    "center_wavelength_nm": 650.0,
                    "fwhm_nm": 30.0,
                    "extensions": {"spectral_library": {"segment": "vnir"}},
                }
            )

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
            mapping_module.SensorBandDefinition.from_dict({"band_id": "b1", "extensions": {"spectral_library": {"segment": "vnir"}}})

        band = mapping_module.SensorBandDefinition.from_dict(
            {
                "band_id": "b1",
                "response_definition": {
                    "kind": "sampled",
                    "wavelength_nm": [445.0, 450.0],
                    "response": [0.2, 1.0],
                },
                "extensions": {"spectral_library": {"segment": "vnir"}},
            }
        )
        band_payload = band.to_dict()
        self.assertEqual(band_payload["extensions"]["spectral_library"]["segment"], "vnir")
        self.assertEqual(band_payload["response_definition"]["kind"], "sampled")
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

    def test_coerce_sensor_input_assigns_segments_and_deterministic_ids(self) -> None:
        schema = coerce_sensor_input(
            SensorInput(
                bands=[
                    BandInput(center_wavelength_nm=550.0, fwhm_nm=20.0),
                    BandInput(center_wavelength_nm=1600.0, fwhm_nm=40.0),
                ]
            )
        )

        self.assertTrue(schema.sensor_id.startswith("custom_"))
        self.assertEqual(schema.band_ids(), ("band_1", "band_2"))
        self.assertEqual(tuple(band.segment for band in schema.bands), ("vnir", "swir"))
        self.assertLess(schema.bands[0].support_max_nm or 0.0, 1000.000001)
        self.assertGreaterEqual(schema.bands[1].support_min_nm or 0.0, 800.0)

    def test_build_mapping_runtime_accepts_in_memory_library_and_target_sensors(self) -> None:
        runtime = build_mapping_runtime(
            library=_in_memory_library_input(),
            source_sensors=[
                SensorInput(
                    sensor_id="sensor_a",
                    bands=[
                        BandInput(
                            band_id="blue",
                            response_definition={
                                "kind": "sampled",
                                "wavelength_nm": [445.0, 450.0, 455.0],
                                "response": [0.2, 1.0, 0.2],
                            },
                            segment="vnir",
                        ),
                        BandInput(
                            band_id="swir",
                            response_definition={
                                "kind": "sampled",
                                "wavelength_nm": [1595.0, 1600.0, 1605.0],
                                "response": [0.2, 1.0, 0.2],
                            },
                            segment="swir",
                        ),
                    ],
                )
            ],
            target_sensors=[
                SensorInput(
                    sensor_id="sensor_b",
                    bands=[
                        BandInput(center_wavelength_nm=1700.0, fwhm_nm=20.0, band_id="target_swir"),
                        BandInput(center_wavelength_nm=500.0, fwhm_nm=20.0, band_id="target_vnir"),
                    ],
                )
            ],
        )
        self.addCleanup(runtime.close)

        self.assertIsInstance(runtime, PreparedRuntime)
        self.assertEqual(runtime.source_sensor_ids, ("sensor_a",))
        self.assertEqual(runtime.target_sensor_ids, ("sensor_b",))
        self.assertEqual(runtime.mapper.manifest.source_sensors, ("sensor_a",))
        self.assertEqual(runtime.get_sensor_schema("sensor_b").band_ids(), ("target_swir", "target_vnir"))

        result = runtime.map_reflectance(
            source_sensor="sensor_a",
            reflectance={"blue": 0.79, "swir": 0.21},
            output_mode="target_sensor",
            target_sensor="sensor_b",
        )

        self.assertEqual(result.target_band_ids, ("target_swir", "target_vnir"))
        assert result.target_reflectance is not None
        self.assertEqual(result.target_reflectance.shape, (2,))

    def test_build_mapping_runtime_reuses_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            kwargs = {
                "library": _in_memory_library_input(),
                "source_sensors": [
                    SensorInput(
                        sensor_id="sensor_a",
                        bands=[
                            BandInput(center_wavelength_nm=450.0, fwhm_nm=10.0, band_id="blue"),
                            BandInput(center_wavelength_nm=1600.0, fwhm_nm=10.0, band_id="swir"),
                        ],
                    )
                ],
                "cache_root": cache_root,
            }
            first = build_mapping_runtime(**kwargs)
            second = build_mapping_runtime(**kwargs)
            self.addCleanup(first.close)
            self.addCleanup(second.close)

            self.assertEqual(first.prepared_root, second.prepared_root)
            self.assertTrue((first.prepared_root / "manifest.json").exists())

    def test_build_mapping_library_rejects_invalid_dtype_and_missing_source_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            with self.assertRaises(PreparedLibraryBuildError):
                build_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["sensor_a"],
                    dtype="int32",
                )
            with self.assertRaises(SensorSchemaError):
                build_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"],
                    fixture["prepared_root"],
                    ["missing_sensor"],
                )

    def test_build_mapping_library_rejects_invalid_sensor_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            (fixture["srf_root"] / "sensor_a.json").write_text(
                json.dumps(
                    _custom_sensor_payload(
                        sensor_id="sensor_a",
                        bands=[
                            _custom_band_payload(
                                band_id="blue",
                                segment="vnir",
                                wavelength_nm=[1200.0, 1205.0],
                                response=[1.0, 0.5],
                            )
                        ],
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SensorSchemaError):
                build_mapping_library(
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

            dup_root = root / "dup"
            dup_root.mkdir()
            payload = _custom_sensor_payload(
                sensor_id="dup_sensor",
                bands=[
                    _custom_band_payload(
                        band_id="b1",
                        segment="vnir",
                        wavelength_nm=[445.0, 450.0],
                        response=[0.2, 1.0],
                    )
                ],
            )
            (dup_root / "a.json").write_text(json.dumps(payload), encoding="utf-8")
            (dup_root / "b.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(SensorSchemaError):
                mapping_module.load_sensor_schemas(dup_root)

            legacy_root = root / "legacy"
            legacy_root.mkdir()
            legacy_payload = _custom_sensor_payload(
                sensor_id="legacy_sensor",
                bands=[
                    {
                        "band_id": "b1",
                        "wavelength_nm": [445.0, 450.0],
                        "rsr": [0.2, 1.0],
                    }
                ],
            )
            (legacy_root / "legacy_sensor.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            with self.assertRaises(SensorSchemaError):
                mapping_module.load_sensor_schemas(legacy_root)

    def test_load_sensor_schemas_resolves_required_rsrf_sensor(self) -> None:
        schema = SensorSRFSchema.from_dict(
            _custom_sensor_payload(
                sensor_id="sentinel-2b_msi",
                bands=[
                    _custom_band_payload(
                        band_id="blue",
                        segment="vnir",
                        wavelength_nm=[450.0, 460.0, 470.0],
                        response=[0.1, 1.0, 0.1],
                    )
                ],
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(schema_module, "_load_rsrf_sensor_schema", return_value=schema) as mocked_loader:
                resolved = mapping_module.load_sensor_schemas(
                    Path(tmpdir) / "missing",
                    required_sensor_ids=["sentinel-2b_msi"],
                )
        self.assertEqual(tuple(resolved), ("sentinel-2b_msi",))
        self.assertEqual(resolved["sentinel-2b_msi"].band_ids(), ("blue",))
        mocked_loader.assert_called_once_with("sentinel-2b_msi")

    def test_build_mapping_library_resolves_required_rsrf_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            schema = SensorSRFSchema.from_dict(
                _custom_sensor_payload(
                    sensor_id="sentinel-2b_msi",
                    bands=[
                        _custom_band_payload(
                            band_id="blue",
                            segment="vnir",
                            wavelength_nm=[450.0, 460.0, 470.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                        _custom_band_payload(
                            band_id="nir",
                            segment="vnir",
                            wavelength_nm=[840.0, 850.0, 860.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                        _custom_band_payload(
                            band_id="swir1",
                            segment="swir",
                            wavelength_nm=[1600.0, 1610.0, 1620.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                    ],
                )
            )
            with patch.object(schema_module, "_load_rsrf_sensor_schema", return_value=schema) as mocked_loader:
                manifest = build_mapping_library(
                    fixture["siac_root"],
                    fixture["srf_root"] / "missing",
                    fixture["prepared_root"],
                    ["sentinel-2b_msi"],
                )

            self.assertEqual(manifest.source_sensors, ("sentinel-2b_msi",))
            self.assertTrue((fixture["prepared_root"] / "source_sentinel-2b_msi_vnir.npy").exists())
            self.assertTrue((fixture["prepared_root"] / "source_sentinel-2b_msi_swir.npy").exists())
            mocked_loader.assert_called_once_with("sentinel-2b_msi")

    def test_spectral_mapper_requires_prepared_sensor_schema_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            with self.assertRaises(SensorSchemaError):
                mapper.get_sensor_schema("snpp_viirs")

            with self.assertRaises(PreparedLibraryValidationError):
                mapper._load_source_matrix("snpp_viirs", "swir")

    def test_build_mapping_library_rejects_missing_and_extra_siac_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            metadata_path = fixture["siac_root"] / "tabular" / "siac_spectra_metadata.csv"
            spectra_path = fixture["siac_root"] / "tabular" / "siac_normalized_spectra.csv"

            with metadata_path.open("r", encoding="utf-8", newline="") as handle:
                metadata_rows = list(csv.DictReader(handle))
            _write_csv(metadata_path, list(metadata_rows[0].keys()), metadata_rows[:-1])
            with self.assertRaises(PreparedLibraryBuildError):
                build_mapping_library(
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
                build_mapping_library(
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

    def test_load_siac_rows_interpolates_sparse_blank_nm_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.csv"
            spectra_path = root / "spectra.csv"
            _write_csv(
                metadata_path,
                ["source_id", "spectrum_id", "sample_name"],
                [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a"}],
            )
            row = {"source_id": "s1", "spectrum_id": "a", "sample_name": "a", **_spectrum_values(0.1, 0.2, 0.3)}
            row["nm_2500"] = ""
            _write_csv(
                spectra_path,
                ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
                [row],
            )

            _, _, hyperspectral, interpolation_summary = mapping_module._load_siac_rows(
                metadata_path,
                spectra_path,
                dtype=np.dtype("float32"),
            )

            self.assertAlmostEqual(float(hyperspectral[0][-1]), float(hyperspectral[0][-2]), places=6)
            self.assertEqual(interpolation_summary["interpolated_row_count"], 1)
            self.assertEqual(interpolation_summary["rows_with_trailing_gaps"], 1)
            self.assertEqual(interpolation_summary["max_trailing_gap_count"], 1)

    def test_load_siac_rows_rejects_rows_with_too_few_numeric_nm_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.csv"
            spectra_path = root / "spectra.csv"
            _write_csv(
                metadata_path,
                ["source_id", "spectrum_id", "sample_name"],
                [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a"}],
            )
            row = {"source_id": "s1", "spectrum_id": "a", "sample_name": "a", **{column: "" for column in NM_COLUMNS}}
            row["nm_400"] = 0.1
            _write_csv(
                spectra_path,
                ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
                [row],
            )

            with self.assertRaises(PreparedLibraryBuildError):
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

    def test_load_siac_rows_rejects_large_internal_gap_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.csv"
            spectra_path = root / "spectra.csv"
            _write_csv(
                metadata_path,
                ["source_id", "spectrum_id", "sample_name"],
                [{"source_id": "s1", "spectrum_id": "a", "sample_name": "a"}],
            )
            row = {"source_id": "s1", "spectrum_id": "a", "sample_name": "a", **_spectrum_values(0.1, 0.2, 0.3)}
            for wavelength in range(900, 909):
                row[f"nm_{wavelength}"] = ""
            _write_csv(
                spectra_path,
                ["source_id", "spectrum_id", "sample_name", *NM_COLUMNS],
                [row],
            )

            with self.assertRaises(PreparedLibraryBuildError) as error_context:
                mapping_module._load_siac_rows(metadata_path, spectra_path, dtype=np.dtype("float32"))

            self.assertEqual(error_context.exception.context["internal_gap_count"], 9)

    def test_spectral_mapper_rejects_incompatible_manifest_and_missing_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            manifest_path = fixture["prepared_root"] / "manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

            manifest_payload["schema_version"] = "1.2.0"
            manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryCompatibilityError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            manifest_path = fixture["prepared_root"] / "manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["schema_version"] = "4.0.0"
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
            os.remove(fixture["prepared_root"] / "checksums.json")
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

    def test_validate_prepared_library_verifies_checksums_and_can_skip_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, manifest = _prepare_fixture(Path(tmpdir))

            validated = validate_prepared_library(fixture["prepared_root"])
            self.assertEqual(validated.to_dict(), manifest.to_dict())

            checksums_path = fixture["prepared_root"] / "checksums.json"
            checksums_payload = json.loads(checksums_path.read_text(encoding="utf-8"))
            checksums_payload["files"]["manifest.json"] = "0" * 64
            checksums_path.write_text(json.dumps(checksums_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                validate_prepared_library(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            os.remove(fixture["prepared_root"] / "checksums.json")
            with self.assertRaises(PreparedLibraryValidationError):
                validate_prepared_library(fixture["prepared_root"], verify_checksums=False)

    def test_prepared_runtime_rejects_malformed_manifest_and_row_index_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            manifest_path = fixture["prepared_root"] / "manifest.json"
            manifest_path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest_payload["row_count"]
            manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            connection = duckdb.connect()
            try:
                relation = connection.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY row_index",
                    [str(fixture["prepared_root"] / "mapping_metadata.parquet")],
                )
                rows = relation.fetchall()
                columns = [description[0] for description in relation.description]
            finally:
                connection.close()
            row_dicts = [dict(zip(columns, row)) for row in rows]
            row_dicts[1]["row_index"] = 3
            temp_csv = fixture["prepared_root"] / "mapping_metadata_broken.csv"
            _write_csv(temp_csv, columns, row_dicts)
            connection = duckdb.connect()
            try:
                connection.execute(
                    f"""
                    COPY (
                      SELECT * FROM read_csv_auto('{str(temp_csv).replace("'", "''")}', HEADER=TRUE, SAMPLE_SIZE=-1)
                    ) TO '{str(fixture["prepared_root"] / "mapping_metadata.parquet").replace("'", "''")}' (FORMAT PARQUET)
                    """
                )
            finally:
                connection.close()
            temp_csv.unlink()
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

    def test_prepared_runtime_rejects_duplicate_sensors_row_mismatches_and_checksum_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))

            mismatched_schema_version = dict(sensor_payload)
            mismatched_schema_version["schema_version"] = "1.2.0"
            sensor_schema_path.write_text(json.dumps(mismatched_schema_version, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))
            bad_canonical_grid = dict(sensor_payload)
            bad_canonical_grid["canonical_wavelength_grid"] = {"start_nm": 401, "end_nm": 2500, "step_nm": 1}
            sensor_schema_path.write_text(json.dumps(bad_canonical_grid, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))
            del sensor_payload["schema_version"]
            sensor_schema_path.write_text(json.dumps(sensor_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))
            del sensor_payload["canonical_wavelength_grid"]
            sensor_schema_path.write_text(json.dumps(sensor_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))
            duplicated_payload = dict(sensor_payload)
            duplicated_payload["sensors"] = [sensor_payload["sensors"][0], sensor_payload["sensors"][0]]
            sensor_schema_path.write_text(json.dumps(duplicated_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            sensor_schema_path = fixture["prepared_root"] / "sensor_schema.json"
            sensor_payload = json.loads(sensor_schema_path.read_text(encoding="utf-8"))
            band_payload = sensor_payload["sensors"][0]["bands"][0]
            band_payload["wavelength_nm"] = band_payload["response_definition"]["wavelength_nm"]
            band_payload["rsr"] = band_payload["response_definition"]["response"]
            del band_payload["response_definition"]
            sensor_schema_path.write_text(json.dumps(sensor_payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            metadata_path = fixture["prepared_root"] / "mapping_metadata.parquet"
            connection = duckdb.connect()
            try:
                connection.execute(
                    f"""
                    COPY (
                      SELECT row_index, source_id, spectrum_id FROM read_parquet('{str(metadata_path).replace("'", "''")}')
                    ) TO '{str(metadata_path).replace("'", "''")}' (FORMAT PARQUET)
                    """
                )
            finally:
                connection.close()
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            metadata_path = fixture["prepared_root"] / "mapping_metadata.parquet"
            connection = duckdb.connect()
            try:
                connection.execute(
                    f"""
                    COPY (
                      SELECT * FROM read_parquet('{str(metadata_path).replace("'", "''")}') LIMIT 3
                    ) TO '{str(metadata_path).replace("'", "''")}' (FORMAT PARQUET)
                    """
                )
            finally:
                connection.close()
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            metadata_path = fixture["prepared_root"] / "mapping_metadata.parquet"
            connection = duckdb.connect()
            try:
                relation = connection.execute(
                    "SELECT * FROM read_parquet(?) ORDER BY row_index",
                    [str(metadata_path)],
                )
                rows = relation.fetchall()
                columns = [description[0] for description in relation.description]
            finally:
                connection.close()
            row_dicts = [dict(zip(columns, row)) for row in rows]
            row_dicts[1]["source_id"] = row_dicts[0]["source_id"]
            row_dicts[1]["spectrum_id"] = row_dicts[0]["spectrum_id"]
            row_dicts[1]["sample_name"] = row_dicts[0]["sample_name"]
            temp_csv = fixture["prepared_root"] / "mapping_metadata_duplicate.csv"
            _write_csv(temp_csv, columns, row_dicts)
            connection = duckdb.connect()
            try:
                connection.execute(
                    f"""
                    COPY (
                      SELECT * FROM read_csv_auto('{str(temp_csv).replace("'", "''")}', HEADER=TRUE, SAMPLE_SIZE=-1)
                    ) TO '{str(metadata_path).replace("'", "''")}' (FORMAT PARQUET)
                    """
                )
            finally:
                connection.close()
            temp_csv.unlink()
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            hyperspectral_path = fixture["prepared_root"] / "hyperspectral_vnir.npy"
            hyperspectral_path.write_bytes(b"not-a-valid-npy")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            source_matrix_path = fixture["prepared_root"] / "source_sensor_a_swir.npy"
            source_matrix_path.write_bytes(b"not-a-valid-npy")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"])

            fixture, _ = _prepare_fixture(Path(tmpdir))
            (fixture["prepared_root"] / "hyperspectral_vnir.npy").write_bytes(b"tampered")
            with self.assertRaises(PreparedLibraryValidationError):
                SpectralMapper(fixture["prepared_root"], verify_checksums=True)

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
                    neighbor_estimator="mean",
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
                    neighbor_estimator="mean",
                    candidate_row_indices=np.array([], dtype=np.int64),
                )

            self.assertIs(mapper._source_queries("sensor_a"), mapper._source_queries("sensor_a"))
            self.assertEqual(mapper._simulate_target_sensor("sensor_b", {})[1], ())
            self.assertIsNone(mapper._simulate_target_sensor("sensor_b", {})[0])
            self.assertGreater(mapper._band_response("sensor_b", mapper.get_sensor_schema("sensor_b").bands[0], segment_only=False).size, 601)

            with patch.object(mapper, "_band_response", return_value=np.zeros(601, dtype=np.float64)):
                with self.assertRaises(SensorSchemaError):
                    mapper._simulate_target_sensor("sensor_b", {"vnir": np.ones(601, dtype=np.float64)})

            with patch.object(schema_module, "_resample_band_response", return_value=np.zeros(601, dtype=np.float64)):
                with self.assertRaises(SensorSchemaError):
                    mapping_module._simulate_segment_matrix(
                        np.ones((2, 601), dtype=np.float32),
                        mapper.get_sensor_schema("sensor_a").bands_for_segment("vnir"),
                        dtype=np.dtype("float32"),
                    )

    def test_output_mode_validation_and_target_sensor_failures_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = _build_fixture(Path(tmpdir))
            sensor_c = _custom_sensor_payload(
                sensor_id="sensor_c",
                bands=[
                    _custom_band_payload(
                        band_id="only_swir",
                        segment="swir",
                        wavelength_nm=[1695.0, 1700.0, 1705.0],
                        response=[0.2, 1.0, 0.2],
                    )
                ],
            )
            (fixture["srf_root"] / "sensor_c.json").write_text(json.dumps(sensor_c, indent=2) + "\n", encoding="utf-8")
            build_mapping_library(
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
                    neighbor_estimator="mean",
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
                        "build-mapping-library",
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
                        "--neighbor-estimator",
                        "distance_weighted_mean",
                        "--report",
                        str(report_path),
                    ]
                )
            self.assertEqual(benchmark_exit, 0)
            benchmark_payload = json.loads(stdout.getvalue())
            self.assertEqual(benchmark_payload["neighbor_estimator"], "distance_weighted_mean")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["neighbor_estimator"], "distance_weighted_mean")
            self.assertEqual(report["target_sensor_id"], "sensor_b")
            self.assertEqual(report["target_sensor"]["band_ids"], ["target_vnir", "target_swir"])

    def test_resolve_prepared_library_root_returns_override_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            override_root = Path(tmpdir) / "custom-runtime"

            with patch("spectral_library.distribution.resolver.download_prepared_library") as download_mock:
                resolved_root = resolve_prepared_library_root(override_root)

            self.assertEqual(resolved_root, override_root)
            download_mock.assert_not_called()

    def test_resolve_prepared_library_root_downloads_default_runtime_when_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_root = Path(tmpdir) / "cache" / "spectral-library" / "prepared-runtime"
            with patch("spectral_library.distribution.resolver.default_prepared_runtime_root", return_value=expected_root):
                with patch("spectral_library.distribution.resolver._cached_prepared_runtime_is_valid", return_value=False):
                    with patch(
                        "spectral_library.distribution.resolver.download_prepared_library",
                        return_value=expected_root,
                    ) as download_mock:
                        resolved_root = resolve_prepared_library_root()

            self.assertEqual(resolved_root, expected_root)
            download_mock.assert_called_once_with(expected_root, tag=f"v{PACKAGE_VERSION}")

    def test_resolve_prepared_library_root_treats_blank_string_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_root = Path(tmpdir) / "cache" / "spectral-library" / "prepared-runtime"
            with patch("spectral_library.distribution.resolver.default_prepared_runtime_root", return_value=expected_root):
                with patch("spectral_library.distribution.resolver._cached_prepared_runtime_is_valid", return_value=False):
                    with patch(
                        "spectral_library.distribution.resolver.download_prepared_library",
                        return_value=expected_root,
                    ) as download_mock:
                        resolved_root = resolve_prepared_library_root("")

            self.assertEqual(resolved_root, expected_root)
            download_mock.assert_called_once_with(expected_root, tag=f"v{PACKAGE_VERSION}")

    def test_resolve_prepared_library_root_redownloads_when_cached_runtime_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_root = Path(tmpdir) / "cache" / "spectral-library" / "prepared-runtime"
            cached_root.mkdir(parents=True, exist_ok=True)
            (cached_root / "stale.txt").write_text("stale", encoding="utf-8")

            with patch("spectral_library.distribution.resolver.default_prepared_runtime_root", return_value=cached_root):
                with patch(
                    "spectral_library.distribution.resolver._cached_prepared_runtime_is_valid",
                    side_effect=[False, False],
                ):
                    with patch(
                        "spectral_library.distribution.resolver.download_prepared_library",
                        return_value=cached_root,
                    ) as download_mock:
                        resolved_root = resolve_prepared_library_root()

            self.assertEqual(resolved_root, cached_root)
            self.assertFalse((cached_root / "stale.txt").exists())
            download_mock.assert_called_once_with(cached_root, tag=f"v{PACKAGE_VERSION}")

    def test_resolve_prepared_library_root_reuses_cached_default_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_root = Path(tmpdir) / "cache" / "spectral-library" / "prepared-runtime"
            cached_root.mkdir(parents=True, exist_ok=True)

            with patch("spectral_library.distribution.resolver.default_prepared_runtime_root", return_value=cached_root):
                with patch("spectral_library.distribution.resolver._cached_prepared_runtime_is_valid", return_value=True):
                    with patch("spectral_library.distribution.resolver.download_prepared_library") as download_mock:
                        resolved_root = resolve_prepared_library_root()

            self.assertEqual(resolved_root, cached_root)
            download_mock.assert_not_called()

    def test_resolve_prepared_library_root_treats_empty_string_as_default_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_root = Path(tmpdir) / "cache" / "spectral-library" / "prepared-runtime"
            with patch("spectral_library.distribution.resolver.default_prepared_runtime_root", return_value=expected_root):
                with patch("spectral_library.distribution.resolver._cached_prepared_runtime_is_valid", return_value=False):
                    with patch(
                        "spectral_library.distribution.resolver.download_prepared_library",
                        return_value=expected_root,
                    ) as download_mock:
                        resolved_root = resolve_prepared_library_root("")

            self.assertEqual(resolved_root, expected_root)
            download_mock.assert_called_once_with(expected_root, tag=f"v{PACKAGE_VERSION}")

    def test_default_prepared_runtime_root_uses_xdg_cache_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"XDG_CACHE_HOME": tmpdir}, clear=False):
                with patch("spectral_library.distribution.resolver.os.name", "posix"):
                    with patch("spectral_library.distribution.resolver.sys.platform", "linux"):
                        resolved_root = default_prepared_runtime_root()

            self.assertEqual(
                resolved_root,
                Path(tmpdir) / "spectral-library" / "prepared-runtime" / f"v{PACKAGE_VERSION}",
            )

    def test_map_reflectance_command_uses_default_runtime_when_prepared_root_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": 0.80},
                    {"band_id": "swir", "reflectance": 0.20},
                ],
            )

            with patch.object(cli, "resolve_prepared_library_root", return_value=fixture["prepared_root"]) as resolver_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance",
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
            resolver_mock.assert_called_once_with(None)
            self.assertTrue(output_path.exists())

    def test_map_reflectance_batch_command_uses_default_runtime_when_prepared_root_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch.csv"
            output_path = root / "mapped.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                ],
            )

            with patch.object(cli, "resolve_prepared_library_root", return_value=fixture["prepared_root"]) as resolver_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
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
            resolver_mock.assert_called_once_with(None)
            self.assertTrue(output_path.exists())

    def test_benchmark_mapping_command_uses_default_runtime_when_prepared_root_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            report_path = root / "benchmark.json"

            with patch.object(cli, "resolve_prepared_library_root", return_value=fixture["prepared_root"]) as resolver_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "benchmark-mapping",
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

            self.assertEqual(exit_code, 0)
            resolver_mock.assert_called_once_with(None)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["prepared_root"], str(fixture["prepared_root"]))
            self.assertEqual(payload["prepared_root_source"], "published_default")

    def test_map_reflectance_batch_command_uses_default_runtime_when_prepared_root_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch.csv"
            output_path = root / "mapped_batch.csv"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance"],
                [
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.80},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.20},
                ],
            )

            with patch.object(cli, "resolve_prepared_library_root", return_value=fixture["prepared_root"]) as resolver_mock:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
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
            resolver_mock.assert_called_once_with(None)
            self.assertTrue(output_path.exists())

    def test_map_reflectance_command_reports_download_failure_when_default_runtime_resolution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "query.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": 0.80},
                ],
            )

            stderr = io.StringIO()
            with patch.object(
                cli,
                "resolve_prepared_library_root",
                side_effect=RuntimeDownloadError("Failed to query GitHub releases: boom"),
            ):
                with contextlib.redirect_stderr(stderr):
                    exit_code = cli.main_with_args(
                        [
                            "--json-errors",
                            "map-reflectance",
                            "--source-sensor",
                            "sensor_a",
                            "--target-sensor",
                            "sensor_b",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "target_sensor",
                            "--output",
                            str(root / "unused.csv"),
                        ]
                    )

            self.assertEqual(exit_code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["command"], "map-reflectance")
            self.assertEqual(payload["error_code"], "download_failed")
            self.assertEqual(payload["context"]["prepared_root"], None)

    def test_prepare_command_accepts_rsrf_sensor_without_srf_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = _build_fixture(root)
            schema = SensorSRFSchema.from_dict(
                _custom_sensor_payload(
                    sensor_id="sentinel-2c_msi",
                    bands=[
                        _custom_band_payload(
                            band_id="blue",
                            segment="vnir",
                            wavelength_nm=[450.0, 460.0, 470.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                        _custom_band_payload(
                            band_id="nir",
                            segment="vnir",
                            wavelength_nm=[840.0, 850.0, 860.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                        _custom_band_payload(
                            band_id="swir1",
                            segment="swir",
                            wavelength_nm=[1600.0, 1610.0, 1620.0],
                            response=[0.1, 1.0, 0.1],
                        ),
                    ],
                )
            )

            stdout = io.StringIO()
            with patch.object(schema_module, "_load_rsrf_sensor_schema", return_value=schema):
                with contextlib.redirect_stdout(stdout):
                    prepare_exit = cli.main_with_args(
                        [
                            "build-mapping-library",
                            "--siac-root",
                            str(fixture["siac_root"]),
                            "--source-sensor",
                            "sentinel-2c_msi",
                            "--output-root",
                            str(fixture["prepared_root"]),
                        ]
                    )
            self.assertEqual(prepare_exit, 0)
            self.assertTrue((fixture["prepared_root"] / "source_sentinel-2c_msi_vnir.npy").exists())

    def test_legacy_prepare_command_alias_still_builds_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = _build_fixture(root)

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli.main_with_args(
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

            self.assertEqual(exit_code, 0)
            self.assertTrue((fixture["prepared_root"] / "manifest.json").exists())

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

    def test_map_reflectance_command_can_exclude_sample_name_from_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"

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
                        "--exclude-sample-name",
                        "vnir_high",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["excluded_sample_names"], ["vnir_high"])
            self.assertNotIn(b"\r\n", output_path.read_bytes())
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["reflectance"]), 0.6, places=6)
            self.assertAlmostEqual(float(rows[1]["reflectance"]), 0.25, places=6)

    def test_map_reflectance_command_emits_json_logs_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": 0.79},
                    {"band_id": "swir", "reflectance": 0.21},
                ],
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-logs",
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
                        "2",
                        "--neighbor-estimator",
                        "distance_weighted_mean",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            stdout_payload = json.loads(stdout.getvalue())
            self.assertEqual(stdout_payload["output_path"], str(output_path))
            self.assertEqual(stdout_payload["neighbor_estimator"], "distance_weighted_mean")
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["reflectance"]), 0.79, places=6)
            self.assertAlmostEqual(float(rows[1]["reflectance"]), 0.21, places=6)
            log_payloads = [json.loads(line) for line in stderr.getvalue().splitlines() if line.strip()]
            self.assertEqual([payload["event"] for payload in log_payloads], ["command_started", "command_completed"])
            self.assertEqual(log_payloads[0]["context"]["input_path"], str(input_path))
            self.assertEqual(log_payloads[0]["context"]["neighbor_estimator"], "distance_weighted_mean")
            self.assertEqual(log_payloads[0]["level"], "info")
            self.assertEqual(log_payloads[1]["context"]["written_rows"], 2)
            self.assertEqual(log_payloads[1]["context"]["segment_statuses"], {})
            self.assertEqual(log_payloads[1]["level"], "info")
            self.assertIsInstance(log_payloads[1]["elapsed_ms"], int)
            self.assertGreaterEqual(log_payloads[1]["elapsed_ms"], 0)

    def test_map_reflectance_command_can_write_diagnostics_and_neighbor_review_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "query.csv"
            output_path = root / "mapped.csv"
            diagnostics_path = root / "diagnostics.json"
            neighbor_review_path = root / "neighbor_review.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": 0.79},
                    {"band_id": "swir", "reflectance": 0.21},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
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
                        "2",
                        "--neighbor-estimator",
                        "simplex_mixture",
                        "--output",
                        str(output_path),
                        "--diagnostics-output",
                        str(diagnostics_path),
                        "--neighbor-review-output",
                        str(neighbor_review_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["diagnostics_output"], str(diagnostics_path))
            self.assertEqual(payload["neighbor_review_output"], str(neighbor_review_path))
            self.assertEqual(payload["knn_backend"], "numpy")
            self.assertAlmostEqual(payload["knn_eps"], 0.0)
            diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics_payload["diagnostics"]["neighbor_estimator"], "simplex_mixture")
            self.assertEqual(diagnostics_payload["diagnostics"]["knn_backend"], "numpy")
            with neighbor_review_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["segment"] for row in rows}, {"vnir", "swir"})
            self.assertEqual({row["sample_id"] for row in rows}, {"sample_000001"})
            self.assertIn("neighbor_weight", rows[0])
            self.assertEqual(rows[0]["knn_backend"], "numpy")
            self.assertEqual(rows[0]["knn_eps"], "0.0")


    def test_map_reflectance_batch_command_accepts_long_input_and_writes_wide_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_long.csv"
            output_path = root / "batch_target.csv"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance", "valid"],
                [
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.80, "valid": "true"},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.20, "valid": "true"},
                    {"sample_id": "beta", "band_id": "blue", "reflectance": 0.80, "valid": "true"},
                    {"sample_id": "beta", "band_id": "swir", "reflectance": 0.90, "valid": "false"},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["sample_count"], 2)
            self.assertEqual(payload["output_columns"], ["target_vnir", "target_swir"])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sample_id"], "alpha")
            self.assertAlmostEqual(float(rows[0]["target_vnir"]), 0.8, places=6)
            self.assertAlmostEqual(float(rows[0]["target_swir"]), 0.2, places=6)
            self.assertEqual(rows[1]["sample_id"], "beta")
            self.assertAlmostEqual(float(rows[1]["target_vnir"]), 0.8, places=6)
            self.assertEqual(rows[1]["target_swir"], "")

    def test_map_reflectance_batch_command_accepts_wide_input_and_writes_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"
            output_path = root / "batch_vnir.csv"
            diagnostics_path = root / "batch_diagnostics.json"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir", "valid_swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20, "valid_swir": "true"},
                    {"sample_id": "beta", "blue": 0.60, "swir": 0.90, "valid_swir": "false"},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
                        "--diagnostics-output",
                        str(diagnostics_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["diagnostics_output"], str(diagnostics_path))
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sample_id"], "alpha")
            self.assertIn("nm_400", rows[0])
            self.assertIn("nm_1000", rows[0])
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostics["sample_ids"], ["alpha", "beta"])
            self.assertEqual(diagnostics["results"][1]["sample_id"], "beta")

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_can_write_zarr_output(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"
            output_path = root / "batch_vnir.zarr"
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "stale.txt").write_text("stale", encoding="utf-8")

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                    {"sample_id": "beta", "blue": 0.15, "swir": 0.25},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
                        "--output-format",
                        "zarr",
                        "--output-chunk-size",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["output_format"], "zarr")
            self.assertEqual(payload["output_columns"][0], "nm_400")
            self.assertEqual(payload["output_columns"][-1], "nm_1000")
            self.assertFalse((output_path / "stale.txt").exists())

            store = zarr.open_group(str(output_path), mode="r")
            self.assertEqual(store.attrs["output_mode"], "vnir_spectrum")
            self.assertEqual(store.attrs["sample_count"], 2)
            self.assertEqual(store.attrs["chunk_size"], 1)
            self.assertEqual(store["reflectance"].shape, (2, len(mapping_module.VNIR_WAVELENGTHS)))
            self.assertEqual(store["source_fit_rmse"].shape, (2,))
            self.assertEqual(store["wavelength_nm"][0], 400)
            self.assertEqual(store["wavelength_nm"][-1], 1000)
            sample_ids = [
                value if isinstance(value, str) else value.decode("utf-8").rstrip("\x00")
                for value in store["sample_id"][:]
            ]
            self.assertEqual(sample_ids, ["alpha", "beta"])
            self.assertTrue(np.isfinite(store["reflectance"][:]).all())
            self.assertTrue(payload["streamed_output"])

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_zarr_path_does_not_call_map_reflectance_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                    {"sample_id": "beta", "blue": 0.15, "swir": 0.25},
                ],
            )

            with patch.object(
                mapping_module.SpectralMapper,
                "map_reflectance_batch",
                side_effect=AssertionError("zarr path should use dense array export"),
            ):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(root / "batch_vnir.zarr"),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["streamed_output"])

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_streams_wide_input_without_materializing_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                    {"sample_id": "beta", "blue": 0.15, "swir": 0.25},
                ],
            )

            with patch.object(cli, "_load_batch_reflectance_input", side_effect=AssertionError("materialized loader should not be used")):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(root / "batch_vnir.zarr"),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["streamed_output"])

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_streams_grouped_long_input_without_materializing_full_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_long.csv"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance"],
                [
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.80},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.20},
                    {"sample_id": "beta", "band_id": "blue", "reflectance": 0.15},
                    {"sample_id": "beta", "band_id": "swir", "reflectance": 0.25},
                ],
            )

            with patch.object(cli, "_load_batch_reflectance_input", side_effect=AssertionError("materialized loader should not be used")):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(root / "batch_vnir.zarr"),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["streamed_output"])

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_streams_interleaved_long_zarr_input_via_disk_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_long_interleaved.csv"
            output_path = root / "batch_vnir_interleaved.zarr"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance"],
                [
                    {"sample_id": "gamma", "band_id": "blue", "reflectance": 0.80},
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.15},
                    {"sample_id": "gamma", "band_id": "swir", "reflectance": 0.20},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.25},
                ],
            )

            with patch.object(cli, "_load_batch_reflectance_input", side_effect=AssertionError("materialized loader should not be used")):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main_with_args(
                        [
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(output_path),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["streamed_output"])

            import zarr

            store = zarr.open_group(str(output_path), mode="r")
            sample_ids = [
                value if isinstance(value, str) else value.decode("utf-8").rstrip("\x00")
                for value in store["sample_id"][:]
            ]
            self.assertEqual(sample_ids, ["gamma", "alpha"])
            self.assertEqual(store.attrs["sample_count"], 2)

    @unittest.skipIf(ZARR_AVAILABLE, "dependency guard only applies when zarr is absent")
    def test_map_reflectance_batch_command_reports_missing_zarr_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                ],
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-errors",
                        "map-reflectance-batch",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--source-sensor",
                        "sensor_a",
                        "--input",
                        str(input_path),
                        "--output-mode",
                        "vnir_spectrum",
                        "--output-format",
                        "zarr",
                        "--output",
                        str(root / "batch_vnir.zarr"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["error_code"], "invalid_mapping_input")
            self.assertEqual(payload["context"]["output_format"], "zarr")

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_cleans_up_temp_zarr_store_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_wide.csv"
            output_path = root / "batch_vnir.zarr"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                ],
            )

            stderr = io.StringIO()
            with patch.object(
                mapping_module.SpectralMapper,
                "_map_reflectance_batch_output_arrays",
                side_effect=MappingInputError("synthetic zarr failure"),
            ):
                with contextlib.redirect_stderr(stderr):
                    exit_code = cli.main_with_args(
                        [
                            "--json-errors",
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(output_path),
                        ]
                    )

            self.assertEqual(exit_code, 2)
            self.assertFalse(output_path.exists())
            self.assertEqual(list(root.glob(f".{output_path.name}.tmp-*")), [])

    @unittest.skipUnless(ZARR_AVAILABLE, "zarr is not installed")
    def test_map_reflectance_batch_command_preserves_existing_zarr_store_when_finalization_fails(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            first_input_path = root / "batch_first.csv"
            second_input_path = root / "batch_second.csv"
            output_path = root / "batch_vnir.zarr"

            _write_csv(
                first_input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                ],
            )
            _write_csv(
                second_input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "beta", "blue": 0.15, "swir": 0.25},
                ],
            )

            exit_code = cli.main_with_args(
                [
                    "map-reflectance-batch",
                    "--prepared-root",
                    str(fixture["prepared_root"]),
                    "--source-sensor",
                    "sensor_a",
                    "--input",
                    str(first_input_path),
                    "--output-mode",
                    "vnir_spectrum",
                    "--output-format",
                    "zarr",
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)

            with patch.object(cli, "_finalize_output_path", side_effect=OSError("synthetic rename failure")):
                with self.assertRaises(OSError):
                    cli.main_with_args(
                        [
                            "map-reflectance-batch",
                            "--prepared-root",
                            str(fixture["prepared_root"]),
                            "--source-sensor",
                            "sensor_a",
                            "--input",
                            str(second_input_path),
                            "--output-mode",
                            "vnir_spectrum",
                            "--output-format",
                            "zarr",
                            "--output",
                            str(output_path),
                        ]
                    )

            store = zarr.open_group(str(output_path), mode="r")
            sample_ids = [
                value if isinstance(value, str) else value.decode("utf-8").rstrip("\x00")
                for value in store["sample_id"][:]
            ]
            self.assertEqual(sample_ids, ["alpha"])
            self.assertEqual(list(root.glob(f".{output_path.name}.tmp-*")), [])

    def test_map_reflectance_batch_command_emits_json_logs_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_long.csv"
            output_path = root / "batch_target.csv"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance"],
                [
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.80},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.20},
                    {"sample_id": "beta", "band_id": "blue", "reflectance": 0.15},
                    {"sample_id": "beta", "band_id": "swir", "reflectance": 0.25},
                ],
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-logs",
                        "map-reflectance-batch",
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
            stdout_payload = json.loads(stdout.getvalue())
            self.assertEqual(stdout_payload["sample_count"], 2)
            log_payloads = [json.loads(line) for line in stderr.getvalue().splitlines() if line.strip()]
            self.assertEqual([payload["event"] for payload in log_payloads], ["command_started", "command_completed"])
            self.assertEqual(log_payloads[0]["context"]["input_path"], str(input_path))
            self.assertEqual(log_payloads[0]["level"], "info")
            self.assertEqual(log_payloads[0]["context"]["knn_backend"], "numpy")
            self.assertEqual(log_payloads[0]["context"]["knn_eps"], 0.0)
            self.assertEqual(log_payloads[1]["context"]["sample_count"], 2)
            self.assertEqual(log_payloads[1]["context"]["output_columns"], ["target_vnir", "target_swir"])
            self.assertEqual(log_payloads[1]["level"], "info")
            self.assertIsInstance(log_payloads[1]["elapsed_ms"], int)
            self.assertGreaterEqual(log_payloads[1]["elapsed_ms"], 0)

    def test_map_reflectance_batch_command_can_write_neighbor_review_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_long.csv"
            output_path = root / "batch_target.csv"
            neighbor_review_path = root / "batch_neighbor_review.csv"

            _write_csv(
                input_path,
                ["sample_id", "band_id", "reflectance"],
                [
                    {"sample_id": "alpha", "band_id": "blue", "reflectance": 0.79},
                    {"sample_id": "alpha", "band_id": "swir", "reflectance": 0.21},
                    {"sample_id": "beta", "band_id": "blue", "reflectance": 0.15},
                    {"sample_id": "beta", "band_id": "swir", "reflectance": 0.25},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
                        "2",
                        "--neighbor-estimator",
                        "simplex_mixture",
                        "--output",
                        str(output_path),
                        "--neighbor-review-output",
                        str(neighbor_review_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["neighbor_review_output"], str(neighbor_review_path))
            self.assertEqual(payload["knn_backend"], "numpy")
            with neighbor_review_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({row["sample_id"] for row in rows}, {"alpha", "beta"})
            self.assertEqual({row["neighbor_estimator"] for row in rows}, {"simplex_mixture"})
            self.assertEqual({row["knn_backend"] for row in rows}, {"numpy"})
            self.assertIn("neighbor_band_values", rows[0])

    def test_map_reflectance_command_emits_failure_log_before_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "invalid_query.csv"

            _write_csv(
                input_path,
                ["band_id", "reflectance"],
                [
                    {"band_id": "blue", "reflectance": "not-a-number"},
                ],
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-logs",
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
                        "--output",
                        str(root / "unused.csv"),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            stderr_text = stderr.getvalue()
            first_newline = stderr_text.find("\n")
            second_newline = stderr_text.find("\n", first_newline + 1)
            self.assertGreaterEqual(first_newline, 0)
            self.assertGreaterEqual(second_newline, 0)
            started_payload = json.loads(stderr_text[:first_newline])
            failed_payload = json.loads(stderr_text[first_newline + 1 : second_newline])
            error_payload = json.loads(stderr_text[second_newline + 1 :])
            self.assertEqual(started_payload["event"], "command_started")
            self.assertEqual(started_payload["level"], "info")
            self.assertEqual(failed_payload["event"], "command_failed")
            self.assertEqual(failed_payload["level"], "error")
            self.assertEqual(failed_payload["command"], "map-reflectance")
            self.assertEqual(failed_payload["context"]["error_code"], "invalid_input_csv")
            self.assertEqual(failed_payload["context"]["context"]["path"], str(input_path))
            self.assertIsInstance(failed_payload["elapsed_ms"], int)
            self.assertGreaterEqual(failed_payload["elapsed_ms"], 0)
            self.assertEqual(error_payload["command"], "map-reflectance")
            self.assertEqual(error_payload["error_code"], "invalid_input_csv")

    def test_map_reflectance_batch_command_reports_sample_context_in_json_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_invalid.csv"
            output_path = root / "batch_invalid_output.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "alpha", "blue": 0.80, "swir": 0.20},
                    {"sample_id": "beta", "blue": "not-a-number", "swir": 0.90},
                ],
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main_with_args(
                    [
                        "--json-errors",
                        "map-reflectance-batch",
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

            self.assertEqual(exit_code, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["command"], "map-reflectance-batch")
            self.assertEqual(payload["error_code"], "invalid_input_csv")
            self.assertEqual(payload["context"]["sample_id"], "beta")

    def test_map_reflectance_batch_command_can_self_exclude_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_self_exclude.csv"
            output_path = root / "batch_self_exclude_output.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "base", "blue": 0.15, "swir": 0.25},
                    {"sample_id": "vnir_high", "blue": 0.80, "swir": 0.20},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
                        "--self-exclude-sample-id",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["self_exclude_sample_id"])
            self.assertNotIn(b"\r\n", output_path.read_bytes())
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["base", "vnir_high"])
            self.assertAlmostEqual(float(rows[0]["target_vnir"]), 0.1, places=6)
            self.assertAlmostEqual(float(rows[0]["target_swir"]), 0.2, places=6)
            self.assertAlmostEqual(float(rows[1]["target_vnir"]), 0.6, places=6)
            self.assertAlmostEqual(float(rows[1]["target_swir"]), 0.25, places=6)

    def test_map_reflectance_batch_command_can_exclude_exact_row_ids_from_input_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_exclude_row_id.csv"
            output_path = root / "batch_exclude_row_id_output.csv"

            _write_csv(
                input_path,
                ["sample_id", "exclude_row_id", "blue", "swir"],
                [
                    {
                        "sample_id": "base_alias",
                        "exclude_row_id": "fixture_source:base:base",
                        "blue": 0.15,
                        "swir": 0.25,
                    },
                    {
                        "sample_id": "vnir_alias",
                        "exclude_row_id": "fixture_source:vnir_high:vnir_high",
                        "blue": 0.80,
                        "swir": 0.20,
                    },
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["input_exclude_row_id_column"])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["base_alias", "vnir_alias"])
            self.assertAlmostEqual(float(rows[0]["target_vnir"]), 0.1, places=6)
            self.assertAlmostEqual(float(rows[0]["target_swir"]), 0.2, places=6)
            self.assertAlmostEqual(float(rows[1]["target_vnir"]), 0.6, places=6)
            self.assertAlmostEqual(float(rows[1]["target_swir"]), 0.25, places=6)

    def test_map_reflectance_batch_self_exclude_ignores_unmatched_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, _ = _prepare_fixture(root)
            input_path = root / "batch_self_exclude_unmatched.csv"
            output_path = root / "batch_self_exclude_unmatched_output.csv"

            _write_csv(
                input_path,
                ["sample_id", "blue", "swir"],
                [
                    {"sample_id": "external_label", "blue": 0.15, "swir": 0.25},
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "map-reflectance-batch",
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
                        "--self-exclude-sample-id",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["self_exclude_sample_id"])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["external_label"])
            self.assertAlmostEqual(float(rows[0]["target_vnir"]), 0.15, places=6)
            self.assertAlmostEqual(float(rows[0]["target_swir"]), 0.25, places=6)

    def test_validate_prepared_library_command_and_version_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture, manifest = _prepare_fixture(root)
            input_path = root / "invalid_query.csv"
            output_path = root / "invalid_output.csv"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli.main_with_args(
                    [
                        "validate-prepared-library",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                    ]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["row_count"], manifest.row_count)
            self.assertTrue(payload["checksums_verified"])

            version_stdout = io.StringIO()
            with self.assertRaises(SystemExit) as version_exit, contextlib.redirect_stdout(version_stdout):
                cli.main_with_args(["--version"])
            self.assertEqual(version_exit.exception.code, 0)
            self.assertIn("0.6.0", version_stdout.getvalue())

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

            os.remove(fixture["prepared_root"] / "checksums.json")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                skip_exit = cli.main_with_args(
                    [
                        "--json-errors",
                        "validate-prepared-library",
                        "--prepared-root",
                        str(fixture["prepared_root"]),
                        "--no-verify-checksums",
                    ]
                )
            self.assertEqual(skip_exit, 2)
            payload = json.loads(stderr.getvalue())
            self.assertEqual(payload["command"], "validate-prepared-library")
            self.assertEqual(payload["error_code"], "invalid_prepared_library")


if __name__ == "__main__":
    unittest.main()
