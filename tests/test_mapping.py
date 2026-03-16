from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np

import spectral_library.mapping as mapping_module
from spectral_library import (
    BatchMappingResult,
    MappingInputError,
    PreparedLibraryCompatibilityError,
    PreparedLibraryManifest,
    PreparedLibraryValidationError,
    SensorSRFSchema,
    SpectralMapper,
    benchmark_mapping,
    cli,
    prepare_mapping_library,
    validate_prepared_library,
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
        if wavelength < 800:
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

            self.assertEqual(manifest.schema_version, "1.2.0")
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

    def test_prepare_mapping_library_can_persist_faiss_indexes(self) -> None:
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
            with patch.object(mapping_module, "_load_faiss_module", return_value=BuildFaissModule):
                manifest = prepare_mapping_library(
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

            with patch.object(mapping_module, "_load_faiss_module", return_value=QueryFaissModule):
                mapper = SpectralMapper(fixture["prepared_root"])
                result = mapper.map_reflectance(
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

    def test_prepare_mapping_library_can_persist_scann_indexes_with_tiny_candidate_sets(self) -> None:
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
            with patch.object(mapping_module, "_load_scann_ops", return_value=FakeScannOps):
                manifest = prepare_mapping_library(
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
                result = mapper.map_reflectance(
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
            by_row = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.80, "swir": 0.20},
                output_mode="target_sensor",
                target_sensor="sensor_a",
                k=1,
                exclude_row_ids=["fixture_source:vnir_high:vnir_high"],
            )
            by_sample = mapper.map_reflectance(
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
            source_sensor = {
                "sensor_id": "sensor_overlap",
                "bands": [
                    {
                        "band_id": "blue",
                        "segment": "vnir",
                        "wavelength_nm": [445.0, 450.0, 455.0],
                        "rsr": [0.2, 1.0, 0.2],
                    },
                    {
                        "band_id": "nir",
                        "segment": "vnir",
                        "wavelength_nm": [845.0, 850.0, 855.0],
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
            (fixture["srf_root"] / "sensor_overlap.json").write_text(
                json.dumps(source_sensor, indent=2) + "\n",
                encoding="utf-8",
            )
            prepare_mapping_library(
                fixture["siac_root"],
                fixture["srf_root"],
                fixture["prepared_root"],
                ["sensor_overlap"],
            )
            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance(
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
            result = mapper.map_reflectance(
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

            mean_result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
            )
            weighted_result = mapper.map_reflectance(
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

    def test_simplex_mixture_neighbor_estimator_can_fit_convex_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))
            mapper = SpectralMapper(fixture["prepared_root"])

            simplex_result = mapper.map_reflectance(
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

    def test_scipy_ckdtree_knn_backend_can_match_numpy_backend(self) -> None:
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
            numpy_result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="simplex_mixture",
            )
            with patch.object(mapping_module, "_load_ckdtree_class", return_value=FakeKDTree):
                scipy_result = mapper.map_reflectance(
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
            numpy_result = mapper.map_reflectance(
                source_sensor="sensor_a",
                reflectance={"blue": 0.79, "swir": 0.21},
                output_mode="target_sensor",
                target_sensor="sensor_b",
                k=2,
                neighbor_estimator="simplex_mixture",
            )

            with patch.object(mapping_module, "_load_faiss_module", return_value=FakeFaissModule):
                faiss_result = mapper.map_reflectance(
                    source_sensor="sensor_a",
                    reflectance={"blue": 0.79, "swir": 0.21},
                    output_mode="target_sensor",
                    target_sensor="sensor_b",
                    k=2,
                    neighbor_estimator="simplex_mixture",
                    knn_backend="faiss",
                    knn_eps=0.2,
                )
            with patch.object(mapping_module, "_load_pynndescent_class", return_value=FakeNNDescent):
                pynndescent_result = mapper.map_reflectance_batch(
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
            with patch.object(mapping_module, "_load_scann_ops", return_value=FakeScannOps()):
                scann_result = mapper.map_reflectance(
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
                mapping_module,
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
                    mapping_module,
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

            result = mapper.map_reflectance(
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

    def test_segment_isolation_keeps_vnir_output_stable_when_only_swir_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            base = mapper.map_reflectance(
                source_sensor="sensor_a",
                target_sensor="sensor_b",
                reflectance={"blue": 0.80, "swir": 0.15},
                output_mode="target_sensor",
                k=1,
            )
            changed_swir = mapper.map_reflectance(
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

            self.assertIsInstance(result, BatchMappingResult)
            self.assertEqual(result.sample_ids, ("alpha", "beta"))
            self.assertEqual(result.results[0].neighbor_ids_by_segment["vnir"], ("fixture_source:vnir_high:vnir_high",))
            self.assertEqual(result.results[0].neighbor_ids_by_segment["swir"], ("fixture_source:vnir_high:vnir_high",))
            self.assertEqual(result.results[1].neighbor_ids_by_segment["vnir"], ("fixture_source:swir_high:swir_high",))
            self.assertEqual(result.results[1].neighbor_ids_by_segment["swir"], ("fixture_source:swir_high:swir_high",))
            assert result.results[0].target_reflectance is not None
            assert result.results[1].target_reflectance is not None
            self.assertTrue(np.allclose(result.results[0].target_reflectance, np.array([0.80, 0.20])))
            self.assertTrue(np.allclose(result.results[1].target_reflectance, np.array([0.10, 0.90])))

    def test_map_reflectance_batch_supports_public_exclusion_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture, _ = _prepare_fixture(Path(tmpdir))

            mapper = SpectralMapper(fixture["prepared_root"])
            result = mapper.map_reflectance_batch(
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

            self.assertEqual(batch.results[0].neighbor_ids_by_segment, single_alpha.neighbor_ids_by_segment)
            self.assertEqual(batch.results[1].neighbor_ids_by_segment, single_beta.neighbor_ids_by_segment)
            assert batch.results[0].reconstructed_full_spectrum is not None
            assert batch.results[1].reconstructed_full_spectrum is not None
            assert single_alpha.reconstructed_full_spectrum is not None
            assert single_beta.reconstructed_full_spectrum is not None
            self.assertTrue(np.allclose(batch.results[0].reconstructed_full_spectrum, single_alpha.reconstructed_full_spectrum))
            self.assertTrue(np.allclose(batch.results[1].reconstructed_full_spectrum, single_beta.reconstructed_full_spectrum))

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
            duplicated_payload = {"sensors": [sensor_payload["sensors"][0], sensor_payload["sensors"][0]]}
            sensor_schema_path.write_text(json.dumps(duplicated_payload, indent=2) + "\n", encoding="utf-8")
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
            self.assertEqual(log_payloads[1]["context"]["segment_statuses"], {"swir": "ok", "vnir": "ok"})
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
            self.assertIn("0.2.0", version_stdout.getvalue())

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
