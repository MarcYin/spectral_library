from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spectral_library import prepare_mapping_library

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_full_library_benchmarks.py"
SMOKE_FIXTURE_SCRIPT_PATH = REPO_ROOT / "scripts" / "create_mapping_smoke_fixture.py"
SMOKE_THRESHOLDS_PATH = REPO_ROOT / "benchmarks" / "smoke_thresholds.json"


def _load_benchmark_runner_module():
    spec = importlib.util.spec_from_file_location("run_full_library_benchmarks", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark runner from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_fixture_module():
    spec = importlib.util.spec_from_file_location("create_mapping_smoke_fixture", SMOKE_FIXTURE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load smoke fixture helper from {SMOKE_FIXTURE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullLibraryBenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_benchmark_runner_module()
        cls.smoke_fixture_module = _load_smoke_fixture_module()

    def test_main_uses_defaults_when_optional_filters_are_omitted(self) -> None:
        with patch.object(self.module, "run_benchmarks", return_value=0) as mock_run:
            exit_code = self.module.main(
                [
                    "--prepared-root",
                    "prepared",
                    "--output-root",
                    "out",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["neighbor_estimators"], ["simplex_mixture"])
        self.assertEqual(mock_run.call_args.kwargs["knn_backends"], ["numpy"])
        self.assertEqual(mock_run.call_args.kwargs["k_values"], [10])

    def test_main_does_not_accumulate_defaults_with_explicit_values(self) -> None:
        with patch.object(self.module, "run_benchmarks", return_value=0) as mock_run:
            exit_code = self.module.main(
                [
                    "--prepared-root",
                    "prepared",
                    "--output-root",
                    "out",
                    "--neighbor-estimator",
                    "distance_weighted_mean",
                    "--knn-backend",
                    "faiss",
                    "--k",
                    "5",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run.call_args.kwargs["neighbor_estimators"], ["distance_weighted_mean"])
        self.assertEqual(mock_run.call_args.kwargs["knn_backends"], ["faiss"])
        self.assertEqual(mock_run.call_args.kwargs["k_values"], [5])

    def test_run_benchmarks_records_numpy_baseline_comparison(self) -> None:
        class Manifest:
            source_sensors = ("sensor_a", "sensor_b")

        def fake_benchmark_mapping(
            prepared_root: Path,
            source_sensor: str,
            target_sensor: str,
            *,
            k: int,
            test_fraction: float,
            max_test_rows: int | None,
            random_seed: int,
            neighbor_estimator: str,
            knn_backend: str,
        ) -> dict[str, object]:
            del prepared_root, test_fraction, max_test_rows, random_seed, neighbor_estimator
            mae = 0.010 if knn_backend == "numpy" else 0.011
            rmse = 0.020 if knn_backend == "numpy" else 0.022
            full_mae = 0.030 if knn_backend == "numpy" else 0.031
            return {
                "source_sensor_id": source_sensor,
                "target_sensor_id": target_sensor,
                "k": int(k),
                "neighbor_estimator": "simplex_mixture",
                "knn_backend": knn_backend,
                "target_sensor": {
                    "retrieval": {
                        "mean_mae": mae,
                        "mean_rmse": rmse,
                    }
                },
                "full_spectrum": {
                    "retrieval": {
                        "mean_mae": full_mae,
                        "mean_rmse": 0.04,
                    }
                },
                "train_rows": 10,
                "test_rows": 2,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "bench"
            with patch.object(self.module, "validate_prepared_library", return_value=Manifest()):
                with patch.object(self.module, "benchmark_mapping", side_effect=fake_benchmark_mapping):
                    exit_code = self.module.run_benchmarks(
                        Path("prepared"),
                        source_sensors=["sensor_a", "sensor_b"],
                        neighbor_estimators=["simplex_mixture"],
                        knn_backends=["faiss"],
                        k_values=[5],
                        test_fraction=0.2,
                        max_test_rows=2,
                        random_seed=0,
                        output_root=output_root,
                        thresholds_path=None,
                        fail_on_thresholds=True,
                    )

            self.assertEqual(exit_code, 0)
            summary_payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["numpy_baseline_metrics"], list(self.module.QUALIFICATION_METRIC_KEYS))
            reports = json.loads((output_root / "reports.json").read_text(encoding="utf-8"))
            self.assertEqual(len(reports), 2)
            self.assertFalse(reports[0]["numpy_baseline"]["is_self_baseline"])
            metric_payload = reports[0]["numpy_baseline"]["metrics"]["target_sensor.retrieval.mean_mae"]
            self.assertAlmostEqual(metric_payload["baseline"], 0.01)
            self.assertAlmostEqual(metric_payload["delta"], 0.001)
            summary_rows = (output_root / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("numpy_baseline_target_mean_mae_delta", summary_rows.splitlines()[0])
            self.assertIn(",False", summary_rows)

    def test_run_benchmarks_can_fail_when_backend_drift_exceeds_threshold(self) -> None:
        class Manifest:
            source_sensors = ("sensor_a", "sensor_b")

        def fake_benchmark_mapping(
            prepared_root: Path,
            source_sensor: str,
            target_sensor: str,
            *,
            k: int,
            test_fraction: float,
            max_test_rows: int | None,
            random_seed: int,
            neighbor_estimator: str,
            knn_backend: str,
        ) -> dict[str, object]:
            del prepared_root, source_sensor, target_sensor, k, test_fraction, max_test_rows, random_seed, neighbor_estimator
            mae = 0.010 if knn_backend == "numpy" else 0.020
            return {
                "target_sensor": {"retrieval": {"mean_mae": mae, "mean_rmse": mae}},
                "full_spectrum": {"retrieval": {"mean_mae": mae, "mean_rmse": mae}},
                "train_rows": 10,
                "test_rows": 2,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "bench"
            thresholds_path = Path(tmpdir) / "thresholds.json"
            thresholds_path.write_text(
                json.dumps(
                    {
                        "defaults": {},
                        "scenarios": {},
                        "baseline_deltas": {
                            "defaults": {"target_sensor.retrieval.mean_mae": 0.001},
                            "scenarios": {},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(self.module, "validate_prepared_library", return_value=Manifest()):
                with patch.object(self.module, "benchmark_mapping", side_effect=fake_benchmark_mapping):
                    exit_code = self.module.run_benchmarks(
                        Path("prepared"),
                        source_sensors=["sensor_a", "sensor_b"],
                        neighbor_estimators=["simplex_mixture"],
                        knn_backends=["pynndescent"],
                        k_values=[5],
                        test_fraction=0.2,
                        max_test_rows=2,
                        random_seed=0,
                        output_root=output_root,
                        thresholds_path=thresholds_path,
                        fail_on_thresholds=True,
                    )

            self.assertEqual(exit_code, 2)
            summary_payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["threshold_failures"][0]["threshold_group"], "numpy_baseline_deltas")

    def test_smoke_fixture_supports_two_sensor_ann_qualification_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self.smoke_fixture_module.create_smoke_fixture(Path(tmpdir) / "smoke")
            prepared_root = Path(fixture["prepared_root"])
            prepare_mapping_library(
                Path(fixture["siac_root"]),
                Path(fixture["srf_root"]),
                prepared_root,
                source_sensors=["sensor_a", "sensor_b"],
            )

            output_root = Path(tmpdir) / "bench"
            exit_code = self.module.run_benchmarks(
                prepared_root,
                source_sensors=["sensor_a", "sensor_b"],
                neighbor_estimators=["simplex_mixture"],
                knn_backends=["numpy"],
                k_values=[1],
                test_fraction=0.5,
                max_test_rows=1,
                random_seed=0,
                output_root=output_root,
                thresholds_path=None,
                fail_on_thresholds=True,
            )

            self.assertEqual(exit_code, 0)
            summary_payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["source_sensors"], ["sensor_a", "sensor_b"])
            reports = json.loads((output_root / "reports.json").read_text(encoding="utf-8"))
            self.assertEqual(len(reports), 2)
            scenarios = {report["scenario_key"] for report in reports}
            self.assertIn("sensor_a->sensor_b|simplex_mixture|numpy|1", scenarios)
            self.assertIn("sensor_b->sensor_a|simplex_mixture|numpy|1", scenarios)

    def test_smoke_thresholds_skip_absolute_quality_gates_for_toy_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture = self.smoke_fixture_module.create_smoke_fixture(Path(tmpdir) / "smoke")
            prepared_root = Path(fixture["prepared_root"])
            prepare_mapping_library(
                Path(fixture["siac_root"]),
                Path(fixture["srf_root"]),
                prepared_root,
                source_sensors=["sensor_a", "sensor_b"],
            )

            output_root = Path(tmpdir) / "bench"
            exit_code = self.module.run_benchmarks(
                prepared_root,
                source_sensors=["sensor_a", "sensor_b"],
                neighbor_estimators=["simplex_mixture"],
                knn_backends=["numpy"],
                k_values=[2],
                test_fraction=0.2,
                max_test_rows=2,
                random_seed=0,
                output_root=output_root,
                thresholds_path=SMOKE_THRESHOLDS_PATH,
                fail_on_thresholds=True,
            )

            self.assertEqual(exit_code, 0)
            summary_payload = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["threshold_failures"], [])


if __name__ == "__main__":
    unittest.main()
