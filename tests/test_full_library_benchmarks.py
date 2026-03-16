from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_full_library_benchmarks.py"


def _load_benchmark_runner_module():
    spec = importlib.util.spec_from_file_location("run_full_library_benchmarks", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark runner from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullLibraryBenchmarkRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_benchmark_runner_module()

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


if __name__ == "__main__":
    unittest.main()
