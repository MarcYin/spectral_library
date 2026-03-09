from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from spectral_library.quality_plots import EXPECTED_GRID_POINTS, generate_quality_plots


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class QualityPlotTests(unittest.TestCase):
    def test_generate_quality_plots_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tabular = root / "normalized" / "tabular"
            write_csv(
                tabular / "source_summary.csv",
                ["source_id", "source_name", "ingest_role", "normalized_spectra", "failure_count", "parsers"],
                [
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "normalized_spectra": 8,
                        "failure_count": 2,
                        "parsers": "csv_row_wide:8",
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "normalized_spectra": 5,
                        "failure_count": 0,
                        "parsers": "usgs_ascii:5",
                    },
                ],
            )
            write_csv(
                tabular / "spectra_metadata.csv",
                [
                    "source_id",
                    "source_name",
                    "ingest_role",
                    "spectrum_id",
                    "sample_name",
                    "input_path",
                    "parser",
                    "native_wavelength_count",
                    "native_min_nm",
                    "native_max_nm",
                    "native_spacing_nm",
                    "value_scale_applied",
                    "normalized_points",
                    "metadata_json",
                ],
                [
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src1_1",
                        "sample_name": "Sample 1",
                        "input_path": "src1.csv",
                        "parser": "csv_row_wide",
                        "native_wavelength_count": 2101,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1.0,
                        "value_scale_applied": 1.0,
                        "normalized_points": EXPECTED_GRID_POINTS,
                        "metadata_json": "{}",
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src2_1",
                        "sample_name": "Sample 2",
                        "input_path": "src2.csv",
                        "parser": "usgs_ascii",
                        "native_wavelength_count": 1800,
                        "native_min_nm": 350,
                        "native_max_nm": 2450,
                        "native_spacing_nm": 1.5,
                        "value_scale_applied": 1.0,
                        "normalized_points": 1900,
                        "metadata_json": "{}",
                    },
                ],
            )
            write_csv(
                tabular / "normalization_failures.csv",
                ["source_id", "input_path", "parser", "stage", "reason"],
                [
                    {
                        "source_id": "src1",
                        "input_path": "bad1.csv",
                        "parser": "file_parse",
                        "stage": "parse",
                        "reason": "no spectra detected",
                    },
                    {
                        "source_id": "src1",
                        "input_path": "bad2.csv",
                        "parser": "file_parse",
                        "stage": "parse",
                        "reason": "spectrum collapsed to fewer than two unique wavelengths",
                    },
                ],
            )

            summary = generate_quality_plots(root / "normalized", top_n_sources=10)

            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["normalized_spectra"], 13)
            self.assertEqual(summary["failure_rows"], 2)
            self.assertIn("metrics_path", summary)
            for path in summary["plots"].values():
                self.assertTrue(Path(path).exists(), path)

    def test_generate_quality_plots_handles_missing_failure_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tabular = root / "tabular"
            write_csv(
                tabular / "source_summary.csv",
                ["source_id", "source_name", "ingest_role", "normalized_spectra", "failure_count", "parsers"],
                [
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "normalized_spectra": 4,
                        "failure_count": 0,
                        "parsers": "csv_row_wide:4",
                    }
                ],
            )
            write_csv(
                tabular / "spectra_metadata.csv",
                [
                    "source_id",
                    "source_name",
                    "ingest_role",
                    "spectrum_id",
                    "sample_name",
                    "input_path",
                    "parser",
                    "native_wavelength_count",
                    "native_min_nm",
                    "native_max_nm",
                    "native_spacing_nm",
                    "value_scale_applied",
                    "normalized_points",
                    "metadata_json",
                ],
                [
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src1_1",
                        "sample_name": "Sample 1",
                        "input_path": "src1.csv",
                        "parser": "csv_row_wide",
                        "native_wavelength_count": 2101,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1.0,
                        "value_scale_applied": 1.0,
                        "normalized_points": EXPECTED_GRID_POINTS,
                        "metadata_json": "{}",
                    }
                ],
            )

            summary = generate_quality_plots(root)
            self.assertEqual(summary["failure_rows"], 0)
            self.assertTrue(Path(summary["plots"]["failure_reasons"]).exists())


if __name__ == "__main__":
    unittest.main()
