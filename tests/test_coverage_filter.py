from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from spectral_library.normalization.coverage import filter_normalized_by_coverage


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CoverageFilterTests(unittest.TestCase):
    def test_filter_normalized_by_coverage_writes_filtered_dataset(self) -> None:
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
                        "normalized_spectra": 2,
                        "failure_count": 1,
                        "parsers": "csv_row_wide:2",
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "normalized_spectra": 1,
                        "failure_count": 0,
                        "parsers": "usgs_ascii:1",
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
                        "spectrum_id": "src1_keep",
                        "sample_name": "Keep",
                        "input_path": "src1_keep.csv",
                        "parser": "csv_row_wide",
                        "native_wavelength_count": 2101,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1.0,
                        "value_scale_applied": 1.0,
                        "normalized_points": 1900,
                        "metadata_json": "{}",
                    },
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src1_drop",
                        "sample_name": "Drop",
                        "input_path": "src1_drop.csv",
                        "parser": "csv_row_wide",
                        "native_wavelength_count": 2101,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1.0,
                        "value_scale_applied": 1.0,
                        "normalized_points": 1000,
                        "metadata_json": "{}",
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src2_keep",
                        "sample_name": "Keep 2",
                        "input_path": "src2_keep.csv",
                        "parser": "usgs_ascii",
                        "native_wavelength_count": 2101,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1.0,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": "{}",
                    },
                ],
            )
            write_csv(
                tabular / "normalized_spectra.csv",
                ["source_id", "spectrum_id", "sample_name", "nm_400", "nm_401"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_keep", "sample_name": "Keep", "nm_400": 0.1, "nm_401": 0.2},
                    {"source_id": "src1", "spectrum_id": "src1_drop", "sample_name": "Drop", "nm_400": 0.3, "nm_401": 0.4},
                    {"source_id": "src2", "spectrum_id": "src2_keep", "sample_name": "Keep 2", "nm_400": 0.5, "nm_401": 0.6},
                ],
            )
            write_csv(
                tabular / "wavelength_grid.csv",
                ["wavelength_nm", "column_name"],
                [{"wavelength_nm": 400, "column_name": "nm_400"}],
            )
            write_csv(
                tabular / "normalization_failures.csv",
                ["source_id", "input_path", "parser", "stage", "reason"],
                [{"source_id": "src1", "input_path": "bad.csv", "parser": "file_parse", "stage": "parse", "reason": "bad"}],
            )

            summary = filter_normalized_by_coverage(
                root / "normalized",
                root / "filtered",
                min_coverage=0.8,
            )

            self.assertEqual(summary["input_spectra"], 3)
            self.assertEqual(summary["retained_spectra"], 2)
            self.assertEqual(summary["written_normalized_rows"], 2)
            self.assertTrue((root / "filtered" / "tabular" / "wavelength_grid.csv").exists())
            self.assertTrue((root / "filtered" / "tabular" / "normalization_failures.csv").exists())

            with (root / "filtered" / "tabular" / "spectra_metadata.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["spectrum_id"] for row in rows], ["src1_keep", "src2_keep"])

            with (root / "filtered" / "tabular" / "normalized_spectra.csv").open("r", encoding="utf-8") as handle:
                spectra_rows = list(csv.DictReader(handle))
            self.assertEqual([row["spectrum_id"] for row in spectra_rows], ["src1_keep", "src2_keep"])

            with (root / "filtered" / "tabular" / "source_summary.csv").open("r", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["normalized_spectra"], "1")
            self.assertEqual(summary_rows[0]["failure_count"], "1")
            self.assertEqual(summary_rows[1]["normalized_spectra"], "1")

            payload = json.loads((root / "filtered" / "coverage_filter_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["retained_sources"], 2)

    def test_filter_normalized_by_coverage_rejects_invalid_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                filter_normalized_by_coverage(root, root / "out", min_coverage=1.2)


if __name__ == "__main__":
    unittest.main()
