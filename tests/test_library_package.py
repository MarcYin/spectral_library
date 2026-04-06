from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from spectral_library.normalization import build_library_package as normalization_build_library_package
from spectral_library.normalization.package import build_library_package
from spectral_library.sources.manifest import SourceRecord


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest_row(source_id: str, name: str, **overrides: str) -> dict[str, str]:
    row = {
        "source_id": source_id,
        "name": name,
        "section": "direct_public",
        "subsection": "soil",
        "spectral_type": "Reflectance",
        "coverage": "400-2500 nm",
        "resource_type": "Direct library",
        "provider": "manual",
        "landing_url": f"https://example.com/{source_id}",
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


class BuildLibraryPackageTests(unittest.TestCase):
    def test_normalization_package_exports_build_library_package(self) -> None:
        self.assertIs(normalization_build_library_package, build_library_package)

    def test_build_library_package_exports_tables_and_prototypes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            normalized_root = root / "normalized"
            output_root = root / "siac"

            _write_csv(
                manifest_path,
                list(SourceRecord.__dataclass_fields__.keys()),
                [
                    _manifest_row("src1", "Source 1"),
                    _manifest_row("src2", "Source 2"),
                    _manifest_row("src3", "Source 3", subsection="urban"),
                    _manifest_row("src4", "Unused Source"),
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "spectra_metadata.csv",
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
                        "spectrum_id": "src1_a",
                        "sample_name": "src1_a",
                        "input_path": "/tmp/src1_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "a"}),
                    },
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src1_b",
                        "sample_name": "src1_b",
                        "input_path": "/tmp/src1_b.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "b"}),
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src2_a",
                        "sample_name": "src2_a",
                        "input_path": "/tmp/src2_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "c"}),
                    },
                    {
                        "source_id": "src3",
                        "source_name": "Source 3",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src3_a",
                        "sample_name": "src3_a",
                        "input_path": "/tmp/src3_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "d"}),
                    },
                    {
                        "source_id": "src4",
                        "source_name": "Unused Source",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src4_a",
                        "sample_name": "src4_a",
                        "input_path": "/tmp/src4_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "e"}),
                    },
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "normalized_spectra.csv",
                ["source_id", "spectrum_id", "sample_name", "nm_400", "nm_401", "nm_402"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "nm_400": 0.1, "nm_401": 0.2, "nm_402": 0.3},
                    {"source_id": "src1", "spectrum_id": "src1_b", "sample_name": "src1_b", "nm_400": 0.3, "nm_401": 0.4, "nm_402": 0.5},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "nm_400": 0.9, "nm_401": 0.8, "nm_402": 0.7},
                    {"source_id": "src3", "spectrum_id": "src3_a", "sample_name": "src3_a", "nm_400": 0.6, "nm_401": 0.6, "nm_402": 0.6},
                    {"source_id": "src4", "spectrum_id": "src4_a", "sample_name": "src4_a", "nm_400": 0.05, "nm_401": 0.06, "nm_402": 0.07},
                ],
            )
            _write_csv(
                normalized_root / "landcover_analysis" / "landcover_labels.csv",
                ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src1", "spectrum_id": "src1_b", "sample_name": "src1_b", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src3", "spectrum_id": "src3_a", "sample_name": "src3_a", "landcover_group": "urban", "classification_rule": "rule"},
                ],
            )

            summary = build_library_package(manifest_path, normalized_root, output_root)

            self.assertEqual(summary["total_spectra"], 5)
            self.assertEqual(summary["classified_spectra"], 4)
            self.assertEqual(summary["unlabeled_spectra"], 1)
            self.assertEqual(summary["source_count"], 4)
            self.assertEqual(summary["labeled_source_count"], 3)
            self.assertEqual(summary["landcover_group_count"], 2)
            self.assertEqual(summary["prototype_rows"], 7)
            self.assertTrue((output_root / "README.md").exists())
            self.assertTrue((output_root / "tabular" / "siac_landcover_prototypes.csv").exists())
            self.assertTrue((output_root / "parquet" / "siac_normalized_spectra.parquet").exists())

            connection = duckdb.connect(str(output_root / "db" / "siac_spectral_library.duckdb"))
            try:
                counts = connection.execute(
                    """
                    SELECT
                      COUNT(*),
                      COUNT(DISTINCT source_id),
                      COUNT(*) FILTER (WHERE landcover_group IS NOT NULL),
                      COUNT(*) FILTER (WHERE landcover_group IS NULL)
                    FROM siac_spectra_metadata
                    """
                ).fetchone()
                self.assertEqual(counts, (5, 4, 4, 1))

                source_summary = connection.execute(
                    """
                    SELECT source_id, spectra_count, labeled_spectra_count, unlabeled_spectra_count
                    FROM siac_source_summary
                    WHERE source_id = 'src4'
                    """
                ).fetchone()
                self.assertEqual(source_summary, ("src4", 1, 0, 1))

                prototypes = connection.execute(
                    """
                    SELECT prototype_level, landcover_group, source_id, ROUND(nm_400, 4)
                    FROM siac_landcover_prototypes
                    WHERE landcover_group = 'soil'
                    ORDER BY prototype_level, COALESCE(source_id, '')
                    """
                ).fetchall()
                self.assertIn(("pooled", "soil", None, 0.4333), prototypes)
                self.assertIn(("source", "soil", "src1", 0.2), prototypes)
                self.assertIn(("source", "soil", "src2", 0.9), prototypes)
                self.assertIn(("source_balanced", "soil", None, 0.55), prototypes)

                wavelength_rows = connection.execute(
                    "SELECT band_name, wavelength_nm FROM siac_wavelength_grid ORDER BY band_index"
                ).fetchall()
            finally:
                connection.close()

            self.assertEqual(wavelength_rows, [("nm_400", 400), ("nm_401", 401), ("nm_402", 402)])

    def test_build_library_package_can_exclude_source_but_keep_metadata_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            normalized_root = root / "normalized"
            output_root = root / "siac"

            _write_csv(
                manifest_path,
                list(SourceRecord.__dataclass_fields__.keys()),
                [
                    _manifest_row("src1", "Source 1"),
                    _manifest_row("src2", "Source 2"),
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "spectra_metadata.csv",
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
                        "spectrum_id": "src1_a",
                        "sample_name": "src1_a",
                        "input_path": "/tmp/src1_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "a"}),
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src2_a",
                        "sample_name": "src2_a",
                        "input_path": "/tmp/src2_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "b"}),
                    },
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "normalized_spectra.csv",
                ["source_id", "spectrum_id", "sample_name", "nm_400", "nm_401", "nm_402"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "nm_400": 0.1, "nm_401": 0.2, "nm_402": 0.3},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "nm_400": 0.9, "nm_401": 0.8, "nm_402": 0.7},
                ],
            )
            _write_csv(
                normalized_root / "landcover_analysis" / "landcover_labels.csv",
                ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "landcover_group": "vegetation", "classification_rule": "rule"},
                ],
            )

            summary = build_library_package(
                manifest_path,
                normalized_root,
                output_root,
                exclude_source_ids=["src2"],
            )

            self.assertEqual(summary["total_spectra"], 1)
            self.assertEqual(summary["source_count"], 2)
            self.assertEqual(summary["spectra_source_count"], 1)
            self.assertEqual(summary["excluded_source_count"], 1)
            self.assertEqual(summary["excluded_spectra_count"], 1)
            self.assertEqual(summary["excluded_source_ids"], ["src2"])

            connection = duckdb.connect(str(output_root / "db" / "siac_spectral_library.duckdb"))
            try:
                included_sources = connection.execute(
                    "SELECT DISTINCT source_id FROM siac_spectra_metadata ORDER BY source_id"
                ).fetchall()
                self.assertEqual(included_sources, [("src1",)])

                excluded_row = connection.execute(
                    """
                    SELECT source_id, available_spectra_count, exclusion_reason
                    FROM siac_excluded_sources
                    """
                ).fetchone()
                self.assertEqual(excluded_row, ("src2", 1, "excluded_by_source_id"))

                source_summary = connection.execute(
                    """
                    SELECT
                      source_id,
                      included_in_spectra,
                      available_spectra_count,
                      spectra_count,
                      excluded_spectra_count,
                      exclusion_reason
                    FROM siac_source_summary
                    WHERE source_id = 'src2'
                    """
                ).fetchone()
                self.assertEqual(source_summary, ("src2", False, 1, 0, 1, "excluded_by_source_id"))

                manifest_rows = connection.execute(
                    """
                    SELECT source_id, included_in_spectra, available_spectra_count
                    FROM siac_manifest_sources
                    ORDER BY source_id
                    """
                ).fetchall()
                self.assertEqual(manifest_rows, [("src1", True, 1), ("src2", False, 1)])
            finally:
                connection.close()

    def test_build_library_package_can_exclude_individual_spectra_but_keep_exclusion_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            normalized_root = root / "normalized"
            output_root = root / "siac"
            excluded_spectra_csv = root / "manifests" / "excluded_spectra.csv"

            _write_csv(
                manifest_path,
                list(SourceRecord.__dataclass_fields__.keys()),
                [
                    _manifest_row("src1", "Source 1"),
                    _manifest_row("src2", "Source 2"),
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "spectra_metadata.csv",
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
                        "spectrum_id": "src1_a",
                        "sample_name": "src1_a",
                        "input_path": "/tmp/src1_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "a"}),
                    },
                    {
                        "source_id": "src1",
                        "source_name": "Source 1",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src1_b",
                        "sample_name": "src1_b",
                        "input_path": "/tmp/src1_b.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "b"}),
                    },
                    {
                        "source_id": "src2",
                        "source_name": "Source 2",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "src2_a",
                        "sample_name": "src2_a",
                        "input_path": "/tmp/src2_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 402,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "c"}),
                    },
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "normalized_spectra.csv",
                ["source_id", "spectrum_id", "sample_name", "nm_400", "nm_401", "nm_402"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "nm_400": 0.1, "nm_401": 0.2, "nm_402": 0.3},
                    {"source_id": "src1", "spectrum_id": "src1_b", "sample_name": "src1_b", "nm_400": 0.3, "nm_401": 0.4, "nm_402": 0.5},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "nm_400": 0.9, "nm_401": 0.8, "nm_402": 0.7},
                ],
            )
            _write_csv(
                normalized_root / "landcover_analysis" / "landcover_labels.csv",
                ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"],
                [
                    {"source_id": "src1", "spectrum_id": "src1_a", "sample_name": "src1_a", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src1", "spectrum_id": "src1_b", "sample_name": "src1_b", "landcover_group": "soil", "classification_rule": "rule"},
                    {"source_id": "src2", "spectrum_id": "src2_a", "sample_name": "src2_a", "landcover_group": "vegetation", "classification_rule": "rule"},
                ],
            )
            _write_csv(
                excluded_spectra_csv,
                ["source_id", "spectrum_id", "reason"],
                [{"source_id": "src1", "spectrum_id": "src1_b", "reason": "manual_review"}],
            )

            summary = build_library_package(
                manifest_path,
                normalized_root,
                output_root,
                exclude_spectra_csv=excluded_spectra_csv,
            )

            self.assertEqual(summary["total_spectra"], 2)
            self.assertEqual(summary["excluded_individual_spectra_count"], 1)
            self.assertEqual(summary["excluded_spectra_count"], 1)

            connection = duckdb.connect(str(output_root / "db" / "siac_spectral_library.duckdb"))
            try:
                included_ids = connection.execute(
                    "SELECT spectrum_id FROM siac_spectra_metadata ORDER BY spectrum_id"
                ).fetchall()
                self.assertEqual(included_ids, [("src1_a",), ("src2_a",)])

                excluded = connection.execute(
                    "SELECT source_id, spectrum_id, exclusion_reason FROM siac_excluded_spectra"
                ).fetchall()
                self.assertEqual(excluded, [("src1", "src1_b", "manual_review")])
            finally:
                connection.close()

    def test_build_library_package_uses_tail_stable_subset_for_urban_pooled_prototypes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            normalized_root = root / "normalized"
            output_root = root / "siac"

            _write_csv(
                manifest_path,
                list(SourceRecord.__dataclass_fields__.keys()),
                [
                    _manifest_row("urban_short", "Urban Short", subsection="urban"),
                    _manifest_row("urban_full", "Urban Full", subsection="urban"),
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "spectra_metadata.csv",
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
                        "source_id": "urban_short",
                        "source_name": "Urban Short",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "urban_short_a",
                        "sample_name": "urban_short_a",
                        "input_path": "/tmp/urban_short_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 2400,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "short"}),
                    },
                    {
                        "source_id": "urban_full",
                        "source_name": "Urban Full",
                        "ingest_role": "primary_raw",
                        "spectrum_id": "urban_full_a",
                        "sample_name": "urban_full_a",
                        "input_path": "/tmp/urban_full_a.csv",
                        "parser": "csv",
                        "native_wavelength_count": 3,
                        "native_min_nm": 400,
                        "native_max_nm": 2500,
                        "native_spacing_nm": 1,
                        "value_scale_applied": 1.0,
                        "normalized_points": 2101,
                        "metadata_json": json.dumps({"sample": "full"}),
                    },
                ],
            )
            _write_csv(
                normalized_root / "tabular" / "normalized_spectra.csv",
                ["source_id", "spectrum_id", "sample_name", "nm_400", "nm_401", "nm_402"],
                [
                    {"source_id": "urban_short", "spectrum_id": "urban_short_a", "sample_name": "urban_short_a", "nm_400": 0.9, "nm_401": 0.9, "nm_402": 0.9},
                    {"source_id": "urban_full", "spectrum_id": "urban_full_a", "sample_name": "urban_full_a", "nm_400": 0.2, "nm_401": 0.2, "nm_402": 0.2},
                ],
            )
            _write_csv(
                normalized_root / "landcover_analysis" / "landcover_labels.csv",
                ["source_id", "spectrum_id", "sample_name", "landcover_group", "classification_rule"],
                [
                    {"source_id": "urban_short", "spectrum_id": "urban_short_a", "sample_name": "urban_short_a", "landcover_group": "urban", "classification_rule": "rule"},
                    {"source_id": "urban_full", "spectrum_id": "urban_full_a", "sample_name": "urban_full_a", "landcover_group": "urban", "classification_rule": "rule"},
                ],
            )

            build_library_package(manifest_path, normalized_root, output_root)

            connection = duckdb.connect(str(output_root / "db" / "siac_spectral_library.duckdb"))
            try:
                pooled = connection.execute(
                    """
                    SELECT ROUND(nm_400, 4), spectra_count, source_count
                    FROM siac_landcover_prototypes
                    WHERE prototype_level = 'pooled' AND landcover_group = 'urban'
                    """
                ).fetchone()
                self.assertEqual(pooled, (0.2, 1, 1))

                balanced = connection.execute(
                    """
                    SELECT ROUND(nm_400, 4), spectra_count, source_count
                    FROM siac_landcover_prototypes
                    WHERE prototype_level = 'source_balanced' AND landcover_group = 'urban'
                    """
                ).fetchone()
                self.assertEqual(balanced, (0.2, 1, 1))

                source_rows = connection.execute(
                    """
                    SELECT prototype_level, source_id, ROUND(nm_400, 4)
                    FROM siac_landcover_prototypes
                    WHERE prototype_level = 'source' AND landcover_group = 'urban'
                    ORDER BY source_id
                    """
                ).fetchall()
                self.assertEqual(
                    source_rows,
                    [
                        ("source", "urban_full", 0.2),
                        ("source", "urban_short", 0.9),
                    ],
                )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
