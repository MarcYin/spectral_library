from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from spectral_library import SensorSRFSchema, SpectralMapper, prepare_mapping_library, validate_prepared_library


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "official_mapping"
SRF_ROOT = EXAMPLES_ROOT / "srfs"
SIAC_ROOT = EXAMPLES_ROOT / "siac"
QUERIES_ROOT = EXAMPLES_ROOT / "queries"
METRICS_ROOT = EXAMPLES_ROOT / "results" / "metrics"
DOC_PATH = REPO_ROOT / "docs" / "official_sensor_examples.md"
COMMON_METRIC_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


class OfficialExamplesTests(unittest.TestCase):
    def test_official_source_manifest_records_hashes(self) -> None:
        payload = json.loads((EXAMPLES_ROOT / "official_source_manifest.json").read_text(encoding="utf-8"))
        with (SIAC_ROOT / "tabular" / "siac_spectra_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
            metadata_rows = list(csv.DictReader(handle))
        self.assertIn("generated_at_utc", payload)
        self.assertEqual(tuple(payload["comparison_band_ids"]), COMMON_METRIC_BANDS)
        self.assertIn("example_design", payload)
        self.assertEqual(payload["example_design"]["strategy"], "held_out_exact_library_spectra")
        self.assertEqual(len(payload["example_design"]["held_out_samples"]), 4)
        self.assertEqual(payload["example_design"]["self_exclusion_policy"], "exclude_matching_sample_name_only")
        self.assertEqual(len(payload["example_design"]["catalogue_samples"]), 10)
        self.assertEqual(
            payload["example_design"]["catalogue_samples"],
            [row["sample_name"] for row in metadata_rows],
        )
        self.assertEqual(len(payload["sensors"]), 4)

        for sensor in payload["sensors"]:
            self.assertIn("source_artifacts", sensor)
            self.assertGreaterEqual(len(sensor["source_artifacts"]), 1)
            for artifact in sensor["source_artifacts"]:
                self.assertIn("sha256", artifact)
                self.assertEqual(len(artifact["sha256"]), 64)
                self.assertGreater(artifact["size_bytes"], 0)
                self.assertIn("downloaded_at_utc", artifact)

    def test_official_sensor_json_examples_load(self) -> None:
        expected_band_ids = {
            "modis_terra": ("blue", "green", "red", "nir", "swir1", "swir2"),
            "sentinel2a_msi": ("ultra_blue", "blue", "green", "red", "nir", "swir1", "swir2"),
            "landsat8_oli": ("ultra_blue", "blue", "green", "red", "nir", "swir1", "swir2"),
            "landsat9_oli": ("ultra_blue", "blue", "green", "red", "nir", "swir1", "swir2"),
        }

        for sensor_id, band_ids in expected_band_ids.items():
            path = SRF_ROOT / f"{sensor_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema = SensorSRFSchema.from_dict(payload)
            self.assertEqual(schema.sensor_id, sensor_id)
            self.assertEqual(schema.band_ids(), band_ids)

    def test_official_example_fixture_prepares_and_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared_root = Path(tmpdir) / "prepared"
            prepare_mapping_library(
                siac_root=SIAC_ROOT,
                srf_root=SRF_ROOT,
                output_root=prepared_root,
                source_sensors=["modis_terra", "sentinel2a_msi", "landsat8_oli", "landsat9_oli"],
            )
            manifest = validate_prepared_library(prepared_root)
            self.assertEqual(manifest.row_count, 10)

            query_path = QUERIES_ROOT / "single" / "asphalt_landsat8_oli.csv"
            reflectance: dict[str, float] = {}
            with query_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    reflectance[row["band_id"]] = float(row["reflectance"])

            mapper = SpectralMapper(prepared_root, verify_checksums=True)
            result = mapper.map_reflectance(
                source_sensor="landsat8_oli",
                reflectance=reflectance,
                output_mode="target_sensor",
                target_sensor="sentinel2a_msi",
                k=3,
            )

            self.assertEqual(
                result.target_band_ids,
                ("ultra_blue", "blue", "green", "red", "nir", "swir1", "swir2"),
            )
            self.assertIsNotNone(result.target_reflectance)
            assert result.target_reflectance is not None
            self.assertEqual(len(result.target_reflectance), 7)
            self.assertEqual(set(result.neighbor_ids_by_segment), {"vnir", "swir"})

    def test_pairwise_metrics_and_generated_doc_are_in_sync(self) -> None:
        with (METRICS_ROOT / "pairwise_band_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
        with (EXAMPLES_ROOT / "results" / "selected" / "dense_vegetation_modis_to_sentinel2a.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            modis_rows = list(csv.DictReader(handle))

        self.assertTrue(metric_rows)
        for row in metric_rows:
            self.assertEqual(tuple(row["evaluated_band_ids"].split("|")), COMMON_METRIC_BANDS)
            self.assertEqual(int(row["evaluated_band_count"]), len(COMMON_METRIC_BANDS))

        lowest = min(metric_rows, key=lambda row: float(row["mean_abs_error"]))
        highest = max(metric_rows, key=lambda row: float(row["mean_abs_error"]))
        doc_text = DOC_PATH.read_text(encoding="utf-8")
        self.assertIn("Generated by scripts/build_official_mapping_examples.py", doc_text)
        self.assertIn(f"`{float(lowest['mean_abs_error']):.4f}` for {lowest['source_label']} -> {lowest['target_label']}", doc_text)
        self.assertIn(f"`{float(highest['mean_abs_error']):.4f}` for {highest['source_label']} -> {highest['target_label']}", doc_text)
        self.assertIn("held-out reconstruction on exact library spectra", doc_text)
        self.assertIn("common comparable subset", doc_text)
        self.assertIn("dense_vegetation", doc_text)
        self.assertIn("landsat8_to_sentinel2a_holdout_batch.csv", doc_text)
        self.assertIn("--self-exclude-sample-id", doc_text)
        self.assertIn("--exclude-sample-name dense_vegetation", doc_text)
        self.assertIn(f"`{float(modis_rows[0]['reflectance']):.4f}`", doc_text)
        self.assertIn("[Mathematical Foundations](theory.md)", doc_text)
        self.assertIn("spectral-library prepare-mapping-library \\\n  --siac-root", doc_text)
        self.assertIn("spectral-library map-reflectance-batch \\\n  --prepared-root", doc_text)


if __name__ == "__main__":
    unittest.main()
