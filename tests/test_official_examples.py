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


class OfficialExamplesTests(unittest.TestCase):
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

            query_path = QUERIES_ROOT / "single" / "veg_soil_mix_landsat8_oli.csv"
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


if __name__ == "__main__":
    unittest.main()
