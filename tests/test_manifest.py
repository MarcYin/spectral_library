from __future__ import annotations

import unittest
from pathlib import Path

from spectral_library.manifest import SourceRecord, filter_sources, load_manifest, manifest_sha256, split_csv_arg


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "manifests" / "sources.csv"


class ManifestTests(unittest.TestCase):
    def test_load_manifest_count(self) -> None:
        records = load_manifest(MANIFEST_PATH)
        self.assertEqual(len(records), 47)

    def test_filter_sources_planned_zenodo(self) -> None:
        records = load_manifest(MANIFEST_PATH)
        filtered = filter_sources(records, statuses=["planned"], adapters=["zenodo_api"])
        source_ids = {record.source_id for record in filtered}
        self.assertIn("ossl", source_ids)
        self.assertIn("slum", source_ids)
        self.assertNotIn("aster_v2", source_ids)

    def test_filter_sources_hits_all_filter_branches(self) -> None:
        records = [
            SourceRecord(
                source_id="a",
                name="A",
                section="direct_public",
                subsection="soil",
                spectral_type="soil",
                coverage="400-2500 nm",
                resource_type="Direct library",
                provider="zenodo",
                landing_url="https://example.com/a",
                download_url="",
                fetch_adapter="zenodo_api",
                auth_mode="public",
                expected_format="csv",
                tier="tier1",
                priority="high",
                ingest_role="primary_raw",
                normalization_eligibility="eligible_full",
                status="planned",
                notes="",
            ),
            SourceRecord(
                source_id="b",
                name="B",
                section="direct_public",
                subsection="soil",
                spectral_type="soil",
                coverage="400-2500 nm",
                resource_type="Direct library",
                provider="manual",
                landing_url="https://example.com/b",
                download_url="",
                fetch_adapter="manual_portal",
                auth_mode="manual_review",
                expected_format="csv",
                tier="tier2",
                priority="medium",
                ingest_role="primary_raw",
                normalization_eligibility="eligible_partial",
                status="review_required",
                notes="",
            ),
            SourceRecord(
                source_id="c",
                name="C",
                section="direct_public",
                subsection="soil",
                spectral_type="soil",
                coverage="400-2500 nm",
                resource_type="Direct library",
                provider="manual",
                landing_url="https://example.com/c",
                download_url="",
                fetch_adapter="manual_portal",
                auth_mode="manual_review",
                expected_format="csv",
                tier="tier1",
                priority="medium",
                ingest_role="primary_raw",
                normalization_eligibility="eligible_partial",
                status="review_required",
                notes="",
            ),
            SourceRecord(
                source_id="d",
                name="D",
                section="direct_public",
                subsection="soil",
                spectral_type="soil",
                coverage="400-2500 nm",
                resource_type="Direct library",
                provider="manual",
                landing_url="https://example.com/d",
                download_url="",
                fetch_adapter="zenodo_api",
                auth_mode="public",
                expected_format="csv",
                tier="tier2",
                priority="medium",
                ingest_role="primary_raw",
                normalization_eligibility="eligible_partial",
                status="review_required",
                notes="",
            ),
        ]
        filtered = filter_sources(
            records,
            source_ids=["b", "c", "d"],
            tiers=["tier2"],
            statuses=["review_required"],
            adapters=["manual_portal"],
        )
        self.assertEqual([record.source_id for record in filtered], ["b"])

    def test_split_csv_arg(self) -> None:
        self.assertEqual(split_csv_arg("tier1,tier2"), ["tier1", "tier2"])
        self.assertEqual(split_csv_arg(""), [])
        self.assertEqual(split_csv_arg(None), [])

    def test_manifest_hash_exists(self) -> None:
        self.assertEqual(len(manifest_sha256(MANIFEST_PATH)), 64)

    def test_specchio_manifest_row_uses_client_adapter(self) -> None:
        records = load_manifest(MANIFEST_PATH)
        specchio = next(record for record in records if record.source_id == "specchio_portal")
        self.assertEqual(specchio.fetch_adapter, "specchio_client")
        self.assertIn("specchio-client.jar", specchio.notes)

    def test_ecostress_manifest_row_uses_web_adapter(self) -> None:
        records = load_manifest(MANIFEST_PATH)
        ecostress = next(record for record in records if record.source_id == "ecostress_v1")
        self.assertEqual(ecostress.fetch_adapter, "ecostress_web")
        self.assertEqual(ecostress.expected_format, "txt_ascii")

    def test_emit_surface_manifest_row_uses_github_archive(self) -> None:
        records = load_manifest(MANIFEST_PATH)
        emit_surface = next(record for record in records if record.source_id == "emit_l2a_surface")
        self.assertEqual(emit_surface.fetch_adapter, "github_archive")
        self.assertIn("#surface", emit_surface.download_url)
        self.assertIn("kurudz_0.1nm.dat", emit_surface.notes)


if __name__ == "__main__":
    unittest.main()
