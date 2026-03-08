from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spectral_library.batch import (
    _classify_output_path,
    _retarget_artifact_paths,
    fetch_batch,
    seed_source_from_existing,
    tidy_source_directory,
)
from spectral_library.fetchers.base import FetchResult
from spectral_library.manifest import SourceRecord


def make_source(**overrides: str) -> SourceRecord:
    values = {
        "source_id": "src1",
        "name": "Source 1",
        "section": "direct_public",
        "subsection": "soil",
        "spectral_type": "Soil",
        "coverage": "400-2500 nm",
        "resource_type": "Direct library",
        "provider": "zenodo",
        "landing_url": "https://example.com/src1",
        "download_url": "",
        "fetch_adapter": "zenodo_api",
        "auth_mode": "public",
        "expected_format": "csv",
        "tier": "tier1",
        "priority": "high",
        "ingest_role": "primary_raw",
        "normalization_eligibility": "eligible_full",
        "status": "planned",
        "notes": "note",
    }
    values.update(overrides)
    return SourceRecord(**values)


def write_manifest(path: Path, rows: list[SourceRecord]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SourceRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


class TidySourceDirectoryTests(unittest.TestCase):
    def test_batch_helpers_cover_edge_cases(self) -> None:
        self.assertEqual(_classify_output_path(Path("record.json")), "metadata")
        self.assertEqual(_classify_output_path(Path("readme.pdf")), "docs")
        self.assertEqual(_classify_output_path(Path("spectra.csv")), "data")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            target = output_dir / "spectra.csv"
            target.write_text("x\n1\n", encoding="utf-8")
            payload = {
                "artifacts": [
                    "invalid",
                    {},
                    {"path": ""},
                    {"path": "/elsewhere/spectra.csv", "size_bytes": 0, "sha256": ""},
                ]
            }
            _retarget_artifact_paths(payload, output_dir)
            self.assertEqual(payload["artifacts"][3]["path"], str(target))
            self.assertGreater(payload["artifacts"][3]["size_bytes"], 0)
            self.assertTrue(payload["artifacts"][3]["sha256"])

            untouched_payload = {"artifacts": "not-a-list"}
            _retarget_artifact_paths(untouched_payload, output_dir)
            self.assertEqual(untouched_payload["artifacts"], "not-a-list")

    def test_tidy_moves_files_and_updates_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()
            metadata = source_dir / "record.json"
            data = source_dir / "spectra.csv"
            doc = source_dir / "readme.pdf"
            metadata.write_text('{"a":1}\n', encoding="utf-8")
            data.write_text("x,y\n1,2\n", encoding="utf-8")
            doc.write_bytes(b"%PDF-1.0")
            payload = {
                "source_id": "src1",
                "source_name": "Source 1",
                "fetch_adapter": "zenodo_api",
                "fetch_mode": "assets",
                "status": "downloaded",
                "landing_url": "https://example.com/src1",
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:01+00:00",
                "notes": [],
                "artifacts": [
                    {"artifact_id": "record", "kind": "metadata", "url": "", "path": str(metadata), "media_type": "application/json", "size_bytes": 0, "sha256": "", "status": "written", "note": ""},
                    {"artifact_id": "spectra", "kind": "data", "url": "", "path": str(data), "media_type": "text/csv", "size_bytes": 0, "sha256": "", "status": "downloaded", "note": ""},
                    {"artifact_id": "readme", "kind": "data", "url": "", "path": str(doc), "media_type": "application/pdf", "size_bytes": 0, "sha256": "", "status": "downloaded", "note": ""},
                ],
            }
            (source_dir / "fetch-result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            summary = tidy_source_directory(source_dir)

            self.assertEqual(summary["metadata_files"], 1)
            self.assertEqual(summary["data_files"], 1)
            self.assertEqual(summary["doc_files"], 1)
            self.assertTrue((source_dir / "metadata" / "record.json").exists())
            self.assertTrue((source_dir / "data" / "spectra.csv").exists())
            self.assertTrue((source_dir / "docs" / "readme.pdf").exists())
            result = json.loads((source_dir / "fetch-result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["artifacts"][0]["path"], str(source_dir / "metadata" / "record.json"))
            self.assertEqual(result["artifacts"][1]["path"], str(source_dir / "data" / "spectra.csv"))
            self.assertEqual(result["artifacts"][2]["path"], str(source_dir / "docs" / "readme.pdf"))

    def test_tidy_replaces_existing_destination_and_skips_invalid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            (source_dir / "data").mkdir(parents=True)
            (source_dir / "data" / "spectra.csv").write_text("old\n", encoding="utf-8")
            source_dir.mkdir(exist_ok=True)
            (source_dir / "spectra.csv").write_text("new\n", encoding="utf-8")
            payload = {
                "source_id": "src1",
                "source_name": "Source 1",
                "fetch_adapter": "zenodo_api",
                "fetch_mode": "assets",
                "status": "downloaded",
                "landing_url": "https://example.com/src1",
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:01+00:00",
                "notes": [],
                "artifacts": ["invalid", {"path": ""}, {"artifact_id": "spectra", "path": "/tmp/spectra.csv"}],
            }
            (source_dir / "fetch-result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            tidy_source_directory(source_dir)

            self.assertEqual((source_dir / "data" / "spectra.csv").read_text(encoding="utf-8"), "new\n")
            result = json.loads((source_dir / "fetch-result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["artifacts"][2]["path"], str(source_dir / "data" / "spectra.csv"))

    def test_tidy_counts_existing_nested_bucket_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            (source_dir / "data" / "nested").mkdir(parents=True)
            (source_dir / "metadata").mkdir(parents=True)
            (source_dir / "data" / "nested" / "a.bin").write_bytes(b"a")
            (source_dir / "data" / "nested" / "b.bin").write_bytes(b"b")
            (source_dir / "metadata" / "record.json").write_text('{"a":1}\n', encoding="utf-8")
            payload = {
                "source_id": "src1",
                "source_name": "Source 1",
                "fetch_adapter": "github_archive",
                "fetch_mode": "assets",
                "status": "downloaded",
                "landing_url": "https://example.com/src1",
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:01+00:00",
                "notes": [],
                "artifacts": [],
            }
            source_dir.mkdir(exist_ok=True)
            (source_dir / "fetch-result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            summary = tidy_source_directory(source_dir)

            self.assertEqual(summary["metadata_files"], 1)
            self.assertEqual(summary["data_files"], 2)
            self.assertEqual(summary["doc_files"], 0)


class SeedSourceTests(unittest.TestCase):
    def test_seed_source_from_existing_copies_and_retargets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_root = root / "seed"
            seed_dir = seed_root / "ghisa"
            seed_dir.mkdir(parents=True)
            data_path = seed_dir / "GHISA.csv"
            data_path.write_text("wavelength,reflectance\n400,0.1\n", encoding="utf-8")
            fetch_payload = {
                "source_id": "ghisa",
                "source_name": "Seed Name",
                "fetch_adapter": "earthdata_lpdaac",
                "fetch_mode": "assets",
                "status": "downloaded",
                "landing_url": "https://example.com/old",
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:01+00:00",
                "notes": ["Seeded local asset"],
                "artifacts": [
                    {
                        "artifact_id": "GHISA",
                        "kind": "data",
                        "url": "https://example.com/old",
                        "path": str(data_path),
                        "media_type": "text/csv",
                        "size_bytes": 0,
                        "sha256": "",
                        "status": "downloaded",
                        "note": "Seeded local asset",
                    }
                ],
            }
            (seed_dir / "fetch-result.json").write_text(json.dumps(fetch_payload, indent=2), encoding="utf-8")
            source = make_source(
                source_id="ghisa",
                name="GHISA",
                fetch_adapter="earthdata_lpdaac",
                landing_url="https://example.com/new",
                auth_mode="earthdata_login",
            )
            output_dir = root / "output" / "ghisa"

            seeded = seed_source_from_existing(source, output_dir, [seed_root])

            self.assertTrue(seeded)
            self.assertTrue((output_dir / "GHISA.csv").exists())
            payload = json.loads((output_dir / "fetch-result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["landing_url"], "https://example.com/new")
            self.assertEqual(payload["artifacts"][0]["path"], str(output_dir / "GHISA.csv"))
            self.assertIn("Copied from local seed directory", payload["notes"][-1])

    def test_seed_source_from_existing_builds_fetch_result_and_replaces_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_root = root / "seed"
            seed_dir = seed_root / "src1"
            (seed_dir / "nested").mkdir(parents=True)
            (seed_dir / "nested" / "seed.csv").write_text("x\n1\n", encoding="utf-8")
            output_dir = root / "output" / "src1"
            (output_dir / "nested").mkdir(parents=True)
            (output_dir / "nested" / "stale.csv").write_text("stale\n", encoding="utf-8")
            source = make_source(source_id="src1", landing_url="https://example.com/new-seed")

            seeded = seed_source_from_existing(source, output_dir, [root / "missing", seed_root])

            self.assertTrue(seeded)
            self.assertFalse((output_dir / "nested" / "stale.csv").exists())
            self.assertTrue((output_dir / "nested" / "seed.csv").exists())
            payload = json.loads((output_dir / "fetch-result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "metadata_only")
            self.assertEqual(payload["artifacts"], [])

    def test_seed_source_from_existing_returns_false_when_no_seed_dir_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = make_source(source_id="src_missing")

            seeded = seed_source_from_existing(source, root / "output" / "src_missing", [root / "seed"])

            self.assertFalse(seeded)


class FetchBatchTests(unittest.TestCase):
    def test_fetch_batch_fetches_seeds_and_records_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            output_root = root / "local_sources"
            seed_root = root / "seed"
            seed_dir = seed_root / "seeded_src"
            seed_dir.mkdir(parents=True)
            (seed_dir / "fetch-result.json").write_text(
                json.dumps(
                    {
                        "source_id": "seeded_src",
                        "source_name": "Seeded",
                        "fetch_adapter": "earthdata_lpdaac",
                        "fetch_mode": "assets",
                        "status": "downloaded",
                        "landing_url": "https://example.com/seed",
                        "started_at": "2024-01-01T00:00:00+00:00",
                        "finished_at": "2024-01-01T00:00:01+00:00",
                        "notes": [],
                        "artifacts": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (seed_dir / "seed.csv").write_text("x\n1\n", encoding="utf-8")
            rows = [
                make_source(source_id="downloaded_src", fetch_adapter="zenodo_api"),
                make_source(source_id="seeded_src", fetch_adapter="earthdata_lpdaac", auth_mode="earthdata_login"),
                make_source(source_id="broken_src", fetch_adapter="static_http"),
            ]
            write_manifest(manifest_path, rows)

            def fake_get_fetcher(name: str):
                if name == "zenodo_api":
                    def fetcher(source, output_dir, fetch_mode, user_agent):
                        del fetch_mode
                        del user_agent
                        (output_dir / "spectra.csv").write_text("x\n1\n", encoding="utf-8")
                        return FetchResult(
                            source_id=source.source_id,
                            source_name=source.name,
                            fetch_adapter=source.fetch_adapter,
                            fetch_mode="assets",
                            status="downloaded",
                            landing_url=source.landing_url,
                            started_at="2024-01-01T00:00:00+00:00",
                            finished_at="2024-01-01T00:00:01+00:00",
                            notes=[],
                            artifacts=[],
                        )

                    return fetcher
                if name == "static_http":
                    def broken_fetcher(source, output_dir, fetch_mode, user_agent):
                        del source, output_dir, fetch_mode, user_agent
                        raise RuntimeError("network failed")

                    return broken_fetcher
                raise AssertionError(name)

            with patch("spectral_library.batch.get_fetcher", side_effect=fake_get_fetcher):
                summary = fetch_batch(
                    manifest_path,
                    output_root,
                    fetch_mode="assets",
                    user_agent="ua",
                    continue_on_error=True,
                    seed_roots=[seed_root],
                    clean_output=True,
                    tidy_downloads=True,
                )

            self.assertEqual(summary["selected_sources"], 3)
            self.assertEqual(summary["seeded_sources"], 1)
            self.assertEqual(summary["status_counts"]["downloaded"], 2)
            self.assertEqual(summary["status_counts"]["error"], 1)
            self.assertTrue((output_root / "downloaded_src" / "data" / "spectra.csv").exists())
            self.assertTrue((output_root / "seeded_src" / "data" / "seed.csv").exists())
            self.assertTrue((output_root / "broken_src" / "metadata" / "fetch_error.json").exists())
            batch_summary = json.loads((output_root / "batch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(batch_summary["rows"]), 3)

    def test_fetch_batch_clean_output_without_tidy_and_raise_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            output_root = root / "local_sources"
            source_dir = output_root / "src1"
            source_dir.mkdir(parents=True)
            (source_dir / "stale.txt").write_text("old\n", encoding="utf-8")
            rows = [make_source(source_id="src1", fetch_adapter="zenodo_api")]
            write_manifest(manifest_path, rows)

            def fetcher(source, output_dir, fetch_mode, user_agent):
                del source, fetch_mode, user_agent
                (output_dir / "spectra.csv").write_text("x\n1\n", encoding="utf-8")
                return FetchResult(
                    source_id="src1",
                    source_name="Source 1",
                    fetch_adapter="zenodo_api",
                    fetch_mode="assets",
                    status="downloaded",
                    landing_url="https://example.com/src1",
                    started_at="2024-01-01T00:00:00+00:00",
                    finished_at="2024-01-01T00:00:01+00:00",
                    notes=[],
                    artifacts=[],
                )

            with patch("spectral_library.batch.get_fetcher", return_value=fetcher):
                summary = fetch_batch(
                    manifest_path,
                    output_root,
                    fetch_mode="assets",
                    user_agent="ua",
                    continue_on_error=False,
                    seed_roots=[],
                    clean_output=True,
                    tidy_downloads=False,
                )

            self.assertFalse((source_dir / "stale.txt").exists())
            self.assertTrue((source_dir / "spectra.csv").exists())
            self.assertEqual(summary["rows"][0]["data_files"], 0)

            def broken_fetcher(source, output_dir, fetch_mode, user_agent):
                del source, output_dir, fetch_mode, user_agent
                raise RuntimeError("boom")

            with patch("spectral_library.batch.get_fetcher", return_value=broken_fetcher):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    fetch_batch(
                        manifest_path,
                        output_root,
                        fetch_mode="assets",
                        user_agent="ua",
                        continue_on_error=False,
                        seed_roots=[],
                        clean_output=False,
                        tidy_downloads=False,
                    )

    def test_fetch_batch_raises_when_continue_on_error_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifests" / "sources.csv"
            output_root = root / "local_sources"
            write_manifest(manifest_path, [make_source(source_id="broken_src", fetch_adapter="static_http")])

            def broken_fetcher(source, output_dir, fetch_mode, user_agent):
                del source, output_dir, fetch_mode, user_agent
                raise RuntimeError("network failed")

            with patch("spectral_library.batch.get_fetcher", return_value=broken_fetcher):
                with self.assertRaises(RuntimeError):
                    fetch_batch(
                        manifest_path,
                        output_root,
                        fetch_mode="assets",
                        user_agent="ua",
                        continue_on_error=False,
                        seed_roots=[],
                        clean_output=True,
                        tidy_downloads=False,
                    )


if __name__ == "__main__":
    unittest.main()
